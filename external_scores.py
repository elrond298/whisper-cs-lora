"""CPU-only external subset scoring; never changes inference or original ASCEND results."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from jiwer import process_words
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer

from metrics import edit_counts
from prepare_external_fleurs import ROOT, sha256, write_json

DATASETS = ('cs_dialogue', 'fleurs_zh', 'fleurs_en')
PAIRS = ((1, 0, 'B_minus_A'), (2, 0, 'C_minus_A'), (2, 1, 'C_minus_B'))
BASIC = BasicTextNormalizer()
ENGLISH = EnglishTextNormalizer({})
POLICY = {
    'cs_dialogue': 'Frozen ASCEND author-compatible jieba MER; no changes to original scorer',
    'fleurs_zh': 'Whisper BasicTextNormalizer; remove whitespace; character edits; no traditional/simplified conversion or number expansion',
    'fleurs_en': 'Whisper EnglishTextNormalizer with empty spelling-alias map; whitespace word edits; normalize references and hypotheses symmetrically',
    'bootstrap': '2000 paired cluster draws, percentile 95% CI, seed42; speaker clusters for CS, sentence-ID clusters for FLEURS',
}


def score_row(reference, prediction, dataset):
    if dataset == 'cs_dialogue':
        return edit_counts(reference, prediction, 'mer')
    normalizer = BASIC if dataset == 'fleurs_zh' else ENGLISH
    ref, hyp = normalizer(reference), normalizer(prediction)
    if dataset == 'fleurs_zh':
        ref, hyp = ' '.join(''.join(ref.split())), ' '.join(''.join(hyp.split()))
    if not ref.split():
        raise ValueError('Empty normalized reference')
    result = process_words(ref, hyp)
    return {'substitutions': result.substitutions, 'deletions': result.deletions,
            'insertions': result.insertions, 'reference_units': len(ref.split())}


def paired_intervals(errors, denominators, clusters, iterations=2000, seed=42):
    errors = np.asarray(errors, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    if (errors.shape != (len(denominators), 3) or len(clusters) != len(denominators)
            or not np.isfinite(errors).all() or not np.isfinite(denominators).all()
            or np.any(errors < 0) or np.any(denominators <= 0) or type(iterations) is not int or iterations <= 0):
        raise ValueError('Invalid paired score arrays')
    keys = sorted(set(clusters))
    if len(keys) < 2:
        raise ValueError('Need at least two clusters')
    grouped = []
    for key in keys:
        selected = np.array(clusters) == key
        grouped.append([*errors[selected].sum(axis=0), denominators[selected].sum()])
    grouped = np.asarray(grouped)
    rng = np.random.default_rng(seed)
    totals = grouped[rng.integers(0, len(keys), size=(iterations, len(keys)))].sum(axis=1)
    percentages = 100 * totals[:, :3] / totals[:, 3, None]
    point = 100 * errors.sum(axis=0) / denominators.sum()
    return {name: {'delta_pp': float(point[left] - point[right]),
                   'ci95_pp': np.quantile(percentages[:, left] - percentages[:, right], [0.025, 0.975]).tolist()}
            for left, right, name in PAIRS}


def regression_verdict(comparison, margin):
    low, high = comparison['ci95_pp']
    if high <= margin:
        return 'supports_within_tolerance'
    if low > margin:
        return 'supports_degradation_beyond_tolerance'
    return 'inconclusive'


def summarize_dataset(dataset):
    from external_eval import identity_matches, load_manifest, prediction_identity, read_predictions
    config = json.loads((ROOT / 'config.json').read_text())
    rows = load_manifest(dataset)
    aligned = {}
    for group in ('A', 'B', 'C'):
        folder = ROOT / 'artifacts' / 'external_eval' / dataset / group
        identity = prediction_identity(dataset, group, config)
        if not identity_matches(json.loads((folder / 'identity.json').read_text()), identity):
            raise ValueError(f'Prediction identity changed: {dataset}/{group}')
        predictions = read_predictions(folder / 'predictions.jsonl', rows, require_complete=True, schema=identity['token_schema'])
        aligned[group] = {r['id']: r for r in predictions}
    metrics, per_example, errors = {}, [], []
    denominators = []
    for row in rows:
        entry = {'id': row['id'], 'cluster_id': row['cluster_id'], 'groups': {}}
        row_errors = []
        for group in ('A', 'B', 'C'):
            prediction = aligned[group][row['id']]
            score = score_row(row['reference'], prediction['prediction'], dataset)
            score['errors'] = sum(score[key] for key in ('substitutions', 'deletions', 'insertions'))
            entry['groups'][group] = score
            row_errors.append(score['errors'])
        if len({v['reference_units'] for v in entry['groups'].values()}) != 1:
            raise ValueError('Mismatched paired denominator')
        denominators.append(entry['groups']['A']['reference_units'])
        errors.append(row_errors)
        per_example.append(entry)
    for group in ('A', 'B', 'C'):
        counts = Counter()
        for entry in per_example:
            counts.update(entry['groups'][group])
        metrics[group] = {**dict(counts), 'percent': 100 * counts['errors'] / counts['reference_units'],
                          'n': len(rows), 'collapse_percent': 100 * sum(bool(aligned[group][r['id']]['collapse_flags']) for r in rows) / len(rows)}
    differences = paired_intervals(errors, denominators, [r['cluster_id'] for r in rows])
    if dataset.startswith('fleurs'):
        margin = 0.5 if dataset == 'fleurs_zh' else 0.8
        for name in ('B_minus_A', 'C_minus_A'):
            differences[name]['tolerance_pp'] = margin
            differences[name]['verdict'] = regression_verdict(differences[name], margin)
    else:
        c = differences['C_minus_B']
        c['verdict'] = 'supports_improvement_on_fixed_subset' if c['ci95_pp'][1] < 0 else (
            'directional_improvement_only' if c['delta_pp'] < 0 else 'no_observed_improvement')
    folder = ROOT / 'artifacts' / 'external_eval' / dataset
    result = {'dataset': dataset, 'metric': {'cs_dialogue': 'MER', 'fleurs_zh': 'CER', 'fleurs_en': 'WER'}[dataset],
              'n': len(rows), 'clusters': len({r['cluster_id'] for r in rows}),
              'duration_seconds': sum(r['duration'] for r in rows), 'manifest_sha256': sha256(ROOT / 'data' / 'external_eval' / dataset / 'manifest.json'),
              'scoring_policy': POLICY[dataset], 'scoring_code_sha256': sha256(__file__),
              'metric_dependency_sha256': sha256(ROOT / 'metrics.py'), 'bootstrap_policy': POLICY['bootstrap'],
              'metrics': metrics, 'comparisons': differences,
              'caveat': 'Fixed-model subset estimates only, not training-seed uncertainty; CS speaker IDs are not verified dialogue IDs; FLEURS sentence clusters do not account for unavailable speaker IDs.'}
    write_json(folder / 'scores.json', per_example)
    write_json(folder / 'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    for name in args.datasets:
        summarize_dataset(name)
