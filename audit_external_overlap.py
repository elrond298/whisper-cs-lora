"""Exact decoded-waveform overlap check against cached ASCEND; no near-duplicate claim."""
import argparse
import hashlib
import io
import json
from collections import defaultdict

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from external_eval import DATASETS, load_manifest
from prepare_external_fleurs import ROOT, sha256, write_json


def pcm_digest(wave, rate):
    wave = np.asarray(wave)
    if rate != 16000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Invalid canonical audio')
    canonical = np.clip(np.rint(wave.astype(np.float64) * 32768), -32768, 32767).astype('<i2')
    return hashlib.sha256(str((rate, len(wave))).encode() + b'\0' + canonical.tobytes()).hexdigest()


def audit(names):
    config = json.loads((ROOT / 'config.json').read_text())
    source = load_dataset(config['dataset'], revision=config['dataset_revision'])
    source = source.cast_column('audio', Audio(decode=False))
    known, counts = defaultdict(list), {}
    for split, records in source.items():
        counts[split] = len(records)
        for index, row in enumerate(records):
            info = row['audio']
            wave, rate = sf.read(io.BytesIO(info['bytes']) if info.get('bytes') else info['path'], dtype='float32')
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            known[pcm_digest(wave, rate)].append({'split': split, 'id': index})
    output = {'source': config['dataset'], 'revision': config['dataset_revision'], 'source_split_counts': counts,
              'canonicalization': '16kHz mono decoded waveform, rounded/clipped signed16 PCM, sample-count bound SHA256',
              'caveat': 'No exact decoded-waveform match does not exclude edited/resampled near-duplicates or base-model pretraining exposure',
              'subsets': {}}
    for name in names:
        rows = load_manifest(name, verify_audio=True)
        matches = []
        for row in rows:
            wave, rate = sf.read(ROOT / row['audio_path'], dtype='float32')
            digest = pcm_digest(wave, rate)
            if digest in known:
                matches.append({'external_id': row['id'], 'ASCEND_matches': known[digest]})
        output['subsets'][name] = {'n': len(rows), 'matches': matches,
                                    'manifest_sha256': sha256(ROOT / 'data' / 'external_eval' / name / 'manifest.json')}
    target = ROOT / 'reports' / 'EXTERNAL_OVERLAP_AUDIT.json'
    write_json(target, output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    if any(item['matches'] for item in output['subsets'].values()):
        raise RuntimeError('Exact overlap detected; review before claiming independent evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', choices=DATASETS, nargs='+', default=list(DATASETS))
    args = parser.parse_args()
    # Canonicalization self-check: equivalent float32/int16-derived signals match.
    wave = np.array([-1, -0.5, 0, 0.5, 32767 / 32768], dtype=np.float32)
    assert pcm_digest(wave, 16000) == pcm_digest(wave.astype(np.float64), 16000)
    assert pcm_digest(wave, 16000) != pcm_digest(wave[:-1], 16000)
    audit(args.datasets)
