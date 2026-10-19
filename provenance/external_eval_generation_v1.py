"""Frozen-weight external inference only; CPU scoring lives in external_scores.py."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import GenerationConfig

import asr
from metrics import collapse_flags
from prepare_external_fleurs import ROOT, sha256, write_json

DATASETS = ('cs_dialogue', 'fleurs_zh', 'fleurs_en')
OUT = ROOT / 'artifacts' / 'external_eval'


def load_manifest(dataset, verify_audio=False):
    if dataset not in DATASETS:
        raise ValueError('Unknown external dataset')
    folder = ROOT / 'data' / 'external_eval' / dataset
    manifest = folder / 'manifest.json'
    if json.loads((folder / 'ready.json').read_text()).get('manifest_sha256') != sha256(manifest):
        raise ValueError('Unbound or changed manifest')
    rows = json.loads(manifest.read_text())
    expected_language = {'cs_dialogue': 'mixed', 'fleurs_zh': 'zh', 'fleurs_en': 'en'}[dataset]
    ids = set()
    if not rows or len(rows) > (500 if dataset == 'cs_dialogue' else 300):
        raise ValueError('External sample count outside protocol')
    for row in rows:
        if not isinstance(row['id'], str) or row['id'] in ids or not row['id']:
            raise ValueError('Duplicate or invalid external ID')
        ids.add(row['id'])
        if row['language'] != expected_language or not row['reference'].strip() or not isinstance(row['cluster_id'], str) or not row['cluster_id']:
            raise ValueError('Invalid reference, language or cluster')
        source, revision = ('BAAI/CS-Dialogue', 'a84770fcdf14072252373a0227321227eaac7dc7') if dataset == 'cs_dialogue' else (
            'google/fleurs', '70bb2e84b976b7e960aa89f1c648e09c59f894dd')
        if row['dataset'] != source or row['revision'] != revision:
            raise ValueError('Wrong dataset source/revision')
        if dataset.startswith('fleurs') and row.get('config') != {'fleurs_zh': 'cmn_hans_cn', 'fleurs_en': 'en_us'}[dataset]:
            raise ValueError('Wrong FLEURS language configuration')
        if row['sample_rate'] != 16000 or type(row['frames']) is not int or not 0 < row['frames'] <= 480000 or not math.isclose(row['duration'], row['frames'] / 16000, abs_tol=1e-8):
            raise ValueError('Invalid duration/sample metadata')
        if not isinstance(row['audio_sha256'], str) or len(row['audio_sha256']) != 64 or any(v not in '0123456789abcdef' for v in row['audio_sha256']):
            raise ValueError('Invalid audio digest')
        audio_path = (ROOT / row['audio_path']).resolve()
        if not audio_path.is_relative_to(folder.resolve()) or Path(row['audio_path']).is_absolute():
            raise ValueError('Audio escapes its dataset directory')
        if verify_audio and sha256(audio_path) != row['audio_sha256']:
            raise ValueError('External audio changed')
    return rows


def ensure_schema(config):
    processor = asr.processor(config)
    generation = GenerationConfig.from_pretrained(config['model'], revision=config['model_revision'], local_files_only=True)
    schema = {'model': config['model'], 'model_revision': config['model_revision'],
              'eos_token_id': processor.tokenizer.eos_token_id,
              'decoder_start_token_id': generation.decoder_start_token_id,
              'task_token_id': generation.task_to_id['transcribe'],
              'language_ids': generation.lang_to_id}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'token_schema.json'
    if path.exists():
        if json.loads(path.read_text()) != schema:
            raise ValueError('Token schema changed')
    else:
        write_json(path, schema)
    return processor


def prediction_identity(dataset, group, config):
    schema = json.loads((OUT / 'token_schema.json').read_text())
    if schema['model'] != config['model'] or schema['model_revision'] != config['model_revision']:
        raise ValueError('Wrong token schema model')
    identity = asr.evaluation_identity(group, 'external_' + dataset, config)
    identity.update({'dataset': dataset, 'manifest_sha256': sha256(ROOT / 'data' / 'external_eval' / dataset / 'manifest.json'),
                     'inference_code_sha256': sha256(__file__), 'asr_code_sha256': sha256(ROOT / 'asr.py'),
                     'collapse_code_sha256': sha256(ROOT / 'metrics.py'), 'token_schema': schema})
    if group != 'A':
        identity['adapter_config_sha256'] = sha256(ROOT / 'artifacts' / group / 'adapter' / 'adapter_config.json')
    return identity


def sequence_flags(text, sequence, language, schema):
    expected_language = schema['language_ids'][f'<|{language}|>']
    flags = collapse_flags(text, sequence, expected_language, set(schema['language_ids'].values()),
                           hit_token_limit=schema['eos_token_id'] not in sequence)
    prefix = [schema['decoder_start_token_id'], expected_language, schema['task_token_id']]
    if sequence[:3] != prefix:
        flags.append('task_prefix')
    return flags


def read_predictions(path, references, require_complete=False, schema=None):
    predictions = []
    expected = {r['id']: r for r in references}
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            index = row['id']
            if index not in expected or index in seen:
                raise ValueError('Duplicate or foreign prediction ID')
            seen.add(index)
            reference = expected[index]
            for key in ('language', 'reference', 'duration', 'audio_sha256', 'cluster_id'):
                if row[key] != reference[key]:
                    raise ValueError('Prediction does not match frozen example')
            if not isinstance(row['prediction'], str) or not isinstance(row['token_ids'], list) or not row['token_ids'] or any(type(v) is not int for v in row['token_ids']):
                raise ValueError('Malformed prediction')
            if schema is not None:
                prompt = {'mixed': 'zh', 'zh': 'zh', 'en': 'en'}[row['language']]
                if row['collapse_flags'] != sequence_flags(row['prediction'], row['token_ids'], prompt, schema):
                    raise ValueError('Collapse flags do not match generated tokens')
            predictions.append(row)
    if require_complete and seen != set(expected):
        raise ValueError('Incomplete predictions')
    return predictions


def infer(dataset, group, smoke=False):
    config = asr.read_json(ROOT / 'config.json')
    rows = load_manifest(dataset, verify_audio=True)
    processor = ensure_schema(config)
    identity = prediction_identity(dataset, group, config)
    folder = OUT / dataset / group
    path, metadata = folder / 'predictions.jsonl', folder / 'identity.json'
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError('Changed inference identity: archive this external run first')
    if path.exists() and not metadata.exists():
        raise ValueError('Existing predictions lack identity')
    schema = identity['token_schema']
    predictions = read_predictions(path, rows, schema=schema)
    if len(predictions) == len(rows):
        print(f'{dataset}/{group}: already complete, verified, no GPU work', flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    write_json(metadata, identity)
    done = {r['id'] for r in predictions}
    pending = [r for r in rows if r['id'] not in done]
    if smoke:
        pending = [r for r in pending if r['id'] in {item['id'] for item in rows[:10]}]
        if not pending:
            print(f'{dataset}/{group}: smoke examples already verified', flush=True)
            return
    asr.seed_all(config['seed'])
    model = asr.base_model(config)
    if group != 'A':
        model = PeftModel.from_pretrained(model, ROOT / 'artifacts' / group / 'adapter')
    model.to(device='cuda', dtype=torch.float16).eval()
    generation = copy.deepcopy(model.generation_config)
    generation.forced_decoder_ids = None
    generation.suppress_tokens = []
    generation.begin_suppress_tokens = config['begin_suppress_tokens']
    generation.do_sample = False
    generation.num_beams = config['num_beams']
    generation.max_new_tokens = config['max_new_tokens']
    language = config['language_prompt'][rows[0]['language']]
    for start in range(0, len(pending), config['eval_batch_size']):
        batch = pending[start:start + config['eval_batch_size']]
        audios = []
        for row in batch:
            audio, rate = sf.read(ROOT / row['audio_path'], dtype='float32')
            if rate != 16000 or audio.ndim != 1 or len(audio) != row['frames'] or not np.isfinite(audio).all():
                raise ValueError('Audio shape/rate changed')
            audios.append(audio)
        inputs = processor.feature_extractor(audios, sampling_rate=16000, padding='max_length',
                                            max_length=480000, truncation=True, return_attention_mask=True, return_tensors='pt')
        with torch.inference_mode():
            sequences = model.generate(input_features=inputs.input_features.to(device='cuda', dtype=torch.float16),
                                       attention_mask=inputs.attention_mask.to(device='cuda'), generation_config=generation,
                                       language=language, task='transcribe', return_timestamps=False,
                                       return_dict_in_generate=True).sequences
        texts = processor.batch_decode(sequences, skip_special_tokens=True)
        generated_ids = sequences.cpu().tolist()
        with open(path, 'a', encoding='utf-8') as handle:
            for row, text, ids in zip(batch, texts, generated_ids):
                item = {key: row[key] for key in ('id', 'language', 'reference', 'duration', 'audio_sha256', 'cluster_id')}
                item.update({'prediction': text, 'token_ids': ids, 'collapse_flags': sequence_flags(text, ids, language, schema)})
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        completed = len(done) + min(start + len(batch), len(pending))
        if start == 0 or completed % 40 < config['eval_batch_size'] or completed == len(rows):
            print(f'{dataset}/{group}: {completed}/{len(rows)}', flush=True)
    read_predictions(path, rows, require_complete=not smoke, schema=schema)
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--groups', nargs='+', choices=('A', 'B', 'C'), default=['A', 'B', 'C'])
    parser.add_argument('--smoke', action='store_true', help='Only the first 10 frozen rows; normal execution resumes the same outputs')
    args = parser.parse_args()
    for dataset in args.datasets:
        for group in args.groups:
            infer(dataset, group, smoke=args.smoke)
