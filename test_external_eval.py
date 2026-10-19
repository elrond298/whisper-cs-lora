import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import external_eval as inference
from external_scores import paired_intervals, regression_verdict, score_row
from prepare_external_fleurs import write_json


SCHEMA = {'decoder_start_token_id': 0, 'task_token_id': 9, 'eos_token_id': 3,
          'language_ids': {'<|zh|>': 1, '<|en|>': 2}}


class ExternalEvaluationTests(unittest.TestCase):
    def test_standard_monolingual_metrics(self):
        for reference, prediction in [("Don't have eight state-of-the-art units", 'Do not have 8 state of the art units'),
                                      ('Colour is different', 'Colour is different')]:
            counts = score_row(reference, prediction, 'fleurs_en')
            self.assertEqual(sum(counts[k] for k in ('substitutions', 'deletions', 'insertions')), 0)
        english = score_row('red blue green', 'red green', 'fleurs_en')
        self.assertEqual((english['deletions'], english['reference_units']), (1, 3))
        chinese = score_row('你好，世界！', '你好世界', 'fleurs_zh')
        self.assertEqual(chinese, {'substitutions': 0, 'deletions': 0, 'insertions': 0, 'reference_units': 4})
        with self.assertRaises(ValueError):
            score_row('!!!', '', 'fleurs_en')

    def test_cluster_bootstrap_is_paired_deterministic_and_corpus_weighted(self):
        counts = [[1, 1, 1], [2, 2, 2], [4, 4, 4]]
        result = paired_intervals(counts, [10, 20, 40], ['x', 'x', 'y'])
        self.assertEqual(result, paired_intervals(counts, [10, 20, 40], ['x', 'x', 'y']))
        for comparison in result.values():
            self.assertEqual(comparison['delta_pp'], 0)
            self.assertEqual(comparison['ci95_pp'], [0, 0])
        result = paired_intervals([[0, 1, 2], [0, 9, 18]], [1, 9], ['x', 'y'])
        self.assertAlmostEqual(result['B_minus_A']['delta_pp'], 100)
        self.assertAlmostEqual(result['C_minus_A']['delta_pp'], 200)
        with self.assertRaises(ValueError):
            paired_intervals(counts, [10, 20, 40], ['x', 'x', 'x'])
        with self.assertRaises(ValueError):
            paired_intervals(counts, [10, 0, 40], ['x', 'x', 'y'])

    def test_uncertainty_is_not_equivalence(self):
        self.assertEqual(regression_verdict({'ci95_pp': [-1, 1]}, 0.5), 'inconclusive')
        self.assertEqual(regression_verdict({'ci95_pp': [-1, 0.5]}, 0.5), 'supports_within_tolerance')
        self.assertEqual(regression_verdict({'ci95_pp': [0.6, 1]}, 0.5), 'supports_degradation_beyond_tolerance')

    def test_prediction_binding_and_completeness(self):
        reference = {'id': 'fleurs:a', 'language': 'zh', 'reference': '你好', 'duration': 0.01,
                     'audio_sha256': 'a' * 64, 'cluster_id': '1'}
        prediction = {**reference, 'prediction': '你好', 'token_ids': [0, 1, 9, 10, 3], 'collapse_flags': []}
        with tempfile.TemporaryDirectory() as temporary, patch.object(inference, 'validation_tokenizer') as factory:
            factory.return_value.__len__.return_value = 20
            factory.return_value.decode.return_value = '你好'
            path = Path(temporary) / 'predictions.jsonl'
            def write(rows):
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            write([prediction])
            self.assertEqual(len(inference.read_predictions(path, [reference], True, SCHEMA)), 1)
            for key, value in [('id', 'foreign'), ('reference', '错误'), ('language', 'en'), ('duration', 1),
                               ('audio_sha256', 'b' * 64), ('cluster_id', '2'), ('collapse_flags', ['empty'])]:
                changed = {**prediction, key: value}
                write([changed])
                with self.assertRaises(ValueError, msg=key):
                    inference.read_predictions(path, [reference], True, SCHEMA)
            write([prediction, prediction])
            with self.assertRaises(ValueError):
                inference.read_predictions(path, [reference], True, SCHEMA)
            write([])
            with self.assertRaises(ValueError):
                inference.read_predictions(path, [reference], True, SCHEMA)

    def test_rejects_text_token_mismatch_and_invalid_vocabulary(self):
        reference = {'id': 'a', 'language': 'zh', 'reference': '你好', 'duration': 1,
                     'audio_sha256': 'a' * 64, 'cluster_id': 's'}
        prediction = {**reference, 'prediction': '你好', 'token_ids': [0, 1, 9, 10, 3], 'collapse_flags': []}
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.__len__.return_value = 20
        tokenizer.decode.return_value = '你好'
        with tempfile.TemporaryDirectory() as temporary, patch.object(inference, 'validation_tokenizer', return_value=tokenizer, create=True):
            path = Path(temporary) / 'p.jsonl'
            for change in ({'prediction': '完全无关'}, {'token_ids': [0, 1, 9, -1, 3]}, {'token_ids': [0, 1, 9, 1000000000, 3]}):
                path.write_text(json.dumps({**prediction, **change}) + '\n')
                with self.subTest(change=change), self.assertRaises(ValueError):
                    inference.read_predictions(path, [reference], True, SCHEMA)

    def test_validation_tokenizer_rejects_asset_corruption(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(inference, 'OUT', Path(temporary)):
            root = Path(temporary); folder = root / 'validation_tokenizer'; folder.mkdir()
            (folder / 'vocab.json').write_text('{}')
            model = {'model': 'pinned', 'model_revision': 'fixed'}
            (root / 'token_schema.json').write_text(json.dumps(model))
            metadata = {**model, 'files': {'vocab.json': inference.sha256(folder / 'vocab.json')}, 'vocab_size': 20}
            (root / 'validation_tokenizer.json').write_text(json.dumps(metadata))
            (folder / 'tokenizer.json').write_text('unlisted')
            with self.assertRaises(ValueError):
                inference.validation_tokenizer()
            (folder / 'tokenizer.json').unlink()
            (folder / 'vocab.json').write_text('changed')
            with self.assertRaises(ValueError):
                inference.validation_tokenizer()

    def test_legacy_identity_is_narrow_and_source_bound(self):
        expected = {'inference_code_sha256': 'current', 'config': {'fixed': 1}}
        legacy = {**expected, 'inference_code_sha256': inference.LEGACY_INFERENCE_SHA256}
        self.assertTrue(inference.identity_matches(expected, expected))
        self.assertTrue(inference.identity_matches(legacy, expected))
        self.assertFalse(inference.identity_matches({**legacy, 'config': {'fixed': 2}}, expected))
        self.assertFalse(inference.identity_matches({**legacy, 'inference_code_sha256': 'unapproved'}, expected))
        with tempfile.TemporaryDirectory() as temporary, patch.object(inference, 'ROOT', Path(temporary)):
            self.assertFalse(inference.identity_matches(legacy, expected))
            archived = Path(temporary) / 'provenance' / 'external_eval_generation_v1.py'
            archived.parent.mkdir()
            archived.write_text('changed')
            self.assertFalse(inference.identity_matches(legacy, expected))

    def test_partial_historical_run_cannot_mix_new_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'fleurs_zh' / 'A'
            folder.mkdir(parents=True)
            metadata = folder / 'identity.json'
            metadata.write_text(json.dumps({'inference_code_sha256': 'legacy'}))
            before = metadata.read_bytes()
            with patch.object(inference, 'OUT', Path(temporary)), patch.object(inference.asr, 'read_json', return_value={}), \
                 patch.object(inference, 'load_manifest', return_value=[{'id': 'a'}, {'id': 'b'}]), \
                 patch.object(inference, 'ensure_schema'), patch.object(inference, 'prediction_identity', return_value={'inference_code_sha256': 'new', 'token_schema': SCHEMA}), \
                 patch.object(inference, 'identity_matches', return_value=True), patch.object(inference, 'read_predictions', return_value=[{'id': 'a'}]), \
                 patch.object(inference.asr, 'base_model') as model:
                with self.assertRaisesRegex(ValueError, 'Historical generation'):
                    inference.infer('fleurs_zh', 'A')
                model.assert_not_called()
            self.assertEqual(metadata.read_bytes(), before)

    def test_language_task_eos_checks(self):
        self.assertEqual(inference.sequence_flags('你好', [0, 1, 9, 10, 3], 'zh', SCHEMA), [])
        self.assertIn('task_prefix', inference.sequence_flags('你好', [0, 1, 8, 10, 3], 'zh', SCHEMA))
        self.assertIn('language_prefix', inference.sequence_flags('你好', [0, 2, 9, 10, 3], 'zh', SCHEMA))
        self.assertIn('token_limit', inference.sequence_flags('你好', [0, 1, 9, 10], 'zh', SCHEMA))

    def test_manifest_hash_source_audio_and_path_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'data' / 'external_eval' / 'fleurs_zh'
            folder.mkdir(parents=True)
            audio = folder / 'a.wav'
            audio.write_bytes(b'example')
            row = {'id': 'fleurs:a', 'language': 'zh', 'reference': '你好', 'duration': 0.01,
                   'frames': 160, 'sample_rate': 16000, 'cluster_id': '1',
                   'audio_path': str(audio.relative_to(root)), 'audio_sha256': hashlib.sha256(b'example').hexdigest(),
                   'config': 'cmn_hans_cn', 'dataset': 'google/fleurs', 'revision': '70bb2e84b976b7e960aa89f1c648e09c59f894dd'}
            def write(rows):
                write_json(folder / 'manifest.json', rows)
                write_json(folder / 'ready.json', {'manifest_sha256': inference.sha256(folder / 'manifest.json')})
            with patch.object(inference, 'ROOT', root):
                write([row])
                self.assertEqual(len(inference.load_manifest('fleurs_zh', True)), 1)
                for key, value in [('duration', 1), ('language', 'en'), ('cluster_id', None), ('frames', 480001),
                                   ('audio_sha256', 'x' * 64), ('audio_path', '../escape.wav'),
                                   ('config', 'en_us'), ('dataset', 'wrong'), ('revision', 'wrong')]:
                    write([{**row, key: value}])
                    with self.assertRaises(ValueError, msg=key):
                        inference.load_manifest('fleurs_zh', True)
                write([row, row])
                with self.assertRaises(ValueError):
                    inference.load_manifest('fleurs_zh')
                write([row])
                (folder / 'manifest.json').write_text('[]')
                with self.assertRaises(ValueError):
                    inference.load_manifest('fleurs_zh')
                write([row])
                audio.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    inference.load_manifest('fleurs_zh', True)


if __name__ == '__main__':
    unittest.main()
