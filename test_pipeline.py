import unittest
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

import asr
import summarize as report

from asr import read_json, sample_indices, ROOT
from metrics import collapse_flags, compare, edit_counts, normalize, summarize, tokens


class PipelineTests(unittest.TestCase):
    def test_mixed_tokenizer(self):
        self.assertEqual(tokens("我 LOVE 苹果。"), ["我", "love", "苹", "果"])
        self.assertEqual(tokens("can't 12 [UNK]"), ["can", "'", "t", "12", "[", "unk", "]"])
        self.assertEqual(tokens("嗯 um 啊 oh"), ["嗯", "um", "啊", "oh"])
        self.assertEqual(normalize("ＡＢＣ，Hello!"), "ａｂｃhello")

    def test_edit_denominator_and_empty(self):
        counts = edit_counts("你好 world", "你好 words extra")
        self.assertEqual(counts, {"substitutions": 1, "deletions": 0, "insertions": 1, "reference_units": 3})
        self.assertEqual(edit_counts("", "extra")["insertions"], 1)
        self.assertEqual(edit_counts("hello", "")["deletions"], 1)
        self.assertEqual(tokens("你好 hi", "cer"), list("你好hi"))

    def test_corpus_weighted_metric(self):
        rows = [{"language": "mixed", "reference": "一 二 三 四 five", "prediction": "一 二 三 四 five", "collapse_flags": []},
                {"language": "mixed", "reference": "hi", "prediction": "bye", "collapse_flags": []}]
        self.assertAlmostEqual(summarize(rows)["mixed"]["percent"], 100 / 6)
        self.assertIsNone(summarize(rows)["en"]["percent"])

    def test_collapse(self):
        self.assertIn("empty", collapse_flags("   "))
        self.assertIn("repetition", collapse_flags("你好 world " * 4))
        self.assertEqual(collapse_flags("hello", [0, 1, 2, 3], 1, {1, 9}), [])
        self.assertIn("language_prefix", collapse_flags("hello", [0, 9, 2, 3], 1, {1, 9}))
        self.assertIn("language_prefix", collapse_flags("hello", [0, 1, 2, 1], 1, {1, 9}))

    def test_sampling_distribution_and_reproducibility(self):
        config = read_json(ROOT / "config.json")
        rows = [{"language": lang} for lang in ["mixed"] * 27 + ["zh"] * 50 + ["en"] * 23]
        for group, target in (("B", .27), ("C", .5)):
            samples = sample_indices(rows, group, config, 50000)
            again = sample_indices(rows, group, config, 50000)
            self.assertTrue((samples == again).all())
            count = Counter(rows[i]["language"] for i in samples)
            self.assertLess(abs(count["mixed"] / len(samples) - target), .01)
            self.assertLess(abs(count["zh"] / count["en"] - 50 / 23), .1)


class ArtifactGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "artifacts"
        self.config = read_json(ROOT / "config.json")
        for module, changes in ((asr, {"ROOT": self.root, "OUT": self.out, "DATA": self.root / "data"}),
                                (report, {"ROOT": self.root, "OUT": self.out})):
            override = patch.multiple(module, **changes)
            override.start()
            self.addCleanup(override.stop)
        asr.save_json(self.root / "config.json", self.config)
        self.rows = [{"id": 0, "language": "zh", "reference": "你好"},
                     {"id": 1, "language": "en", "reference": "hello"},
                     {"id": 2, "language": "mixed", "reference": "你好 hello"}]
        for row in self.rows:
            row.update(duration=1.0, prediction=row["reference"], collapse_flags=[])
        asr.save_json(self.out / "data_audit.json", {"splits": {"test": {
            "n": 3, "counts": {"zh": 1, "en": 1, "mixed": 1}}}})
        for group in ("A", "B", "C"):
            folder = self.out / group
            folder.mkdir(parents=True)
            identity = {"config": self.config, "group": group, "split": "test", "limit_per_language": None}
            if group != "A":
                adapter = folder / "adapter" / "adapter_model.safetensors"
                adapter.parent.mkdir()
                adapter.write_bytes(group.encode())
                identity["adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
                asr.save_json(folder / "config.json", self.config)
                asr.save_json(folder / "training_complete.json", {
                    "completed_steps": 800, "config_sha256": asr.signature(self.config)})
            asr.save_json(folder / "test_eval_config.json", identity)
            self.write_predictions(group, self.rows)
        (self.root / "reports").mkdir()
        for group, steps in (("pilot", 50), ("B", 800), ("C", 800)):
            folder = self.out / group
            asr.save_json(folder / "initialization.json", {"adapter_sha256": "same", "gpu": "RTX 4060"})
            asr.save_json(folder / "progress.json", {"completed_steps": steps})
            asr.save_json(folder / "sampling.json", {"sample_count": steps * 8,
                "language_counts": {"mixed": steps * 2, "zh": steps * 4, "en": steps * 2},
                "actual_speech_hours": 0.1, "padded_input_hours": steps * 8 * 30 / 3600,
                "unique_examples": 3})
            events = [{"step": i + 1, "seconds": 1., "loss": 1.,
                       "peak_allocated_MiB": 1., "peak_reserved_MiB": 1.} for i in range(steps)]
            (folder / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))

    def write_predictions(self, group, rows):
        (self.out / group / "test_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_summary_accepts_complete_bound_predictions(self):
        self.assertEqual(report.main(), 0)
        comparison = read_json(self.out / "comparison.json")
        self.assertTrue(comparison["acceptance"]["all_hard_gates_pass"])

    def test_summary_rejects_wrong_ids_references_and_languages_before_writes(self):
        for field, value in (("id", 2000), ("id", 0), ("reference", "wrong reference"),
                             ("language", "zh"), ("duration", 999.0)):
            with self.subTest(field=field, value=value):
                rows = copy.deepcopy(self.rows)
                rows[2][field] = value
                self.write_predictions("C", rows)
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    report.main()
                self.assertEqual(before, self.snapshot())

    def test_summary_rejects_stale_checkpoint_or_decoding_identity(self):
        metadata = self.out / "C" / "test_eval_config.json"
        original = read_json(metadata)
        for field, value in (("adapter_sha256", "stale"), ("limit_per_language", 1)):
            with self.subTest(field=field):
                asr.save_json(metadata, {**original, field: value})
                before = self.snapshot()
                with self.assertRaises(RuntimeError):
                    report.main()
                self.assertEqual(before, self.snapshot())

    def test_evaluate_rejects_changed_training_config_before_loading_or_writing(self):
        (self.out / "B" / "test_eval_config.json").unlink()
        (self.out / "B" / "test_predictions.jsonl").unlink()
        for field, value in (("model_revision", "wrong"), ("max_new_tokens", 128)):
            with self.subTest(field=field), patch.object(asr, "load_split", side_effect=AssertionError("loaded data")):
                before = self.snapshot()
                with self.assertRaisesRegex(RuntimeError, "training config"):
                    asr.evaluate("B", "test", {**self.config, field: value})
                self.assertEqual(before, self.snapshot())

    def test_evaluate_rejects_unbound_or_changed_existing_predictions(self):
        path = self.out / "B" / "test_eval_config.json"
        identity = path.read_bytes()
        for missing_metadata in (False, True):
            with self.subTest(missing_metadata=missing_metadata):
                path.write_bytes(identity)
                rows = copy.deepcopy(self.rows)
                rows[0]["reference"] = "not the prepared reference"
                self.write_predictions("B", rows)
                if missing_metadata:
                    path.unlink()
                before = self.snapshot()
                with patch.object(asr, "load_split", return_value=(self.rows, None)), \
                     patch.object(asr, "processor", side_effect=AssertionError("loaded processor")):
                    with self.assertRaises((ValueError, RuntimeError)):
                        asr.evaluate("B", "test", self.config)
                self.assertEqual(before, self.snapshot())

    def test_cache_requires_matching_preparation_identity(self):
        folder = asr.DATA / "train"
        folder.mkdir(parents=True)
        asr.save_json(folder / "manifest.json", self.rows[:1])
        np.save(folder / "features.npy", np.zeros((1, 80, 3000), dtype=np.float16))
        identity = {key: self.config[key] for key in
                    ("dataset", "dataset_revision", "model", "model_revision", "language_prompt")}
        asr.save_json(folder / "ready.json", {"preparation_identity": identity})
        rows, features = asr.load_split("train", self.config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(features.shape, (1, 80, 3000))
        changed = copy.deepcopy(self.config)
        changed["language_prompt"]["mixed"] = "en"
        for wrong in ({**self.config, "dataset_revision": "wrong"},
                      {**self.config, "model_revision": "wrong"}, changed):
            with self.subTest(config=wrong):
                with self.assertRaisesRegex(RuntimeError, "preparation identity"):
                    asr.load_split("train", wrong)
                with patch.object(asr, "processor", side_effect=AssertionError("loaded processor")):
                    with self.assertRaisesRegex(RuntimeError, "preparation identity"):
                        asr.prepare(wrong)
        asr.save_json(folder / "ready.json", {"model_revision": self.config["model_revision"]})
        with self.assertRaisesRegex(RuntimeError, "preparation identity"):
            asr.load_split("train", self.config)

    def test_training_rejects_invalid_targets_before_writes(self):
        torch.save({"step": 800}, self.out / "B" / "resume.pt")
        for group, steps in (("B", 0), ("B", -1), ("B", 801), ("B", 50), ("A", 800)):
            with self.subTest(group=group, steps=steps), \
                 patch.object(asr, "load_split", side_effect=AssertionError("loaded data")), \
                 patch.object(asr.torch.cuda, "is_available", return_value=True):
                before = self.snapshot()
                with self.assertRaises((ValueError, RuntimeError)):
                    asr.train(group, steps, self.config)
                self.assertEqual(before, self.snapshot())

    def test_training_rejects_orphan_or_conflicting_completion_before_writes(self):
        folder = self.out / "B"
        (folder / "adapter" / "adapter_model.safetensors").unlink()
        (folder / "adapter").rmdir()
        for checkpoint_step in (None, 400):
            with self.subTest(checkpoint_step=checkpoint_step):
                if checkpoint_step is not None:
                    torch.save({"step": checkpoint_step}, folder / "resume.pt")
                before = self.snapshot()
                with patch.object(asr.torch.cuda, "is_available", side_effect=AssertionError("reached GPU")):
                    with self.assertRaises(RuntimeError):
                        asr.train("B", 800, self.config)
                self.assertEqual(before, self.snapshot())
        (folder / "training_complete.json").unlink()
        before = self.snapshot()
        self.assertEqual(asr.load_training_state("B", 800, self.config)["step"], 400)
        self.assertEqual(before, self.snapshot())

    def test_completed_training_is_a_real_noop(self):
        state = {"step": 800, "scheduler": {"last_epoch": 800},
                 "optimizer": {"state": {0: {"step": torch.tensor(800.)}}}}
        torch.save(state, self.out / "B" / "resume.pt")
        before = self.snapshot()
        with patch.object(asr.torch.cuda, "is_available", return_value=False):
            asr.train("B", 800, self.config)
        self.assertEqual(before, self.snapshot())

    def test_cli_does_not_replace_explicit_zero_steps_with_full_budget(self):
        with patch("sys.argv", ["asr.py", "train", "--group", "B", "--steps", "0"]), \
             patch.object(asr, "train") as train:
            asr.main()
        train.assert_called_once_with("B", 0, self.config)


class GenerationTests(unittest.TestCase):
    def test_whisper_generation_preserves_prefix(self):
        import torch
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        cfg = WhisperConfig(vocab_size=32, num_mel_bins=4, d_model=8,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
            decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
            max_source_positions=10, max_target_positions=20,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
        model = WhisperForConditionalGeneration(cfg).eval()
        gen = model.generation_config
        gen.is_multilingual = True
        gen.lang_to_id = {"<|en|>": 3, "<|zh|>": 6}
        gen.task_to_id = {"transcribe": 4}
        gen.no_timestamps_token_id = 5
        gen.suppress_tokens = []
        gen.begin_suppress_tokens = []
        with torch.inference_mode():
            sequence = model.generate(torch.zeros(1, 4, 20), language="en", task="transcribe",
                return_timestamps=False, max_new_tokens=2, return_dict_in_generate=True).sequences[0].tolist()
        self.assertEqual(sequence[:4], [1, 3, 4, 5])
        self.assertNotIn("language_prefix", collapse_flags("hello", sequence[:4], 3, {3, 6}))

    def test_verdict_respects_overall_threshold(self):
        def score(value):
            return {k: {"percent": value, "collapse_percent": 0} for k in ["overall", "mixed", "zh", "en"]}
        a, b, c = score(30), score(20), score(20)
        c["mixed"]["percent"] = 19
        self.assertEqual(compare(a, b, c)["verdict"], "oversampling_effective_single_seed")
        c["overall"]["percent"] = 20.4
        self.assertEqual(compare(a, b, c)["verdict"], "tradeoff")
        b["overall"]["percent"] = 31
        self.assertEqual(compare(a, b, c)["verdict"], "hard_gate_failed")


if __name__ == "__main__":
    unittest.main()
