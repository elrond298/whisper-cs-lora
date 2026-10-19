# Focused adversarial verification of Sol guard fixes

## Result

**F2–F4 are resolved within the audited scope. F1's original wrong-ID/stale-checkpoint reproductions are blocked, but F1 is not fully resolved against its stated duration-alignment requirement.** One low-severity omission remains; no additional regression was established. This does not invalidate the delivered results: current A/B/C files pass the new guards and their durations also align.

Read `SOL_REVIEW.md` F1–F4, `REVIEW_FIXES.md`, the actual `asr.py`/`summarize.py` changes and `test_pipeline.py`, and the recorded before/after test evidence. Used ripwire connected-symbol/caller/edit-check navigation and targeted source reads. No HEAD comparison was assumed.

## F1 — Partly resolved; duration identity omitted

`verified_test_results()` (`summarize.py:18–31`) now validates every group before `main()` writes metrics: exact official ID range, subset counts, cross-group reference/language identity, full-evaluation metadata, frozen configuration, completion budget and current adapter SHA-256. Independent temporary-fixture probes rejected foreign, duplicate and boolean IDs; changed references/languages; missing/extra rows; wrong group/split; partial-evaluation scope; and stale adapter identity. Every rejection preserved fixture bytes.

**Confirmed remaining issue [P3]:** `check_predictions()` (`asr.py:75–85`) compares only `(reference, language)`. Sol's F1 minimum explicitly required `(reference, language, duration)`. Reproduction using otherwise valid three-example A/B/C fixtures:

1. Assign duration `1.0` to every row in all groups.
2. Change only C's ID-0 duration to `999.0`.
3. Call `summarize.main()` with output writers intercepted.

Observed **return code 0 and `all_hard_gates_pass=true`**, including `official_test_identity_pass=true`. No model or generation was involved. Duration does not affect the current error-rate calculation, so this is an incomplete provenance check, not a demonstrated scoring error. The new tests' fixture rows omit duration, explaining the coverage gap.

**Smallest fix:** include duration in the expected per-ID tuple and prediction comparison; give fixtures real duration fields and add one mismatch rejection asserting no output writes. No retraining or prediction regeneration is needed. Real delivered A/B/C `(reference, language, duration)` mappings were separately confirmed equal.

## F2 — Resolved

`evaluation_identity()` validates the exact saved training config and its completion-marker hash; B/C require the frozen step count. `evaluate()` invokes it before loading prepared data, constructing processors/models or writing metadata. Nine independent negative probes covered changed model/dataset revisions, prefix mapping, learning rate, completion counts 0/50/799/801 and a wrong config hash. Mocks made any data/model load fatal; all nine failed at the guard without file changes. A correctly bound 50-step pilot remained eligible.

The 16-test suite additionally verified rejection of existing JSONL without metadata or with changed sample references. Source tracing confirms existing predictions undergo identity checks before metadata publication/model loading.

## F3 — Resolved

`preparation_identity()` binds dataset and model names/revisions plus `language_prompt`; `prepare()` checks existing ready splits before processor/dataset construction, and `load_split()` checks before opening manifests/features.

Independent fixtures changed each of the five identity inputs separately. Both preparation reuse and direct loading rejected all ten cases without side effects; an identity-less ready marker was also rejected. The ongoing real cache rebuild was neither opened for validation nor modified. Its final byte-for-byte comparison remains the main agent's responsibility.

## F4 — Resolved

`load_training_state()` runs before GPU checks, data loading or metadata writes. Independent probes rejected illegal groups, zero/negative/above-budget targets, short B/C targets, changed frozen config, saved-step overrun, inconsistent optimizer/scheduler counters and missing completion metadata. A consistent completed step-800 rerun returned with identical fixture bytes even when GPU/data access would raise. A legal step-400 checkpoint targeting 800 was accepted without writes; this verifies preflight, not actual resumed optimization. Existing tests also confirm CLI zero is preserved and rejected rather than replaced with 800.

## Evidence and limits

- Offline CPU command: `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest -v test_pipeline`: **16/16 passed**.
- Additional independent fixture probes: **44 rejected cases**, unchanged fixture files, plus the positive cases and confirmed duration counterexample above.
- Read-only real-artifact guard invocation accepted **1,315 rows per group**; scoring was mocked to avoid repeating experiment recomputation.

Only this report was written in the project; temporary fixtures were cleaned up. No training, full prediction generation, cache mutation, archive rebuild, GPU execution or real optimizer-state reload was performed. Saved-prediction metric recomputation is CPU-only with the pinned scoring dependencies, metadata and adapters; it does not require `data/`. Fresh generation requires separately prepared, identity-bound data/model assets and the evaluator's CUDA path. The archive excludes `data/`; archive-only checks therefore do not establish fresh-generation reproducibility. Non-gating ripwire warnings and the lack of a HEAD baseline are not evidence of behavioral correctness.
