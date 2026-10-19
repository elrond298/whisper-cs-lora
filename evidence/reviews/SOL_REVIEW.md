# Independent review: completed ASCEND Whisper-small LoRA experiment

## Result first

**The delivered numerical results and 800-update completion evidence withstand independent checks. I found four confirmed pipeline guard defects, but no confirmed scoring error, skipped-update counting error, split contamination, or invalid adapter in the delivered B/C experiment.** The four defects can mislabel or incorrectly accept a future rerun; none was observed to have corrupted the present results.

Read-only review: no project code was changed and no ASCEND training was restarted. This report is the only project file intentionally created. Tests, offline source-data checks, in-memory fault reproductions, a tiny synthetic AMP test, and six short checkpoint-reload generations were run. Existing reviewer conclusions were not used as substitutes for those checks. There is no usable HEAD baseline, so findings refer to the inspected working-tree files, whose existing 90-entry SHA256SUMS passed before this report was added.

**Recommended disposition:** fix the guards and add regression tests; retain the measured results and the carefully qualified scientific conclusion. Do not retrain merely to fix these guards. Rebuild the checksums and delivery archive after implementation and documentation edits.

Priorities below use P2 for a reproducible correctness/provenance defect worth fixing before another run or handoff. No P0/P1 defect affecting the completed experiment was established.

## Confirmed defects, in recommended fix order

### F1 — Final acceptance can certify the wrong test-example identities and does not bind predictions to their checkpoint [P2]

**Location:** `summarize.py:18–42`, especially `main()` at lines 22–29 and 31–39; output success at line 69.

**Cause → effect:** the summarizer checks uniqueness within each file and the total row count, but not that IDs are the official test IDs, that A/B/C refer to the same examples/references/language labels, or that prediction metadata describes the current adapter. It never reads `*_eval_config.json` or hashes the current adapters. Therefore a same-sized, internally unique but mismatched prediction file can still receive `all_hard_gates_pass=true` and exit status 0. Correct predictions left beside a replaced adapter can likewise be presented as that adapter's results; the evaluator's metadata guard does not protect a direct `summarize.py` invocation.

**Concrete reproduction, without disk edits:** intercepted only the reads of `artifacts/C/test_predictions.jsonl`, added 100000 to every C prediction ID, and intercepted `save_json`/`Path.write_text`. All texts, scores, training state and configuration were unchanged. Calling `summarize.main()` produced:

```text
C_ids_min: 100000
exit_code: 0
all_hard_gates_pass: true
eval_metadata_files_read: []
```

Those C IDs are outside the official prepared test IDs `0..1314`. The acceptance claim at line 29 is stronger than its count-only check.

**Minimal fix:** before writing any metrics or acceptance outputs:

1. Require the exact test ID set `set(range(audit['splits']['test']['n']))` for this frozen pipeline, not just cardinality. Require identical `id -> (reference, language, duration)` mappings across A/B/C, and match language counts to the audit.
2. Read each test evaluation identity; verify group/split/full-test scope and frozen config. For B/C, compare its recorded adapter SHA-256 with the adapter actually delivered. Reject stale/mismatched metadata rather than silently scoring it as the current run.
3. Do not publish partially refreshed result files before these checks finish. Currently lines 25–26 write per-group metrics before later validation can fail.

This does not need a new framework. The pipeline already produces the needed metadata, audit and hashes; the read-only review verified that the current files satisfy these stronger checks. If the dataset representation is later generalized beyond contiguous row IDs, replace the range check with a saved official ID/reference fingerprint, rather than weakening back to a count check.

**Regression tests:** reject a C file with shifted IDs; reject one changed reference or language with unchanged count; reject an adapter hash mismatch. Assert no result outputs are written on failure.

**Existing-result repair:** code/tests only. No present metric recomputation, generation or retraining is required. If an actual mismatched prediction file is discovered in another run, regenerate only its evaluation with the intended adapter, then recompute comparison outputs.

### F2 — Evaluation accepts a configuration that does not match the trained adapter [P2]

**Location:** `asr.py:270–306`, `evaluate()`, particularly lines 285–292 and the subsequent `base_model(config)`/adapter load.

**Cause → effect:** a non-A evaluation requires only that `training_complete.json` exists. It does not read that marker's contents or compare the requested config with `artifacts/<group>/config.json`. Its identity guard compares against an existing file for the *same evaluation output name*, not the adapter's training identity. On a fresh output name, changing the base-model revision can reach model loading and be recorded as an apparently valid evaluation of the existing adapter. A real, compatible-but-unintended base revision could silently change the model; an invalid revision would fail later for an unrelated reason. This is precisely the kind of pairing error the fixed-revision delivery should catch locally.

**Concrete reproduction, without generation or disk edits:** copied the config in memory, changed `model_revision` to a sentinel string, called `evaluate('B', 'test', wrong_config, limit=1)` so there was no old same-name metadata, intercepted metadata writes, and stopped at `processor()` before any model load. Observed:

```text
accepted_wrong_base_identity: true
saved_metadata_revision: different-base-model
STOP before any model load or prediction write
```

The existing B training config and completion marker were untouched. The sentinel tests the missing local guard; it is not a claim that a nonexistent revision can actually be downloaded.

**Minimal fix:** for B/C/pilot, read and validate the saved run config and completion marker before creating evaluation metadata. Under the documented immutable-config protocol, require config equality and `training_complete.config_sha256 == signature(config)`. For formal B/C require the expected 800 completed steps; permit the explicitly designated pilot's shorter completion. Then construct/load the base model from that verified identity. If alternate decoding experiments are desired later, separate decoding fields deliberately; do not silently treat the entire incoming config as the training truth.

**Regression tests:** fresh-output evaluation rejects an altered base revision or language-prefix mapping before loading a model or writing metadata; B/C rejects a short/invalid completion marker, while the 50-step pilot remains evaluable.

**Existing-result repair:** code/tests only. I checked that all delivered evaluation configs match the frozen config and that their adapter hashes match the current files. No metrics, predictions or training need replacement. A genuinely mispaired evaluation would need prediction regeneration and metric recomputation, not adapter retraining unless its training identity is itself invalid.

### F3 — Prepared-data cache can be silently reused under a different preparation identity [P2]

**Location:** `asr.py:60–75`, `prepare()`, especially the `ready.exists()` branch at lines 72–74; `load_split()` at lines 127–130.

**Cause → effect:** `ready.json` contains model/dataset revisions, but `prepare()` accepts it without comparing those revisions to the requested config. `load_split()` checks only its existence. Changing the dataset/model/prefix preparation settings while retaining `data/` can therefore train on old references/features/labels while new run metadata and even the top-level audit claim the new identity. The cache also does not record all label-defining inputs, notably `language_prompt`.

**Concrete reproduction, without disk edits:** changed the model and dataset revisions in a config copy, mocked dataset loading/processor construction, retained the real ready files, and captured the audit that `prepare()` would write. All three cached summaries still held:

```text
model_revision: 973afd24965f72e36ca33b3055d56a652f456b4d
dataset_revision: 737e9800ae31be9932ba8464c80366559bd28424
```

Yet the new top-level audit claimed `revision: different-dataset-revision`, and no stale-cache error was raised. A second attempt at this reproduction corrected the test harness's mock to implement `cast_column`; the initial harness AttributeError was not a project bug.

**Minimal fix:** save a preparation identity covering dataset ID/revision, model/processor ID/revision, and the label-prefix mapping. Compare it before reusing a ready cache and when training/evaluating load it. Fail with an instruction to use a fresh data directory or rebuild the cache; do not delete data automatically. For current older ready files, either explicitly verify/backfill their identity from the frozen configuration and existing evidence, or require a rebuild—never silently label them with arbitrary new settings.

**Regression tests:** stale dataset revision, model revision, or language mapping must be rejected both by `prepare()` reuse and by loading prepared data directly; unchanged identities must reuse the cache.

**Existing-result repair:** the completed experiment's cached data was independently compared to the pinned official payload (see verification section), so no present predictions, metrics or training need replacement. A future run actually trained with stale labels/data under a different claimed identity would need corrected preparation **and retraining**, followed by prediction/metric regeneration. Merely recomputing scores would not repair that run.

### F4 — A lower requested step target overwrites completed-run provenance without doing training [P2]

**Location:** `asr.py:184–204`, `train()`; resumed step at line 228; loop at line 237; unconditional completion write at line 267; CLI dispatch at line 359.

**Cause → effect:** the requested `steps` value is used to overwrite sampling metadata before the saved step is checked. If the checkpoint is already beyond the requested target, `range(first, steps)` is empty, but the code still overwrites `training_complete.json` with the lower requested target. The adapter/resume checkpoint remains at the old step while the sampling/completion files now describe a different run. The exposed `--steps` CLI has no group/range validation to prevent this.

**Concrete reproduction, without disk edits:** retained the real training rows/config, mocked the model/optimizer and all writes, supplied a resume state at step 800, and called `train('B', 50, config)`. No optimizer loop iteration ran. Captured writes included:

```text
sampling.json: sample_count=400, padded_input_hours=3.3333333333333335
training_complete.json: completed_steps=50
actual unmodified artifacts/B/resume.pt: step=800
```

Thus even a non-training invocation can make the delivered provenance contradictory. Targets beyond `config['steps']` are also not validated although only that many sample indices are generated; reject them rather than eventually reaching an empty batch.

**Minimal fix:** validate `0 < steps <= config['steps']` immediately. In `main()`, distinguish `args.steps is None` from zero instead of using `args.steps or config['steps']`, so `--steps 0` is rejected rather than silently becoming a full run. For this fixed experiment, the smallest public contract is to allow shorter `--steps` only for `pilot`, with B/C fixed to `config['steps']`. Read/check an existing saved step before overwriting any run metadata; reject a target below it, and make an equal-target completed rerun a genuine no-op after consistency validation. Write completed counts from the verified executed/restored state, not merely the requested integer.

**Regression tests:** step-800 resume + target 50 fails before any write; step-800 resume + target 800 preserves metadata; invalid zero/negative/above-budget targets fail immediately. Test a legal interrupted resume separately.

**Existing-result repair:** code/tests only. Both actual B/C checkpoints, completion files and sampling statistics are currently consistent at 800/6400; nothing was overwritten in this review. No present metric recomputation, generation or retraining is needed. If only metadata has been damaged elsewhere, restore it from verified checkpoint state and deterministic sampling before considering any retraining.

## Independent verification of the delivered experiment

### Scope actually read

Read the complete `asr.py` (365 lines), `metrics.py`, `summarize.py`, `test_pipeline.py`, `config.json`, `pyproject.toml`, and `README.md`. Read the complete relevant reports: `EXPERIMENT.md`, `protocol_sources.md`, `M1_BASELINE.md`, `M2_FEASIBILITY.md`, `B_GATE.md`, `B_INTERRUPTION.md`, `RESULTS.md`, `FINAL_REPORT.md`, `ERROR_ANALYSIS.md`, `VERIFICATION.md`, `environment.json`, `verification_details.json`, and all 407 lines of `error_analysis_40.json`. Inspected the saved configurations, prediction metadata, audits, sampling arrays, train logs, adapters and project-created resume states as needed.

Used ripwire from the repository root for ranked navigation and callers of `train`, `evaluate`, and `summarize`, then read the implementations and actual callers. No absence claim depends on zero ripwire results. Only trusted, locally project-created `artifacts/{pilot,B,C}/resume.pt` pickle checkpoints were loaded; adapter weights were inspected through safetensors.

### Tests, dependency lock, and integrity

- `python -m unittest -v test_pipeline` and a later quiet rerun: **7 tests passed**.
- `uv lock --check --offline`: passed.
- `sha256sum -c SHA256SUMS`: **90/90 entries passed** before adding this report.
- Pinned versions in the project are intentional. The research memo's discussion of newer Transformers APIs is not the runtime authority; actual checks used the installed pinned 4.53.3 stack.
- The archive was already independently verified before this review; I did not unpack/rebuild it again. Current source/checkpoint/metric checksums were rechecked here. This new report and future fixes will need a new manifest/archive; the old archive is not magically updated by local edits.

### Data, splits, labels, and features

Loaded the cached official dataset offline at the exact pinned release. Compared **all 12,314 examples** against the prepared manifests, not just report totals:

| Split | Rows | mixed / zh / en |
|---|---:|---|
| train | 9,869 | 2,739 / 4,799 / 2,331 |
| validation | 1,130 | 213 / 655 / 262 |
| test | 1,315 | 373 / 685 / 257 |

- Every prepared reference and language matched the official corresponding row.
- Decoded every stored waveform with soundfile: valid finite samples, 16 kHz, nonempty and at most 30 s. Exact waveform durations matched the manifests. Per-language duration sums reproduced the audit, including **3.6168703646 h** of train mixed audio.
- Regenerated every training/evaluation label sequence with the pinned processor and configured prefix mapping; all saved labels matched. Removing the initial decoder-start token from labels is appropriate because Whisper shifts decoder inputs internally.
- Checked all decoded waveform hashes: **zero exact waveform duplicate groups**, within or across the splits.
- Checked speaker sets: train/validation/test have **18/3/2 distinct speakers**, with **no pairwise overlap** in `original_speaker_id`. This supports the official speaker-independent split; it does not prove absence of all conceivable pretraining contamination.
- Cached feature arrays have shape `(n, 80, 3000)` and FP16 storage. Re-extracted the first/middle/last example of each split: all **9** feature arrays matched exactly after FP16 conversion, and saved frame-mask lengths matched the processor's attention masks. This was spot-checking of features, not a rehash/re-extraction of all feature values.
- The `[UNK]` language-inference discrepancies, paper-vs-release hours, and retained official language labels are disclosed, not grounds to invent a new split or silently clean references after seeing scores.

### Scoring and collapse handling

Independently fetched the authors' pinned `utils.py` and `eval.py` at commit `8c50c2e2c65f8555eb1d7655310dd585dc1f6710`:

- <https://raw.githubusercontent.com/HLTCHKUST/ASCEND/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/utils.py>
- <https://raw.githubusercontent.com/HLTCHKUST/ASCEND/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/eval.py>

Their MER token rule retains ASCII jieba tokens and splits other tokens into characters; their corpus error calculation sums edit distances and divides by summed reference tokens. The local implementation follows that rule with the explicitly documented symmetric normalization. It correctly does **not** call `jiwer.mer`, which denotes another measure. Contractions, retained ASCII brackets in `[UNK]`, numbers and fillers follow the frozen tokenizer, not a newly chosen cleanup. CER and overall MER need not have identical denominators on Chinese-labeled rows containing ASCII annotations; the report distinguishes them.

For every saved A/B/C test row and B/C validation row, independently checked ID coverage, reference/language/duration against the manifest, tokenizer decoding against the saved prediction, and collapse flags from the raw token IDs. Recomputed every saved metric and matched the JSON objects exactly.

| Group/split | Overall MER % | Mixed MER % |
|---|---:|---:|
| A test | 42.141152404 | 48.706431908 |
| B test | 16.067166723 | 15.918074021 |
| C test | 16.007500852 | 15.630614445 |
| B validation | 17.526507359 | 19.854227405 |
| C validation | 17.486944137 | 20.145772595 |

A has two affected test outputs, each flagged for repetition and token limit; B/C have no flagged validation/test outputs. The union, not the sum of overlapping flag counts, produces the stated collapse rate. Prefix tokens are retained in returned sequences and checked before decoding; saved token sequences support this. Zero flagged outputs is not a claim of zero recognition/semantic errors.

### Sampling and the accepted padded-input budget

Regenerated every saved sample-index array from the exact train manifest, seed and sampling function; all matched. Independently recomputed the consumed-prefix statistics:

| Quantity | B | C |
|---|---:|---:|
| Effective sample selections | 6,400 | 6,400 |
| mixed / zh / en | 1,812 / 3,085 / 1,503 | 3,220 / 2,147 / 1,033 |
| Unique examples | 4,698 | 4,437 |
| Actual speech hours | 5.6811437674 | 6.5310126215 |
| Padded-input hours | 53.3333333333 | 53.3333333333 |

Uniform-with-replacement B sampling and mixed-probability-0.5 C sampling are implemented correctly; C preserves uniformity among the pooled monolingual rows. The same numeric random stream does not make the two groups' selected examples identical, nor should it.

The user explicitly selected **8 × 30 padded seconds per effective update**. Actual speech exposure is about 15% higher for C; this is an interpretation limitation already disclosed, **not an equal-budget implementation defect**. Discarded interruption/overflow work also means physical computation was not exactly equal; the final report explicitly limits the matching claim to the effective-update padded-input budget.

### Optimization, resumption, and 800-step evidence

Independently loaded each trusted project-created resume state and compared every adapter tensor with the safetensors delivery:

| Run | Saved step | All 144 AdamW parameter counters | Scheduler last_epoch | Log rows / unique steps |
|---|---:|---:|---:|---|
| pilot | 50 | 50 | 50 | 50 / 50 |
| B | 800 | 800 | 800 | 841 / 800 |
| C | 800 | 800 | 800 | 800 / 800 |

All adapter tensors were finite and bitwise equal between each resume state and its delivered safetensors file. Logs covered every expected step and had finite recorded loss/gradient norms. B's extra **41** rows are replays, not evidence of 841 steps in the final optimizer trajectory. Scheduler state, optimizer counters and resume step are independent evidence beyond a completion JSON or line count. FP16 scaler growth trackers were 800 in both final states, with scale 65536; these are consistent with the claimed retained trajectories.

Rebuilt the seed-42 initial model/LoRA state from the pinned base. Its initialization digest was exactly `5d3bb69622c631477f8c113df9908c2aaa7da3c34c8cfc448f50c1630f86ae55`, matching both B/C records; counted **884,736** trainable parameters and **72** LoRA injection sites (144 A/B matrices), including decoder cross-attention. The frozen config digest reproduced `df98034d4fc7675394f54e46c7b1ec4a7939a395e8a86beb49583b8e91e1ca0f`.

Reviewed `optimizer_update()` (`asr.py:152–181`) and exercised it with a tiny synthetic CUDA model, not the ASCEND model. Starting loss scale 65536 caused ten forced overflow retries, then succeeded at scale 64. Exactly **one** optimizer step was counted. Parameters and post-update CPU/CUDA RNG states exactly matched a single low-scale successful attempt from the same initial parameters/random state. This supports the custom same-batch/same-randomness retry behavior, but the repository should retain an equivalent regression test rather than rely on this review's one-off check.

The B CUDA unknown error's external cause remains unproven. Saved RNG/optimizer/scheduler/scaler restoration and coherent final states support a valid continued run; they do not establish bitwise equivalence to an uninterrupted run on all CUDA kernels. Different debugging synchronization/shared loads invalidate a B-versus-C speed claim, which the final report correctly avoids.

### Checkpoint reload and actual generation

Freshly loaded each B/C adapter onto the pinned base through `PeftModel.from_pretrained`. Checked every loaded adapter tensor against its safetensors file. On the available CUDA GPU, regenerated the first test example of each language (`mixed: id 4`, `zh: id 0`, `en: id 2`) for both groups using the saved decoding policy. All **6/6 texts matched exactly**, and token IDs matched through EOS after ignoring only batch-padding EOS tokens. These were read-only short inference checks; no saved prediction file was appended.

The existing seven tests alone do not establish all of this. Conversely, the retained optimizer counters and live adapter-reload checks mean the completed-run evidence is materially stronger than its automatic acceptance script currently enforces.

### Forty-example analysis

Recreated the seed-42 sample of 40 C mixed test predictions: IDs, references and predictions matched both the review selection and analysis records. Recomputed **12 zero-edit examples**, **4 with `[UNK]`**, and the multi-label category counts. These checks validate the selection and textual accounting, not the truth of every acoustic interpretation. Calling it AI text-only review, rather than human listening, is correct.

## Scientific limitations and conclusions — not new code bugs

1. **Single seed, small C−B difference.** Test mixed errors fall from 886 to 870 (16 net errors over 5,566 units); overall errors improve by only 7. Validation mixed goes the other way, 19.854% → 20.146%. The data support the report's narrowly phrased directional observation, not robust superiority. No extra training is needed to report this observation honestly; additional seeds would be a new scientific scope.
2. **Oracle language prefixes.** Ground-truth language categories select zh/en prefixes. This is an explicitly stated experimental condition, not automatic language-detection performance. It can make results unlike a deployment that lacks those labels.
3. **Actual speech exposure is not matched.** C receives more true audio while effective padded inputs and update counts match. This prevents attributing the difference solely to language ratio under an equal true-speech budget; it does not violate the user's selected budget.
4. **Test gating.** B test scores are used to decide whether to proceed to C. This is not an untouched-holdout protocol even if no failing restart was triggered. The fixed final checkpoint and absence of a visible hyperparameter search reduce the concern but cannot erase it. The code and logs cannot prove that no unrecorded human/model choices were influenced by test outputs; retain the disclosure. A future confirmatory experiment should gate on validation, not revise this completed protocol retrospectively.
5. **Paper comparison.** The 27.05% paper value belongs to wav2vec 2.0, not Whisper. No exact Whisper paper-baseline replication claim should be restored. The current correction predates the documented formal training and is scientifically appropriate.
6. **Limited speaker population and pretraining provenance.** Only two official test speakers constrain generalization. Official split fidelity/no exact waveform duplicates do not establish absence of original Whisper pretraining overlap; that remains unverified, not an accusation of leakage in this pipeline.
7. **Frozen text normalization.** Traditional/simplified forms, fillers, English spelling and `[UNK]` affect absolute scores. No evidence supports changing the primary scorer after seeing the outcome. A separately labeled sensitivity analysis is possible later, but is not a defect fix or prerequisite for this delivery.
8. **Collapse is a fixed heuristic.** Its repetition window and special-prefix checks do not detect every semantically wrong/hallucinated output. Current docs make this distinction, including an explicit negation-error example.
9. **Acceptance wording.** The final report appropriately distinguishes preset directional acceptance from statistical proof. The machine label `oversampling_effective_single_seed` should always be shown with its caveat and the contrary validation result, not promoted as an unconditional win. The automatic acceptance script has the provenance defects above; the current result survives stronger independent validation.

## Important missing regression tests

Highest value additions, without a large test harness or any full finetuning in CI:

- The four guard-failure tests listed in F1–F4, using in-memory mocks and asserting that rejection happens before any writes/model loads.
- **AMP overflow/retry:** a tiny model or faithful scaler mock must show skipped attempts do not advance optimizer/scheduler/completed steps, the same batch and random state are replayed, and the persistent-overflow limit raises without recording success. Also cover a non-finite loss.
- **Interrupted resume parity:** a tiny deterministic run comparing uninterrupted execution with checkpoint/restoration of optimizer, scheduler, scaler and all RNGs. Check the consumed sample suffix and actual optimizer counters, not only completion markers. Document that production GPU kernels may still be nondeterministic.
- **Prediction-resume integrity:** missing IDs, duplicate IDs, changed reference/language and changed evaluation identity must reject before accepting results. Add a trailing-incomplete-JSONL-line policy test if crash-tolerant append recovery is promised; torn-line recovery was not established in this review.
- Boundary coverage for collapse: no-EOS token limit, expected language-token position, extra language token, union-of-flags counting, and a non-collapsed row. Existing tests cover several prefix/repetition cases but not the whole persisted-flag path.
- An opt-in, non-CI full-weight smoke check can load an adapter and compare a small saved prediction. It should not make routine unit tests download a base model or need a GPU.

These tests are more valuable than adding new abstraction layers. Existing metric/tokenizer/sampling tests are useful and should remain.

## Minimal implementation and documentation handoff

1. Fix F1–F4 in the existing pipeline modules and focused test file, with focused tests; do not change metric definitions, seed, sampling policy, or the trained adapters.
2. Rerun unit tests and the read-only identity/count/hash checks. For the present coherent artifacts, regenerate derived reports only as needed for changed acceptance fields; no recognition-score change is expected.
3. Perform the requested separate documentation UX pass. Keep a result-first summary, the test/validation contradiction, exact resume-versus-retrain behavior, the oracle-prefix disclosure, and a clear matrix of what can be reproduced on CPU versus GPU/network. Make the fixed-base-revision adapter-loading example explicit. This review does not substitute for the requested DeepSeek Flash pass.
4. Rebuild `SHA256SUMS` and the delivery archive after final changes, then verify a fresh extraction. Include this review if it is part of the intended handoff.

**Recompute / regenerate / retrain decision:** none of the four guard fixes requires changing the current measured metrics, regenerating predictions, or retraining the verified B/C adapters. For a different run that actually exercised the defect, F1/F2 can require evaluation regeneration; F3 can require retraining; F4 may need only metadata repair after checking the underlying optimizer state. Do not apply those expensive remedies speculatively to the completed experiment.

## Explicit limits

No full 800-step rerun, full validation/test regeneration, all-feature re-extraction, archive rebuild/extraction, human audio listening, multi-seed study, production deployment test, or resolution of the external CUDA fault was performed. No untrusted pickle was loaded. The current results are independently supported; future-run provenance is not sufficiently fail-closed until F1–F4 are fixed.
