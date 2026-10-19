# ASCEND / Whisper-small protocol sources

Research checked 2026-10-19. Primary sources only. No training was run; this report is the only repository file written by this research task.

## 1. Authoritative dataset and download format

**Use `CAiRE/ASCEND`, config `main`, revision `737e9800ae31be9932ba8464c80366559bd28424`.** The authors' repository links directly to this Hugging Face dataset:

- Authors' repository: https://github.com/HLTCHKUST/ASCEND
- Authoritative dataset: https://huggingface.co/datasets/CAiRE/ASCEND
- Pinned dataset card: https://huggingface.co/datasets/CAiRE/ASCEND/blob/737e9800ae31be9932ba8464c80366559bd28424/README.md
- Live repository metadata (includes `sha`, feature schema, counts, download size): https://huggingface.co/api/datasets/CAiRE/ASCEND
- Pinned file listing: https://huggingface.co/datasets/CAiRE/ASCEND/tree/737e9800ae31be9932ba8464c80366559bd28424/main

The current release is **Parquet**, not a legacy executable dataset-loading script. Files:

- `main/train-00000-of-00003.parquet`
- `main/train-00001-of-00003.parquet`
- `main/train-00002-of-00003.parquet`
- `main/validation-00000-of-00001.parquet`
- `main/test-00000-of-00001.parquet`

A direct download URL is `https://huggingface.co/datasets/CAiRE/ASCEND/resolve/737e9800ae31be9932ba8464c80366559bd28424/main/<filename>`. With Hugging Face Datasets, select the dataset ID, config, and pinned revision above; no `trust_remote_code` is needed for these Parquet files. The card declares **1,223,536,062 download bytes (~1.22 GB decimal)**.

Features: `id` (string), `path` (string), `audio` (Audio, **16,000 Hz**), `transcription` (string), `duration` (float32 seconds), `language` (string), `original_speaker_id` and `session_id` (int64), `topic` (string). Decode the `audio` feature, not the historical absolute filesystem paths in the separate `path` column. Current Datasets audio decoding uses TorchCodec/FFmpeg and may return an `AudioDecoder`, unlike older releases' `audio['array']` dictionary; pin and test the library interface actually used. `Audio(decode=False)` exposes stored audio bytes/path without this decoder.

Sources for audio loading/interface: https://huggingface.co/docs/datasets/en/audio_load and https://huggingface.co/docs/datasets/en/about_dataset_features

License declared in the dataset card: **CC-BY-SA-4.0**. Preserve attribution and review redistribution obligations if publishing data-derived artifacts.

### Official splits, hours, and language values

| Split | Current official HF utterances | Paper-reported hours | `zh` | `en` | `mixed` |
|---|---:|---:|---:|---:|---:|
| train | **9,869** | **8.78** | 4,799 | 2,331 | **2,739** |
| validation | **1,130** | **0.92** | 655 | 262 | 213 |
| test | **1,315** | **0.92** | 685 | 257 | 373 |
| total | **12,314** | **10.62** | 6,139 | 2,850 | 3,325 |

**The actual `language` strings are exactly `zh`, `en`, and `mixed`; not `chinese`, `english`, or `code-switching`.** Natural training has 2,739 / 9,869 = **27.7536% mixed utterances**.

Counts/schema: pinned dataset card above. Language frequencies are independently inspectable in the official dataset-viewer statistics:

- https://datasets-server.huggingface.co/statistics?dataset=CAiRE%2FASCEND&config=main&split=train
- https://datasets-server.huggingface.co/statistics?dataset=CAiRE%2FASCEND&config=main&split=validation
- https://datasets-server.huggingface.co/statistics?dataset=CAiRE%2FASCEND&config=main&split=test

These statistics endpoints follow the live dataset, not a revision parameter; snapshot their responses or recompute counts from the pinned payload for the experiment record.

**Paper discrepancy:** Table 4's validation *total row* says **1,129**, but its female/male validation rows are 484 + 646 = **1,130**, its corpus total is 12,314, and the HF release has 1,130. This appears to be an internal table-total typo; do not drop a validation item or invent a new split to match 1,129. Preserve the released splits.

Hours in the table above are the paper's rounded values, not a fresh audit of the pinned audio. Current viewer rounded mean `duration` values imply approximately **8.77483 / 0.92322 / 0.91749 h**; sum exact pinned durations for machine-readable run metadata rather than treating the paper's two-decimal hours as exact.

## 2. Original paper reference: **not Whisper-small**

- Published paper: https://aclanthology.org/2022.lrec-1.788/
- Paper PDF: https://aclanthology.org/2022.lrec-1.788.pdf
- Versioned HTML (Table 4, Section 4 annotation, Section 6 evaluation, Table 12): https://arxiv.org/html/2112.06223v6
- Submission history: https://arxiv.org/abs/2112.06223

ASCEND v1 was submitted **2021-12-12**, the final arXiv v6 **2022-05-03**, and LREC publication was **June 2022**. Whisper was publicly introduced **2022-09-21**: https://openai.com/index/whisper/

**There is no Whisper-small baseline in the original ASCEND paper.** Its Table 12 contains fine-tuned **wav2vec 2.0 CTC** models, with these full-split results:

| Pre-training language | Validation MER % | Validation CER % | Test MER % | Test CER % |
|---|---:|---:|---:|---:|
| Chinese | **30.37** | **25.72** | **27.05** | **22.69** |
| English | 35.77 | 28.07 | 28.72 | 22.78 |
| Multilingual | 35.30 | 28.68 | 29.35 | 24.31 |

**27.05% test MER is a wav2vec 2.0 reference, not a Whisper-small target or zero-shot score.** The paper fine-tunes the whole baseline using CTC, learning rate 5e-5, batch size 16, up to 100 epochs with early stopping patience 5, and a 3090-class GPU. Whisper-small LoRA has different pre-training, architecture, decoder, trainable parameter count, augmentation, and optimization. Treat comparison to the paper as contextual, not a controlled architecture- or LoRA-only comparison. A newly measured frozen Whisper-small score must be labeled a new baseline, not attributed to this paper.

## 3. Exact released scoring protocol and recommended normalization

The paper defines **mixed error rate (MER)** as Chinese-character error rate combined with word/token error rate for other text. Operational details are in the authors' code, pinned to commit **`8c50c2e2c65f8555eb1d7655310dd585dc1f6710`** (2022-07-12):

- Character removal and tokenizers: https://github.com/HLTCHKUST/ASCEND/blob/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/utils.py#L9-L28
- Evaluation preprocessing: https://github.com/HLTCHKUST/ASCEND/blob/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/eval.py#L91-L110
- Metric computation: https://github.com/HLTCHKUST/ASCEND/blob/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/eval.py#L155-L182
- Training preprocessing (uppercase rather than evaluation's lowercase): https://github.com/HLTCHKUST/ASCEND/blob/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/train.py#L147-L166
- Unpinned original requirements: https://github.com/HLTCHKUST/ASCEND/blob/8c50c2e2c65f8555eb1d7655310dd585dc1f6710/requirements.txt

### What that code actually does

1. Evaluation references: concatenate the entries of the exact `CHARS_TO_IGNORE` list below, escape them into a regex character class, remove matching characters, lowercase, and append a space. ASCII apostrophes, ASCII hyphens, and square brackets are **not** in this list; curly apostrophes and several Unicode dashes are. The released list does **not** define special atomic `[UNK]` removal.

```text
CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "\uFFFD", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]
```
2. For MER, call **`jieba.lcut(text)`**, discard whitespace-only tokens, retain each ASCII token intact (`tok.isascii()`), and split every non-ASCII token into individual characters. Compute Levenshtein edit distance directly between these token lists (or WER on their space-joined representations). Example: `我 love 苹果` becomes `我 | love | 苹 | 果`. This is not Chinese word segmentation for scoring: Chinese multi-character jieba tokens are split back into characters. Apostrophes/punctuation may form separate ASCII tokens; do not replace the official tokenizer with whitespace splitting while calling it an exact reproduction.
3. For CER, join all whitespace-separated text and split it into individual characters, including the English letters.
4. Aggregate the entire split as **100 × sum(substitutions + deletions + insertions) / sum(reference units)**, not mean per-utterance error rates.

**Do not call `jiwer.mer` to implement ASCEND MER.** Jiwer's `mer` means *match error rate*, with denominator including insertions, and is a different metric. The original code uses `editdistance.distance` directly on token lists, then divides summed distances by summed reference-token counts. Equivalent contemporary implementations can use word edit-distance/WER on those pretokenized strings. Primary Jiwer evidence: https://jitsi.github.io/jiwer/reference/measures/ and https://github.com/jitsi/jiwer/blob/master/src/jiwer/process.py (the `wer` and `mer` denominators differ).

### Recommendation for this Whisper experiment

Use one explicit **ASCEND-compatible lowercase/character-removal normalizer on both hypotheses and references**, then the released MER/CER tokenizers and corpus-level aggregation. Applying the reference cleanup to Whisper hypotheses is an intentional symmetric adaptation: the original CTC system's hypotheses came from a corpus-normalized vocabulary. Save raw hypotheses as well as scored normalized text; pin `jieba` and scoring-library versions and include a few tokenizer fixtures (mixed text, contractions, numbers, unknown markers, empty output).

Do **not** silently use Whisper's English normalizer, expand contractions, convert spoken number words to digits, delete fillers/repetitions, or convert Simplified/Traditional Chinese. Those are not operations in the released scoring code. Section 4 says numbers are written as words, contractions and repetitions are retained, fillers are annotated as `ah`/`oh`/`um`, and sentence punctuation is omitted. The code does not specify Unicode NFKC or Chinese script conversion. If a more linguistically forgiving normalizer is desired, report it as a separately named secondary metric, not as the paper-exact MER.

Exact published-number reproduction is not guaranteed: the paper gives a higher-level description, the released code has uppercase training versus lowercase evaluation, and dependencies were unpinned. Describe results as **ASCEND-compatible** unless the full original pipeline is reproduced and checked.

## 4. Current stable multilingual Whisper generation API

At research time, both PyPI and GitHub's latest non-prerelease report **Transformers 5.20.1** (release 2026-10-16):

- https://pypi.org/pypi/transformers/json
- https://api.github.com/repos/huggingface/transformers/releases/latest
- https://github.com/huggingface/transformers/releases/tag/v5.20.1
- Versioned Whisper docs: https://huggingface.co/docs/transformers/v5.20.1/en/model_doc/whisper

Versioned implementation is the authority for the following behavior:

- Language/task selection and legacy prefix handling: https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/whisper/generation_whisper.py#L1430-L1610
- Suppression processors: https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/whisper/generation_whisper.py#L1784-L1809
- Tokenizer training-label prefix API: https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/whisper/tokenization_whisper.py#L585-L655
- Checkpoint generation defaults: https://huggingface.co/openai/whisper-small/blob/main/generation_config.json

**Use the multilingual checkpoint `openai/whisper-small`, not `whisper-small.en`.** Prefer the supported generation controls **`language='zh'` (or `'chinese'`) and `task='transcribe'`**, with `return_timestamps=False` for transcript-only evaluation. `'translate'` is not the desired task. `mixed` is a dataset label, **not a Whisper language token**; a Chinese prefix does not restrict the output vocabulary to Chinese.

**`forced_decoder_ids` is a legacy generation-config fallback.** In 5.20.1 it is read only when neither language nor task is set. Explicit language/task controls take priority. Do not build a new implementation around historical `model.config.forced_decoder_ids` recipes or combine contradictory forced IDs with language/task arguments. For training labels, the tokenizer exposes `set_prefix_tokens(language=..., task='transcribe', predict_timestamps=False)`; this is separate from generation configuration.

**Keep the checkpoint's `suppress_tokens` and `begin_suppress_tokens` defaults unless intentionally testing a different decoding policy.** They remain supported and exclude non-speech/special output tokens; `suppress_tokens=[]` deliberately disables this suppression rather than modernizing the API. `whisper-small` currently declares `begin_suppress_tokens=[220, 50257]` and a nonempty suppression list. Keep settings identical between B and C.

For automatic language detection instead of a fixed Chinese prompt, ensure `generation_config.language` is actually unset and use the same transcription task policy in both runs. Passing `language=None` does **not** clear a previously configured language: the setter updates only non-`None` arguments. Record fixed-language versus auto-detection as part of the protocol; do not choose a test item's language from its ground-truth label without calling that an oracle condition.

These API facts are source-verified, **not a tested guarantee that Transformers 5.20.1 plus any particular PEFT/CUDA/TorchCodec stack fits or works on an RTX 4060 8 GB**. Pin the actual stack and smoke-test it; if using a different Transformers version, verify against that version rather than assuming this behavior is identical.

## 5. Minimum fair B-versus-C comparison

- **B:** unchanged natural training split. **C:** oversample only training rows whose `language == 'mixed'`; record the sampling weight, replacement policy, effective mixed fraction, and actual example exposures.
- Keep official validation and test splits untouched and naturally distributed. No oversampling, random resplitting, test-derived vocabulary, or test-based checkpoint choice.
- Match model initialization, LoRA targets/rank, data normalization, training-label prompt, seed, optimizer, decoding settings, checkpoint-selection rule, and hardware/precision. Prefer equal **optimizer-update budget**; equal epochs can hide a larger C training budget if its sampler draws more rows per epoch. Report both steps and exposures.
- Primary comparison: full official test-set corpus MER, with CER and mixed-only MER as clearly labeled secondary results. Do not compare mixed-only MER against the paper's full-test 27.05% MER.
- Select checkpoints/settings on validation, evaluate final test once per selected run, and distinguish a single-seed measured difference from a robust improvement. No speed, memory-fit, or MER claim has been experimentally verified by this research.
