# External evaluation: dataset feasibility and source audit

Planning research only. Checked **2026-10-19 UTC** against primary sources; no audio downloaded, no models/evaluation run, and no code, config or checkpoints changed. Counts and file sizes below were verified by direct public HTTP requests to the pinned repositories; only repository metadata, transcript/index text and papers were retrieved.

## Decision in brief

- **The proposed units are valid:** up to **500 CS-Dialogue mixed test utterances**, plus **300 Mandarin-labelled and 300 US-English-labelled FLEURS test recordings**, evaluated unchanged by A/B/C. Do not feed whole CS conversations into the short-utterance experiment.
- CS-Dialogue has **1,034 author-tagged `<MIX>` test utterances across all 30 test speakers** before a duration filter. Its release indices differ from the paper: use the pinned release counts below. Durations and the final number eligible at ≤30 seconds remain unverified without audio.
- **CS-Dialogue is the download bottleneck:** the published short-audio archive consists of 19 numbered parts totalling **9,595,542,742 bytes**. No independently addressable utterance files, Parquet audio or gzip seek index are published in the inspected tree. A 500-example evaluation does **not** imply a 500-file download. A small-byte, representative test subset cannot currently be promised.
- **FLEURS is feasible without the entire corpus or either complete language:** current official `main` has Parquet, and legacy split-specific archives remain available. Both full test archives total **815,197,822 bytes**; both test Parquet files total **1,097,396,719 bytes**. Those are bounded download options, not the size of the proposed 600 examples. Range-streaming Parquet may transfer less; exact network bytes are not established.
- FLEURS Mandarin is a **language-labelled**, not strictly Han-only, regression set: **164/945** normalized test references contain ASCII Latin letters, including names. One Mandarin recording is **31.12 s**; all English recordings are ≤30 s according to published sample counts. Apply the agreed duration rule explicitly; report the Latin-containing fraction rather than silently calling every Mandarin example purely monolingual.

## 1. CS-Dialogue

**Official source:** `BAAI/CS-Dialogue`, revision `a84770fcdf14072252373a0227321227eaac7dc7`. This is a new collection of spontaneous Mandarin–English dialogues, not an ASCEND repackaging: the authors describe 200 speakers, 100 two-person conversations, 200 participant recordings and 104.02 hours. Speakers were recruited across mainland China; recordings span seven topics. [CS1–CS3]

### Available splits and audio units

| Split | Published short-audio index: unique utterance IDs | Paper v2, Table 4 | Long-audio index entries |
|---|---:|---:|---:|
| `train` | **26,239** | 26,428 | 140 |
| `dev` | **6,186** | 6,196 | 30 |
| `test` | **6,257** | 6,293 | 30 |
| Total | **38,682** | 38,917 | 200 |

Release counts were independently counted from both `text` and `wav.scp`: each split has identical ID sets, no repeated IDs, and no ID intersection between splits. The information table also has 38,682 records and covers every test ID. **Do not substitute paper counts for downloadable release counts.** The reason for the discrepancy is not documented in the inspected sources. [CS2–CS4]

- `short_wav` contains already segmented **utterances**. Its `text` lines are `utterance_id transcript`; corresponding `wav.scp` lines are `utterance_id archive_relative_wav_path`.
- `long_wav` contains long participant recordings with associated TextGrid annotation paths; **30 long test entries are not 30 conversations**. The paper describes two participants per conversation. Do not infer an exact partner/dialogue mapping merely from adjacent speaker IDs.
- For diversity, stratify the short test pool by published speaker and topic; use a verified conversation/recording mapping if one is established later. The public information table does **not** have an explicit dialogue-ID column. No waveform durations or archive member positions are supplied in these text indices. [CS2–CS4]

### Actual annotation fields and scoring

The file is a **tab-separated text table**, despite its `.txt` extension: `data/index/total_information/Information_Index.txt`. Actual columns are `Speaker ID`, `Gender`, `Age`, `Region`, `Device`, `Topic`, `WAVE ID`, `TAG`, `SCRIPT`. Join `WAVE ID` to the index ID plus `.wav`. `TAG` is utterance-level `<CN>`, `<EN>` or `<MIX>`, not word-level language alignment. In test: **3,390 `<CN>` + 1,833 `<EN>` + 1,034 `<MIX>` = 6,257**. All 30 test speakers have mixed utterances. Use `<MIX>` for the proposed sampling pool rather than an unexplained ASCII/Chinese heuristic. The README's generic `audio`/`transcription` feature declarations are not a ready-made loaded dataset schema. [CS1, CS4]

Manual character-level Mandarin plus English-word transcripts are usable for **mixed error rate (MER: Mandarin character errors and English word errors on a combined token sequence)**. The paper itself evaluates CER/WER/MER. References preserve punctuation and annotation conventions, so use a frozen, documented normalization policy consistently for A/B/C. Paper Table 2 defines `**` as unintelligible speech and `<FIL/>`, `<SPK/>`, `<NON/>`, `<NPS/>` as filled pauses, speaker noises, non-speech noises and other-person noises. Exclude annotation markup from spoken-token scoring by a declared rule; do not count the literal tag names as recognized words. This audit did not listen to or validate test annotations. [CS1–CS2]

### Access, license and partial retrieval

- Official label: **CC BY-NC-SA 4.0**. Additionally, the card states academic/non-commercial use only; no re-identification; honour participant withdrawal; no sharing/redistribution to third parties; research derivatives must stay inside the research group without written permission. It lists access-request fields. **The public API currently says `gated: false`, but public retrieval is not permission to ignore those stated restrictions.** Confirm the intended research use is compliant; do not bundle external audio/transcripts into the existing deliverable. [CS1, CS3]
- Metadata-first selection is cheap: test `text` **810,268 B**, test `wav.scp` **466,155 B**, and the full information table **6,746,727 B**; combined **8,023,150 B**. The table's directory is actually `total_information`, not the README tree's misspelled `total_infomation`. [CS3–CS4]
- Short audio: `short_wav.tar.gz00`–`18`; **18 × 524,288,000 B + 158,358,742 B = 9,595,542,742 B** (about 9.60 GB / 8.94 GiB).
- Long audio: `long_wav.tar.gz00`–`30`; **30 × 524,288,000 B + 16,674,199 B = 15,745,314,199 B**. It is unnecessary for this short-utterance evaluation. [CS3]
- These names indicate multipart gzip archives, not demonstrated independent sampleable tar shards. **Archive contents/extraction behaviour were not tested**, because that would require audio retrieval. Conservatively budget all short-audio parts. A later streaming extraction could retain only chosen WAV files on disk, but must consume the compressed prefix through the last selected member; there is no verified bound below the complete archive and no public member-to-part map. `allow_patterns` cannot select unseen members inside compressed archive parts. If small network transfer is mandatory, obtain a maintainer-provided test subset or independently accessible utterances; do not promise random access to this release.

## 2. FLEURS: Mandarin Chinese and US English

**Current official repository:** `google/fleurs`, revision **`70bb2e84b976b7e960aa89f1c648e09c59f894dd`**. Exact config names are **`cmn_hans_cn`** and **`en_us`**. Current `main` declares Parquet data under `parquet-data/<config>/`; the older loader script was removed in the May 2026 conversion. Legacy `data/<config>/audio/{train,dev,test}.tar.gz` and TSV metadata remain. Do not assume the old script-only layout is still current. [F1–F4]

| Config | `train` recordings | `validation` recordings | `test` recordings | Distinct test text IDs |
|---|---:|---:|---:|---:|
| `cmn_hans_cn` | **3,246** | **409** | **945** | **349** |
| `en_us` | **2,602** | **394** | **647** | **350** |

Official card counts agree with direct test-TSV counts; test audio filenames are unique. The evaluation unit is **one read-speech recording of a sentence**, not a dialogue and not a unique sentence: several recordings share `id`. Avoid presenting repeated readings as independent texts; a paired uncertainty estimate can cluster by sentence ID. No speaker-ID column is available in the public schema. Audio is **16,000 Hz**. [F1, F4–F5]

**Fields:** `id` (sentence ID), `num_samples`, `path`, `audio`, **`transcription`** (normalized reference), **`raw_transcription`** (original reference), `gender`, `lang_id`, `language`, `lang_group_id`. The normalized text is lowercased and has some punctuation removed; do not assume every punctuation character disappears (English hyphens remain in examples). Mandarin characters are not space-separated, and Latin names can remain. Use `transcription` for the primary published-style score and retain raw text for inspection; Mandarin CER and English WER are the natural separate regression metrics. If reporting the project's MER as well, label it as the project's normalization/tokenization, not an official FLEURS MER benchmark. [F1, F4–F5]

The original TSV has seven tab-separated fields, interpreted by the legacy official script as ID, audio filename, raw transcription, normalized transcription, an unused character-level transcription field, sample count, and gender. `num_samples / 16000` permits the ≤30 s filter **before any audio download**. Mandarin has 944 eligible recordings by that rule; English has 647. Proposed 300+300 recording subsets are therefore feasible. [F4–F5]

### Retrieval choices and honest byte budgets

| Official test-only file | Exact published size |
|---|---:|
| `data/cmn_hans_cn/audio/test.tar.gz` | **525,346,466 B** |
| `data/en_us/audio/test.tar.gz` | **289,851,356 B** |
| Both legacy test archives | **815,197,822 B** |
| `parquet-data/cmn_hans_cn/test-00000-of-00001.parquet` | **695,674,033 B** |
| `parquet-data/en_us/test-00000-of-00001.parquet` | **401,722,686 B** |
| Both test Parquet files | **1,097,396,719 B** |

These are **file sizes from the pinned tree**, not card `num_bytes` (logical dataset size) or whole-config `download_size`. Both test TSVs together are only **859,351 B**. No audio requests were made to check range or compression behaviour. [F3–F5]

1. **Simple bounded plan:** choose seeded filename subsets from the TSVs, then retrieve only the two test archives and extract only chosen utterances. Network budget is ~815 MB even though only 600 recordings are evaluated/retained. No train/dev audio or other languages are needed. Archive member layout must be checked safely at retrieval time.
2. **Current Parquet route:** specify only the pinned test Parquet URL for each language through the generic `parquet` loader's `data_files`, with `streaming=True` if wanted. Explicit test files avoid downloading other splits/configs. Use `Audio(decode=False)` while selecting IDs or handling encoded bytes.
3. **Do not equate `take(300)` / `test[:300]` with 300 audio downloads.** Non-streaming selection normally happens after preparing input files; default config loading may prepare other splits too. Parquet streaming uses HTTP ranges/row groups, not individual audio objects. A randomly dispersed 300-ID selection may scan most/all of the single test shard. Prefix sampling can save bytes but is not a representative random sample. Actual row-group layout, client caching and transferred bytes remain unmeasured. [D1]

**Python compatibility:** Hugging Face Datasets **4.0.0 removed dataset scripts** and switched audio decoding to **TorchCodec**, with its FFmpeg/PyTorch compatibility requirements. Consequently, old pinned `fleurs.py` revisions can fail on `datasets>=4`; `trust_remote_code=True` does not restore removed script support. Conversely, current pinned Parquet does not require remote dataset code. Older `datasets` 3.6 uses SoundFile/libsndfile and normally returns NumPy audio dictionaries; v4 returns an AudioDecoder, so downstream code must not assume `audio['array']` works unchanged. Explicit Parquet with decode disabled, or native WAV extraction from the test archives, avoids accidental dependence on script execution/the new decoder. **The local installation and end-to-end loading were not tested by this research task**; the parent agent owns interface inspection. [D2–D4]

### Provenance, overlap and license

FLEURS is **read speech built on translated, parallel FLoRes-101 sentence prompts**, with roughly 12 hours per language and 102 languages; it is not spontaneous ASCEND conversation. The paper describes multiple readings of a prompt and speaker-disjoint splits. Mandarin/English test sets are parallel in text content, so they are not wholly independent domains. Official FLEURS license: **CC BY 4.0**; public API reports `gated: false`. [F1–F2]

No inspected primary source reports ASCEND audio reuse in either CS-Dialogue or FLEURS. CS-Dialogue cites ASCEND as a comparison dataset, not its recording source; ASCEND's official card describes a separate collection. **That is provenance evidence, not a completed overlap audit**: no audio hashes, participant identity matching, transcript deduplication against ASCEND, or Whisper pretraining-contamination check was performed. Describe this as external-corpus evaluation, not proven contamination-free evaluation. [CS2, A1, F2]

## Primary sources

- **CS1 — official card and access terms:** https://huggingface.co/datasets/BAAI/CS-Dialogue/blob/a84770fcdf14072252373a0227321227eaac7dc7/README.md
- **CS2 — authors' paper, Tables 2–4 and ASR experiments:** https://arxiv.org/html/2502.18913v2
- **CS3 — official repository API/tree (file sizes and paths):** https://huggingface.co/api/datasets/BAAI/CS-Dialogue and https://huggingface.co/api/datasets/BAAI/CS-Dialogue/tree/a84770fcdf14072252373a0227321227eaac7dc7/data?recursive=true&expand=false
- **CS4 — exact release indices:** https://huggingface.co/datasets/BAAI/CS-Dialogue/tree/a84770fcdf14072252373a0227321227eaac7dc7/data/index ; raw files are accessible through `/resolve/<revision>/data/index/...`, including `short_wav/{train,dev,test}/{text,wav.scp}` and `total_information/Information_Index.txt`.
- **F1 — current official card/configs/features/counts:** https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md and https://huggingface.co/api/datasets/google/fleurs
- **F2 — authors' FLEURS paper:** https://arxiv.org/abs/2205.12446 and official card above.
- **F3 — pinned file-size APIs:** https://huggingface.co/api/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd/parquet-data/cmn_hans_cn ; replace the final config with `en_us` for English. Legacy files: https://huggingface.co/api/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/cmn_hans_cn?recursive=true&expand=false and the corresponding `en_us` path.
- **F4 — official legacy parser and migration history:** https://huggingface.co/datasets/google/fleurs/blob/d7c758a6dceecd54a98cac43404d3d576e721f07/fleurs.py and https://huggingface.co/datasets/google/fleurs/commits/main
- **F5 — pinned test metadata (text only):** https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/cmn_hans_cn/test.tsv and https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/en_us/test.tsv
- **D1 — official streaming/Parquet documentation:** https://huggingface.co/docs/datasets/en/stream
- **D2 — official Datasets 4.0.0 release notes:** https://github.com/huggingface/datasets/releases/tag/4.0.0
- **D3 — current audio loading/decoding docs:** https://huggingface.co/docs/datasets/en/audio_load
- **D4 — versioned 3.6 audio decoder behaviour:** https://huggingface.co/docs/datasets/v3.6.0/en/audio_load
- **A1 — original ASCEND collection:** https://huggingface.co/datasets/CAiRE/ASCEND/blob/main/README.md
