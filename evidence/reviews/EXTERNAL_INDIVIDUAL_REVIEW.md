# Independent external-evaluation review

The following is the independent reviewer's final repair-verification record. The first-round finding, repair, original identities and numerical-preservation checks are summarized in [EXTERNAL_REVIEW_FIXES.md](EXTERNAL_REVIEW_FIXES.md).

## Result: repair confirmed; no remaining confirmed issue

I independently verified the final repair and provenance bridge. **The original P2 is resolved, and I found no confirmed blocker in the requested scope.**

### Prediction validation

In `external_eval.py`, `read_predictions()` now rejects:
- Prediction text inconsistent with decoding the stored tokens.
- Negative token IDs.
- Token IDs outside the tokenizer vocabulary.

I replayed all three original attacks using temporary copies and the **real exported tokenizer**, not a mocked decoder: **all rejected**.

All **3,150 actual predictions** pass the repaired validator. Vocabulary size is computed once per parsed file, rather than inside the per-token bounds check.

### Exported tokenizer integrity

`validation_tokenizer()` verifies:
- Model and revision against `token_schema.json`.
- Exact directory asset-name inventory.
- SHA256 of every declared asset.
- Loaded vocabulary size.

I tested temporary copies of the actual export:
- Corrupted each of the **six assets** independently: **all rejected**.
- Changed tokenizer metadata model/revision: **rejected**.
- Added a valid, unlisted `tokenizer.json` containing an altered decoder: **rejected by the final inventory check**.

### Provenance bridge

`identity_matches()` permits only exact current identity, or identity differing solely in `inference_code_sha256` to the explicitly approved historical hash, with the retained historical source matching that hash.

Confirmed:
- `provenance/external_eval_generation_v1.py` has exactly SHA256  
  `06611afbb60ee79d0d7a366f1b17d73c7a42b6c822210b46e155bdbfc680d01e`.
- All nine recorded historical identities pass.
- **102 independent top-level identity-field mutations** were rejected, including changed config, dataset, manifest binding, token schema, adapter hashes, and unapproved inference hashes.
- Missing or corrupted historical source causes rejection.
- New expected identities contain the current code hash; historical metadata is not relabelled.
- A real, temporarily copied **partial historical prediction file** was rejected before model creation or metadata writing.
- Complete historical reuse returned successfully with model creation and output writes explicitly forbidden.

The exception therefore does **not** relax the other model/config/dataset bindings. `external_scores.py` uses the same narrow identity check and still requires complete predictions.

### Tests and preservation

On the final code:
- **39 tests passed**, offline with CUDA hidden.
- Revalidated all **3,150 predictions**.
- Executed CPU rescoring with output writes intercepted in memory: all three datasets’ saved per-example scores and complete comparison objects—including confidence intervals—matched exactly.
- Snapshotted the nine identity files and nine prediction files before verification: **all 18 remained byte-unchanged during verification**.
- Root `SHA256SUMS`: **all 101 original files passed**.

No repository/documentation edits, downloads, training, or GPU inference were performed.

**Limit:** I did not retain byte hashes of the 18 external files from the first audit, so I cannot independently certify byte continuity across the intervening repair; I verified their unchanged state during this recheck, historical identity bindings, and current contents. No new methodological issue was identified in this repair-only review.
