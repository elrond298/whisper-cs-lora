"""Frozen corpus metrics: Chinese characters + English words, never sentence averages."""
import re
import jieba
from collections import Counter

from jiwer import process_words

CHARS_TO_IGNORE = ',?¿.!¡;；:"%\uFFFDʿ·჻~՞؟،।॥«»„“”「」‘’《》(){}=`_+<>…–°´ʾ‹›©®—→。、﹂﹁‧～﹏，｛｝（）［］【】‥〽『』〝〟⟨⟩〜：！？♪؛/\\º−^ʻˆ'
REMOVE = re.compile("[" + re.escape(CHARS_TO_IGNORE) + "]")
HAN = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"


def normalize(text):
    return REMOVE.sub("", text).lower()


def tokens(text, kind="mer"):
    text = normalize(text)
    if kind == "cer":
        return list("".join(text.split()))
    result = []
    for word in jieba.lcut(text):
        if word.strip():
            result.extend([word] if word.isascii() else list(word))
    return result


def edit_counts(reference, hypothesis, kind="mer"):
    ref, hyp = tokens(reference, kind), tokens(hypothesis, kind)
    aligned = process_words(" ".join(ref), " ".join(hyp))
    return {"substitutions": aligned.substitutions, "deletions": aligned.deletions,
            "insertions": aligned.insertions, "reference_units": len(ref)}


def collapse_flags(text, generated_ids=None, expected_language=None, language_ids=None,
                   hit_token_limit=False):
    flags = []
    units = tokens(text)
    if not units:
        flags.append("empty")
    # Fixed heuristic: any >=3-unit ngram repeated >=4 times consecutively.
    for size in range(3, len(units) // 4 + 1):
        if any(units[start:start + size] * 4 == units[start:start + size * 4]
               for start in range(len(units) - size * 4 + 1)):
            flags.append("repetition")
            break
    if generated_ids is not None and language_ids is not None:
        positions = [(i, token) for i, token in enumerate(generated_ids) if token in language_ids]
        if positions != [(1, expected_language)]:
            flags.append("language_prefix")
    if hit_token_limit:
        flags.append("token_limit")
    return flags


def summarize(rows):
    result = {}
    for subset in ("overall", "mixed", "zh", "en"):
        selected = [row for row in rows if subset == "overall" or row["language"] == subset]
        kind = "cer" if subset == "zh" else "mer"
        counts = Counter()
        failures = Counter()
        for row in selected:
            counts.update(edit_counts(row["reference"], row["prediction"], kind))
            failures.update(row["collapse_flags"])
        denominator = counts["reference_units"]
        errors = sum(counts[k] for k in ("substitutions", "deletions", "insertions"))
        result[subset] = {"metric": {"zh": "CER", "en": "WER"}.get(subset, "MER"),
                          "percent": 100 * errors / denominator if denominator else None,
                          "n": len(selected), **dict(counts),
                          "collapse_percent": 100 * sum(bool(r["collapse_flags"]) for r in selected) / len(selected)
                          if selected else None, "collapse_counts": dict(failures)}
    return result


def compare(a, b, c):
    delta = {key: c[key]["percent"] - b[key]["percent"] for key in c}
    hard = (b["overall"]["percent"] <= a["overall"]["percent"]
            and b["overall"]["collapse_percent"] < 5
            and c["overall"]["collapse_percent"] < 5)
    improved = delta["mixed"] < 0
    mono_ok = delta["zh"] <= 0.5 and delta["en"] <= 0.8
    overall_ok = delta["overall"] <= 0.3
    if not hard:
        verdict = "hard_gate_failed"
    elif improved and mono_ok and overall_ok:
        verdict = "oversampling_effective_single_seed"
    elif improved:
        verdict = "tradeoff"
    else:
        verdict = "no_detectable_benefit_or_worse"
    return {"C_minus_B_percentage_points": delta, "metric_hard_gates_pass": hard,
            "mixed_improved": improved, "monolingual_thresholds_pass": mono_ok,
            "overall_threshold_pass": overall_ok, "verdict": verdict,
            "caveat": "Single seed; any decrease is descriptive, not proof of statistical significance."}
