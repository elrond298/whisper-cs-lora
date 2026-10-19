"""Recompute all metrics from saved predictions; prepare the final comparison and review set."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys

from asr import OUT, ROOT, check_predictions, evaluation_identity, read_json, save_json
from metrics import compare, summarize


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def verified_test_results(config):
    audit = read_json(OUT / "data_audit.json")["splits"]["test"]
    results, baseline = {}, None
    for group in ("A", "B", "C"):
        folder = OUT / group
        if read_json(folder / "test_eval_config.json") != evaluation_identity(group, "test", config):
            raise RuntimeError(f"Test evaluation identity mismatch: {group}; refuse stale or partial predictions")
        rows = read_rows(folder / "test_predictions.jsonl")
        seen = check_predictions(rows, rows if baseline is None else baseline)
        if seen != set(range(audit["n"])) or Counter(row["language"] for row in rows) != audit["counts"]:
            raise ValueError(f"Official test IDs/subset counts mismatch: {group}")
        if baseline is None:
            baseline = rows
        results[group] = summarize(rows)
    return results


def main():
    config = read_json(ROOT / "config.json")
    results = verified_test_results(config)
    for group, metrics in results.items():
        save_json(OUT / group / "test_metrics.json", metrics)
    b, c = (OUT / group for group in ("B", "C"))
    same_config = read_json(b / "config.json") == read_json(c / "config.json") == config
    same_init = read_json(b / "initialization.json")["adapter_sha256"] == read_json(c / "initialization.json")["adapter_sha256"]
    fixed_steps = all(read_json(folder / "training_complete.json")["completed_steps"] == config["steps"] for folder in (b, c))
    pilot = read_json(OUT / "pilot" / "progress.json")
    hardware_pass = pilot["completed_steps"] >= 50 and "4060" in read_json(OUT / "pilot" / "initialization.json")["gpu"]
    verdict = compare(results["A"], results["B"], results["C"])
    verdict.update(same_BC_config=same_config, same_BC_initialization=same_init,
                   fixed_BC_steps=fixed_steps, hardware_50_steps_pass=hardware_pass,
                   official_test_identity_pass=True, prediction_checkpoint_identity_pass=True)
    verdict["all_hard_gates_pass"] = all((same_config, same_init, fixed_steps, hardware_pass, verdict["metric_hard_gates_pass"]))
    if not verdict["all_hard_gates_pass"]:
        verdict["verdict"] = "hard_gate_failed"
    save_json(OUT / "comparison.json", {"metrics": results, "acceptance": verdict})
    training = {}
    for group in ("pilot", "B", "C"):
        events = read_rows(OUT / group / "train.jsonl")
        by_step = {row["step"]: row for row in events}
        events = [by_step[step] for step in sorted(by_step)]
        training[group] = {"steps": len(events), "median_step_seconds": statistics.median(r["seconds"] for r in events),
                           "mean_step_seconds": statistics.mean(r["seconds"] for r in events),
                           "peak_allocated_MiB": max(r["peak_allocated_MiB"] for r in events),
                           "peak_reserved_MiB": max(r["peak_reserved_MiB"] for r in events),
                           "loss_first10_mean": statistics.mean(r["loss"] for r in events[:10]),
                           "loss_last10_mean": statistics.mean(r["loss"] for r in events[-10:]),
                           "sampling": read_json(OUT / group / "sampling.json")}
    save_json(OUT / "training_summary.json", training)
    test_rows = read_rows(c / "test_predictions.jsonl")
    mixed = [r for r in test_rows if r["language"] == "mixed"]
    review = random.Random(config["seed"]).sample(mixed, min(40, len(mixed)))
    save_json(OUT / "error_review_40.json", [{**r, "manual_category": None, "manual_note": None} for r in review])
    labels = {"A": "A 原始 Whisper-small", "B": "B 自然采样 LoRA", "C": "C mixed 过采样 LoRA"}
    keys = ("mixed", "zh", "en", "overall")
    counts = Counter(row["language"] for row in test_rows)
    lines = ["# A/B/C 测试集对比（自动汇总；AI 文本复核另附）", "",
             "> 本文件由 `summarize.py` 生成，请勿手工编辑，改动会在下次运行时丢失。主报告见 `FINAL_REPORT.md`，逐例错误见 `ERROR_ANALYSIS.md`，审查与修复状态见 `REVIEW_FIXES.md`。", "",
             "所有指标为百分比，越低越好；`C−B` 为差值，单位是**绝对百分点（pp）**，不是百分比。MER = 编辑错误数 ÷ 参考计分单位数（中文按字、英文按词混算）；中文 CER 为纯中文子集（按非空白字符），英文 WER 为纯英文子集（按英文词）。", "",
             "| 组别 | mixed MER | 中文 CER | 英文 WER | 整体 MER | 崩溃率 |", "|---|---:|---:|---:|---:|---:|"]
    for group, metrics in results.items():
        values = [metrics[k]["percent"] for k in keys]
        lines.append("| " + labels[group] + " | " + " | ".join(f"{v:.3f}" for v in values) + f" | {metrics['overall']['collapse_percent']:.3f} |")
    deltas = [results["C"][k]["percent"] - results["B"][k]["percent"] for k in keys]
    collapse_delta = results["C"]["overall"]["collapse_percent"] - results["B"]["overall"]["collapse_percent"]
    lines.append("| C−B | " + " | ".join(f"{d:+.3f}pp".replace("-", "−") for d in deltas) + f" | {collapse_delta:.3f} |")
    lines += ["",
              f"测试集 {len(test_rows):,} 条：mixed {len(mixed)}、中文 {counts['zh']}、英文 {counts['en']}。结论以验收结果中的 `verdict` 为准；通过预设方向性规则不等于统计显著。",
              "", "验收结果：", "```json", json.dumps(verdict, ensure_ascii=False, indent=2), "```", "",
              "训练实测：", "```json", json.dumps(training, ensure_ascii=False, indent=2), "```", "",
              "限制：单随机种子、无显著性检验，差值只是描述性结果；两组匹配的是每步 8×30 秒**补齐输入**预算，不是真实语音秒数；解码使用已知语言标签前缀（oracle），不是自动语言检测；论文 27.05% MER 属于 wav2vec 2.0，不是 Whisper 基线；测试集被用于 B→C 进入门槛，存在选择偏差风险；0.3pp 是预设容忍范围，不是统计噪声界；40 条逐例复核为 AI 文本对照，不是真人听音。"]
    (ROOT / "reports" / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict["all_hard_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
