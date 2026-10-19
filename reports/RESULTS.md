# A/B/C 测试集对比（自动汇总；AI 文本复核另附）

> 本文件由 `summarize.py` 生成，请勿手工编辑，改动会在下次运行时丢失。主报告见 `FINAL_REPORT.md`，逐例错误见 `ERROR_ANALYSIS.md`，审查与修复状态见 `REVIEW_FIXES.md`。

所有指标为百分比，越低越好；`C−B` 为差值，单位是**绝对百分点（pp）**，不是百分比。MER = 编辑错误数 ÷ 参考计分单位数（中文按字、英文按词混算）；中文 CER 为纯中文子集（按非空白字符），英文 WER 为纯英文子集（按英文词）。

| 组别 | mixed MER | 中文 CER | 英文 WER | 整体 MER | 崩溃率 |
|---|---:|---:|---:|---:|---:|
| A 原始 Whisper-small | 48.706 | 37.414 | 34.290 | 42.141 | 0.152 |
| B 自然采样 LoRA | 15.918 | 14.422 | 24.356 | 16.067 | 0.000 |
| C mixed 过采样 LoRA | 15.631 | 14.775 | 24.062 | 16.008 | 0.000 |
| C−B | −0.287pp | +0.353pp | −0.294pp | −0.060pp | 0.000 |

测试集 1,315 条：mixed 373、中文 685、英文 257。结论以验收结果中的 `verdict` 为准；通过预设方向性规则不等于统计显著。

验收结果：
```json
{
  "C_minus_B_percentage_points": {
    "overall": -0.05966587112171595,
    "mixed": -0.2874595759971257,
    "zh": 0.3527702842913474,
    "en": -0.2943340691685066
  },
  "metric_hard_gates_pass": true,
  "mixed_improved": true,
  "monolingual_thresholds_pass": true,
  "overall_threshold_pass": true,
  "verdict": "oversampling_effective_single_seed",
  "caveat": "Single seed; any decrease is descriptive, not proof of statistical significance.",
  "same_BC_config": true,
  "same_BC_initialization": true,
  "fixed_BC_steps": true,
  "hardware_50_steps_pass": true,
  "official_test_identity_pass": true,
  "prediction_checkpoint_identity_pass": true,
  "all_hard_gates_pass": true
}
```

训练实测：
```json
{
  "pilot": {
    "steps": 50,
    "median_step_seconds": 3.19741517050079,
    "mean_step_seconds": 3.2611265322001053,
    "peak_allocated_MiB": 1122.52490234375,
    "peak_reserved_MiB": 1252.0,
    "loss_first10_mean": 1.1763307036831976,
    "loss_last10_mean": 0.8806582322809845,
    "sampling": {
      "sample_count": 400,
      "language_counts": {
        "mixed": 113,
        "zh": 189,
        "en": 98
      },
      "actual_speech_hours": 0.3362023090277778,
      "padded_input_hours": 3.3333333333333335,
      "unique_examples": 388
    }
  },
  "B": {
    "steps": 800,
    "median_step_seconds": 8.729476702499596,
    "mean_step_seconds": 9.077013717147468,
    "peak_allocated_MiB": 1134.6376953125,
    "peak_reserved_MiB": 1278.0,
    "loss_first10_mean": 1.176463527046144,
    "loss_last10_mean": 0.37015310605056584,
    "sampling": {
      "sample_count": 6400,
      "language_counts": {
        "mixed": 1812,
        "zh": 3085,
        "en": 1503
      },
      "actual_speech_hours": 5.6811437673611325,
      "padded_input_hours": 53.333333333333336,
      "unique_examples": 4698
    }
  },
  "C": {
    "steps": 800,
    "median_step_seconds": 2.4802103780002653,
    "mean_step_seconds": 2.474994986826241,
    "peak_allocated_MiB": 1136.3798828125,
    "peak_reserved_MiB": 1270.0,
    "loss_first10_mean": 1.0659259538166226,
    "loss_last10_mean": 0.35594399625551887,
    "sampling": {
      "sample_count": 6400,
      "language_counts": {
        "zh": 2147,
        "mixed": 3220,
        "en": 1033
      },
      "actual_speech_hours": 6.531012621527767,
      "padded_input_hours": 53.333333333333336,
      "unique_examples": 4437
    }
  }
}
```

限制：单随机种子、无显著性检验，差值只是描述性结果；两组匹配的是每步 8×30 秒**补齐输入**预算，不是真实语音秒数；解码使用已知语言标签前缀（oracle），不是自动语言检测；论文 27.05% MER 属于 wav2vec 2.0，不是 Whisper 基线；测试集被用于 B→C 进入门槛，存在选择偏差风险；0.3pp 是预设容忍范围，不是统计噪声界；40 条逐例复核为 AI 文本对照，不是真人听音。
