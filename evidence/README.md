# evidence/ — 过程记录，不是结果

本目录保存“当时到底发生了什么、怎么核验的”。结论、指标、验收与复现路径仍以 `reports/` 和 `artifacts/` 为准；不要把这里的文件当作交付结论引用。

## 组织

| 子目录 | 内容 | 整理前位置 |
|---|---|---|
| `logs/` | 22 个原始运行日志：A 基线、B（含两次恢复）/C 训练与评测、pilot、数据准备、外部数据准备与计分、smoke 与重试 | 仓库根目录，文件名不变 |
| `milestones/` | M1 数据审计与零样本基线、M2 本地可行性标定、B 正式训练门槛、B 的 CUDA 中断与恢复 | `reports/`，文件名不变 |
| `reviews/` | 独立代码审查（SOL 初审、二次复核）、两轮文档可用性审查、外部评测的独立审查与修复复核、修复前后的原始测试输出与耗时记录 | `reports/`，文件名不变 |
| `checks/` | 机器可读核验件：环境快照、一次性核验明细、运行时回归、缓存重建比对、40 条错误复核数据 | `reports/`，文件名不变 |

所有文件名都没有改动，所以“整理前位置 + 文件名”就是旧路径。

## 边界

- `logs/` 是原始输出，未经编辑；其中的失败记录反映的是修复前行为，不是当前测试状态。
- `reviews/SOL_REVIEW.md`、`reviews/SOL_FIX_VERIFICATION.md` 是**当时**的快照，按原样保留、不改写。当前状态以 `reports/REVIEW_FIXES.md` 为准。
- `checks/error_analysis_40.json` 是 40 条 mixed 样例的 AI 文本复核数据件，不是真人听音标注。
- 本目录内的文档保留各自的原始措辞，正文里出现的 `reports/xxx` 或裸文件名指的是目录整理前的位置；`README.md` 与 `reports/*.md` 中的对外引用已更新为 `evidence/...`。

## 整理时间

2026-10-19，在两个交付归档（`delivery/asr-finetune-complete.tar.gz`、`delivery/asr-external-eval.tar.gz`）打包完成之后。归档内部仍是最初的扁平结构，其清单 `SHA256SUMS`、`EXTERNAL_SHA256SUMS` 也描述归档内部路径；核验请对归档解压目录执行。

唯一保留的旧路径引用在 `artifacts/external_eval/verification.json` 的 `record` 字段（`reports/EXTERNAL_INDIVIDUAL_REVIEW.md`）：该文件已被 `EXTERNAL_SHA256SUMS` 固定哈希，改路径会让交付包校验失败，因此保持原样，实际文件在 `evidence/reviews/EXTERNAL_INDIVIDUAL_REVIEW.md`。
