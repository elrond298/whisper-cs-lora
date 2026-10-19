# 外部验证补充交付

**状态：1050 段录音、3150 条 A/B/C 预测、独立流程审查、缺陷修复与 DeepSeek Flash 文档修订均已完成；公开补充包已通过隔离离线核验。**

本轮不重训：A 用原 Whisper-small，B/C 用既有最终 800 步 LoRA，只在新的域外子集上推理。CS-Dialogue mixed 抽 450 条，FLEURS 中文、英文各 300 条。

| 测试 | A 原始 | B 自然采样微调 | C mixed 过采样微调 |
|---|---:|---:|---:|
| CS mixed MER | 26.181% | **12.275%** | 13.231% |
| FLEURS 中文 CER | 19.093% | **10.692%** | 10.981% |
| FLEURS 英文 WER | **6.813%** | 7.860% | 7.727% |

越低越好。三行是三个不同测试，单位和口径不同，不要合成一个总分，也不能当成公开榜单名次。

- 混合语音和中文：B/C 都明显好于 A。
- 英文：B/C 分别比 A 差 1.047、0.914 个百分点。区间不含 0，但跨过事先固定的 +0.8 个百分点容忍界限，所以“是否超过容忍范围”证据不足；退化方向不能隐去。
- CS 上 C 比 B 差 0.956 个百分点，区间跨 0：本次没有复现过采样优势，也不能证明 C 一定更差。

## 各主题与详细文档

| 主题 | 结论 | 展开 |
|---|---|---|
| 完整结果、区间与限制 | CS 上 B 最好；FLEURS 英文 A 最好，B/C 的退化方向不能隐去 | [完整报告](reports/EXTERNAL_EVAL_REPORT.md) |
| 数据来源与抽样规则 | CS-Dialogue 450 条（覆盖 30/30 测试说话人）、FLEURS 中英各 300 条，固定 revision | [抽样前计划](reports/EXTERNAL_EVAL_PLAN.md) · [官方来源核查](reports/EXTERNAL_EVAL_SOURCES.md) |
| 离线复算与重新推理 | 已有预测可在 CPU 上重算指标与区间；重新生成需要加载模型 | [查看、复算、重新推理](reports/EXTERNAL_REPRODUCE.md) §3 / §4–5 |
| 许可与转交边界 | FLEURS 可带归属转发；CS 逐条材料只留在本地获准目录 | [数据许可与转交边界](reports/EXTERNAL_DATA_LICENSES.md) |
| 独立审查与修复 | 首轮发现与修复复核、修复后的独立核验 | [审查与修复](reports/EXTERNAL_REVIEW_FIXES.md) · [独立复核](reports/EXTERNAL_INDIVIDUAL_REVIEW.md) |

路径约定：本页链接和正文里的路径都以仓库（或解压后的包）根目录为基准；`reports/` 内的文档也用同一基准，不相对 `reports/` 解析。

## 边界

本次为独立补充，未重训，原 ASCEND 主交付包保持不变。公开补充包为 `delivery/asr-external-eval.tar.gz`（约 7.7 MiB），含 70 个文件；根目录 `EXTERNAL_SHA256SUMS` 校验其中 69 个文件，清单不包含自身。包内有代码、方法、汇总指标、FLEURS 复算材料、6 个校验 tokenizer 文件和 B/C 各 4 个身份文件，但不含基础模型权重或任何音频。CS 的参考、逐条预测和逐条评分仅留在获准本地研究目录，不能把两个 `external_eval/` 目录整个公开打包；因此公开包不能独立复算 CS。最终核验证据为交付目录的 `delivery/external-verification.json`，并在外部报告中汇总。

FLEURS 为 CC-BY-4.0：再分发其清单、参考文本或预测时，须保留归属、许可链接和修改说明。CS-Dialogue 官方标注 CC BY-NC-SA 4.0，并另有仅限学术/非商业、不得向第三方转发的条款，详见许可说明。

## 局限

- 未做真人听音、多种子训练或全量评测；区间只反映当前固定模型和子集，不含训练随机性。
