# 外部数据来源、修改与交付边界

一句话：FLEURS 材料可以带归属转发；CS-Dialogue 的逐条材料只留在本地获准研究目录。公开安全补充包只含代码、方法、汇总指标和 FLEURS 相关材料（含离线复算用的校验 tokenizer/词表与 B/C 身份文件），计划归档为 `delivery/asr-external-eval.tar.gz`，完整性清单为 `EXTERNAL_SHA256SUMS`，尚未定稿。

## 能否对外转交

| 材料 | FLEURS | CS-Dialogue |
|---|---|---|
| 原始音频 | 可转发，须署名并附许可 | 不随包提供，仅限本地研究 |
| 参考转写 | 可转发，须署名并说明修改 | 不随包提供 |
| 逐条预测、逐条评分 | 可转发，须署名并说明修改 | 不随包提供 |
| 评测代码、方法、汇总指标 | 可 | 可 |

公开安全补充包按设计只含 FLEURS 复算所需材料，因此不能只凭它复算 CS；这不是打包失误，是许可边界。

## FLEURS

- 来源：[Google FLEURS](https://huggingface.co/datasets/google/fleurs)，固定 revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`。
- 引用：Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*（[论文](https://arxiv.org/abs/2205.12446)）。
- 许可：[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。原数据归其创作者；本项目未获其背书。
- 使用范围：`cmn_hans_cn`（普通话）与 `en_us`（美式英语）官方 test，各 300 条，均为 ≤30 秒录音。中文从 945 条 test 中排除 164 条含拉丁字母、1 条 >30 秒，剩 780 条后抽 300 条（286 个句子 ID，3,454.76 秒）；英文 647 条全部合格，抽 300 条（300 个句子 ID，3,009.88 秒）。
- 修改：筛选、固定种子抽样、参考与预测对称规范化、计算识别错误。原始参考和计分口径保留在清单与审计中。
- 补充包可以包含带来源信息的 FLEURS 清单、参考文本、模型预测、指标和代码；不包含音频或整个原始数据集。再分发这些材料时，保留本归属说明、许可链接和修改说明。

## CS-Dialogue

- 来源：[BAAI CS-Dialogue](https://huggingface.co/datasets/BAAI/CS-Dialogue)，固定 revision `a84770fcdf14072252373a0227321227eaac7dc7`。作者论文：[arXiv:2502.18913](https://arxiv.org/html/2502.18913v2)。
- 官方许可标注为 **CC BY-NC-SA 4.0**（[固定版本数据卡](https://huggingface.co/datasets/BAAI/CS-Dialogue/blob/a84770fcdf14072252373a0227321227eaac7dc7/README.md)）。数据卡另列：仅限学术/非商业用途、不得重新识别说话人、须遵守参与者撤回要求、未经书面许可不得向研究组之外转发、研究衍生物留在研究组内。公开可下载（数据卡 `gated: false`）不等于可以商用或再分发；具体要求以该固定版本的数据卡和权利人授权为准。
- 使用范围：官方 short_wav 的 test 索引与转写。test 6257 条中有 1034 条 mixed；排除 31 条含隐私/不可辨识遮蔽（`**`）、6 条清理标签后不再是 mixed、97 条 >30 秒，剩 900 条合格池，固定抽 450 条（30 位说话人，6,014.544 秒）。
- 修改：去除标注标签、保留口语内容与官方隐私遮蔽，不使用另一份表格中较少遮蔽的文本恢复内容。含 `**` 的样本直接排除，原因记录在审计中。
- **CS 音频、参考文本、逐条预测及逐条衍生评分保留在获准的本地研究目录，不进入公开安全补充包。** 补充包只保留评测代码、方法与汇总指标，它不是 CS 数据授权文件。
- 不得把整个 `data/external_eval/` 或 `artifacts/external_eval/` 不加筛选地公开打包。需要对外转交 CS 逐条材料时，先取得所需许可，不要用改名或去掉路径来规避限制。

## 模型与原实验

- A 为 `openai/whisper-small`，固定 revision `973afd24965f72e36ca33b3055d56a652f456b4d`；B/C 沿用 ASCEND 微调后的最终 LoRA 权重，本轮不训练。基础模型权重不随外部补充包提供；评测所用的 6 个 tokenizer 文件同样来自这个固定 revision，随包提供时按模型卡的 [Apache-2.0 许可](https://huggingface.co/openai/whisper-small/blob/973afd24965f72e36ca33b3055d56a652f456b4d/README.md)署名，本仓库不另造许可文本。
- 补充包包含 B/C 各 4 个身份文件（`config.json`、`training_complete.json`、`adapter/adapter_config.json`、`adapter/adapter_model.safetensors`）。它们是本项目的微调产物，只用于身份校验与复算，不是 ASCEND 的原始音频或转写，也不能用来还原受限数据。
- 原模型与 ASCEND 的来源、版本和复现说明见原主包。
- 外部精确波形审计未发现所选音频与本项目 ASCEND 训练/验证/测试数据重叠。它不证明没有裁切、改速或重采样后的近重复，也不保证基础模型预训练从未接触 FLEURS 或其它相关内容。
