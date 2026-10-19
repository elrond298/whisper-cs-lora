# 外部评测文档可读性与使用流程审查（DeepSeek Flash）

日期：2026-10-19。范围：`EXTERNAL_EVAL.md`、`reports/EXTERNAL_EVAL_REPORT.md`、`reports/EXTERNAL_REPRODUCE.md`、`reports/EXTERNAL_DATA_LICENSES.md`。只改文档，不改数值、代码、配置、权重、数据或预测。

## 0. 执行者与写入边界

| 项目 | 内容 |
|---|---|
| 本次角色 | 文档可读性/使用流程（UX）审查与直接修订 |
| 模型/通道 | 会话运行时配置为 `PI_PROVIDER=deepseek`、`PI_MODEL=deepseek-flash`；`~/.pi/agent/models-store.json` 中该条目 `baseUrl` 是 `https://api.deepseek.com`，即直连 DeepSeek，不经 OpenRouter。**未做独立网络探测或供应商回执核对**，路由身份以本机配置为准 |
| 允许写入 | 上述 4 份文档；本文件 |
| 明确未改 | 冻结计划 `EXTERNAL_EVAL_PLAN.md`、来源记录 `EXTERNAL_EVAL_SOURCES.md`、`protocol.json`、任何代码/测试/配置/权重/预测/数据；原 `README.md` 与 `FINAL_REPORT.md` |
| 未执行 | 推理、训练、重新计分、重新打包、归档核验、测试套件、上游许可再核验、数值复核 |
| 审阅时并发状态 | 本轮文档审阅时独立修复复审与隔离包核验尚在进行，故下表保留当时的待核验用语；其后父级已完成，冻结状态更新见文末。本次 DeepSeek 实际写入仅上述 5 个文件 |

## 1. 发现的问题与处理

| 编号 | 问题 | 读者影响 | 处理 |
|---|---|---|---|
| U1 | 三篇文档混着讲“看报告 / CPU 复算 / 重新推理”，入口按文件而不是按目的 | 读者容易把复算当重训，或直接去跑 GPU | 根页加“按目的选入口”表；复算文档第 1 节加三操作对比表（GPU / 基础模型 / 音频 / 改写对象） |
| U2 | 复算前置条件只写成"需要身份文件、manifest、token_schema 和 adapter" | 缺根 `config.json`、`protocol.json` 或 B/C adapter 哈希时会在中途报错，读者以为只要预测 JSONL | 复算文档第 2 节列逐文件清单与作用，并说明**只跑 FLEURS 也要 B/C 的 adapter 与 `training_complete.json`**，因为三组一起核对身份 |
| U3 | 离线推理只说"需先按原 README 获取基础模型" | 读者不知道下载哪些文件、是否会自动下载、为何离线失败 | 补固定 revision 的 `snapshot_download` 命令（`allow_patterns=['*.json','*.txt','model.safetensors']`，共 11 个文件、约 927 MB），并写明设置 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` 后不会自动下载 |
| U4 | `--smoke` 的真实语义没写清 | 读者可能把冒烟预测当一次性试跑丢掉，或以为正式运行会重算这 10 条 | 说明冒烟只处理每组冻结清单前 10 条、写进同一个 `predictions.jsonl`、正式运行接着续、这 10 条是最终结果的一部分；并写明默认续跑与 `already complete, verified, no GPU work` |
| U5 | 受限材料边界写成"external_eval 目录" | 容易只防住 `data/` 或只防住 `artifacts/` 一边 | 根页、报告、许可文档统一写成 `data/external_eval/` 与 `artifacts/external_eval/` 两个目录，并说明整包不加筛选不可公开 |
| U6 | 公开补充包写得像已经交付 | 读者会去找不存在的归档 | 统一为"仍在组装，尚未作为成品归档"，并按设计说明它只带 FLEURS 复算材料、CS 不可只凭它复算 |
| U7 | 路径基准不清：`reports/` 里的路径实际相对仓库根 | 读者在 `reports/` 下执行会找不到文件 | 根页与复算文档各加一行路径约定；报告第 7 节也标明"以下路径均相对仓库根" |
| U8 | "35 项测试通过"是快照，但写成当前状态 | 审查新增测试后数字会过期 | 改为"此前本地运行 35 项通过；独立审查可能新增测试，以最终核验记录为准"（报告与复算文档） |
| U9 | 术语“配对”“前缀标记”未解释 | 非统计读者看不懂区间怎么来、异常是什么 | 补“配对”定义；把 `language_prefix` 说清：开头 zh/transcribe/无时间戳前缀在全部样本中都正确，标记记录的是 EOS 之前多出的内部语言 token，不是初始语言设置失败，也不代表转写内容错 |
| U10 | 许可文档只写"按非商业研究处理"，没写官方许可标签，也缺"能否转交"速查 | 无法判断哪些材料能给别人；CS 的转发限制被弱化 | 补 CS 官方标注 CC BY-NC-SA 4.0（固定版本数据卡链接）与不得再识别/不得向第三方转发等条款；补 FLEURS/CS 材料转交速查表；把 FLEURS 945→780→300、CS 1034→900→450 的筛选数写进同一处 |
| U11 | 不知道 CS 复算到底缺什么 | 读者以为必须拿到 CS 音频才能复算 | 说明复算不读音频、不加载权重；CS 真正的缺口是受限的参考文本与逐条预测（及清单身份），而非音频或评分代码 |
| U12 | 重新生成的危险动作（移走原 `artifacts/B`、手改 identity）散在多段 | 容易破坏原实验或把评测改成重训 | 合并为 3 步流程 + 常见错误清单，明确只备份目标外部组、保留 adapter 与训练元数据、不删不利样本 |
| U13 | 报告结论段"英文表现变差"可能被读成已确认超过容忍界限 | 与第 1、4 节的谨慎表述不一致 | 结论段补"点估计 +0.9～1.0 个百分点，区间跨容忍界限，见第 4 节" |
| U14 | 百分比与百分点、三行不同测试易混 | 读者可能把 12.275% 与 0.956pp 相加或排名 | 报告开头加"单位与读法"：百分数是错误率，对比用百分点，三行不合成总分，区间跨 0/跨界限不等于"没有退化" |
| U15 | 离线复算还要校验 token 与文本一致、token 在词表内，但文档没说这些资产从哪来 | 读者以为有预测和 `token_schema.json` 就够，离线时又读不到 HF 缓存 | 复算文档第 2 节写明 6 个随包 tokenizer 文件、`validation_tokenizer.json` 哈希、词表上界与逐条解码校验；并说明推理会从固定缓存自动导出 |
| U16 | 补充包只有“组装中”，没有归档名和完整性清单 | 读者不知道将来核对什么 | 统一写明计划归档 `delivery/asr-external-eval.tar.gz`、清单 `EXTERNAL_SHA256SUMS` 在仓库根；文件数量和哈希未定稿 |
| U17 | 复算文档原写“B/C 的 adapter 和训练元数据来自主包，不在补充材料里” | 读者会去找并不需要的旧包，或以为无法独立复算 | 改为补充包内含 B/C 各 4 个身份文件（共 8 个）；只有基础 Whisper 权重不随包提供 |
| U18 | 没有环境安装步骤，推理入口只写成整体运行 | 读者不知道先 `uv sync`，也不知道可只跑某个子集/模型 | 复算文档补 `uv sync`，并给出 `--datasets/--groups` 的每数据集/每组循环；说明已完成的组合不占 GPU |
| U19 | “35 项测试”已过时，且未说明新增校验内容 | 读者无法判断修复是否覆盖 token 绑定 | 更新为当前 39 项，注明含 3150 条序列/token/文本校验、原始分数与区间不变；复审未完不宣称完成 |
| U20 | 随包 tokenizer 的再分发许可未写 | 读者不知道该按什么许可署名 | 许可文档补固定 revision 模型卡的 Apache-2.0 链接与署名要求，不另造许可文本 |

## 2. 逐文件改动

| 文件 | 改动 |
|---|---|
| `EXTERNAL_EVAL.md` | 重写为结果速览 + 按目的选入口 + 边界三段；补英文退化的量级与区间位置、路径约定、两份受限目录、补充包状态与组成（含 B/C 身份文件）、FLEURS 归属要求 |
| `reports/EXTERNAL_EVAL_REPORT.md` | 加“单位与读法”；补 A 的固定 revision；音频同时给分钟与秒；补压缩分卷合计字节；解释“配对”“前缀标记”；修正新增目录写法；补序列校验说明与 39 项测试；结论段补退化量级 |
| `reports/EXTERNAL_REPRODUCE.md` | 重组为 6 节：三操作对比与环境安装（`uv sync`）、复算逐文件清单（6 个随包 tokenizer 文件与哈希、B/C 各 4 个身份文件、v1 身份来源）、离线复算、取数与按子集/组合推理（固定 revision 下载、`--smoke` 语义、离线失败行为、`--verify`）、重新生成流程与常见错误、自检与解释范围 |
| `reports/EXTERNAL_DATA_LICENSES.md` | 加“能否对外转交”速查表；补 CS 的 CC BY-NC-SA 4.0 标签与固定版本数据卡链接、作者论文链接；补两套数据的筛选数量与时长；补模型 revision、tokenizer 的 Apache-2.0 署名、B/C 身份文件说明、补充包归档名/清单 |

## 3. 实际执行的检查

| 检查 | 方法 | 结果 |
|---|---|---|
| 链接有效性 | 脚本解析 5 份文档（含本文件）的相对 Markdown 链接并检查目标存在 | 0 个失效 |
| 引号风格 | 脚本检查代码块外的直引号 | 0 处（全部为中文引号） |
| 必需事实保留 | 脚本核对 revision、1050/3150、450/900/1034、排除数、时长、9 个指标、9 行区间、9.6 GB/927 MB 等字符串 | 全部存在 |
| 命令与代码一致 | 逐条对照 `external_eval.py`、`external_scores.py`、`prepare_external_cs.py`、`prepare_external_fleurs.py`、`asr.py` 的参数与校验顺序 | 参数名、默认行为、报错条件与文档一致 |
| 修复后核对 | 对照 `validation_tokenizer.json` 与 6 个资产、`LEGACY_INFERENCE_SHA256`、`provenance/external_eval_generation_v1.py`、`delivery/EXTERNAL_FILELIST.txt`、`--datasets/--groups/--smoke` 参数和测试方法计数 | 文档与代码一致；文件清单仍未含 tokenizer、provenance 和校验清单条目，待父级定稿 |
| 下载清单 | 对照本机已缓存的固定 revision 文件列表 | 11 个文件全部匹配 `allow_patterns`，合计约 927 MB |
| 未改动原始产物 | 本次仅调用文档写入工具；未运行推理/计分/打包 | 无新增或改写 `data/`、`artifacts/`、代码 |

## 4. 未检查与待父级核验

- **没有**重新推理、重新计分或复跑测试；上述数字属于方法计数，最终执行证据由父级另行完成，见文末。
- **没有**独立访问上游确认许可标签。CS 的 CC BY-NC-SA 4.0 及附加条款取自本仓库 `EXTERNAL_EVAL_SOURCES.md` 的固定版本审计，建议父级再对固定 revision 数据卡核实一次。
- **没有**构建或解压补充包。`delivery/EXTERNAL_FILELIST.txt` 当前 59 项，尚未包含 tokenizer 资产（7 个）、`provenance/external_eval_generation_v1.py` 和 `EXTERNAL_SHA256SUMS`；数量与哈希未定稿，需父级按文档第 2 节核对。
- **没有**复核任何数值或区间，只核对文档与 `comparison.json`/`config.json` 是否一致；评分代码哈希、身份绑定和原始预测均未重算。
- “只复算 FLEURS 也要 B/C 8 个身份文件”来自 `external_eval.py` 的 `prediction_identity` 与 `asr.evaluation_identity` 校验；若父级再调整校验范围，需同步文档。
- 随包 tokenizer 的 6 个文件与 `validation_tokenizer.json` 路径已核对；最终隔离核验进一步确认包内路径、文件哈希及解码行为有效。
- 父级描述的 `external_eval.py infer --dataset … --group …` 子命令在当前工作树中不存在：入口是 `--datasets/--groups/--smoke`。文档按已核对的参数写，并在复算文档中注明两种写法的对应关系；若子命令随后加入，需把该段改为子命令形式。

## 交付冻结后的事实更新（父级）

DeepSeek 文档修订完成后，独立修复复核确认 P2 已解决，无剩余已确认问题；39 项测试通过。公开补充包共 70 文件、69 个哈希条目，已隔离解压，在禁用 GPU、阻断网络/DNS、HF 缓存初始为空时精确复算 FLEURS 指标和区间，1800 条 token/文本校验通过。CS 逐条数据不进入公开包。复用了现有固定依赖，未声称全新机器安装验证。完整证据见交付目录 `delivery/external-verification.json`。前面的待核验措辞是文档审阅时的历史状态，不代表当前仍未交付。
