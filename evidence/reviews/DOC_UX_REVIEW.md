# 文档可读性与使用流程审查（DeepSeek Flash）

日期：2026-10-19。范围：`README.md`、`reports/FINAL_REPORT.md`、`reports/ERROR_ANALYSIS.md`、`reports/REVIEW_FIXES.md`、`reports/RESULTS.md` 的生成模板。

> **状态（后续修订）**：下方第 1–4 节保留第一轮审查的原始记录——其中的 16 项测试、81/90 哈希通过、包内 7 项记录都是**当时**的实测快照，未按后续结果改写。之后仍有一轮修正与本轮补充，当前 17 项测试见第 5 节。

## 0. 执行者与写入边界

| 项目 | 内容 |
|---|---|
| 本次角色 | 文档可读性/使用流程（UX）审查与直接修订 |
| 模型/通道 | 配置中的 **Direct DeepSeek**，模型 `deepseek/deepseek-flash`；未切换模型或服务商（依据本次会话配置，未做独立网络探测） |
| 允许写入 | `README.md`；`reports/FINAL_REPORT.md`、`ERROR_ANALYSIS.md`、`REVIEW_FIXES.md`；本文件；`summarize.py` 的 Markdown 渲染部分（`RESULTS.md` 由程序重新生成） |
| 明确未改 | 计算与校验逻辑（`asr.py`、`metrics.py`、`test_pipeline.py`、`config.json`）、依赖、权重、原始预测、`data/`、日志、`reports/EXPERIMENT.md`、`SOL_REVIEW.md`、`SOL_FIX_VERIFICATION.md`、`VERIFICATION.md` |
| 未执行 | 训练、完整推理、重新打包、额外实验、新文档框架或新依赖 |

## 1. 发现的问题与处理

编号 U1–U14 按对读者的影响排序。

| 编号 | 问题 | 读者影响 | 处理 |
|---|---|---|---|
| U1 | README 没有按“我想做什么”组织的入口 | 新读者要读完 88 行才知道该去哪 | 增加“先看这一页”导航表，按目的、文档、是否需要 GPU 三列给出入口 |
| U2 | 把 GPU 安装/自检与 CPU 复算混在同一节 | 只想核对指标的人会误以为必须有 GPU | 拆成 4 种使用方式（读结果 / CPU 重算 / 重新生成 / 从头训练），每种列前置条件与联网需求 |
| U3 | 旧 README 称“保留预测文件即可在 CPU 上重算” | **实质使用错误**：`summarize.py` 还校验评测身份文件、B/C `adapter` 哈希、训练完成标记、pilot 与 `data_audit.json`，只留 JSONL 会报错 | 明确列出 `summarize.py` 实际读取的文件；单组指标重算给出只需要 `metrics.py` 的独立命令 |
| U4 | 未记录 CLI 参数 | 读者会猜 `evaluate`、`--limit-per-language`、`--config` | 新增参数表（子命令 `prepare/train/eval`，`--group/--split/--steps/--limit-per-language/--config`、A 不可训练、0 步被拒绝） |
| U5 | 历史“7 项测试”与当前 16 项、以及交付包状态没有区分 | 会把过期数字当现状 | README、FINAL_REPORT、REVIEW_FIXES 明确标注历史快照与当前 16 项 |
| U6 | README 称可执行 `sha256sum -c SHA256SUMS` | 实际当前 **81/90 通过、9 项 FAILED**（4 项在审查修复后已过期：`asr.py`、`test_pipeline.py`、`comparison.json`、`data_audit.json`；5 项因本次文档/渲染器修订：`README.md`、`summarize.py`、`RESULTS.md`、`FINAL_REPORT.md`、`ERROR_ANALYSIS.md`） | 改为“审查前校验快照，当前会失败，随包重建时重算”，不再暗示现可通过 |
| U7 | 缓存/身份不匹配没有操作指引 | 读者可能盲删缓存或手工改 `ready.json` | 新增“归档后重新准备、比对审计数值、不手工补写身份”流程，并指向 `CACHE_REBUILD_VERIFICATION.json` 先例 |
| U8 | “已完成 B/C 再跑不会重训”“续跑需要精确配置”散落在多段 | 容易误以为重跑等于重训，或混用两份配置 | 合并为“中断、续跑与重复执行”一节，说明 50 步恢复点、no-op、配置哈希核对 |
| U9 | 自动生成的 RESULTS 缺 C−B 行、子集数、单位说明与“勿手改”提示 | 读者要在多个文件间拼装对比；手改会被下次运行覆盖 | 修改 `summarize.py` 渲染：加导航、单位定义、带正负号的 C−B 行、子集计数、完整限制清单；重新生成 `RESULTS.md` |
| U10 | FINAL_REPORT 开头没有阅读路线，指标/单位只在第 3 节出现 | 直接看结论的人可能把 pp 当百分比 | 标题后加阅读路线与单位/指标定义 |
| U11 | ERROR_ANALYSIS 没有摘要，40 行表格没有读法说明 | 要读完 40 行才知道范围与限制 | 顶部加一句话结论与相关文档；表前加“标签可复选、[UNK] 含义、疑似含义”；40 行逐条内容保持不动 |
| U12 | REVIEW_FIXES 把“历史报告说 F1 未修”与“现已修复”埋在长段落 | 读者可能继续认为 duration 校验缺失 | 重写为当前状态表 + “历史报告与最新结论的关系” + 明确标注未重建交付包 |
| U13 | `0.3pp` 出现但未定义“百分点” | 百分比与百分点混淆 | README、FINAL_REPORT、RESULTS 统一给出定义与一个换算例子（15.918%→15.631% 为 −0.287pp，约 −1.8% 相对变化） |
| U14 | “交付包不含依赖/基础模型/数据”未说明新机器仍需网络 | 误以为全新机器完全离线可用 | 说明 `uv sync` 通常需要网络，依赖不随包提供；“完全离线”仅适用于依赖与资产已就位 |

## 2. 逐文件改动

| 文件 | 改动性质 |
|---|---|
| `README.md` | 重组为 8 节：导航表、结果速览与结论边界、四种使用方式、精确命令与参数表、中断/续跑、缓存与身份处理、交付与校验、解释边界。补入 CLI 事实、CPU 复算的真实前置条件、no-op 与归档式重训流程、交付包待重建与 `SHA256SUMS` 过期说明 |
| `reports/FINAL_REPORT.md` | 标题后加阅读路线与单位定义；验收表“7 项自检”改为“16 项（历史快照 7 项）”；第 9 节补充交付包状态、`SHA256SUMS` 过期、CPU 复核真实依赖 |
| `reports/ERROR_ANALYSIS.md` | 顶部加一句话结论与相关文档；表前加读法说明；标题补“阶段 M5”。**40 行逐条证据逐字保留**（脚本计数仍为 40 行） |
| `reports/REVIEW_FIXES.md` | 重写结构：当前状态一览、历史报告关系、逐项修复、证据与复现命令、未改变内容、仍未验证与待办 |
| `reports/RESULTS.md` | 由更新后的 `summarize.py` 重新生成（不手工编辑） |
| `summarize.py` | 只改 Markdown 渲染字符串与 C−B 行计算；不触及指标计算、校验或文件身份逻辑 |

## 3. 实际执行的检查

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| 测试套件 | `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest test_pipeline` | `Ran 16 tests … OK`（0.876s） |
| 语法 | `.venv/bin/python -m py_compile summarize.py` | 通过 |
| 真实 CLI 复算 | `uv run python summarize.py` | 退出码 0，所有硬门槛 `true` |
| 产物未被改写 | 文档修改前后对 `artifacts/` 全部 56 个文件做 SHA-256 快照并逐一 diff | 完全一致（含 `comparison.json`、`error_review_40.json`） |
| 渲染器输出 | 先将 `summarize.OUT` 指向 `/tmp` 副本运行两次，再比对生成的派生 JSON 与线上文件 | 字节一致；确认渲染改动不影响数据写入 |
| 链接有效性 | 脚本解析 4 份文档的 Markdown 相对链接并检查文件存在 | README 11 个链接全部存在；其余文档使用反引号路径，无失效链接 |
| 40 行证据 | `grep -cE '^\| [0-9]+ / [0-9]+ \|' reports/ERROR_ANALYSIS.md` | 40 |
| 过期表述 | grep `7项自检`、`第 5 节`、`只复现评测` | 已清除或改为带历史标注的表述 |
| 校验清单现状 | `sha256sum -c SHA256SUMS` | 81 OK / 9 FAILED；归属见 U6 |

## 4. 未验证与限制

- 未重新训练、未重跑完整推理；只做了 CPU 指标重算与单元测试。
- **交付包未重建**：包内仍是审查前的 7 项测试与 90 个哈希记录；包内测试数、文件数、压缩包哈希需由主执行者重建后填写，本文件不给出新计数。
- `SHA256SUMS` 仍为过期清单，需随新包重算；当前不应引用为通过证据。
- 未逐页目视检查 Markdown 渲染效果（表格换行、中文折行），只验证结构、链接与内容一致性；未新增英文版或图表。
- 只检查了链接目标是否存在，未校验文内锚点跳转。
- 未独立探测模型路由；`deepseek/deepseek-flash` 与 Direct DeepSeek 通道来自本次会话配置。
- 未审查本文件之外的报告（`VERIFICATION.md`、`SOL_*.md`、`EXPERIMENT.md` 等）的文字，仅在链接与事实引用层面保持一致；这些文件按约束保持原样。
- 文档修订不构成新的实验证据：所有结论仍受单种子、无显著性检验、AI 文本复核（非听音）、测试集门槛选择偏差等既有边界限制。

## 5. 后续修订补充（当前 17 项测试）

本节记录第 1–4 节之后的修改。上面的历史检查（16 项测试、81/90 哈希、包内 7 项记录）保持原样，不改写成 17 项。

| 项目 | 现状 |
|---|---|
| 测试数 | 17 项（第 3 节记录的是第一轮 16 项） |
| 真正重新生成 | 只归档 `<split>_predictions.jsonl`、`<split>_eval_config.json`、`<split>_metrics.json` 三个测试输出文件，保留该组 `adapter/`、`config.json`、`training_complete.json` 等训练权重与元数据；整组移走会因缺少身份文件而失败 |
| 从零复跑 | 在隔离副本中归档整个 `artifacts/`（含 A 与 pilot），否则只重训 B/C 会沿用旧的 50 步标定与零样本基线 |
| 已完成 vs 续跑 | 完成且 `resume.pt` 步数一致的运行是 no-op；步数不足才从恢复点续跑；两种情况都不重训 800 步 |
| adapter 哈希 | adapter SHA-256 的绑定属于**评测与汇总**阶段，训练续跑不核对 |
| 复现边界 | 40 条逐例分析标注为 **AI 文本对照**，不是真人听音 |
| shell 片段 | README 两段保护性备份命令改为 `( ... )` 子 shell 包裹，`set -e` 在块内生效，块内失败只中止本块、不影响读者的交互 shell；`mkdir` 仍故意不用 `-p`，仍只移动 3 个测试输出文件，训练元数据保留，CLI 命令未改 |
| 完成标记加固 | 主执行者在后续修复中、**任何 GPU 访问与写入之前**补齐了孤儿/冲突完成标记的拒绝逻辑，新增第 17 项测试。该改动不涉及指标数学：原始 `artifacts/` 的 54 个哈希未变，36 个原始 fresh-generation 与 9 个缓存等值证据也未变 |

历史关系：`reports/SOL_FIX_VERIFICATION.md` 仍是 16 项测试时的快照；当前状态以 `reports/REVIEW_FIXES.md` 的 17 项为准。

本节完成时，交付包与 `SHA256SUMS` 尚待主执行者重建；后续实测状态记录如下。

## 6. 归档核验补录（主执行者，非 DeepSeek 的文档检查）

- 同版候选归档已在独立解压目录通过 17 项测试与 101 个文件哈希，复算前后文件未变；A/B/C 测试指标和 B/C 验证指标精确一致。
- 核验禁用 GPU、阻断网络连接与 DNS、使用空模型缓存；只复用了本机固定依赖，不是全新机器安装验证。
- B/C 优化器计数各为 800，144 个 adapter 张量各自与恢复点完全一致。
- 正式归档后的大小、哈希、独立复验结果另附于 `delivery/verification.json`；包内不嵌入自身归档哈希。历史 7 项/90 文件记录保留在 `delivery/pre-review/`，不是当前校验状态。
