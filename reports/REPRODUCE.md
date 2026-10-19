# 复现与操作手册

本文件给出 ASCEND 主实验的复现步骤、命令序列、续跑语义与交付内容校验。结论与指标见 `FINAL_REPORT.md` 与 `RESULTS.md`，外部评测的操作见 `EXTERNAL_REPRODUCE.md`。

## 1. 四种使用方式

| 方式 | 操作 | 前置条件 | 是否需要网络 |
|---|---|---|---|
| 1 读结果 | 阅读报告 | 无 | 否 |
| 2 CPU 复算 | 由已保存预测复算指标 | 仓库 + `artifacts/` + 固定版本依赖 | 首次安装依赖可能需要 |
| 3 重新生成 | 以已训练 adapter 重新生成预测 | 方式 2 的全部 + 基础模型 + `data/` + CUDA | 准备数据/模型时需要 |
| 4 从头训练 | 在隔离副本中重训 A/pilot/B/C | 方式 3 的全部 + GPU 时间 | 是 |

`uv sync --frozen` 需要访问软件包源；依赖不随交付包提供，因此新机器首次安装通常需要网络。完全离线安装仅适用于已具备固定依赖与所需资产的机器。

### 方式 1：只读结果

本方式不需要 GPU、依赖或数据。README 第 1 节的对应表将各结论链接至详细文档；可复算数值见 [`reports/RESULTS.md`](RESULTS.md)，计分口径与验收项见 [`reports/FINAL_REPORT.md`](FINAL_REPORT.md)。

### 方式 2：CPU 复算指标（不需要 GPU、基础模型或 `data/`）

复算单组指标只需该组预测文件与 `metrics.py`：

```bash
uv run python -c 'import json; from metrics import summarize; rows=[json.loads(x) for x in open("artifacts/B/test_predictions.jsonl", encoding="utf-8")]; print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))'
```

完整复核（A/B/C 全部指标、验收标记、训练汇总）执行：

```bash
uv run python summarize.py
```

`summarize.py` 除预测文件外还会读取并校验以下内容，仅保留预测文件不足以完成复核：

- `artifacts/{A,B,C}/test_eval_config.json` 与 `test_predictions.jsonl`；
- `artifacts/B|C/{config,initialization,training_complete}.json` 与 `adapter/adapter_model.safetensors`（用于 SHA-256 身份绑定）；
- `artifacts/data_audit.json`、`artifacts/{pilot,B,C}/train.jsonl`、`artifacts/pilot/progress.json`。

它会重写派生的 `test_metrics.json`、`comparison.json`、`training_summary.json`、`error_review_40.json` 和 `reports/RESULTS.md`，不改原始预测、adapter 或训练记录。所以缺少 adapter 或身份文件时，正确做法是恢复完整 `artifacts/`，而不是删掉校验后继续。

### 方式 3：用已训练 adapter 重新生成预测

需要方式 2 的全部文件，外加固定 revision 的基础模型和已准备、身份匹配的 `data/`。生成前会核对冻结训练配置、完成步数、配置哈希和当前 adapter 哈希；不符会在加载数据/模型或写文件之前报错。预测固定写入 `artifacts/<group>/<split>_predictions.jsonl`（同目录还有 `<split>_eval_config.json`、`<split>_metrics.json`），**没有可指定输出目录的参数**。

若要确认生成为真实重新推理而非空跑，须先移走旧的测试输出。要点如下：

- 整组复制后重跑无效：文件仍存在时按样本 ID 续跑，1,315 条均已完成，故全部跳过，不产生新预测。
- 整组移走会直接失败：`adapter/`、`config.json`、`training_complete.json` 是评测身份校验的输入。
- 正确做法是仅归档三个测试输出文件，保留该组 adapter 与训练元数据，再执行同一命令。

在解压后的隔离副本（`artifacts/` 完整）中执行：

```bash
(                                        # 用子 shell 包住：块内失败只中止本块，不改动当前 shell
  set -e                                 # 任一步失败即停止，不保留部分状态
  mkdir artifacts/B/backup-test-20261019 # 不用 -p：备份目录已存在即报错，避免覆盖既有备份
  mv artifacts/B/test_predictions.jsonl artifacts/B/test_eval_config.json artifacts/B/test_metrics.json artifacts/B/backup-test-20261019/
  uv run python asr.py eval --group B --split test   # predictions 已不在，会重新生成全部 1,315 条
  uv run python summarize.py
)
```

判断依据是日志中按语言推进的生成进度，而非退出码：续跑跳过全部行时退出码同样为 0，故命令成功退出不表明生成了新预测。

### 方式 4：从头重训 A/pilot/B/C（在隔离副本中）

当前目录中的 B/C 已完成 800 步，以相同训练命令重跑会读取恢复点并立即结束，不会重新训练 800 步；A 与 pilot 同理：已完成的 pilot 训练为 no-op，已有 `test_predictions.jsonl` 的 A 评测会跳过全部行。因此仅归档 B/C 不构成整条流水线的重跑——50 步显存与时耗标定（pilot）与 A 的零样本基线将沿用旧结果，新硬件未经重新标定。

应将交付包解压至**新位置**（其中不预留 `.venv`、模型缓存或备份目录），先将整个 `artifacts/` 一次性归档，再按第 2 节执行：

```bash
(                                        # 用子 shell 包住：块内失败只中止本块，不改动当前 shell
  set -e
  test ! -e artifacts.delivered          # 已有归档则停止，不覆盖上一次交付结果
  mv artifacts artifacts.delivered       # 一次归档 A/pilot/B/C 全部结果，原工作区不被破坏
  uv run python asr.py prepare           # 交付包不含 data/：首次需下载或构建；仅复用已存在且身份匹配的缓存
)
```

归档之后 A、pilot、B、C 才会重新执行：A 重算零样本基线，pilot 重做 50 步标定，B/C 重新训练 800 步。若需重建数据特征，须一并归档 `data/` 后再执行 `prepare`（需要网络与磁盘空间）。完整顺序见第 2 节；正式 B/C 只接受冻结的 800 步预算，50 步标定须使用 `pilot` 组。

不得以删除单个文件的方式重置某次运行：删除 `resume.pt` 而保留 `adapter/` 或完成标记时，训练会在任何 GPU 访问与写入之前报错；完成标记与恢复点步数不一致（如标记 800、恢复点 400）同样被拒绝。旧的预测或评测记录也不会因训练重跑而成为新证据。

## 2. 完整命令序列

`asr.py` 只支持以下子命令和参数（`--group` 默认 A，`--split` 默认 test）：

| 参数 | 取值 | 说明 |
|---|---|---|
| 子命令 | `prepare` / `train` / `eval` | 注意是 `eval`，不是 `evaluate` |
| `--group` | `A` / `B` / `C` / `pilot` | A 不允许训练 |
| `--split` | `validation` / `test` | 默认 test |
| `--steps` | 整数 | 正式 B/C 只接受冻结预算；0、负数、超预算均被拒绝 |
| `--limit-per-language` | 整数 | 每种语言限条数，用于小规模冒烟评测 |
| `--config` | 路径 | 默认 `config.json` |

命令按以下顺序执行：

```bash
uv sync --frozen
uv run python -m unittest -v test_pipeline
uv run python asr.py prepare

# A：测试集全量（1,315 条）
uv run python asr.py eval --group A --split test

# 独立 50 步可行性标定（不用于初始化 B/C）
uv run python asr.py train --group pilot --steps 50
uv run python asr.py eval --group pilot --split validation --limit-per-language 10

# 核查 pilot 的显存、时耗、loss 与生成结果，确认无需降级后再启动 B
uv run python asr.py train --group B
uv run python asr.py eval --group B --split validation
uv run python asr.py eval --group B --split test
# 确认 B 整体测试 MER ≤ A 且输出崩溃率 < 5%，否则停止并诊断，不得进入 C
uv run python asr.py train --group C
uv run python asr.py eval --group C --split validation
uv run python asr.py eval --group C --split test
uv run python summarize.py
```

所有正式训练均为 800 次**优化器更新**，而非 800 个 microbatch。每步 8 条、每条补齐至 30 秒。B/C 的初始化 SHA-256、学习率、scheduler、步数、LoRA、数据池与解码配置必须一致；正式运行时不得修改根目录 `config.json`。

正常训练不启用同步 CUDA 诊断；只有出现 CUDA 异常时才临时设置 `CUDA_LAUNCH_BLOCKING=1` 定位（见 [`evidence/milestones/B_INTERRUPTION.md`](../evidence/milestones/B_INTERRUPTION.md)），定位后 `unset CUDA_LAUNCH_BLOCKING` 再继续。

## 3. 中断、续跑与重复执行

- 训练每 50 步保存 adapter、optimizer、scaler、scheduler、PyTorch CPU/CUDA、NumPy、Python 随机状态。中断后**用原命令原配置**重跑即可从最后保存点继续。
- 续跑要求配置完全一致：配置哈希、数据集/模型 revision、前缀映射、完成步数均按冻结值核对，续跑状态以 `resume.pt` 为准；adapter SHA-256 不参与训练核对，只在评测与汇总阶段绑定。如需修改配置，应重新训练，不得拼接两套配置的产物。
- **`resume.pt` 是续跑的权威来源**，完成标记 `training_complete.json` 只有在 `resume.pt` 存在且步数一致时才被采信。出现「有 `adapter/` 或完成标记但没有 `resume.pt`」或「完成标记步数与恢复点不符」时，训练会在**任何 GPU 访问与写入之前**报错，不会静默当成已完成。
- 日志里可能出现最后保存点之后的重复 step（被丢弃的更新）。汇总时按最后一次相同步数取值；每个日志 step 都对应真实更新。遇到 FP16 跳步会降低 loss scale、用同一批数据重试且不增加完成步数；连续 16 次失败或 loss 非有限才停止。
- 已完成的 B/C 以相同命令重跑为字节级 no-op；`--steps 0` 不会被替换为 800，而是直接拒绝。
- 生成按样本 ID 断点续跑；同一个预测文件不允许混用不同 adapter 或解码配置。

## 4. 缓存与身份不匹配的处理

准备缓存和评测结果都带身份校验（数据集/模型名称与 revision、语言前缀、完整配置哈希、完成步数、adapter SHA-256）。缺少新身份字段的旧缓存不会被自动认可。

遇到 `Cache preparation identity mismatch` 或 `Evaluation must match the frozen training config` 时：

1. 先归档旧目录，不要删除，也不要手工补写 `ready.json` 或删掉校验字段来“升级”；
2. 重新 `prepare`（需要网络与存储），再用 `artifacts/data_audit.json` 的条数、语言比例、时长核对数值是否一致；
3. 评测身份不符时，恢复完整且匹配的 `artifacts/`，或重建对应 adapter；不要把新旧预测混在同一文件。

参考先例：本次审查后曾完整重建缓存，3 个 split 的 9 个文件 SHA-256 逐一相同、审计数值不变（`evidence/checks/CACHE_REBUILD_VERIFICATION.json` 记录了这 9 个哈希）。旧交付包与核验快照仍保存在 `delivery/pre-review/`，未就地改写；旧缓存副本与原件哈希一致，已在整理时删除，只保留记录，不保留重复数据。

## 5. 交付与校验

- `artifacts/data_audit.json`：split 条数、语言比例、实际时长、标签不一致计数。
- `artifacts/{A,B,C}/test_predictions.jsonl`：参考、原始预测、token IDs、语言/时长、崩溃标记。
- `artifacts/{B,C}/adapter/`：可直接加载的 PEFT LoRA checkpoint（必须配合固定 revision 的基础模型）。
- `artifacts/{pilot,B,C}/`：`config.json`、`initialization.json`、`sampling.json`、`progress.json`、`training_complete.json`、`train.jsonl`、`sample_indices.npy`、`resume.pt`。
- `artifacts/comparison.json`、`artifacts/training_summary.json`：统一重算的指标、门槛与对照可比性。
- `artifacts/error_review_40.json`：固定种子抽取的 40 条 mixed 样例；AI 复核报告在 `reports/ERROR_ANALYSIS.md`。
- `reports/RESULTS.md`：由 `summarize.py` 生成；`reports/FINAL_REPORT.md`、`ERROR_ANALYSIS.md`、`VERIFICATION.md` 为 AI 整理的非自动汇总文档。
- `evidence/`：过程记录，不是结论——训练/评测原始日志（`logs/`）、里程碑与中断说明（`milestones/`）、审查与测试快照（`reviews/`）、机器核验 JSON（`checks/`）。引用结论仍以 `reports/` 与 `artifacts/` 为准，索引见 `evidence/README.md`。
- 校验清单：每个归档包内部各带一份清单（主包 `SHA256SUMS` 101 项、外部补充包 `EXTERNAL_SHA256SUMS` 69 项，均不含自身）。清单描述归档内部的路径，仓库工作树不保留副本——本版归档后目录已整理（见 [`evidence/README.md`](../evidence/README.md)），工作树与清单不再对应，核验须在归档解压目录执行 `sha256sum -c`。
- `delivery/asr-finetune-complete.tar.gz`：可移交包，不含 `.venv`、基础模型缓存和 `data/` 特征。
- `delivery/asr-external-eval.tar.gz`：外部验证补充包（约 7.7 MiB，70 个文件；包内 `EXTERNAL_SHA256SUMS` 校验其中 69 个，不含清单自身）。含代码、方法、汇总指标、FLEURS 复算材料、6 个校验 tokenizer 文件与 B/C 各 4 个身份文件；不含基础模型权重、音频，也不含 CS-Dialogue 逐条参考与预测，因此公开包不能独立复算 CS。许可与转交边界见 [EXTERNAL_DATA_LICENSES](EXTERNAL_DATA_LICENSES.md)。包内部的入口页是归档里的 `EXTERNAL_EVAL.md`（冻结副本，本仓库不再保留同一份）。

**交付核验**：审查后的同版代码、权重与预测已在隔离解压目录通过 **17 项测试、101 个文件哈希和 CPU 指标精确复算**；验证时禁用 GPU、阻断网络连接与 DNS，并使用空模型缓存。正式归档后的结果、大小与哈希另附于 `delivery/verification.json` 和 `.sha256` 文件，避免归档自引用；审查前的 7 项/90 文件快照保存在 `delivery/pre-review/`。核验复用了本机已安装的固定依赖；重新生成仍需准备数据与基础模型。
