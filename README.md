# ASCEND × Whisper-small LoRA（RTX 4060 Laptop 8GB）

在 ASCEND 官方语音数据上用 LoRA 微调 Whisper-small，检验“把中英混说（mixed）的采样概率提到 50%”能否比自然采样更好。主指标是测试集 mixed 子集的 MER（编辑错误率）。实验协议在训练前冻结，数据来源与论文口径见 [`reports/protocol_sources.md`](reports/protocol_sources.md)。

## 1. 组别、结论与对应文档

后面反复出现 A、B、C，先定义清楚。**mixed** 指同一句话里中英混说（code-switching），是 ASCEND 官方训练集里的一个子集：2,739 条，占 27.8%。

| 组别 | 是什么 | 训练量 |
|---|---|---|
| **A** | 原始 Whisper-small，零样本，不训练 | — |
| **B** | LoRA 微调，按训练集的自然语言比例有放回抽样 | 800 次优化器更新 |
| **C** | LoRA 微调，把 mixed 的采样概率提到 50%，其余与 B 相同 | 800 次优化器更新 |
| pilot | 50 步可行性标定，只看显存和耗时够不够，不进任何结论 | 50 次优化器更新 |

B、C 从同一个原模型出发、LoRA 初始化相同，C 不接续 B。每组 800 次优化器更新 = 800 × 8 条 = 6,400 次抽样，不是 6,400 个不同样本。

| 主题 | 结论 | 展开 |
|---|---|---|
| 测试集主结果 | 整体 MER 从 42.141% 降到 16.067%（B）/ 16.008%（C）；C 比 B 的 mixed MER 低 0.287pp，中文 CER 高 0.353pp | 第 2 节 · [FINAL_REPORT](reports/FINAL_REPORT.md) §3 · [RESULTS](reports/RESULTS.md) |
| 稳定性边界 | 验证集 mixed 方向相反（B 19.854% → C 20.146%，+0.292pp）；单种子、无显著性检验 | [FINAL_REPORT](reports/FINAL_REPORT.md) §4 |
| 实验协议与冻结变量 | 采样分布是唯一预设变量；步数、学习率、LoRA、种子在训练前冻结 | [EXPERIMENT](reports/EXPERIMENT.md) |
| 数据与计分口径 | ASCEND 官方划分；MER 按中文计字、英文计词混合计数 | [protocol_sources](reports/protocol_sources.md) · [FINAL_REPORT](reports/FINAL_REPORT.md) §2 |
| 错误分析 | 40 条 C mixed 逐条文本复核：12 条零编辑错误、28 条有错；是 AI 文本对照，不是真人听音 | [ERROR_ANALYSIS](reports/ERROR_ANALYSIS.md) |
| 代码审查与修复 | 4 项确认问题全部关闭、17 项测试通过；权重、预测与评分未因此改变 | [REVIEW_FIXES](reports/REVIEW_FIXES.md) |
| 交付核验 | 隔离解压（禁用 GPU、断网）下 17 项测试、101 个文件哈希、CPU 精确复算通过 | [VERIFICATION](reports/VERIFICATION.md) · 第 7 节 |
| 外部子集评测 | CS-Dialogue 450 条与 FLEURS 中英各 300 条上的泛化与单语保留结果 | [EXTERNAL_EVAL](EXTERNAL_EVAL.md) |
| 重跑与重训 | 四种使用方式：只读结果、CPU 重算、重新生成、隔离副本从头重训 | 第 3 节 |

## 2. 结果速览

测试集 1,315 条（mixed 373、中文 685、英文 257）。单位是百分比，越低越好；差值一律为**绝对百分点（pp）**。

| 组别 | mixed MER | 中文 CER | 英文 WER | 整体 MER | 崩溃率 |
|---|---:|---:|---:|---:|---:|
| A 原始 Whisper-small | 48.706 | 37.414 | 34.290 | 42.141 | 0.152 |
| B 自然采样 LoRA | 15.918 | 14.422 | 24.356 | 16.067 | 0.000 |
| C mixed 过采样 LoRA | 15.631 | 14.775 | 24.062 | 16.008 | 0.000 |
| C−B | −0.287pp | +0.353pp | −0.294pp | −0.060pp | 0pp |

- 指标口径：MER = 编辑错误数 ÷ 参考计分单位数，中文按字、英文按词混合计数；CER 只统计纯中文子集（按非空白字符）；WER 只统计纯英文子集（按英文词）。`jiwer.mer` 不是本实验的 MER。
- pp 与百分比的差别：15.918% → 15.631% 是 **−0.287pp**（约 −1.8% 的相对变化），不是“降低 0.287%”。

**结论边界**（这四条限定了上表数字能支持什么）：

1. 预设方向性门槛通过，但收益很小：mixed 净减少 16 个编辑错误，整体只净减少 7 个。
2. **验证集 mixed 方向相反**：B 19.854% → C 20.146%（+0.292pp）。单随机种子、无显著性检验，因此只能说“本次运行有微弱方向性收益”，不能说过采样稳定更优。
3. 两组匹配的是**每步 8 条 × 30 秒补齐输入**（用户预先选定），不是真实语音秒数；C 实际多约 0.850 小时语音，仍是混杂因素。
4. 论文的 27.05% 测试 MER 来自 wav2vec 2.0，不是原始 Whisper-small 基线；A 是本项目实测的零样本基线。

### 2.1 这次到底怎样训练的

下表数值全部从 `config.json`、`artifacts/` 下的产物和训练日志导出，不是按常见教程写法补的。两组唯一的差别是 mixed 的采样概率。

| 项目 | 实际取值 | 出处 |
|---|---|---|
| 数据池 | 官方 train 9,869 条 / 8.775 h：mixed 2,739（27.8%，3.617 h）、中文 4,799（48.6%，3.505 h）、英文 2,331（23.6%，1.653 h） | `artifacts/data_audit.json` |
| 实际入模条数 | 800 步 × 每步 8 条 = **6,400 次抽样**；有放回采样，去重后 B 4,698 条、C 4,437 条。B 抽到 mixed 1,812 / 中文 3,085 / 英文 1,503（28.3 / 48.2 / 23.5%）；C 为 3,220 / 2,147 / 1,033（50.3 / 33.5 / 16.1%） | `artifacts/{B,C}/sampling.json` |
| 语音时长 | 两组补齐输入相同：800 × 8 × 30 s = **53.33 h**；其中真实语音 B 5.681 h、C 6.531 h | 同上 |
| LoRA 作用位置 | `q_proj`、`v_proj`：encoder 12 层 self-attn、decoder 12 层 self-attn 与 cross-attn，共 72 个模块 / 144 个张量；rank 8、alpha 16、dropout 0.05 | `config.json`、`artifacts/{B,C}/adapter/adapter_model.safetensors` |
| 可训练参数 | **884,736**（占 241.73M 基座权重的 0.366%），B/C 相同 | `artifacts/{B,C}/initialization.json` |
| 优化 | AdamW，lr 1e-4，线性衰减 + 50 步 warmup，weight decay 0.01，梯度裁剪 1.0，每组 800 次优化器更新，seed 42 | `config.json`、`asr.py` |
| 显存与精度 | fp16 autocast + GradScaler；microbatch 1 × 梯度累积 8；启用 gradient checkpointing（non-reentrant） | `config.json`、`asr.py` |
| 显存峰值 | B allocated 1,134.6 MiB / reserved 1,278 MiB；C 1,136.4 / 1,270 MiB | `artifacts/training_summary.json` |
| 训练耗时 | 单步中位数 B 8.73 s、C 2.48 s；按每步日志累加 B 2.02 h、C 0.55 h | `evidence/logs/B_train*.log`、`evidence/logs/C_train.log` |
| 语言前缀 | mixed 与中文都喂 `zh` 语言 token，英文喂 `en`；task 固定 `transcribe`；训练与推理读同一份映射，不存在训练用中文前缀、评测换标签的情况 | `config.json`、`asr.py` |

两个测量口径，避免误读：

- **显存**：数字来自 PyTorch 进程的 `max_memory_allocated/reserved`，不含驱动、桌面和其它进程。B 训练期间全卡占用约 6.4 GiB，不能整体算作 Whisper 的需求。
- **耗时**：B 的中断与两次续跑使墙钟达到 2 h 13 min，且同期有外部 GPU ASR 进程和视频转码争用，per-step 时间被拉长。因此 B/C、B/pilot 的耗时差异反映的是环境争用，不是采样策略的开销差异，**不要拿这组数字做性能对比**。

## 3. 四种使用方式

| 方式 | 做什么 | 需要什么 | 是否需要网络 |
|---|---|---|---|
| 1 读结果 | 直接看报告 | 无 | 否 |
| 2 CPU 重算 | 从已保存预测重算指标 | 仓库 + `artifacts/` + 固定版本依赖 | 首次装依赖可能需要 |
| 3 重新生成 | 用已训练 adapter 重跑预测 | 方式 2 的全部 + 基础模型 + `data/` + CUDA | 准备数据/模型时需要 |
| 4 从头训练 | 在隔离副本重训 A/pilot/B/C | 方式 3 的全部 + GPU 时间 | 是 |

`uv sync --frozen` 需要能访问软件包源；依赖不随交付包提供，所以新机器首次安装通常需要网络。全新机器上“完全离线”只适用于已装好固定依赖且资产已就位的情况。

### 方式 1：只读结果

不需要 GPU、依赖或数据。第 1 节的“主题 / 结论 / 展开”表把每个结论对应到详细文档；可重算的数字在 [`reports/RESULTS.md`](reports/RESULTS.md)，口径与验收逐项在 [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)。

### 方式 2：CPU 重算指标（不需要 GPU、基础模型或 `data/`）

只重算某一组指标，只需要该组的预测文件和 `metrics.py`：

```bash
uv run python -c 'import json; from metrics import summarize; rows=[json.loads(x) for x in open("artifacts/B/test_predictions.jsonl", encoding="utf-8")]; print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))'
```

完整复核（A/B/C 全部指标、验收标记、训练汇总）用：

```bash
uv run python summarize.py
```

`summarize.py` 不是“只看预测文件”就够，它还会读取并校验：

- `artifacts/{A,B,C}/test_eval_config.json` 与 `test_predictions.jsonl`；
- `artifacts/B|C/{config,initialization,training_complete}.json` 与 `adapter/adapter_model.safetensors`（用于 SHA-256 身份绑定）；
- `artifacts/data_audit.json`、`artifacts/{pilot,B,C}/train.jsonl`、`artifacts/pilot/progress.json`。

它会重写派生的 `test_metrics.json`、`comparison.json`、`training_summary.json`、`error_review_40.json` 和 `reports/RESULTS.md`，不改原始预测、adapter 或训练记录。所以缺少 adapter 或身份文件时，正确做法是恢复完整 `artifacts/`，而不是删掉校验后继续。

### 方式 3：用已训练 adapter 重新生成预测

需要方式 2 的全部文件，外加固定 revision 的基础模型和已准备、身份匹配的 `data/`。生成前会核对冻结训练配置、完成步数、配置哈希和当前 adapter 哈希；不符会在加载数据/模型或写文件之前报错。预测固定写入 `artifacts/<group>/<split>_predictions.jsonl`（同目录还有 `<split>_eval_config.json`、`<split>_metrics.json`），**没有可指定输出目录的参数**。

要确认是**真的重新推理**而不是空跑，必须先把旧的测试输出挪走。关键点：

- 整组复制后再跑同一命令没有用：文件还在就按样本 ID 续跑，1,315 条都已完成 → 全部跳过，看不到任何新预测。
- 整组移走会直接失败：`adapter/`、`config.json`、`training_complete.json` 正是评测身份校验的输入。
- 正确做法是**只归档三个测试输出文件**，保留该组的 adapter 与训练元数据，再跑同一命令。

在解压后的隔离副本（`artifacts/` 完整）中执行：

```bash
(                                        # 用子 shell 包住：块内失败只中止本块，不改动你的交互 shell
  set -e                                 # 任一步失败即停止，不带着半截状态继续
  mkdir artifacts/B/backup-test-20261019 # 故意不用 -p：备份目录已存在就报错，避免覆盖上一次备份
  mv artifacts/B/test_predictions.jsonl artifacts/B/test_eval_config.json artifacts/B/test_metrics.json artifacts/B/backup-test-20261019/
  uv run python asr.py eval --group B --split test   # predictions 已不在，会重新生成全部 1,315 条
  uv run python summarize.py
)
```

判断依据是日志里按语言推进的生成进度，不是退出码：续跑跳过全部行时退出码同样是 0，所以“命令跑完且成功”不能证明这次真的生成了新预测。

### 方式 4：从头重训 A/pilot/B/C（在隔离副本中）

当前目录里的 B/C 已经完成 800 步，直接再跑同一训练命令会读取恢复点而立即结束，**不会重训 800 步**。A 和 pilot 也一样：完成的 pilot 训练是 no-op，已有 `test_predictions.jsonl` 的 A 评测会跳过全部行。所以**只归档 B/C 不等于重跑整条流水线**——50 步显存/耗时标定（pilot）和 A 的零样本基线会沿用旧结果，新硬件并没有真正重新标定。

正确做法是把交付包解压到一个**新位置**（里面不要预留 `.venv`、模型缓存或备份目录），先把整个 `artifacts/` 一次性归档，再按第 4 节顺序跑：

```bash
(                                        # 用子 shell 包住：块内失败只中止本块，不改动你的交互 shell
  set -e
  test ! -e artifacts.delivered          # 已有归档就停下，绝不覆盖上一次交付结果
  mv artifacts artifacts.delivered       # 一次归档 A/pilot/B/C 全部结果，原工作区不被破坏
  uv run python asr.py prepare           # 交付包不含 data/：首次会下载/构建；只有已存在且身份匹配的缓存才复用
)
```

归档之后 A、pilot、B、C 才会真正重跑：A 重算零样本基线，pilot 重做 50 步标定，B/C 重新训练 800 步。若连数据特征也要重建，把 `data/` 也归档后再 `prepare`（需要网络与磁盘空间）。完整顺序见第 4 节；正式 B/C 只接受冻结的 800 步预算，50 步标定请用 `pilot` 组。

**不要**靠删单个文件“重置”某个运行：删掉 `resume.pt` 却留着 `adapter/` 或完成标记，训练会在任何 GPU 访问与写入之前直接报错；完成标记与保存的 checkpoint 步数不一致（例如标记 800、恢复点 400）同样会被拒绝。旧预测或旧评测记录也不会因为训练重跑而变成新证据。

## 4. 精确命令

`asr.py` 只支持以下子命令和参数（`--group` 默认 A，`--split` 默认 test）：

| 参数 | 取值 | 说明 |
|---|---|---|
| 子命令 | `prepare` / `train` / `eval` | 注意是 `eval`，不是 `evaluate` |
| `--group` | `A` / `B` / `C` / `pilot` | A 不允许训练 |
| `--split` | `validation` / `test` | 默认 test |
| `--steps` | 整数 | 正式 B/C 只接受冻结预算；0、负数、超预算均被拒绝 |
| `--limit-per-language` | 整数 | 每种语言限条数，用于小规模冒烟评测 |
| `--config` | 路径 | 默认 `config.json` |

严格顺序：

```bash
uv sync --frozen
uv run python -m unittest -v test_pipeline
uv run python asr.py prepare

# A：完整官方测试集
uv run python asr.py eval --group A --split test

# 独立 50 步可行性标定（不用于初始化 B/C）
uv run python asr.py train --group pilot --steps 50
uv run python asr.py eval --group pilot --split validation --limit-per-language 10

# 检查 pilot 显存、耗时、loss 和生成，确认不需要降级后再开始 B
uv run python asr.py train --group B
uv run python asr.py eval --group B --split validation
uv run python asr.py eval --group B --split test
# 确认 B 整体测试 MER <= A、输出崩溃率 <5%，否则停下诊断，不能直接进入 C
uv run python asr.py train --group C
uv run python asr.py eval --group C --split validation
uv run python asr.py eval --group C --split test
uv run python summarize.py
```

所有正式训练是 800 个**优化器更新**，不是 800 个 microbatch。每步 8 条、每条补齐到 30 秒。B/C 的初始化 SHA-256、学习率、scheduler、步数、LoRA、数据池与解码配置必须一致；正式运行时不要改根目录 `config.json`。

正常训练不启用同步 CUDA 诊断；只有出现 CUDA 异常时才临时设置 `CUDA_LAUNCH_BLOCKING=1` 定位（见 [`evidence/milestones/B_INTERRUPTION.md`](evidence/milestones/B_INTERRUPTION.md)），定位后 `unset CUDA_LAUNCH_BLOCKING` 再继续。

## 5. 中断、续跑与重复执行

- 训练每 50 步保存 adapter、optimizer、scaler、scheduler、PyTorch CPU/CUDA、NumPy、Python 随机状态。中断后**用原命令原配置**重跑即可从最后保存点继续。
- 续跑要求配置完全一致：配置哈希、数据集/模型 revision、前缀映射、完成步数都按冻结值核对，续跑状态以 `resume.pt` 为准；adapter SHA-256 不参与训练核对，只在评测与汇总阶段绑定。想改配置就重训，不要拼接两个配置的产物。
- **`resume.pt` 是续跑的权威来源**，完成标记 `training_complete.json` 只有在 `resume.pt` 存在且步数一致时才被采信。出现「有 `adapter/` 或完成标记但没有 `resume.pt`」或「完成标记步数与恢复点不符」时，训练会在**任何 GPU 访问与写入之前**报错，不会静默当成已完成。
- 日志里可能出现最后保存点之后的重复 step（被丢弃的更新）。汇总时按最后一次相同步数取值；每个日志 step 都对应真实更新。遇到 FP16 跳步会降低 loss scale、用同一批数据重试且不增加完成步数；连续 16 次失败或 loss 非有限才停止。
- 已完成的 B/C 再用相同命令重跑是**真正的 no-op**（文件字节不变）；`--steps 0` 不会被悄悄替换成 800，而是直接拒绝。
- 生成按样本 ID 断点续跑；同一个预测文件不允许混用不同 adapter 或解码配置。

## 6. 缓存与身份不匹配怎么处理

准备缓存和评测结果都带身份校验（数据集/模型名称与 revision、语言前缀、完整配置哈希、完成步数、adapter SHA-256）。缺少新身份字段的旧缓存不会被自动认可。

遇到 `Cache preparation identity mismatch` 或 `Evaluation must match the frozen training config` 时：

1. 先归档旧目录，不要删除，也不要手工补写 `ready.json` 或删掉校验字段来“升级”；
2. 重新 `prepare`（需要网络与存储），再用 `artifacts/data_audit.json` 的条数、语言比例、时长核对数值是否一致；
3. 评测身份不符时，恢复完整且匹配的 `artifacts/`，或重建对应 adapter；不要把新旧预测混在同一文件。

参考先例：本次审查后曾完整重建缓存，3 个 split 的 9 个文件 SHA-256 逐一相同、审计数值不变（`evidence/checks/CACHE_REBUILD_VERIFICATION.json` 记录了这 9 个哈希）。旧交付包与核验快照仍保存在 `delivery/pre-review/`，未就地改写；旧缓存副本与原件哈希一致，已在整理时删除，只保留记录，不保留重复数据。

## 7. 交付与校验

- `artifacts/data_audit.json`：split 条数、语言比例、实际时长、标签不一致计数。
- `artifacts/{A,B,C}/test_predictions.jsonl`：参考、原始预测、token IDs、语言/时长、崩溃标记。
- `artifacts/{B,C}/adapter/`：可直接加载的 PEFT LoRA checkpoint（必须配合固定 revision 的基础模型）。
- `artifacts/{pilot,B,C}/`：`config.json`、`initialization.json`、`sampling.json`、`progress.json`、`training_complete.json`、`train.jsonl`、`sample_indices.npy`、`resume.pt`。
- `artifacts/comparison.json`、`artifacts/training_summary.json`：统一重算的指标、门槛与对照可比性。
- `artifacts/error_review_40.json`：固定种子抽取的 40 条 mixed 样例；AI 复核报告在 `reports/ERROR_ANALYSIS.md`。
- `reports/RESULTS.md`：由 `summarize.py` 生成；`reports/FINAL_REPORT.md`、`ERROR_ANALYSIS.md`、`VERIFICATION.md` 为 AI 整理的非自动汇总文档（不是人工撰写，也不是人工听音核验）。
- `evidence/`：过程记录，不是结论——训练/评测原始日志（`logs/`）、里程碑与中断说明（`milestones/`）、审查与测试快照（`reviews/`）、机器核验 JSON（`checks/`）。引用结论仍以 `reports/` 与 `artifacts/` 为准，索引见 `evidence/README.md`。
- `SHA256SUMS`：本版 101 个交付文件的 SHA-256 清单。解压后运行 `sha256sum -c SHA256SUMS` 核对完整性；修改代码、文档或派生文件后需要重新生成清单，不能继续引用旧校验结论。该清单与 `EXTERNAL_SHA256SUMS` 描述的是两个归档包内部的结构；仓库工作树在本版交付后做过目录整理（见 `evidence/README.md`），路径可能与清单不同，核验请在归档解压目录执行。
- `delivery/asr-finetune-complete.tar.gz`：可移交包，不含 `.venv`、基础模型缓存和 `data/` 特征。

**交付核验**：审查后的同版代码、权重与预测已在隔离解压目录通过 **17 项测试、101 个文件哈希和 CPU 指标精确复算**；验证时禁用 GPU、阻断网络连接与 DNS，并使用空模型缓存。正式归档后的结果、大小与哈希另附于 `delivery/verification.json` 和 `.sha256` 文件，避免归档自引用；审查前的 7 项/90 文件快照保存在 `delivery/pre-review/`。核验复用了本机已安装的固定依赖，并不证明全新机器可以完全离线安装或重训；重新生成仍需准备数据与基础模型。

## 8. 解释边界

- ASCEND 论文的 27.05% MER 属于 wav2vec 2.0，不是 Whisper-small；本项目不宣称复现论文 Whisper 结果。
- 计分使用 ASCEND 作者兼容的对称标点清理、小写与 jieba 分词，保留语气词与重复；未做繁简转换或数字展开。
- 解码用已知语言标签提供 zh/en 前缀（oracle 设置），不是自动语言检测评估。
- 测试集被用于 B→C 的进入门槛，存在选择偏差风险；0.3pp 是事先约定的容忍范围，不是统计验证过的噪声界。
- 40 条逐例分析是 **AI 文本对照**，不是真人听音；`0%` 输出崩溃率不代表语义正确或可生产部署。
- 未做多随机种子、显著性检验、SenseVoiceSmall 对比、40% 敏感性组、真人听音，也未验证部署端自动语言检测。硬门槛失败不等于完成。
- RTX 4060 Laptop 8GB 的显存数字是 PyTorch 进程的 allocated/reserved 峰值，不含驱动、桌面及其它进程占用，不能把全卡占用都归给本次训练。
