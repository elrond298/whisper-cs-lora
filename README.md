# Whisper-small LoRA 微调：ASCEND 中英混说语音（RTX 4060 Laptop 8GB）

本研究在 ASCEND 中英混说语音语料上以 LoRA 微调 Whisper-small，检验提高混合语种样本的采样概率能否优于自然采样。主指标为测试集 mixed 子集的 MER（编辑错误率）。实验协议在训练开始前冻结，数据来源与计分口径见 [`reports/protocol_sources.md`](reports/protocol_sources.md)。

## 1. 数据、组别、结论与对应文档

**数据**：ASCEND 论文作者发布于 Hugging Face 的 [`CAiRE/ASCEND`](https://huggingface.co/datasets/CAiRE/ASCEND)，固定 revision `737e9800ae31be9932ba8464c80366559bd28424`，共 12,314 条、10.62 h。该数据集自带 train / validation / test 三个划分（依次 9,869 / 1,130 / 1,315 条），本研究直接沿用，未重新切分；下文所述“训练集”“测试集”均指上述划分。

**mixed** 指同一句内中英混说（code-switching），为数据集 `language` 字段的一个取值，在 train 划分中 2,739 条，占 27.8%；另两个取值为 `zh` 与 `en`。

实验包含三个被比较的系统（A、B、C）与一次独立的可行性标定（pilot），定义如下：

| 组别 | 定义 | 训练量 |
|---|---|---|
| **A** | 原始 Whisper-small，零样本，不训练 | — |
| **B** | LoRA 微调，按训练集自然语种比例有放回抽样 | 800 次优化器更新 |
| **C** | LoRA 微调，mixed 采样概率提至 50%，其余设置与 B 相同 | 800 次优化器更新 |
| pilot | 50 步可行性标定，仅核查显存与时耗是否满足要求，不参与结论 | 50 次优化器更新 |

B、C 均从同一基础模型出发，LoRA 初始化一致，C 不接续 B 的训练状态。每组 800 次优化器更新对应 800 × 8 = 6,400 次样本抽样；因有放回，去重后的独立样本数为 B 4,698 条、C 4,437 条。

| 主题 | 结论 | 展开 |
|---|---|---|
| 测试集主结果 | 整体 MER 从 42.141% 降到 16.067%（B）/ 16.008%（C）；C 比 B 的 mixed MER 低 0.287pp，中文 CER 高 0.353pp | 第 2 节 · [FINAL_REPORT](reports/FINAL_REPORT.md) §3 · [RESULTS](reports/RESULTS.md) |
| 稳定性边界 | 验证集 mixed 方向相反（B 19.854% → C 20.146%，+0.292pp）；单种子、无显著性检验 | [FINAL_REPORT](reports/FINAL_REPORT.md) §4 |
| 实验协议与冻结变量 | 采样分布是唯一预设变量；步数、学习率、LoRA、种子在训练前冻结 | [EXPERIMENT](reports/EXPERIMENT.md) |
| 数据与计分口径 | 沿用数据集自带划分；MER 按中文计字、英文计词混合计数 | [protocol_sources](reports/protocol_sources.md) · [FINAL_REPORT](reports/FINAL_REPORT.md) §2 |
| 错误分析 | 40 条 C mixed 逐条文本复核：12 条零编辑错误、28 条有错；是 AI 文本对照，不是真人听音 | [ERROR_ANALYSIS](reports/ERROR_ANALYSIS.md) |
| 代码审查与修复 | 4 项确认问题全部关闭、17 项测试通过；权重、预测与评分未因此改变 | [REVIEW_FIXES](reports/REVIEW_FIXES.md) |
| 交付核验 | 隔离解压（禁用 GPU、断网）下 17 项测试、101 个文件哈希、CPU 精确复算通过 | [VERIFICATION](reports/VERIFICATION.md) · 第 8 节 |
| CS-Dialogue 外部验证 | 450 条域外混合语音：A 26.181%、B 12.275%、C 13.231%，C 未复现过采样优势 | 第 3 节 · [EXTERNAL_EVAL_REPORT](reports/EXTERNAL_EVAL_REPORT.md) |
| FLEURS 单语保留 | 中文 CER 降约 8pp；英文 WER 分别 +1.047、+0.914pp | [EXTERNAL_EVAL_REPORT](reports/EXTERNAL_EVAL_REPORT.md) |
| 重跑与重训 | 四种使用方式：只读结果、CPU 复算、重新生成、隔离副本从头重训 | 第 4 节 |

## 2. 主要结果

测试集 1,315 条（mixed 373、中文 685、英文 257）。单位是百分比，越低越好；差值一律为**绝对百分点（pp）**。

| 组别 | mixed MER | 中文 CER | 英文 WER | 整体 MER | 崩溃率 |
|---|---:|---:|---:|---:|---:|
| A 原始 Whisper-small | 48.706 | 37.414 | 34.290 | 42.141 | 0.152 |
| B 自然采样 LoRA | 15.918 | 14.422 | 24.356 | 16.067 | 0.000 |
| C mixed 过采样 LoRA | 15.631 | 14.775 | 24.062 | 16.008 | 0.000 |
| C−B | −0.287pp | +0.353pp | −0.294pp | −0.060pp | 0pp |

- 指标口径：MER = 编辑错误数 ÷ 参考计分单位数，中文按字、英文按词混合计数；CER 只统计纯中文子集（按非空白字符）；WER 只统计纯英文子集（按英文词）。`jiwer.mer` 不是本实验的 MER。
- pp 与百分比的差别：15.918% → 15.631% 是 **−0.287pp**（约 −1.8% 的相对变化），不是“降低 0.287%”。

**结论**：预设方向性门槛通过，但效应量很小：mixed 子集净减少 16 个编辑错误，整体净减少 7 个；验证集 mixed 子集方向相反（B 19.854% → C 20.146%）。局限见第 9 节。

### 2.1 实际训练设置

下表数值均由 `config.json`、`artifacts/` 下的产物与训练日志导出。两组唯一差异为 mixed 的采样概率。

| 项目 | 实际取值 | 出处 |
|---|---|---|
| 数据池 | train 划分 9,869 条 / 8.775 h：mixed 2,739（27.8%，3.617 h）、中文 4,799（48.6%，3.505 h）、英文 2,331（23.6%，1.653 h） | `artifacts/data_audit.json` |
| 实际入模条数 | 800 步 × 每步 8 条 = **6,400 次抽样**；有放回采样，去重后 B 4,698 条、C 4,437 条。B 抽到 mixed 1,812 / 中文 3,085 / 英文 1,503（28.3 / 48.2 / 23.5%）；C 为 3,220 / 2,147 / 1,033（50.3 / 33.5 / 16.1%） | `artifacts/{B,C}/sampling.json` |
| 语音时长 | 两组补齐输入相同：800 × 8 × 30 s = **53.33 h**；其中真实语音 B 5.681 h、C 6.531 h | 同上 |
| LoRA 作用位置 | `q_proj`、`v_proj`：encoder 12 层 self-attn、decoder 12 层 self-attn 与 cross-attn，共 72 个模块 / 144 个张量；rank 8、alpha 16、dropout 0.05 | `config.json`、`artifacts/{B,C}/adapter/adapter_model.safetensors` |
| 可训练参数 | **884,736**（占 241.73M 基座权重的 0.366%），B/C 相同 | `artifacts/{B,C}/initialization.json` |
| 优化 | AdamW，lr 1e-4，线性衰减 + 50 步 warmup，weight decay 0.01，梯度裁剪 1.0，每组 800 次优化器更新，seed 42 | `config.json`、`asr.py` |
| 显存与精度 | fp16 autocast + GradScaler；microbatch 1 × 梯度累积 8；启用 gradient checkpointing（non-reentrant） | `config.json`、`asr.py` |
| 显存峰值 | B allocated 1,134.6 MiB / reserved 1,278 MiB；C 1,136.4 / 1,270 MiB | `artifacts/training_summary.json` |
| 训练耗时 | 单步中位数 B 8.73 s、C 2.48 s；按每步日志累加 B 2.02 h、C 0.55 h | `evidence/logs/B_train*.log`、`evidence/logs/C_train.log` |
| 语言前缀 | mixed 与中文都喂 `zh` 语言 token，英文喂 `en`；task 固定 `transcribe`；训练与推理读同一份映射，不存在训练用中文前缀、评测换标签的情况 | `config.json`、`asr.py` |

两处指标的量测范围如下：

- **显存**：PyTorch 进程的 `max_memory_allocated/reserved` 峰值，不含驱动、桌面和其它进程；B 训练期间全卡占用约 6.4 GiB。
- **时耗**：B 经历一次中断与两次续跑，墙钟 2 h 13 min，同期存在其它 GPU ASR 与视频转码进程；B 最后 300 步平均 2.80 s/步。两组的每步时耗不用于性能比较。

## 3. CS-Dialogue 外部混合语音验证

为检验结论的跨语料可迁移性，将 A、B、C 三个模型用于未参与训练与选模的 CS-Dialogue 混合语音子集。该子集含 450 条混合短句、30 位官方测试说话人，共 100.2 分钟音频；不重新训练，直接加载 B、C 第 800 步的 adapter 推理。

| 组别 | mixed MER | 相对 A | 异常样本 |
|---|---:|---:|---:|
| A 原始 Whisper-small | 26.181% | — | 3/450 |
| B 自然采样 LoRA | **12.275%** | −13.906pp | 2/450 |
| C mixed 过采样 LoRA | 13.231% | −12.950pp | 3/450 |
| C−B | +0.956pp | 95% 区间 [−0.480, 3.781] | — |

- **微调收益可迁移**：B、C 在该域外混合语音上分别比原模型降低 13.906 与 12.950 个百分点。
- **过采样优势未迁移**：C 比 B 高 0.956pp，配对区间跨 0；方向与 ASCEND 测试集（C 低 0.287pp）相反，两次差异均在 1pp 以内。
- **计分口径**：MER 沿用本项目 ASCEND 作者兼容口径，非 CS-Dialogue 论文的原生评测程序，两者数值不可直接比较。
- **异常输出**：重复、截断至上限、正文前的语言前缀标记三类异常，A、B、C 分别为 3/450、2/450、3/450。
- FLEURS 中文与英文的单语保留结果、配对区间及数据许可边界见 [EXTERNAL_EVAL_REPORT](reports/EXTERNAL_EVAL_REPORT.md)。

## 4. 四种使用方式

| 方式 | 操作 | 前置条件 | 是否需要网络 |
|---|---|---|---|
| 1 读结果 | 阅读报告 | 无 | 否 |
| 2 CPU 复算 | 由已保存预测复算指标 | 仓库 + `artifacts/` + 固定版本依赖 | 首次安装依赖可能需要 |
| 3 重新生成 | 以已训练 adapter 重新生成预测 | 方式 2 的全部 + 基础模型 + `data/` + CUDA | 准备数据/模型时需要 |
| 4 从头训练 | 在隔离副本中重训 A/pilot/B/C | 方式 3 的全部 + GPU 时间 | 是 |

`uv sync --frozen` 需要访问软件包源；依赖不随交付包提供，因此新机器首次安装通常需要网络。完全离线安装仅适用于已具备固定依赖与所需资产的机器。

### 方式 1：只读结果

本方式不需要 GPU、依赖或数据。第 1 节的对应表将各结论链接至详细文档；可复算数值见 [`reports/RESULTS.md`](reports/RESULTS.md)，计分口径与验收项见 [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)。

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

应将交付包解压至**新位置**（其中不预留 `.venv`、模型缓存或备份目录），先将整个 `artifacts/` 一次性归档，再按第 5 节执行：

```bash
(                                        # 用子 shell 包住：块内失败只中止本块，不改动当前 shell
  set -e
  test ! -e artifacts.delivered          # 已有归档则停止，不覆盖上一次交付结果
  mv artifacts artifacts.delivered       # 一次归档 A/pilot/B/C 全部结果，原工作区不被破坏
  uv run python asr.py prepare           # 交付包不含 data/：首次需下载或构建；仅复用已存在且身份匹配的缓存
)
```

归档之后 A、pilot、B、C 才会重新执行：A 重算零样本基线，pilot 重做 50 步标定，B/C 重新训练 800 步。若需重建数据特征，须一并归档 `data/` 后再执行 `prepare`（需要网络与磁盘空间）。完整顺序见第 5 节；正式 B/C 只接受冻结的 800 步预算，50 步标定须使用 `pilot` 组。

不得以删除单个文件的方式重置某次运行：删除 `resume.pt` 而保留 `adapter/` 或完成标记时，训练会在任何 GPU 访问与写入之前报错；完成标记与恢复点步数不一致（如标记 800、恢复点 400）同样被拒绝。旧的预测或评测记录也不会因训练重跑而成为新证据。

## 5. 完整命令序列

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

正常训练不启用同步 CUDA 诊断；只有出现 CUDA 异常时才临时设置 `CUDA_LAUNCH_BLOCKING=1` 定位（见 [`evidence/milestones/B_INTERRUPTION.md`](evidence/milestones/B_INTERRUPTION.md)），定位后 `unset CUDA_LAUNCH_BLOCKING` 再继续。

## 6. 中断、续跑与重复执行

- 训练每 50 步保存 adapter、optimizer、scaler、scheduler、PyTorch CPU/CUDA、NumPy、Python 随机状态。中断后**用原命令原配置**重跑即可从最后保存点继续。
- 续跑要求配置完全一致：配置哈希、数据集/模型 revision、前缀映射、完成步数均按冻结值核对，续跑状态以 `resume.pt` 为准；adapter SHA-256 不参与训练核对，只在评测与汇总阶段绑定。如需修改配置，应重新训练，不得拼接两套配置的产物。
- **`resume.pt` 是续跑的权威来源**，完成标记 `training_complete.json` 只有在 `resume.pt` 存在且步数一致时才被采信。出现「有 `adapter/` 或完成标记但没有 `resume.pt`」或「完成标记步数与恢复点不符」时，训练会在**任何 GPU 访问与写入之前**报错，不会静默当成已完成。
- 日志里可能出现最后保存点之后的重复 step（被丢弃的更新）。汇总时按最后一次相同步数取值；每个日志 step 都对应真实更新。遇到 FP16 跳步会降低 loss scale、用同一批数据重试且不增加完成步数；连续 16 次失败或 loss 非有限才停止。
- 已完成的 B/C 以相同命令重跑为字节级 no-op；`--steps 0` 不会被替换为 800，而是直接拒绝。
- 生成按样本 ID 断点续跑；同一个预测文件不允许混用不同 adapter 或解码配置。

## 7. 缓存与身份不匹配的处理

准备缓存和评测结果都带身份校验（数据集/模型名称与 revision、语言前缀、完整配置哈希、完成步数、adapter SHA-256）。缺少新身份字段的旧缓存不会被自动认可。

遇到 `Cache preparation identity mismatch` 或 `Evaluation must match the frozen training config` 时：

1. 先归档旧目录，不要删除，也不要手工补写 `ready.json` 或删掉校验字段来“升级”；
2. 重新 `prepare`（需要网络与存储），再用 `artifacts/data_audit.json` 的条数、语言比例、时长核对数值是否一致；
3. 评测身份不符时，恢复完整且匹配的 `artifacts/`，或重建对应 adapter；不要把新旧预测混在同一文件。

参考先例：本次审查后曾完整重建缓存，3 个 split 的 9 个文件 SHA-256 逐一相同、审计数值不变（`evidence/checks/CACHE_REBUILD_VERIFICATION.json` 记录了这 9 个哈希）。旧交付包与核验快照仍保存在 `delivery/pre-review/`，未就地改写；旧缓存副本与原件哈希一致，已在整理时删除，只保留记录，不保留重复数据。

## 8. 交付与校验

- `artifacts/data_audit.json`：split 条数、语言比例、实际时长、标签不一致计数。
- `artifacts/{A,B,C}/test_predictions.jsonl`：参考、原始预测、token IDs、语言/时长、崩溃标记。
- `artifacts/{B,C}/adapter/`：可直接加载的 PEFT LoRA checkpoint（必须配合固定 revision 的基础模型）。
- `artifacts/{pilot,B,C}/`：`config.json`、`initialization.json`、`sampling.json`、`progress.json`、`training_complete.json`、`train.jsonl`、`sample_indices.npy`、`resume.pt`。
- `artifacts/comparison.json`、`artifacts/training_summary.json`：统一重算的指标、门槛与对照可比性。
- `artifacts/error_review_40.json`：固定种子抽取的 40 条 mixed 样例；AI 复核报告在 `reports/ERROR_ANALYSIS.md`。
- `reports/RESULTS.md`：由 `summarize.py` 生成；`reports/FINAL_REPORT.md`、`ERROR_ANALYSIS.md`、`VERIFICATION.md` 为 AI 整理的非自动汇总文档。
- `evidence/`：过程记录，不是结论——训练/评测原始日志（`logs/`）、里程碑与中断说明（`milestones/`）、审查与测试快照（`reviews/`）、机器核验 JSON（`checks/`）。引用结论仍以 `reports/` 与 `artifacts/` 为准，索引见 `evidence/README.md`。
- `SHA256SUMS`：本版 101 个交付文件的 SHA-256 清单。解压后运行 `sha256sum -c SHA256SUMS` 核对完整性；修改代码、文档或派生文件后需要重新生成清单，不能继续引用旧校验结论。该清单与 `EXTERNAL_SHA256SUMS` 描述的是两个归档包内部的结构；仓库工作树在本版交付后做过目录整理（见 `evidence/README.md`），路径可能与清单不同，核验请在归档解压目录执行。
- `delivery/asr-finetune-complete.tar.gz`：可移交包，不含 `.venv`、基础模型缓存和 `data/` 特征。
- `delivery/asr-external-eval.tar.gz`：外部验证补充包（约 7.7 MiB，70 个文件，`EXTERNAL_SHA256SUMS` 校验其中 69 个，不含清单自身）。含代码、方法、汇总指标、FLEURS 复算材料、6 个校验 tokenizer 文件与 B/C 各 4 个身份文件；不含基础模型权重、音频，也不含 CS-Dialogue 逐条参考与预测，因此公开包不能独立复算 CS。许可与转交边界见 [EXTERNAL_DATA_LICENSES](reports/EXTERNAL_DATA_LICENSES.md)。包内部的入口页是归档里的 `EXTERNAL_EVAL.md`（冻结副本，本仓库不再保留同一份）。

**交付核验**：审查后的同版代码、权重与预测已在隔离解压目录通过 **17 项测试、101 个文件哈希和 CPU 指标精确复算**；验证时禁用 GPU、阻断网络连接与 DNS，并使用空模型缓存。正式归档后的结果、大小与哈希另附于 `delivery/verification.json` 和 `.sha256` 文件，避免归档自引用；审查前的 7 项/90 文件快照保存在 `delivery/pre-review/`。核验复用了本机已安装的固定依赖；重新生成仍需准备数据与基础模型。

## 9. 局限

- **单随机种子**：每组只跑一次（seed 42），没有多种子复现、显著性检验或置信区间。
- **效应量小**：mixed 子集净减少 16 个编辑错误，整体净减少 7 个；验证集 mixed 子集方向相反（B 19.854% → C 20.146%）。
- **输入预算按补齐秒数匹配**：每步 8 条 × 30 秒补齐，不是真实语音秒数；C 实际多约 0.850 h 语音，暴露量差异与采样比例混在一起。
- **测试集参与决策**：测试集用于 B→C 的进入门槛，存在选择偏差；0.3pp 是事先约定的容忍范围，不是统计噪声界。
- **解码给定语言标签**：mixed 与中文用 `zh`、英文用 `en`，是 oracle 设置，不评估自动语言检测。
- **错误分析是文本对照**：40 条由 AI 读文本归类，未听音；音近字与疑似翻译的成因只标“疑似”。
- **计分口径**：ASCEND 作者兼容的对称标点清理、小写与 jieba 分词，保留语气词与重复；未做繁简转换或数字展开。
- **与论文不可直接对比**：论文 27.05% 的测试 MER 出自 wav2vec 2.0；A 是本项目实测的 Whisper-small 零样本基线。
- **硬件指标的范围**：显存为 PyTorch 进程峰值；时耗受同期共享负载影响，不反映采样策略的开销。
- **外部验证只覆盖子集**：CS-Dialogue 450 条、FLEURS 各 300 条，区间不含训练随机性；CS 的逐条参考与预测受许可限制，不随公开包分发。
- **未覆盖**：SenseVoiceSmall 对比、40% 敏感性组、真人听音、部署端自动语言检测。
