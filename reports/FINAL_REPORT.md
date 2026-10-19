# ASCEND × Whisper-small LoRA：最终实验报告

> **组别**：A = 原始 Whisper-small，零样本、不训练；B = LoRA 微调、按自然语言比例采样；C = LoRA 微调、mixed 采样概率 50%，其余同 B。**mixed** 指同一句内中英混说（code-switching）。B、C 均指各自第 800 步的 checkpoint。
>
> 单位约定：指标均为百分比，越低越好；差值一律为**绝对百分点（pp）**。MER = 编辑错误数 ÷ 参考计分单位数（中文按字、英文按词混合计数）；CER 为纯中文子集（按非空白字符），WER 为纯英文子集（按英文词）。

## 结论先行

**B、C均已实际完成800个有效优化器更新，完成全量验证集和测试集生成。所有数值硬门槛通过；按事先冻结的测试集方向性规则，本次单种子实验通过。**

C的mixed MER比B低 **0.287个百分点**，中文CER退化 **0.353个百分点**（未超0.5），英文WER改善 **0.294个百分点**，整体MER改善 **0.060个百分点**。结论是在相同补齐输入预算下，本次运行有微弱方向性收益。**验证集mixed方向相反：B 19.854% → C 20.146%。**

## 1. 冻结配置与实际训练

| 项目 | 实际设置 |
|---|---|
| 硬件 | NVIDIA RTX 4060 Laptop GPU，8GB |
| 基础模型 | openai/whisper-small；精确revision见config.json |
| 数据 | CAiRE/ASCEND（论文作者发布，固定 revision 见 config.json）；沿用其自带 train/validation/test 划分，未重新切分 |
| 有效更新 | B=800，C=800；每组6,400个有效训练样本选择 |
| 单步输入 | batch=1，梯度累积8；每条补齐30秒，240补齐秒/更新 |
| 精度与内存 | FP16 autocast，梯度检查点；可训练参数保持FP32 |
| LoRA | rank8 / alpha16 / dropout0.05；所有encoder/decoder q_proj、v_proj（含交叉注意力）；884,736可训练参数 |
| 优化 | AdamW，学习率1e-4，50步warmup后线性衰减，weight decay0.01，梯度裁剪1 |
| 随机种子 | 42；B/C同一原模型、相同LoRA初始化；C不接续B |
| 唯一预设训练变量 | B逐样本均匀、有放回；C mixed总采样概率50%，单语内部保持均匀 |
| 最终模型 | 固定第800步，不选测试最优checkpoint |
| 降级 | 未降级：未使用rank4或QLoRA |

B/C相同配置SHA-256：`df98034d4fc7675394f54e46c7b1ec4a7939a395e8a86beb49583b8e91e1ca0f`。

相同LoRA初始化SHA-256：`5d3bb69622c631477f8c113df9908c2aaa7da3c34c8cfc448f50c1630f86ae55`。

两个最终resume.pt均复核：144个AdamW参数状态计数器全为800，scheduler为800，保存step为800。

## 2. 数据审计与评测口径

- 数据集自带 train/validation/test 划分分别 9,869/1,130/1,315 条，实际 waveform 合计 8.774909/0.923225/0.917496 小时；未重新切分。
- train mixed 2,739条，占27.75%，实际3.616870小时；与计划估计2.88小时不同，以固定release实测为准。
- language与文本汉字/拉丁字母启发式差异87/15/6条，主要由[UNK]里的字母造成；language与参考文本保持数据集原样。
- MER为语料级编辑错误数/参考计分单位数；中文按字、英文按词，采用作者兼容的对称标点清理及jieba口径。纯中文CER按非空白字符、纯英文WER按英文计分单位；三组相同。
- greedy，num_beams=1，max_new_tokens=256；forced_decoder_ids=None、suppress_tokens=[]；默认begin_suppress_tokens保留。mixed/zh用zh前缀，en用en前缀，这是已知标签的oracle前缀。
- 崩溃按空输出、预设连续重复、语言前缀错误/正文特殊前缀、无EOS且达到长度上限的并集统计；保存完整sequences和原始token IDs。
- 论文27.05%测试MER来自wav2vec 2.0，没有对应的论文Whisper-small基线；A是本项目实测的原始Whisper-small。

## 3. 测试集主结果

全部百分比均越低越好；差值是**绝对百分点**。

| 组别 | mixed MER | 中文 CER | 英文 WER | 整体 MER | 崩溃率 |
|---|---:|---:|---:|---:|---:|
| A 原始Whisper-small | 48.706% | 37.414% | 34.290% | 42.141% | 0.152% |
| B 自然采样LoRA | 15.918% | 14.422% | 24.356% | 16.067% | 0.000% |
| C mixed过采样LoRA | 15.631% | 14.775% | 24.062% | 16.008% | 0.000% |
| C−B | −0.287pp | +0.353pp | −0.294pp | −0.060pp | 0pp |

测试总样本1,315；mixed **373条**，中文685条，英文257条。mixed参考计分单位5,566；B编辑错误886，C为870，**净减少16个编辑错误**。整体MER仅净减少7个编辑错误。

B对A的提升远大于C对B的增益，说明本次域内LoRA微调有效；新增过采样收益仅是小幅观察值。

## 4. 验证集结果与稳定性边界

| 组别 | mixed MER | 中文 CER | 英文 WER | 整体 MER | 崩溃率 |
|---|---:|---:|---:|---:|---:|
| B | 19.854% | 15.341% | 25.395% | 17.527% | 0% |
| C | 20.146% | 15.219% | 26.049% | 17.487% | 0% |

验证集 1,130 条，其中 mixed 213 条。C 的验证 mixed 比 B 差 0.292pp，与测试集方向相反。没有据此调整采样比例、学习率、步数或主指标。

## 5. 采样暴露量：匹配的量与未匹配的量

| 项目 | B | C |
|---|---:|---:|
| 有效样本选择 | 6,400 | 6,400 |
| mixed/zh/en条数 | 1,812 / 3,085 / 1,503 | 3,220 / 2,147 / 1,033 |
| 实际mixed占比 | 28.3125% | 50.3125% |
| 独立样本数 | 4,698 | 4,437 |
| 累计真实语音小时 | 5.681144 | 6.531013 |
| 有效更新的补齐输入小时 | 53.333333 | 53.333333 |

匹配的是**每步8×30秒补齐输入**，不是实际语音秒数。C实际多约0.850小时（约15%）的语音，暴露量差异与采样比例混在一起。中断/overflow的丢弃重算不计入最终模型的800次有效更新预算。

## 6. 显存、耗时与过程记录

| 阶段 | 有效步数 | 平均秒/步 | 中位秒/步 | PyTorch峰值分配MiB | 峰值预留MiB |
|---|---:|---:|---:|---:|---:|
| 独立pilot | 50 | 3.261 | 3.197 | 1122.525 | 1252 |
| B（有效步日志） | 800 | 9.077 | 8.729 | 1134.638 | 1278 |
| C | 800 | 2.475 | 2.480 | 1136.380 | 1270 |

- pilot无OOM；loss前10步均值1.176、后10步0.881；30条验证生成自检无崩溃。未将pilot权重用于B/C。
- B/C最后10步loss均值分别0.370/0.356。最终指标来自生成评测，不用训练loss替代识别质量。
- B在第68步附近报CUDA unknown error，从完整第50步状态恢复；为定位临时启用同步CUDA，在第500步恢复点后关闭诊断同步。总841条成功更新日志包含17+24条被丢弃后重算的更新，最终模型仍严格800步。
- B异常当时有其它GPU/CPU负载；未报CUDA OOM，外部根因未确认。未终止其它用户进程，未改数值超参数。FP16重放的loss差异很小但非零。
- B的同步诊断与共享负载使其与C不同；B最后300步平均2.802秒/步。表中是有效更新记录的step耗时，不含装载、保存、评测与丢弃重算。
- 显存为PyTorch进程分配/预留峰值，约1.11/1.25GiB，不含驱动、桌面和其它进程；全卡时间序列在artifacts/gpu_monitor.csv。
- 原始故障和恢复记录见evidence/milestones/B_INTERRUPTION.md；B门槛检查见evidence/milestones/B_GATE.md。

## 7. 错误分析

已按固定随机种子逐条文本复核40条C mixed样例，12条规范化后无编辑错误，28条存在编辑错误；其中4条含[UNK]，未知区域不作声学定论。

残留问题包括英文专名/拼写、词边界、英文被中文替代、中文近音字、语气词跨文字差异。例：privacy→private school，badminton→Bankman腾，Hong Kong→香港。尤其“并不需要→必须要”改变了否定含义。切换处的断句错误无法仅凭文本统计。

40条完整参考、预测、分类、备注见 [ERROR_ANALYSIS](ERROR_ANALYSIS.md) 及 evidence/checks/error_analysis_40.json；主判断使用完整373条mixed指标。

## 8. 验收逐项

| 项目 | 证据/结果 |
|---|---|
| 4060 8GB至少50步不溢出 | 独立pilot50步通过，最终B/C各800步；无容量降级 |
| 评测可复现 | 固定 revision、计分与解码；17 项测试、101 文件哈希和隔离 CPU 复算通过；论文模型差异已在正式训练前记录 |
| B/C输出不崩溃 | 两组测试均0/1,315，验证均0/1,130，低于5% |
| B不劣于A | 整体测试MER 16.067% < 42.141% |
| 对照可比 | 相同配置/初始化哈希、optimizer/scheduler计数800、每组6,400样本与53.333补齐小时；只有预设采样分布不同（其它差异见第 10 节） |
| C mixed改善 | −0.287pp，通过 |
| 中文退化≤0.5pp | +0.353pp，通过 |
| 英文退化≤0.8pp | −0.294pp，通过 |
| 整体退化≤0.3pp | −0.060pp，通过 |

最终判定：**预设数值验收通过，本次单种子过采样方向性有效；稳定性与统计显著性未获证实。** 按用户要求使用B测试门控；未因测试表现改超参数或重复搜寻C。

## 9. 交付与复现

- 代码：asr.py（准备、采样、训练、生成）、metrics.py、summarize.py、test_pipeline.py。
- 协议、依赖与配置：[EXPERIMENT](EXPERIMENT.md)、[protocol_sources](protocol_sources.md)、根目录 config.json/pyproject.toml/uv.lock。
- checkpoint：artifacts/B/adapter/、artifacts/C/adapter/；相应resume.pt包含optimizer/scheduler/scaler/RNG；A为固定原模型，没有LoRA adapter。
- 预测与指标：artifacts/{A,B,C}/test_predictions.jsonl、test_metrics.json；B/C另有完整validation预测与指标。
- 过程证据：artifacts/{pilot,B,C}/train.jsonl、sampling.json、initialization.json、progress.json、training_complete.json 及 `evidence/logs/` 下的原始训练/评测日志。
- 审查与核验：[REVIEW_FIXES](REVIEW_FIXES.md)（4 项确认问题及关闭状态）、[VERIFICATION](VERIFICATION.md)（隔离解压下的实际核验记录）。
- 汇总：artifacts/comparison.json、training_summary.json；[reports/RESULTS.md](RESULTS.md) 可由 summarize.py 重生成；`artifacts/error_review_40.json` 会由该脚本重新抽样生成，AI 文本复核结论（非真人听音）保存在 `evidence/checks/error_analysis_40.json` 与 [ERROR_ANALYSIS](ERROR_ANALYSIS.md)。
- 可移交包：`delivery/asr-finetune-complete.tar.gz`；不含 `.venv`、基础模型缓存或 `data/`。本版 101 个文件的校验清单为 `SHA256SUMS`；正式归档后的隔离核验结果及包哈希记录在归档外的 `delivery/verification.json` 与 `.sha256` 文件。
- **本轮核验**：同版代码、权重与预测的候选归档已在无 GPU、网络连接与 DNS 被阻断、模型缓存为空的解压目录通过 17 项测试；A/B/C 测试指标、B/C 验证指标精确重算，101 个文件哈希在复算前后均一致。两组各 144 个 optimizer 计数均为 800，导出的 144 个 adapter 张量分别与恢复点完全相同。正式归档后的核验由上述外部记录给出；审查前 7 项测试/90 文件的旧快照保存在 `delivery/pre-review/`。

复现评测无需GPU、基础模型或`data/`：安装固定依赖后，可按 [REPRODUCE](REPRODUCE.md) 方式 2 从保存预测重算单组指标，或运行`uv run python summarize.py`做全量复核。注意`summarize.py`还需要评测身份文件、B/C adapter与训练元数据，只保留预测JSONL不够。从头重训须在隔离解压副本中先把整个`artifacts/`一次性归档（不只是B/C：只归档B/C会让pilot走no-op、A的评测跳过全部行，新硬件的50步标定与零样本基线都不会真正重跑），再按`prepare`→A→pilot→B→C顺序执行；要确认评测是真重跑，应只归档该组三个测试输出文件而保留adapter与训练元数据。完整命令见 [REPRODUCE](REPRODUCE.md) 方式 3、方式 4；新机器准备数据/模型需要网络与相应存储。

## 10. 局限

- **单随机种子**：B、C 各只跑一次（seed 42）；没有多种子复现、显著性检验或置信区间，验证集 mixed 方向相反（+0.292pp）。
- **输入预算按补齐秒数匹配**：每步 8 条 × 30 秒补齐，不是真实语音秒数；C 实际多约 0.850 h 语音，暴露差异与采样比例混在一起。
- **测试集参与决策**：测试集用于 B→C 的进入门槛，存在选择偏差；0.3pp 是事先约定的容忍范围。
- **解码给定语言标签**：mixed 与中文用 `zh`、英文用 `en`，是 oracle 设置，不评估自动语言检测。
- **错误分析是文本对照**：40 条由 AI 读文本归类，未听音；音近字与疑似翻译只标“疑似”。
- **与论文不可直接对比**：论文 27.05% 的测试 MER 出自 wav2vec 2.0；A 是本项目实测的 Whisper-small 零样本基线。
- **硬件数字的范围**：显存是 PyTorch 进程峰值；B 的耗时含同期共享负载（最后 300 步 2.80 s/步）。
- **核验范围**：复用了本机已安装的固定依赖，未验证全新机器离线安装；未做 SenseVoiceSmall 对比、40% 敏感性组、真人听音或部署端自动语言检测。

