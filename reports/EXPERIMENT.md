# ASCEND Whisper-small LoRA：预先固定的实验协议与决策记录

## 目的与范围
在 RTX 4060 Laptop GPU 8GB 上，以 ASCEND（论文作者发布于 Hugging Face 的 `CAiRE/ASCEND`，固定 revision 见 `config.json`）自带的 train 划分微调 Whisper-small，比较自然采样 B 与 mixed 过采样 C。仅句内中英混合，不扩展到远场、会议或多人重叠语音。A 为不训练的原模型。主指标固定为测试 mixed 子集 MER。

## 固定变量
机器可执行配置见根目录 `config.json`，依赖见 `uv.lock`。基础模型、数据分别固定到 Hugging Face commit，保留官方 train/validation/test。种子 42；B/C 都从同一原模型及相同 LoRA 随机初始化出发，保存 adapter 初始化 SHA-256 验证，不从 B 继续训练 C。

- 800 次优化器更新，每次 8 个 microbatch，每个 microbatch 一条音频。
- AdamW，学习率 1e-4，线性衰减，50 步 warmup，weight decay 0.01，梯度范数上限 1。
- LoRA rank 8 / alpha 16 / dropout 0.05，encoder+decoder 所有 q_proj/v_proj；bias 不训练。
- FP16 autocast、FP32 可训练参数与梯度、non-reentrant gradient checkpointing。
- 同一 train 数据池，有放回抽样。B 为逐样本均匀抽样（期望保持自然语言比例）；C 将 mixed 总概率设为 50%，剩余单语样本彼此仍等概率。
- 主比较仅最终第 800 步 checkpoint，不选测试集最佳 checkpoint，不为改善 C 的结果加步数或提高学习率。

## 音频时长预算：用户已选择的解释
两组每次更新均为 8×30 秒补齐后的 Whisper 输入，计算/补齐音频预算一致；不强制真实语音秒数一致。分别报告实际累计语音时长、采样语言比例、独立样本数。**真实语音时长仍可能是混杂因素，不声称严格匹配真实语音时长。** 这是用户在开始正式训练前明确选择的口径。

## 数据审计
核实官方 split 数、实际音频秒数、16kHz 采样率、最长 30 秒、标签长度、language 与文字汉字/英文字母启发式一致性。字段不一致样本只列出，不擅自重标或剔除；英文字母可能来自标注或拼写词，启发式差异不自动代表标注错误。准备完成后见 `artifacts/data_audit.json` 与 `data/<split>/language_mismatches.json`。

## 评测定义与生成配置
- 中文字符+ASCII 英文 token 的语料级编辑错误数 / 参考单位总数 ×100，称 MER；纯中文子集 CER 则按所有非空白字符计分，纯英文使用同样英文 token 口径 WER。不使用 jiwer.mer（它代表另一种指标）。
- 采用 ASCEND 作者的标点移除列表、lowercase、jieba 0.42.1 分词；ASCII token 保留，其余按字拆开。对预测与参考对称规范化；保留语气词、重复、数字、缩写，不做繁简转换或英文数字展开。
- greedy decoding，num_beams=1，max_new_tokens=256，task=transcribe，return_timestamps=False。
- forced_decoder_ids=None，suppress_tokens=[]；保留默认 begin_suppress_tokens=[220,50257]，仅在首个生成位置抑制空格与 EOS，避免首 token 直接停止。三组完全相同。
- 依据已知样本 language 分组给予前缀：mixed/zh 用 zh，en 用 en。这是使用数据标签的 oracle 前缀设置，不是自动语言检测性能。
- 空输出、长度至少 3 个计分单位的 ngram 连续重复至少 4 次、特殊语言 token 前缀缺失/错误/在正文重复出现、达到生成上限而无 EOS，任一即计为崩溃。保存原始 token IDs 与逐例标记。重复检测是预设启发式，非人工识别全部崩溃。

## 基线量级参考的修正
ASCEND 原论文早于 Whisper；其中 27.05% 测试 MER 是微调 wav2vec 2.0 的结果，不存在可精确复现的原论文 Whisper-small 数值。A 将作为本项目重新测得的零样本基线。论文数字只作量级与计分自检参考，不把模型差异误报为计分错误。来源与细节见 `reports/protocol_sources.md`。正式训练之前需通过评测单元测试并检查 A 生成样例。

## 标定与降级
独立 pilot 从初始化训练 50 步（scheduler 仍按正式 800 步），记录 loss、每步耗时、PyTorch peak allocated/reserved 显存；在 validation 每种语言前 10 条做生成评测。pilot 不作正式模型、不用其权重初始化 B/C。OOM 时 rank8→rank4；仍 OOM 再 4-bit NF4 QLoRA。任何降级必须重跑 pilot，并对 B/C 一起冻结新配置。不得只改一组。

## 门槛与预设重跑
B 完成后先评 validation，再评 test。B/C 测试崩溃率均需 <5%，B 整体测试 MER 不高于 A，否则不进入 C。若失败，先检查实现/前缀/标签/学习曲线，允许按照用户计划预设将学习率降至 5e-5、重新独立做 100 步 pilot；若证据支持，再将 B/C 一起改为新配置并从头训练。保留失败记录，不反复试探测试集直到过关。测试集门控/重跑会产生测试泄漏风险，须在最终报告注明。

## 结论规则（绝对百分点）
C mixed MER < B；纯中文 CER 退化 ≤0.5pp；纯英文 WER 退化 ≤0.8pp；整体 MER 退化 ≤0.3pp。全部满足且硬门槛通过，描述为本次单种子实验过采样有效；mixed 改善而其它超阈值则为存在权衡；无改善则如实报告无可检测收益或变差。0.3pp 是事先约定的容忍区间，不等于统计证明的噪声区间。单种子结果不做显著性宣称。

M5 固定随机抽取 40 条 C mixed 测试预测，由 AI 逐条按用户列出的英文漏词/音近字/切换断句/翻译等错误分析，并披露 mixed 测试子集样本数。SenseVoiceSmall 不在必需范围，先不做，避免额外模型下载和无关算力。
错误分析采用参考/预测文本逐条复核，不假称真人听音标注；真实声学歧义保留不确定性。因不再向用户提问，未要求真人参与，也不新增外包步骤。最终报告须把此项与真人听音审核区分。
## 数据实测与实施决策补充

- 官方固定 revision 实测 train/validation/test 分别 9,869/1,130/1,315 条，8.774909/0.923225/0.917496 小时。train mixed 2,739 条（27.75%），按真实 waveform 求和为 **3.616870 小时**，不是计划估计的 2.88 小时；本项目以固定 release 的实测值为准，不为匹配估计数修改数据。
- language 与文本汉字/拉丁字母启发式有 87/15/6 条不一致；其中 train 86 条、validation 15 条、test 6 条来自 `[UNK]` 标注中的英文字母，train 仅剩 1 条非此原因。保留官方 language，不把 `[UNK]` 错判为真实 code-switching，也不擅自清洗计分参考。
- A 初始样例存在繁体输出与简体参考差异；不在见到结果后新增繁简转换。此因素会影响绝对 MER，因此报告同时保留原始文字，B/C 仍按同一固定口径比较。
- 固定 Transformers 4.53.3，生成必须返回 return_dict_in_generate=True 的完整 sequences 才可检查前缀和 EOS。相关回归测试通过。
- FP16 overflow 降低 scale 后重试相同 microbatch、恢复 CPU/CUDA/NumPy/Python 随机状态；成功更新才推进 scheduler 与步数。连续 16 次失败即停止，避免无休止重试。断点保存全部上述随机状态。


## 完成约束
仅安装依赖、写完脚本、完成 pilot、或启动后台训练都不是完成。只有 B/C 正式微调实际结束且完成统一评测、对比与复现交付，才能报告项目完成。若无法满足硬门槛，明确报告未完成与阻塞证据。
