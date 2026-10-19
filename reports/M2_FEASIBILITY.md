# M2 本地可行性标定：通过

RTX 4060 Laptop GPU 8GB 完成 **50 个完整优化器更新**，每步 8 个单样本 microbatch，FP16、non-reentrant gradient checkpointing。未发生 OOM，未使用 rank4 或 QLoRA。

| 项目 | 实测 |
|---|---:|
| LoRA rank / alpha | 8 / 16 |
| 可训练参数 | 884,736 |
| 50 步训练 wall time | 164.996 秒 |
| 平均每步 | 3.261 秒 |
| 中位每步 | 3.197 秒 |
| PyTorch peak allocated | 1122.525 MiB |
| PyTorch peak reserved | 1252.000 MiB |
| 前 10 步平均 loss | 1.17633 |
| 后 10 步平均 loss | 0.88066 |

上述显存为 PyTorch allocator 统计，不含桌面、CUDA context 或其它程序。额外的 nvidia-smi 监控从 pilot 后段开始，正式 B/C 将完整覆盖，见 `artifacts/gpu_monitor.csv`。总训练时长除以 50 包含 checkpoint I/O，因此略大于平均 step time。

validation 各语言固定前 10 条、合计 30 条生成自检：整体 MER 21.3855%，mixed MER 26.5823%，中文 CER 11.6883%，英文 WER 20.6186%，崩溃率 0%。这只是小样本生成自检，不与完整 A 测试集做效果比较，不用于正式结论。

示例：

- mixed 参考：`小朋友我回想一下when i was in elementary school`；预测：`小朋友我回想一下`（漏英文，提示核心问题仍然存在，并非已解决）。
- zh 参考/预测：`嗯` / `嗯`。
- en 参考/预测：`like year three` / ` like year three`。

最终正式配置冻结为根目录 `config.json`：800 步、学习率 1e-4、rank8、alpha16、dropout0.05、q_proj/v_proj、种子42、每步8×30s补齐输入。B/C 不复用 pilot 权重，均从原模型和相同随机 LoRA 初始化出发。预计单组训练约 40–50 分钟，仅为按 pilot 推算，正式耗时以后续日志实测为准。

初始化 adapter SHA-256（不含配置编码差异）：`5d3bb69622c631477f8c113df9908c2aaa7da3c34c8cfc448f50c1630f86ae55`。
