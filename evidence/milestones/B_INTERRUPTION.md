# B 首次 CUDA 中断与恢复记录

B 首次运行完成 step 1–67，在接下来的一次反向传播中报 `RuntimeError: CUDA error: unknown error`，最终栈位于 non-reentrant checkpoint 重算退出后的 CUDA RNG state 恢复。原始第一异常位于 Whisper encoder fc2 的重算路径；异步 CUDA 报错可能使栈不精确，不能据此证明 RNG 或 fc2 是根因。

最后原子保存 checkpoint 为 step 50，包含 adapter/optimizer/scheduler/scaler/CPU+CUDA+NumPy+Python RNG。已完成但未持久保存的 step 51–67 将重算；只按最终连续的 800 次真实优化器更新验收，不将重复计算计作额外步数。

观测：PyTorch peak reserved 1266 MiB，未报 CUDA OOM；退出后新进程 CUDA 矩阵乘法成功。系统检查可见其它 GPU ASR 模型进程及大量占 CPU 的视频转码，B step 时长由 pilot 的约3.3秒升至7–16秒。**资源争用仅为候选解释，不是已验证的异常根因。** `dmesg` 因权限不足无法读取，不擅自重启 WSL、重置 GPU 或结束他人进程。

恢复决策：保持 config.json、数据、采样序列、种子与 optimizer 状态不变；从 step50 恢复，并设置 `CUDA_LAUNCH_BLOCKING=1` 以便在复现时获得更精确栈。此环境变量只影响 CUDA 执行同步与诊断，不增加训练步数或改变学习率；C 同样设置。原始失败日志保留 `B_train.log`，恢复日志另存 `B_train_resume1.log`。

恢复验证：已从第50步恢复，跨过原出错点并保存第100步 checkpoint。恢复后重算17步的 loss 最大绝对差约0.000662，说明恢复数值接近但非 bitwise 一致；本实验不声称 GPU FP16 bitwise 确定性。实际更新预算与 scheduler 连续性保持。同期 GPU 总占用约6.4GiB，包含其它进程；不把总占用误认为 Whisper 的独立显存需求。
后续执行调整：恢复后已稳定超过第500步，诊断同步模式已完成用途。该模式使每个 CUDA 调用同步等待，不适合作为正式运行性能设置；主动在已落盘的 step500 处恢复并取消 `CUDA_LAUNCH_BLOCKING`，日志为 `B_train_resume2.log`，C 正式运行也取消该变量。采样序列、optimizer/scheduler/scaler/RNG、800个有效更新及所有数值超参数不变。中断前未落盘的24步会重算，不额外增加最终 checkpoint 的有效更新数。运行时间不可当作受控性能对比。
最终状态：B、C均已完成800个有效更新及完整验证/测试评测，证据见 `FINAL_REPORT.md` 与 `artifacts/*/training_complete.json`。CUDA unknown error 的外部根因仍未确认；成功恢复不等于已定位根因。
