# 交付前实际核验记录

## 本轮审查后的隔离核验

- 同版候选归档解压后，禁用 GPU、使用空 Hugging Face 缓存，阻断 Python socket 连接与 DNS：17 项测试通过，A/B/C 测试指标与 B/C 验证指标精确复算。
- 101 个交付文件的 SHA-256 在复算前后均一致；B/C 各 144 个 optimizer 状态计数均为 800，144 个 adapter 张量分别与恢复点完全相同。
- 重建缓存的 9 个文件与原件字节一致；A/B/C 各新生成 12 条、共 36 条预测与原件逐 token 相同；54 个原始实验产物未改动。详见 `evidence/checks/CACHE_REBUILD_VERIFICATION.json` 与 `evidence/checks/REVIEW_RUNTIME_VERIFICATION.json`。
- 修复见 `REVIEW_FIXES.md`，文档可读性与操作流程由 DeepSeek Flash 修订。失败反例日志记录的是**修复前**表现，不是当前测试失败。
- 本轮没有重新执行 800 步训练；完成证据来自保留的真实 checkpoint 与原始日志。正式归档后的独立核验与归档哈希放在包外的 `delivery/verification.json` 和 `.sha256` 中，避免自引用。
- 使用的是本机已安装的固定依赖；未验证全新机器离线安装，也未做真人听音、多种子或显著性检验。

## 已执行且通过

1. `uv run python -m py_compile asr.py metrics.py summarize.py test_pipeline.py`：语法检查通过。
2. `uv run python -m unittest -v test_pipeline`：全部 17 项测试通过；原 7 项指标/采样/生成检查，加上 10 项样本与权重身份、冻结配置、缓存身份、非法步数及完成/恢复标记保护回归。
3. `uv run python summarize.py`：从保存预测重新计算A/B/C，所有硬门槛为true，预设方向性判定为`oversampling_effective_single_seed`。
4. A/B/C 各 1,315 条测试预测的 ID、参考文本、language、duration 均与当前官方 test manifest 一致；B/C 验证集各 1,130 条也完成同样逐例检查。不是只核对文件行数。
5. B/C各1,130条完整验证预测，ID完整且唯一，验证指标亦精确重算。
6. B/C各自resume.pt的step与scheduler.last_epoch均为800；144个AdamW参数状态计数器全为800。
7. B/C最终adapter_model.safetensors的所有张量与对应resume.pt中adapter状态逐张量完全相等。
8. 训练缓存shape为[9869,80,3000]，测试为[1315,80,3000]，即30秒补齐输入。重建固定seed采样序列，核对两组均6,400条选择、语言计数、独立样本数、实际语音小时与补齐53.333333小时全部相符。
9. 40条AI逐例文本复核与固定抽样原件逐条ID/参考/预测对齐；12条零编辑例与计分器一致；4条参考含[UNK]。该复核不是听音标注。
10. B/C配置与初始化SHA-256一致；独立pilot50步完成，无容量降级；完整测试与验证均无B/C输出崩溃。

上述checkpoint、ID、采样与复核对齐的一次性核验输出保存在`evidence/checks/verification_details.json`，不是仅根据训练进度文本推断完成。

## 静态工具的范围与未消除提示

- 使用ripwire导航，并执行`--situ=asr.py,metrics.py,summarize.py,test_pipeline.py`；推荐测试为test_pipeline.py，已实际运行。
- `ripwire . --edit-check=./summarize.py:main --legend=compact`：没有发现不兼容调用；仓库尚无HEAD基线，工具返回`no-baseline`，因此不能据此声称已证明历史接口无回归。
- `ripwire . --quality-delta --legend=compact`：门禁级回归 gating=0，但仍有非门禁的复杂度、长度与静态 dead-code 提示。测试执行已证明相关 unittest 会运行；未为了消除静态提示而改变训练或评测逻辑。
- ripwire动态分派/未索引文件是盲区；零结果不代表不存在依赖。

## 故障与边界

- B发生一次CUDA unknown error并完整恢复，详见evidence/milestones/B_INTERRUPTION.md。最终800步与adapter核验已通过，外部根因未定位。
- B训练结束后的评测启动曾误用`evaluate`子命令；改为正确的`eval`后，完整validation/test进程成功退出，预测和指标通过上面的逐项校验。原始日志保留，未把shell退出码2当作训练失败或成功证明。
- 未重新跑第二组独立随机种子、没有显著性检验、没有真人听音。测试mixed的小幅收益与验证mixed反向结果均披露于FINAL_REPORT.md。
- 交付包解压后的CPU复算、压缩包SHA-256和文件校验结果随包另附于`delivery/verification.json`与`.sha256`文件；压缩包内部本文件记录源码/原始结果核验，外部文件记录归档后核验，避免自引用归档校验值。
