# 审查后的修复记录（当前状态）

代码审查最初发现 4 项确认问题，都属于“后续复跑/复算的保护缺陷”，没有证据表明它们污染了已完成的实验。**当前 4 项全部关闭，17 项测试通过。** B/C 权重（自然采样与 mixed 过采样两个 LoRA adapter）、原始预测、评分口径、800 步预算和采样分布均未改变，也没有为修复保护逻辑而重训。

> 初审（4 项发现）与二次复核（当时 16 项测试）作为历史快照原样保留在 [`evidence/reviews/`](../evidence/reviews/)，不改写；两者与当前结论的差异见第 2 节。

## 1. 当前状态一览

| 项目 | 状态 | 证据 |
|---|---|---|
| F1 汇总身份绑定（含 duration 对齐） | 已关闭 | 修复前 duration 反例失败：`evidence/reviews/review_duration_before.txt`；修复后 17 项测试通过 |
| F2 评测与训练配置一致性 | 已关闭 | 配置/步数/前缀/哈希不符均在加载前报错，测试覆盖 |
| F3 准备缓存身份校验 | 已关闭 | 9 个缓存文件重建后 SHA-256 相同：`evidence/checks/CACHE_REBUILD_VERIFICATION.json` |
| F4 非法步数与已完成复跑 | 已关闭 | 完成 800 步后重跑为字节级 no-op；`--steps 0` 被拒绝；孤儿/冲突完成标记在 GPU 访问前被拒绝：`evidence/reviews/review_completion_before.txt` |
| 自动化测试 | 17/17 通过 | `evidence/reviews/review_tests_after.txt` |
| 运行时回归 | A/B/C 各新生成 12 条、共 36 条与原件逐条一致 | `evidence/checks/REVIEW_RUNTIME_VERIFICATION.json` |
| 原始产物保护 | 54 个原始 artifact 字节不变 | 同上 |
| 交付归档核验 | 同版候选包通过：17 项测试、101 个文件哈希、CPU 精确复算 | 归档后的最终证据在 `delivery/verification.json`；旧包保存在 `delivery/pre-review/` |

## 2. 历史报告与最新结论的关系

- `evidence/reviews/SOL_REVIEW.md` 是初次审查，列出 F1–F4。
- `evidence/reviews/SOL_FIX_VERIFICATION.md` 是**当时**的二次复核快照：F2–F4 通过，F1 仍缺 duration 对齐。该报告按原样保留，不改写。
- 在那之后，主执行者补齐了 duration 反例与共享检查：修复前该反例失败（`evidence/reviews/review_duration_before.txt`），`evidence/reviews/SOL_FIX_VERIFICATION.md` 记录的是**当时**的 16 项测试快照；F4 的完成标记守卫是之后再补的，所以当前应为 17 项（`evidence/reviews/review_tests_after.txt`、`evidence/checks/REVIEW_RUNTIME_VERIFICATION.json`），并用完整重建的缓存重载 A/B/C 验证生成结果未变。
- 因此：**不要**根据历史报告推断“duration 校验仍未修”；也**不要**把历史报告里的 7 项测试当成当前测试数。

## 3. 逐项修复

| 审查项 | 修复 | 验证 |
|---|---|---|
| F1 汇总仅检查行数，未绑定样本与权重 | 检查完整官方 ID 集合、三子集计数、A/B/C 参考/语言/**时长**逐 ID 一致；绑定完整评测配置和当前 adapter SHA-256；全部通过才写派生指标 | 伪造或重复 ID、改参考/语言/时长、改权重哈希或子集评测标记均被拒绝，且不修改原文件 |
| F2 评测可与训练配置不一致 | 生成前核对冻结训练配置、完成标记配置哈希和正式训练步数；已有预测必须有相符来源记录，逐条身份也须与准备数据一致 | 配置不符在加载数据/模型或写文件前报错；无来源或已改参考的续评文件被拒绝 |
| F3 旧准备缓存缺少身份校验 | `ready.json` 绑定数据集及模型名称、固定版本和语言前缀；`prepare`/`load_split` 均校验，不自动认可缺少新身份字段的旧缓存 | 原缓存保存在 `delivery/pre-review/data`；完整重建后 3 个 split 的 features/manifest/差异清单共 9 文件 SHA-256 逐一相同，原审计数值不变 |
| F4 非法步数或已完成复跑会改写记录 | 正式 B/C 只允许冻结预算；0/负数/超预算或 A 训练立即拒绝；`resume.pt` 是续跑的权威来源，完成标记仅在 `resume.pt` 存在且步数一致时才被采信——孤儿标记（有 adapter/标记但无 `resume.pt`）与标记步数冲突（标记 800、恢复点 400）都在任何 GPU 访问与写入之前拒绝；相同步数真正不操作 | 800 步完成后在无 GPU（CUDA 禁用）条件下重跑保持文件字节不变；`--steps 0` 不再被 CLI 替换为 800；两个新反例修复前均复现失败：`review_completion_before.txt`；未标记的合法 400→800 恢复仍被接受 |

缓存身份是**新增校验字段**，不代表原训练已使用该字段。旧缓存不直接补写身份“升级”为新缓存；选择保留旧目录后重新准备，再验证数值一致。旧交付包及校验记录保留在 `delivery/pre-review/`。

## 4. 测试与证据

- 先添加 9 个回归测试方法：旧代码复现 14 个失败和 3 个错误子场景，另有合法输入正向用例；原始输出为 `evidence/reviews/review_tests_before.txt`。
- 修复后原 7 项加新增 10 项（先 9 项，再补 1 项完成标记孤儿/冲突守卫），**共 17 项全部通过**：`evidence/reviews/review_tests_after.txt`。
- 独立反例探针：44 个被拒绝的用例，fixture 文件保持不变（见 `evidence/reviews/SOL_FIX_VERIFICATION.md` 的证据段）。
- `uv run python summarize.py` 通过新增身份保护；A/B/C 主指标、差值与结论与原交付相同，并新增官方样本身份与预测/权重身份检查标记。
- 缓存完整重建与一致性核验通过（9 文件 SHA-256 相同）。
- 新增第 17 项回归（`test_training_rejects_orphan_or_conflicting_completion_before_writes`）覆盖 F4 的完成标记守卫：孤儿完成标记与“标记 800 / 恢复点 400”冲突在修复前都能被复现（`evidence/reviews/review_completion_before.txt`），修复后在任何 GPU 访问与写入之前报错；合法但未标记的 400→800 恢复仍被接受。数学与评测代码未变，因此 36 条全新生成一致性与 9 个数据文件重建一致性仍然成立；B/C 的 CUDA 禁用重跑仍是 no-op，54 个原始 artifact 哈希不变。
- 完整重载 A/B/C，各在临时副本中新生成 12 条（每种语言 4 条，保持正式评测 batch=4），共 36 条的预测、token IDs、崩溃标记及样本身份与原正式输出完全一致；54 个原始 artifact 字节不变。

复现这些检查：

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest -v test_pipeline
uv run python summarize.py
```

## 5. 未改变的内容

- B/C adapter 权重与 `resume.pt`、A/B/C 原始预测、`test_metrics.json` 数值、计分口径、800 步预算、采样分布、随机种子。
- 不需要重新训练或重新生成预测来“让修复生效”；修复针对的是**以后**的复跑/复算是否会被错误产物污染。

## 6. 仍未验证与待办

- **归档核验已补齐**：审查后的同版候选包通过 17 项测试、101 个文件哈希和隔离 CPU 指标复算；正式归档后的大小、哈希与独立解压检查记录在归档外的 `delivery/verification.json`，避免归档自引用。旧的 7 项/90 文件快照保存在 `delivery/pre-review/`。
- 本次文档修订（`evidence/reviews/DOC_UX_REVIEW.md`）只改可读性与使用流程，不改任何数值、判定或代码计算逻辑。
- 未做真人听音、多随机种子、显著性检验；这些是实验设计边界，不是本次代码审查的遗漏。
- 静态工具（ripwire）的 `no-baseline` 与 dead-code 提示不是行为正确性证据；零结果不代表不存在依赖。
