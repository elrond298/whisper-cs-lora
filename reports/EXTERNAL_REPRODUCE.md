# 外部子集评测：查看结果、离线复算与重新推理

本文只涉及新增的 CS-Dialogue / FLEURS 评测。**不重新训练，不覆盖原 ASCEND 成果。** 数值见 `EXTERNAL_EVAL_REPORT.md`，抽样规则与数据源见 `EXTERNAL_EVAL_PLAN.md`、`EXTERNAL_EVAL_SOURCES.md`。

路径都以仓库（或解压后的包）根目录为基准，`reports/` 内的文档同样如此。

## 1. 先分清三种操作

| 我想做什么 | 需要 GPU | 需要基础模型权重 | 需要音频 | 会改写什么 |
|---|---|---|---|---|
| 看报告和 `comparison.json` | 否 | 否 | 否 | 不改写 |
| 用已有预测重算指标和区间 | 否 | 否 | 否 | 重写 `scores.json`、`comparison.json` |
| 重新生成预测 | 是 | 是（固定 revision） | 是 | 续写 `predictions.jsonl` |

装环境：

```bash
uv sync          # 按 uv.lock 建 .venv，需要 Python 3.11
```

之后所有命令都用 `uv run python ...`。

公开补充包已完成隔离核验，包含：评测代码、方法、汇总指标；FLEURS 的清单、参考、逐条预测和逐条评分；6 个离线 tokenizer 资产与哈希；以及 **B/C 各 4 个身份文件**（`config.json`、`training_complete.json`、`adapter/adapter_config.json`、`adapter/adapter_model.safetensors`）。**不包含**基础 Whisper 权重、CS 逐条材料和任何音频。

## 2. 离线复算要读哪些文件

下表中的路径都要在。缺一项会报错；不要改 `identity.json`、`ready.json`、`validation_tokenizer.json` 或根 `config.json` 去凑。

| 位置 | 文件 | 作用 |
|---|---|---|
| 根目录 | `config.json` | 冻结训练配置；B/C 训练元数据的 `config_sha256` 必须与它一致 |
| `artifacts/external_eval/` | `token_schema.json`、`protocol.json`、`verification.json` | token 规则；协议；生成期校验记录 |
| `artifacts/external_eval/` | `validation_tokenizer.json` + `validation_tokenizer/`（6 个文件） | 校验 token 与文本一致；每份资产哈希见 JSON |
| `data/external_eval/<dataset>/` | `manifest.json`、`ready.json`、`audit.json` | 冻结样本、清单绑定哈希、排除原因 |
| `artifacts/external_eval/<dataset>/<group>/` | `predictions.jsonl`、`identity.json` | 预测及权重/代码/清单身份 |
| `artifacts/B/`、`artifacts/C/` | `config.json`、`training_complete.json`、`adapter/adapter_config.json`、`adapter/adapter_model.safetensors` | 训练完成 800 步、adapter 哈希；共 8 个文件，已在补充包内 |
| 仓库根 | `external_scores.py`、`external_eval.py`、`metrics.py`、`asr.py` | 评分与身份代码 |
| `provenance/` | `external_eval_generation_v1.py` | 仅在核对原始 9 份 v1 身份时需要 |

`<dataset>` 取 `cs_dialogue`、`fleurs_zh`、`fleurs_en`；`<group>` 取 `A`、`B`、`C`。

复算不只是比字符串。读取每条预测时会做四件事：

1. 记录必须与清单逐字段一致（ID、语言、参考、时长、音频哈希、分组）。
2. 每个 token ID 必须是整数且落在词表范围内（Whisper-small 为 51,865）。
3. `tokenizer.decode(token_ids, skip_special_tokens=True)` 必须与保存的文本完全相同。
4. 异常标记必须能由 token 序列重新推出。

词表来自随包的 6 个 tokenizer 文件（`vocab.json`、`merges.txt`、`tokenizer_config.json`、`added_tokens.json`、`special_tokens_map.json`、`normalizer.json`），哈希登记在 `validation_tokenizer.json`，并与固定 model/revision 绑定。**所以 CPU 复算不需要 HF 缓存、不联网、不下载、不加载权重**；只靠补充包里的文件就能完成 FLEURS 复算。

原始 9 份 `identity.json` 和 9 份 `predictions.jsonl` 保持字节不变，其中 `inference_code_sha256` 是当时的 `06611afbb60e…`。校验代码把这一份已归档的生成源 `provenance/external_eval_generation_v1.py` 视为唯一可接受的旧身份；不改标签、不重写预测。历史未完成的运行不能续跑，需要归档或换成新身份重新生成。

## 3. 离线 CPU 复算

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python external_scores.py --datasets fleurs_zh fleurs_en
```

在本地受限材料齐全时，可复算三个子集（不加 `--datasets` 即全部）：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python external_scores.py
```

命令会原地重写 `artifacts/external_eval/<dataset>/scores.json` 和 `comparison.json`，不训练、不生成预测、不改 identity。想核对原交付文件，先校验，再在只读副本或解压副本里复算，别在唯一的交付副本上直接跑。

`comparison.json` 里的 `scoring_code_sha256` 是评分代码的哈希。改过代码后新旧文件字节不同是正常的，应比较指标和区间，不要把哈希变化当成模型结果变化；要还原原始字节，就重新解压或用原包文件覆盖。

## 4. 获取子集并生成预测

这一步要联网，且必须先确认数据使用范围：FLEURS 为 CC-BY-4.0；CS-Dialogue 官方标注 CC BY-NC-SA 4.0，并另有仅限学术/非商业、不得向第三方转发的条款。公开可下载不等于可商用或可再分发。

```bash
HF_HUB_DISABLE_XET=1 uv run python prepare_external_fleurs.py
HF_HUB_DISABLE_XET=1 uv run python prepare_external_cs.py
```

FLEURS 只取中英文 test 包，压缩音频载荷约 815 MB；已准备好的子集再次运行会先核验身份与音频，不会重复下载。CS 短音频是 19 个连续分卷、约 9.6 GB，不能当成 19 个独立小数据集；准备程序只保留所选短句，不取 long_wav。

下载完成后可在离线状态复核 CS：`HF_HUB_OFFLINE=1 uv run python prepare_external_cs.py --verify`。注意不带 `--verify` 的默认命令会先联网取元数据，即使 `ready.json` 已存在。

基础模型只在重新生成预测时需要，且必须是指定 revision。补充包不含权重；下面只取必需的 11 个文件（约 927 MB），不拉别的 revision：

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="openai/whisper-small",
    revision="973afd24965f72e36ca33b3055d56a652f456b4d",
    allow_patterns=["*.json", "*.txt", "model.safetensors"],
)
PY
```

运行这条命令时不要设 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`；下载完成后再离线推理。

音频和固定模型都就位后，先冒烟，再按子集/模型组合续跑：

```bash
# 每个清单的前 10 条，检查加载、对齐和计分
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python external_eval.py --smoke

# 正式续跑；已完成的预测不会重算
for ds in cs_dialogue fleurs_zh fleurs_en; do
  for g in A B C; do
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      uv run python external_eval.py --datasets "$ds" --groups "$g"
  done
done
```

`--smoke` 只处理每个数据集/组冻结清单的前 10 条，写进同一个 `predictions.jsonl`，正式运行接着往下续。这 10 条就是最终结果的一部分，不是可以丢掉的试跑。全部完成时打印 `already complete, verified, no GPU work`，不再占用 GPU。

即使全部完成，再跑一次仍会先核验音频哈希、读取基础模型的配置和 tokenizer/处理器（不需要完整权重）。所以“预测已完整但没有音频或模型配置”时会先报错；这是身份校验，不是需要重跑。

推理固定使用已知语言前缀（mixed/中文 zh，英文 en），不做自动语言检测。离线推理不会自动下载基础模型；缓存缺失会直接失败。当前包的入口是 `--datasets` / `--groups`（没有 `infer` 子命令）；以随包代码的 `--help` 为准。

## 5. 真要重新生成预测

默认命令只续跑，**不会自动清空已完成结果**。只想看结果时，不要移动任何文件。

确实要重生成某个模型/子集：

1. 在隔离副本里备份整个目标目录，例如 `artifacts/external_eval/fleurs_zh/B/`，连同它的 `identity.json` 一起留存。
2. 保留 `data/external_eval/fleurs_zh/{manifest,ready}.json`、`artifacts/external_eval/{token_schema.json,validation_tokenizer.json,validation_tokenizer/}`、根 `config.json`，以及 `artifacts/B/` 的 4 个身份文件。
3. 再跑对应命令。

不要移动原 `artifacts/B/` 或 `artifacts/C/`，不要把外部评测改成重训。历史未完成的运行不能续跑：连同旧 `identity.json` 一起归档，再以新身份重生成。

常见错误：

- 身份不符（配置、权重、代码、清单、tokenizer 任一）：停下查原因，不要手改 `identity.json`、`ready.json`、`validation_tokenizer.json` 或 `config.json` 去凑。
- JSONL 损坏、重复、token 与文本不一致或缺少 `identity.json`：保留故障原件，在隔离副本里重生成。
- 只想重算指标：用 `external_scores.py`，不要启动 GPU 推理。
- 结果不利：保留样本，不能静默删行或换清单。

## 6. 自检与解释范围

```bash
uv run python -m unittest -v test_pipeline test_external_eval test_external_fleurs test_external_cs
```

最终 39 项测试全部通过；独立修复复核确认 P2 已解决。公开包隔离解压后，69 个文件哈希通过，1800 条 FLEURS 预测通过 token/文本校验，六组指标与差值区间精确复算；GPU 禁用、网络/DNS 阻断、HF 缓存初始为空。复用了已安装的固定依赖。CS 全部 1350 条预测另已在获准本地目录复核，公开包不含其逐条材料。最终证据见交付目录 `delivery/external-verification.json`。

口径：CS MER 用本项目原有作者兼容口径；FLEURS 中文用 BasicTextNormalizer 后按字 CER；英文用 EnglishTextNormalizer（空拼写别名映射）后按空格分词 WER。参考和预测使用相同规范化。原 ASCEND 历史数值不改口径；不同口径的绝对分不能混成总分，也不能直接当公开榜单分数。

配对区间只描述当前固定模型和子集。CS 按说话人分组，但说话人 ID 未验证为独立对话；FLEURS 缺说话人 ID，只按句子 ID 分组。区间不覆盖训练随机种子变化。
