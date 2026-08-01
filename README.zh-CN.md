# trace2train

> **把 agent 的工具调用与行为失败轨迹，变成干净的 SFT/DPO 训练数据。**

[![PyPI](https://img.shields.io/pypi/v/trace2train)](https://pypi.org/project/trace2train/)
[![CI](https://github.com/wane528/trace2train/actions/workflows/ci.yml/badge.svg)](https://github.com/wane528/trace2train/actions/workflows/ci.yml)
![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-green)

**[English](README.md) · 简体中文**

你的 agent 选错工具、传错参数、该返回 JSON 却返回散文、过度拒绝一个正常请求、或
无视系统规则。trace2train 把这些失败轨迹变成训练数据，专门修复这些问题——已做 PII
脱敏、去重、完整 provenance。**无需服务器，本地跑在你的笔记本上。**

```console
$ trace2train inspect --demo
19 traces → 16 failed → 38% dirty (PII/dupes/noise) → 14 trainable

failure types (trainable):
  wrong_tool        8
  bad_args          2
  over_refusal      2
```

`inspect` 是纯规则、**无需 LLM、无需 API key**——它告诉你有哪些失败、有多脏、多少条真正
可用，**在你花一分钱之前**。

## 目录

- [特性](#特性)
- [安装](#安装)
- [快速上手](#快速上手)
- [命令](#命令)
- [配置](#配置)
- [支持的输入](#支持的输入)
- [工作原理](#工作原理)
- [范围](#范围它修什么不修什么)
- [Langfuse](#langfuse)
- [设计原则](#设计原则)
- [常见问题](#常见问题)
- [贡献与许可](#贡献与许可)

## 特性

- **免费质量报告** —— `inspect` 用纯规则给你的 trace 打分：多少失败、多脏（PII/重复/
  噪声）、多少可训练，以及按失败类型的分布。无 LLM、无 key。
- **诚实纠正** —— `convert` 只在纠正答案可从 trace 本身推导时才产出数据；需要外部真值
  的失败会标记为 `skipped`，绝不编造。
- **每条 trace 只调一次 LLM** —— 失败归因和纠正在一次调用里完成，成本约减半。运行前
  会打印成本预估。
- **人在环中** —— `convert --review` 让你在写入前逐条 approve/reject。
- **可断点续跑** —— `convert --resume` 跳过已处理的 trace，限流后重跑不会重复付费。
- **数据集健康体检** —— `convert` 后展示失败类型分布、长度分布，并在数据集倾斜/过小/
  噪声过高时告警。
- **可审计输出** —— LLaMA-Factory 可直接用的 JSONL，每条带 `_provenance`，另有
  `meta.json` 审计文件。
- **可脚本化** —— `inspect` 和 `convert` 支持 `--json`，方便接入 CI/流水线。
- **本地优先** —— 无服务器、无账号、无遥测。

## 安装

需要 Python 3.11+。

```bash
pip install trace2train         # 正常使用
pip install -e ".[dev]"         # 开发（含测试 + lint）
```

## 快速上手

**30 秒试用** —— 无需数据、无需 API key：

```bash
# 1. 对内置示例数据做即时质量报告
trace2train inspect --demo

# 2. 转换（离线：原始 trace 写到 out/needs_review/ 供人工整理）
trace2train convert --demo --no-llm -o out

# 3. 看看结果
trace2train review -o out
```

**用你自己的数据 + 一个 LLM**（推荐——纠正后的训练数据从这里来）：

```bash
cp .env.example .env          # 填入 T2T_LLM_API_KEY（DeepSeek 很便宜）

trace2train inspect traces.jsonl          # 免费、即时
trace2train convert traces.jsonl -o out   # LLM 纠正的 SFT/DPO
```

> **离线 vs. LLM。** 没有 API key 时，`convert` 无法推导出*纠正后*的答案，于是把原始
> 失败 trace 写到 `out/needs_review/raw_traces.jsonl` 供你手动修正——它绝不会把未经
> 验证的答案当作训练数据。设置 `T2T_LLM_API_KEY` 才能得到纠正后的
> `train_sft.jsonl` / `train_dpo.jsonl`。

## 命令

| 命令 | 作用 |
|---|---|
| `trace2train inspect [FILE]` | 即时、纯规则的质量报告（无 LLM）。 |
| `trace2train convert [FILE]` | 把失败转成 LLaMA-Factory 格式的 SFT/DPO JSONL。 |
| `trace2train review` | 漂亮打印生成的样本，让你肉眼判断质量。 |
| `trace2train langfuse pull [OUT]` | 把 Langfuse v4 observations 快照到本地 JSONL。 |
| `trace2train --version` | 打印已安装版本。 |

任何命令加 `--help` 查看完整选项。关键 flag：

**`inspect`**

| Flag | 用途 |
|---|---|
| `--demo` | 使用内置示例数据集。 |
| `--format auto\|langsmith\|langfuse\|messages` | 强制指定输入格式（默认自动识别）。 |
| `--export PATH` | 额外写一份可分享的 Markdown 报告。 |
| `--json` | 输出机器可读的 JSON 报告而非表格。 |

**`convert`**

| Flag | 用途 |
|---|---|
| `-o, --out-dir PATH` | 输出目录（默认 `out`）。 |
| `--no-llm` | 不用 LLM 运行（原始 trace → `needs_review/`）。 |
| `--review` | 写入前逐条 approve/reject（需要 LLM）。 |
| `--resume` | 跳过上次运行已输出的 trace。 |
| `--redact / --no-redact` | PII 脱敏（默认开启）。 |
| `--leak-file PATH` | 排除命中 eval 集指纹的样本。 |
| `--max-traces N` | 限制处理的 trace 数量。 |
| `--json` | 输出机器可读的 JSON 摘要。 |

`--review` 中，用 `k`/`d` 保留/丢弃单条，`A`/`D` 对剩余全部生效。

**`review`**

| Flag | 用途 |
|---|---|
| `-o, --out-dir PATH` | `convert` 输出文件所在目录（默认 `out`）。 |
| `-n, --limit N` | 展示多少条样本（默认 5）。 |
| `--kind sft\|dpo\|both` | 展示哪类记录（默认 both）。 |

## 配置

在 `.env` 文件（复制 `.env.example`）或环境变量里设置：

| 变量 | 用途 | 默认值 |
|---|---|---|
| `T2T_LLM_API_KEY` | 启用 LLM 纠正的 `convert`。不填则离线运行。 | *(无)* |
| `T2T_LLM_BASE_URL` | OpenAI 兼容的接口地址。 | `https://api.deepseek.com` |
| `T2T_LLM_MODEL` | 模型名。 | `deepseek-chat` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | `langfuse pull` 的鉴权。 | *(无)* |
| `LANGFUSE_BASE_URL` | Langfuse 主机地址。 | `https://cloud.langfuse.com` |

换 `T2T_LLM_BASE_URL` / `T2T_LLM_MODEL` 即可用任何 OpenAI 兼容的 provider
（OpenAI、Moonshot、Qwen 等）。

## 支持的输入

| 输入 | 用法 | 状态 |
|---|---|---|
| LangSmith JSONL 导出 | `trace2train inspect traces.jsonl` | 支持 |
| 通用 messages JSONL | `trace2train inspect messages.jsonl` | 支持 |
| Langfuse v4 Public API v2 快照 | `trace2train langfuse pull` → `inspect` | 支持 |

**输出**（在 `out/`）：`train_sft.jsonl` + `train_dpo.jsonl`（LLaMA-Factory ShareGPT
格式）+ `meta.json` 审计。还没有自己的 trace？`scripts/fetch_dataset.py` 能从
HuggingFace 拉取公开的 agent 轨迹数据集——见 [`scripts/README.md`](scripts/README.md)。

## 工作原理

```
traces (LangSmith、Langfuse 快照，或 messages JSONL)
   │
   ├─▶ inspect ──▶ 纯规则质量报告                    (免费、即时、无 LLM)
   │
   └─▶ convert
         ① 检测失败（规则）
         ② 一次 LLM 调用里同时归因 + 纠正 —— 为什么失败以及怎么修，
            仅当修法可从 trace 推导时
         ③ 净化：PII 脱敏 · 去重 · eval 泄漏过滤
         ④ 生成 SFT + DPO
         ⑤ 导出 LLaMA-Factory JSONL + provenance + meta.json
         ⑥ 数据集健康体检 —— 失败类型分布、长度分布、倾斜告警
```

## 范围——它修什么，不修什么

trace2train 纠正那些**纠正答案可从 trace 本身推导**的**行为失败**：

- ✅ 选错工具 · 参数错误 · 上下文丢失 · 输出格式错误 · 明显违背常识的答案 · 过度拒绝

它会**跳过**（并告诉你）那些正确性需要外部真值的失败，而不是编造答案：

- ❌ “代码通过测试了吗？” · “这个事实准确吗？” · “任务真的完成了吗？” —— 光看 trace
  无法判断，所以这些会被标记为 `skipped`。

> 这份诚实正是重点：一个塞满“看似合理但其实错误”纠正的训练集，比没有训练集更糟。
> 借助外部真值的纠正是规划中的未来特性。

## Langfuse

Langfuse 支持采用**两阶段快照流程**——先拉一份本地快照，再像其他输入一样 inspect/convert：

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

trace2train langfuse pull langfuse_observations.jsonl
trace2train inspect langfuse_observations.jsonl
trace2train convert langfuse_observations.jsonl -o out
```

**包含：** 通过官方 Public API v2 observations 端点访问的 Langfuse Cloud 与自托管
**v4**。**不含：** v3 旧版 API、blob 存储导出、UI 下载的 JSON 形态、OpenTelemetry、
同步/守护进程行为，以及回写。

> ⚠️ **隐私：** 拉取的快照包含原始 prompt 内容、工具参数和输出。脱敏发生在 `convert`
> 阶段，**而非**拉取时——请据此存储和分享快照。

已针对真实的 Langfuse Cloud **v4.2.0** 项目、使用合成且非敏感的 observations 做过端到端
验证。细节与脱敏证据：[`docs/validation/langfuse-cloud-v4.md`](docs/validation/langfuse-cloud-v4.md)。
更多公开验证证据：[`docs/validation/agentforge.md`](docs/validation/agentforge.md)。

## 设计原则

- **本地优先 CLI。** 无服务器、无账号、无遥测。
- **诚实优于讨好。** 不产出数据，好过产出误导性数据。
- **可审计。** 每条记录都带 `_provenance`（来源 trace、run id、原始错误、归因）。
- **默认便宜。** DeepSeek 每条 trace 只花几分之一美分；换 `base_url`/`model` 即可用任何
  OpenAI 兼容 provider。

<sub>在非 UTF-8 的 Windows 控制台上，输出会自动降级为 ASCII 边框/符号。想要 Unicode
方框，用 UTF-8 运行：`python -X utf8 -m trace2train.cli ...`。</sub>

## 常见问题

**怎么把 agent 的失败变成 fine-tuning 数据？**
把 trace2train 指向你的 trace 导出：`trace2train convert traces.jsonl -o out`。它会检测
行为失败（选错工具、参数错误、过度拒绝……）并写出可喂给 trainer 的 SFT/DPO JSONL。

**能把 LangSmith / Langfuse 日志转成 SFT 或 DPO 数据集吗？**
可以。LangSmith JSONL 导出直接支持。Langfuse 则先用 `trace2train langfuse pull` 快照 v4
observations，再 `convert`。见[支持的输入](#支持的输入)。

**怎么从 agent 轨迹构建工具调用 / function-calling 的 fine-tuning 数据集？**
这正是核心场景——trace2train 专注于工具调用和 agent 行为失败（选错工具、参数畸形、
上下文丢失、输出格式错误、过度拒绝），产出正好纠正这些问题的训练对。

**离线 / 没有 API key 能用吗？**
`inspect` 完全离线（纯规则）。`convert` 需要 LLM 来推导纠正答案；没有 key 时，它把原始
失败 trace 写到 `needs_review/` 供你手动修正，而不是编造数据。

**输出是什么格式，能配 LLaMA-Factory 用吗？**
输出是 LLaMA-Factory 可直接用的 ShareGPT JSONL（`train_sft.jsonl` / `train_dpo.jsonl`），
外加一个 `meta.json` 审计。

**这和直接把日志丢进 trainer 有什么区别？**
它会净化（PII 脱敏、去重、eval 集泄漏过滤），并且关键在于——只在修法可从 trace 推导时
才产出数据；需要外部真值的失败会被跳过，而不是编造。

## 贡献与许可

欢迎那些让 trace2train 保持真实、测试完善、聚焦于工具调用 / agent 行为失败的贡献——见
[`CONTRIBUTING.md`](CONTRIBUTING.md)。以 [MIT](LICENSE) 许可发布。
