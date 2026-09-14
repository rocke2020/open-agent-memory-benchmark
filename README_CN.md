# Open Agent Memory Benchmark

[English](README.md) | 简体中文 | [日本語](README_JA.md)

**一个可复现、证据优先的基准测试，通过各系统的原生 API 比较自托管 Agent Memory 系统。**

## v0.1.0 结果

**在平衡的 LME-60 筛选中，Hindsight 的答案准确率最高，为 57/60（95.0%）；Mem0 使用的答案可见上下文最少，为 392.6k tokens，比 Hindsight 少 59%，比 OpenViking 少 58%。**

| 提供方 | 答案准确率 | 答案可见上下文 tokens |
|---|---:|---:|
| Hindsight | **57/60 (95.0%)** | 总计 954.4k / 均值 15.91k |
| Mem0 | 52/60 (86.7%) | **总计 392.6k / 均值 6.54k** |
| OpenViking | 51/60 (85.0%) | 总计 927.6k / 均值 15.46k |

**[打开完整的 v0.1.0 报告](https://rocke2020.github.io/open-agent-memory-benchmark/eval_results/v0.1.0/comparison-20260913-133717-82378/report.html)** · [浏览完整的公开评测快照](eval_results/v0.1.0/)

**覆盖范围：** 三个提供方共同评测 60 个 LongMemEval 问题，产生 180/180 个已判定的提供方结果。

在这次 60 题筛选中，Hindsight 的准确率最高。Mem0 最突出的结果是上下文用量较低：它以 392.6k 个答案可见上下文 tokens 达到 52/60（86.7%），比 Hindsight 少 59%。OAMB 不根据这一样本宣布明确的胜者；配对统计分析请参阅完整报告。答案可见上下文 tokens 衡量的是实际展示给答案模型的检索证据，而不是提供方内部的 token 用量。这是一次平衡的 60 题筛选比较，并非完整的 500 题 LongMemEval 复现，也不是通用的提供方排名。

## 生成式模型：DeepSeek V4.1 Flash

OAMB v0.1.0 在三个提供方的记忆处理、答案生成和评判中，**统一只使用 DeepSeek V4.1 Flash 这一种生成式模型**。这个成本较低的选择提供了足以支撑 v0.1.0 比较的质量，同时让复现实验更经济。语义嵌入向量由独立的非生成式模型 `qwen3-embedding:0.6b` 完成。

## 概览

Open Agent Memory Benchmark（OAMB）在同一套评测协议下，通过原生 REST API 比较 Hindsight、Mem0 和 OpenViking。v0.1.0 对每个提供方使用同一组平衡选取的 60 个 LongMemEval 问题：六种问题类型各十题，共产生 180 个提供方结果。

OAMB 的亮点之一是五指标视图：**答案准确率**、**答案可见上下文 tokens**、**索引 tokens**、**检索延迟**和**索引耗时**。定义和详细结果请参阅[完整的 v0.1.0 报告](https://rocke2020.github.io/open-agent-memory-benchmark/eval_results/v0.1.0/comparison-20260913-133717-82378/report.html)。

在 OAMB 之前，我使用 AMB 完成了[完整 500 题 Hindsight 复现](https://github.com/rocke2020/agent-memory-benchmark/tree/deepseek-provider)。该实验复现了公开的 Hindsight 0.4.17 设置，仅替换了外部模型栈：抽取和评判使用 DeepSeek V4 Flash 0731，答案生成使用 DeepSeek V4 Pro 0813。这项研究进一步表明，模型选择会改变被评测的系统，而逐次运行的非确定性也会改变结果；LongMemEval 中少数题目存在歧义，但该数据集整体上仍足以支持广泛比较。为控制成本，OAMB 使用平衡的 LME-60 比较固定版本的提供方（包括 Hindsight 0.9.2），而没有重新运行全部 500 题。

OAMB 在执行前固定问题、模型角色、答案/评判策略和比较设置。各提供方保留其原生存储和索引行为。重试、失败、部分摄取、成本和不可用的测量值均保持可见；OAMB 不生成通用综合分数。检索允许查询嵌入，但禁用生成式查询改写、反思和重排。

## 提供方版本

OAMB 当前固定以下正式发布版本和不可变源码提交：

| 提供方 | 发布版本 | 源码提交 |
|---|---|---|
| Hindsight | [0.9.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.9.2) | [`ebad478240d3171bb88201ececda5e8d9883d22d`](https://github.com/vectorize-io/hindsight/commit/ebad478240d3171bb88201ececda5e8d9883d22d) |
| Mem0 | [2.0.19](https://github.com/mem0ai/mem0/releases/tag/v2.0.19) | [`dc82354e143c2581d505d581a00286d6ef8c3605`](https://github.com/mem0ai/mem0/commit/dc82354e143c2581d505d581a00286d6ef8c3605) |
| OpenViking | [0.4.19](https://github.com/volcengine/OpenViking/releases/tag/v0.4.19) | [`f3afef11637f2d7c11e4b1f36ed2f90630737cdc`](https://github.com/volcengine/OpenViking/commit/f3afef11637f2d7c11e4b1f36ed2f90630737cdc) |

[`provider-services/versions.env`](provider-services/versions.env) 是权威的运行时版本固定来源；其中的镜像摘要和源码归档哈希确保提供方准备过程可复现。

## 快速开始

你需要 Git、Python 3.11+、`uv`、带 Compose 的 Docker、`curl`、`jq`、`shasum`，以及 OpenAI 兼容模型端点的凭据。准备和评测过程可能产生收费的模型调用。它们会创建隔离的提供方状态，并保留现有的提供方/数据库数据。

### 1. 配置

克隆仓库并准备私有配置：

```bash
git clone https://github.com/rocke2020/open-agent-memory-benchmark.git
cd open-agent-memory-benchmark
cp .env.example .env
chmod 600 .env
```

编辑 `.env`：设置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_LIGHT_MODEL` 和 `LLM_DEEP_MODEL`；保持 `LLM_URL_TYPE=openai_chat`。[模板](.env.example)为两个模型配置统一只使用 `deepseek-flash`（DeepSeek-V4.1-Flash）这一种生成式模型。轻量配置负责提供方抽取和评判，深度配置负责答案生成。[基准配置](configs/benchmark.yml)是角色分配、思考强度和执行控制的权威来源。

如需复用嵌入服务，请设置其 OpenAI 兼容的基础 URL。默认配置要求 `qwen3-embedding:0.6b`、1,024 维向量，以及足以容纳提供方源会话输入的上下文容量。对于已有的宿主机本地服务：

```dotenv
OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:18000/v1
OAMB_EMBEDDING_API_KEY=
```

对于无密钥服务，嵌入 API 密钥可以缺失或为空。OAMB 会为 Docker 容器转换这个 HTTP IPv4 回环地址，同时保持宿主机探测地址不变。也支持远程 HTTPS 端点；HTTPS 和 IPv6 回环地址会被拒绝。

如果 URL 未设置或仍为 `change-me`，预检会使用本地回退方案：macOS 需要同级目录中的 `vllm-metal` 安装和已缓存的模型文件；Linux 需要 Ollama，并可能在首次使用时下载嵌入模型。请参阅[本地嵌入辅助工具](scripts/start_local_embedding)。

### 2. 预检与运行

准备依赖、数据集、固定版本的提供方服务和模型就绪状态：

```bash
./precheck.sh
```

仅在看到 `precheck: PASS` 后继续。然后运行完整比较：

```bash
./run.sh --full_test
```

各提供方会在配置的限制内并发运行。每个提供方每 30 秒输出一次状态、已用时间和已保存问题数。完成的结果会原子保存；最终比较要求全部 180 个结果齐全并通过最新证据验证。

其他运行模式：

```bash
./run.sh --full_test --resume  # 复用已完成问题；在新隔离范围中重跑缺失问题
./run.sh --smoke_test         # 可选：对三个提供方执行单题诊断
./run.sh --dry-run            # 不调用模型或提供方，仅检查固定计划
```

直接运行 `./run.sh` 也会选择 smoke 模式。Smoke 结果单独保存，完整运行不依赖它们。

在 [configs/benchmark.yml](configs/benchmark.yml) 的 `execution` 下设置并发：

| 设置 | 限制对象 |
|---|---|
| `max_parallel_providers_per_dataset` | 并发的提供方评测 |
| `max_parallel_history_ingestions_per_provider` | 每个提供方并发摄取的独立历史 |
| `max_parallel_questions_per_provider` | 每个提供方并发的检索、答案和评判流水线 |

在同一配置中通过 `retrieval.top_k` 设置共享的原生候选项上限。该值会固定到解析后的计划中，并由每个提供方适配器使用：Mem0 将其作为 `top_k` 发送，OpenViking 将其作为 `limit` 发送；由于 Hindsight 的 recall 端点没有数量参数，它只保留提供方排序后、规范化候选项中的前 N 个。

同一历史中的会话保持串行。这些限制不会约束提供方内部的嵌入/模型请求。全新运行使用已预检的计划。每次 `--resume` 都会从当前 YAML 读取历史和问题的并发上限，用于尚未完成的工作；更改这两个值不会使已完成结果失效。其他计划设置保持固定，已经运行的进程不会重新加载 YAML。不要编辑 `resolved-plan.json`。

重试和单次尝试超时设置与这些控制项一起记录在配置中。它们不是整个运行或支出的上限。本地超时并不能证明远端处理或计费已经停止。

### 3. 停止提供方服务

在活动的完整运行中按一次 `Ctrl-C`，即可停止该运行及其提供方容器。若要直接停止本仓库的提供方服务容器：

```bash
./provider-services/bin/provider-services stop
```

停止操作会保留 Docker 卷、提供方/数据库数据、已保存结果、日志和证据。它会在提供方执行结束后清除临时生命周期标记。使用上面的 resume 命令继续中断的完整运行。

## 结果与故障排查

完整报告写入 `outputs/full-test/<run-label>/`，smoke 报告写入 `outputs/smoke-test/<run-label>/`。自包含 HTML 会自动打开。无界面环境可设置 `OAMB_NO_OPEN=1`；报告仍会生成并输出其路径。

### 从各提供方的独立结果生成报告

当 Hindsight、Mem0 和 OpenViking 已分别完成并作为独立快照保存在同一目录下时，可以在不重新运行提供方的情况下生成报告：

```bash
./run.sh --generate-report --result-dir=./eval_results/v0.1.0
```

该目录必须恰好包含每个提供方的一份完整 60 题快照，并且每份快照中都包含 `resolved-plan.json` 和 `results/<provider>.json`。带版本的历史快照还可以在根目录包含 `case-manifest.json`，以便在新运行使用当前清单时保留当时评测的问题 ID。该命令需要根目录 `.env` 提供固定评判模型的连接；如果没有匹配的分析缓存，它会进行一次有界的 LLM 分析调用。当报告输入和分析配置不变时，它会复用 `<result-dir>/report-analysis-cache` 中按内容寻址的缓存，不再调用模型。该命令不需要预检或提供方服务，不会修改源快照，并会创建一次写入的 `comparison/` 目录（如果该名称已存在，则创建带时间戳的后继目录）。它会记录每个源计划/结果的哈希；如果保存的计划不能证明使用了相同的比较控制项，则抑制受影响的领先者声明。

HTML 会保持比较聚焦：省略全部不可用的次要计量项和 Wilson 区间文字；“问题结果”部分只显示至少有一个提供方答错的问题。完整的 60 题矩阵和统计证据仍保存在 `report.json` 中。

### 检查已保存结果的准确率

答案准确率是主要指标之一。对于完整或中断的完整运行，使用已保存的结果映射计算 `correct / judged`；将 `saved / 60` 作为完成覆盖率单独报告。

```bash
result_file=".../$provider.json"
jq -r '
  [.[] | select(.evaluation.disposition == "judged")] as $judged
  | ($judged | map(.evaluation.numerator) | add // 0) as $correct
  | ($judged | map(.evaluation.denominator) | add // 0) as $total
  | if $total == 0 then "accuracy: unavailable; completion: \(length)/60"
    else "accuracy: \($correct)/\($total) = \(((10000 * $correct / $total | round) / 100))%; completion: \(length)/60"
    end
' "$result_file"
```

每次运行都会输出 `run: log=<path>`，并将终端输出保存到 `outputs/tmp/`。随后可执行 `tail -f <path>`。完整运行的进度来自已保存的 `results/{hindsight,mem0,openviking}.json`；resume 计数包含先前已完成的问题。

有关已观察到的提供方行为和失败，请参阅[调查记录](docs/investigations/README.md)。

## 致谢

感谢 Vectorize 的 [Agent Memory Benchmark（AMB）](https://github.com/vectorize-io/agent-memory-benchmark)及其贡献者公开评测工具、方法、提示词和结果。AMB 是 OAMB 评测流程和设计的重要参考；OAMB 为独立实现。

我们也感谢 [LongMemEval 作者](https://github.com/xiaowu0162/LongMemEval)提供数据集和评判标准，感谢 [Hindsight](https://github.com/vectorize-io/hindsight)、[Mem0](https://github.com/mem0ai/mem0) 和 [OpenViking](https://github.com/volcengine/OpenViking) 社区提供开源记忆系统。直接复用的提示词材料保留其[来源归属和声明](prompt-packs/README.md)。

## 项目规范

请参阅[安全说明](SECURITY.md)、[数据集来源](DATASETS.md)和[第三方组件](THIRD_PARTY.md)。

OAMB 采用 [Apache-2.0](LICENSE) 许可。数据集和第三方制品各自遵循其原有许可证。
