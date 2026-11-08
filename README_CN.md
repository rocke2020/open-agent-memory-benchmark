# Open Agent Memory Benchmark

[English](README.md) | 简体中文 | [日本語](README_JA.md)

## 概览

Open Agent Memory Benchmark（OAMB）是 Open Agent Memory（OAM）的独立评测仓库，提供公开的 LME-500 和 BEAM-10M 评测流程。仓库涵盖数据集准备、执行和报告，并提供与 Hindsight、Mem0 和 OpenViking 的独立对比评测。

已保存的 OAM v0.0.1 开发版评测覆盖 [LongMemEval 全部 500 题](eval_results/lme500/README.md)和 [BEAM-10M 的 10 段完整对话、200 题](eval_results/beam10m/README.md)。LME-500 达到 **94.0%（470/500）**；BEAM-10M 按重试策略确定的最终平均评分为 **61.56%**，其中 **142/200** 题达到 0.5 阈值。这些报告复用已完成的评测，无需配置环境或发起新的评测调用。

离线 HTML 报告可浏览已保存的问题、答案、评判及完整注入上下文：[LME-500](eval_results/lme500/report.html) · [BEAM-10M，含 Hindsight 已保存上下文的回放](eval_results/beam10m/report.html)。下载任一文件后，即可在当前浏览器中直接打开。

**已发布的 OAM v0.0.1 评测在抽取、回答和评判阶段均使用 `deepseek-v4.1-flash`，侧重日常 Agent 任务，而非追求排行榜最高分。模型选择会影响结果：** Hindsight 使用 **Gemini 3.1 Pro**（`gemini-3.1-pro-preview`）回答时，[报告的 BEAM-10M 分数为 **64.1%**](https://vectorize.io/benchmarks)；我们使用 DeepSeek 回放其已保存上下文，[经过 43 次符合条件的 max 思考强度答案重试后，分数为 **52.16%**](eval_results/beam10m/README.md)，平均评分相差约 **12 个百分点**。

[原生 OAM 工作流](docs/native-oam.md)通过 `./precheck.sh --oam` 和 `./run.sh --oam` 执行 LME-500 和 BEAM-10M 评测。仓库提供固定版本的数据集下载/导出脚本、记录的回答与评判提示词、可配置的回答/评判模型，以及经审查后允许单次 max 思考强度答案重试的流程。报告分别保留首次与策略最终分数。小规模诊断验证执行路径；完整数据集上的分数接近程度仍需单独测量。

**完整 LME-500 和 BEAM-10M 的原生 OAM 评测是 OAMB 的主要任务。** 次要任务是通过 Hindsight、Mem0 和 OpenViking 的原生 REST API，在同一组平衡选取的 60 个 LongMemEval 问题上进行比较：六种问题类型各十题，共产生 180 个提供方结果。

离线报告分别呈现五项指标：

- **答案准确率**：使用检索到的记忆回答问题的正确程度。
- **上下文 tokens**：展示给答案模型的检索证据量。
- **索引 tokens**：提供方报告的记忆构建生成式模型 tokens；不包含嵌入和提供方用户初始化。
- **检索延迟**：提供方记忆查询请求的耗时。
- **索引耗时**：从首次写入历史到可查询就绪的耗时。

OAMB 在执行前确定问题、模型角色、回答/评判策略和比较设置。各提供方保留其原生存储和索引行为。重试、失败、部分摄取、成本和不可用的测量值均保持可见；OAMB 不生成通用综合分数。检索允许查询嵌入，但禁用生成式查询改写、反思和重排。

## 数据集

| 数据集 | 评测覆盖 | 执行工作流 | 评分 |
|---|---|---|---|
| LongMemEval S，完整 LME-500 | 500 题 | 原生 OAM，使用 [LME-500 配置](configs/oam-lme500.yml) | 按问题类别判定正确性 |
| BEAM-10M | 10 段完整对话，十种类别共 200 题 | 原生 OAM，使用 [BEAM-10M 配置](configs/oam-beam10m.yml) | 先计算每题评分项的均值，再对所有问题等权平均 |
| LongMemEval S，平衡 LME-60 | 六种问题类型各十题，共 60 题 | 次要的 Hindsight、Mem0 和 OpenViking REST 比较；180 个提供方结果 | 按问题类别判定正确性 |

BEAM 覆盖拒答、矛盾消解、事件排序、信息提取、指令遵循、知识更新、多会话推理、偏好遵循、总结和时间推理。平均评分与得分至少为 0.5 的问题比例是两个不同指标。

比较工作流通过 `./precheck.sh` 下载 LongMemEval，也可以单独运行 `./scripts/download/longmemeval.sh`。运行 `./scripts/download/beam10m.sh` 下载固定版本的官方 BEAM-10M Parquet 分片，并按评测使用的精确 JSON 序列化和 SHA-256 导出 `datasets/beam/10m.json`。两个工具都会保留并拒绝现有的不匹配文件。下载的数据集不进入 Git；参见[数据集来源](DATASETS.md)。

使用[原生 OAM 指南](docs/native-oam.md)准备隔离部署，再依次执行 `./precheck.sh --oam` 和 `./run.sh --oam` 运行任一完整数据集。两个数据集均提供合成诊断配置。新运行记录被评测的 OAM 版本和设置；已保存的 [LME-500](eval_results/lme500/README.md) 和 [BEAM-10M](eval_results/beam10m/README.md) 报告保留历史结果。

## 提供方版本

OAMB 首先面向 **OAM v0.0.1**，同时保留以下对比提供方版本和源码提交：

| 提供方 | 发布版本 | 源码提交 |
|---|---|---|
| Open Agent Memory（OAM） | **0.0.1 开发快照** | `c0f2d2e1f62decf3af188d7c5dff2a529134610e` |
| Hindsight | [0.9.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.9.2) | [`ebad478240d3171bb88201ececda5e8d9883d22d`](https://github.com/vectorize-io/hindsight/commit/ebad478240d3171bb88201ececda5e8d9883d22d) |
| Mem0 | [2.0.19](https://github.com/mem0ai/mem0/releases/tag/v2.0.19) | [`dc82354e143c2581d505d581a00286d6ef8c3605`](https://github.com/mem0ai/mem0/commit/dc82354e143c2581d505d581a00286d6ef8c3605) |
| OpenViking | [0.4.19](https://github.com/volcengine/OpenViking/releases/tag/v0.4.19) | [`f3afef11637f2d7c11e4b1f36ed2f90630737cdc`](https://github.com/volcengine/OpenViking/commit/f3afef11637f2d7c11e4b1f36ed2f90630737cdc) |

[`provider-services/versions.env`](provider-services/versions.env) 固定对比提供方的镜像和源码归档。OAM 开发镜像按下述步骤从上表的源码提交构建。

### OAM 开发镜像

从包含记录提交的 OAM 检出目录构建 v0.0.1 开发镜像：

```bash
./provider-services/oam/build-image.sh /path/to/open-agent-memory c0f2d2e1f62decf3af188d7c5dff2a529134610e
docker run --rm oamb/oam:0.0.1-c0f2d2e1f62d --version
```

构建使用已提交源码和 OAM 锁文件，不包含本地未提交文件。生成的本地镜像是开发快照，并非正式发布镜像。可用的 OAM 部署还需要隔离的 PostgreSQL 数据库、相应凭据和已配置的模型服务。参见[原生 OAM 设置与执行指南](docs/native-oam.md)，准备隔离的 PostgreSQL/API/worker，并了解输入与模型绑定检查、检查点和续跑。

`provider-services/oam/` 中的评测覆盖配置为 LME-500 选择共享 `small` 向量索引，为 BEAM-10M 选择 `large` 用户专属索引。两者均禁用 Insights 和 Profile，并使用 low 抽取思考强度。评测器须对 LME 请求 fast 召回，对 BEAM 请求 normal 召回；这些模式是请求参数，不是索引设置。其余设置来自镜像中的 OAM 权威配置；宣称与历史流程匹配前，须对照历史协议检查。

历史评测使用输入上限为 32K tokens 的 Qwen3 嵌入和 OAM 的近似 tokenizer。当前 OAM 默认使用 `toktok-rs` 的 `o200k_base` 进行摄取打包，因此仅靠提供的镜像和覆盖配置无法重现历史分块。为提高分块计数准确性，优先使用嵌入模型自己的 tokenizer，并配置实际输入与请求限制。例如，`cl100k_base` 匹配 OpenAI 的 `text-embedding-3` 模型。使用前须检查具体模型的 tokenizer、输入格式和限制；默认编码并不适用于所有模型。

当前 OAM 通过 `tokenizer.type` 支持摄取打包所需的 `o200k_base`、`cl100k_base`、旧版 `oam-unicode-quarter-v1` 计数器和模型专用 `tokenizer_json` 资源。嵌入输入校验另有 `models.embedding.tokenizer` 设置：`auto` 为已识别的 OpenAI 和 Qwen3-Embedding 模型名称选择匹配的 tokenizer；`openai` 显式选择 `cl100k_base`，`qwen` 选择内置 Qwen3-Embedding tokenizer。优先为可识别的模型名称使用 `auto`，以选择正确变体。也可用 tokenizer 对象指定自定义 `tokenizer_json` 资源。

固定的开发镜像包含这些自动嵌入配置及 `openai`/`qwen` 选项。应按服务要求配置 `models.embedding.max_input_tokens`、`max_request_tokens` 和 `max_request_items`，并验证嵌入维度。如果没有支持的计数器匹配模型，应将计数视为近似值并预留余量；它不能保证输入符合提供方限制。更换嵌入模型或 tokenizer 可能改变分块、嵌入和结果，应披露这些变化，不能称为与历史评测完全相同。

原生 BEAM 工作流在每段对话的嵌入检索单元生成后创建专属索引，再执行检索、回答和评判。小规模合成样本已覆盖真实摄取、召回、回答、评判和离线 HTML 检查；这些诊断与历史结果及完整数据集结果分别记录。具体命令、制品边界和当前限制见[工作流指南](docs/native-oam.md)。

## 在 LME-500 和 BEAM-10M 上评测 OAM

按[原生 OAM 设置指南](docs/native-oam.md)构建 OAM 镜像，在私有 `.env.oam` 中配置模型端点，并为两个数据集分别启动隔离部署。独立的 OAM 检出目录提供产品镜像；评测在本仓库运行。

抽取、回答和评判均使用 **`deepseek-v4.1-flash`**；嵌入使用 **`qwen3-embedding:0.6b` 和 1,024 维向量**。将 [.env.oam.example](.env.oam.example) 复制为私有 `.env.oam`，配置端点和密钥，并使两个模型角色与所选 YAML 中的 `model` 保持一致。`deepseek-flash` 等网关别名仅在解析到目标模型时适用于该复现目标。抽取使用 low 思考强度；首次回答和评判使用 high。失败问题经过审查后，可在上下文和提示词不变的情况下进行一次 max 思考强度答案重试，再以 high 强度评判。

准备好输入和两个部署后，运行以下完整配置。这些命令会产生收费调用。请将客户端环境路径替换为各部署输出的路径。

```bash
./precheck.sh --oam configs/oam-lme500.yml \
  --env-file outputs/oam-services/lme-example/client.env \
  --output outputs/oam-lme500
./run.sh --oam outputs/oam-lme500

./precheck.sh --oam configs/oam-beam10m.yml \
  --env-file outputs/oam-services/beam-example/client.env \
  --output outputs/oam-beam10m
./run.sh --oam outputs/oam-beam10m
```

每个输出目录包含 `report.html`、`summary.json` 和保留的逐题证据。使用 `./run.sh --oam <output-directory> --resume` 继续中断的评测。完整运行前，可用对应的 `*-smoke.yml` 配置和全新输出目录执行小规模合成诊断。

对于首次回答失败的问题，须先完成生成的 `retry-eligibility.json` 中的源数据、上下文和标准答案审查，再续跑。明显错误的标准答案禁止重试。完成的重试决定策略最终结果，即使分数降低也不例外。待审查问题保持可见，且不能宣称策略最终评测已完成；参见[审查与续跑流程](docs/native-oam.md#3-full-datasets-and-resume)。

## 快速开始：LME-60 比较

以下命令启动次要的提供方比较。主要的完整数据集 OAM 任务请使用[上面的原生评测工作流](#在-lme-500-和-beam-10m-上评测-oam)。阅读已保存的 [LME-500](eval_results/lme500/README.md) 和 [BEAM-10M](eval_results/beam10m/README.md) 评测无需配置环境。


你需要 Git、Python 3.11+、`uv`、带 Compose 的 Docker、`curl`、`jq`、`shasum`，以及 OpenAI 兼容模型端点的凭据。准备和评测过程可能产生收费的模型调用。它们会创建隔离的提供方状态，并保留现有的提供方/数据库数据。

### 1. 配置

克隆仓库并准备私有配置：

```bash
git clone https://github.com/rocke2020/open-agent-memory-benchmark.git
cd open-agent-memory-benchmark
cp .env.example .env
chmod 600 .env
```

编辑 `.env`：设置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_LIGHT_MODEL` 和 `LLM_DEEP_MODEL`；保持 `LLM_URL_TYPE=openai_chat`。[模板](.env.example)的两个模型配置均默认使用 `deepseek-flash`。轻量配置负责提供方抽取和评判，深度配置负责答案生成。[基准配置](configs/benchmark.yml)是角色分配、思考强度和执行控制的权威来源。

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

请求与验证边界见[提供方检索语义](docs/designs/retrieval-top-k-provider-semantics.md)。

同一历史中的会话保持串行。这些限制不会约束提供方内部的嵌入/模型请求。全新运行使用已预检的计划。每次 `--resume` 都会从当前 YAML 读取历史和问题的并发上限，用于尚未完成的工作；更改这两个值不会使已完成结果失效。其他计划设置保持固定，已经运行的进程不会重新加载 YAML。不要编辑 `resolved-plan.json`。

重试和单次尝试超时设置与这些控制项一起记录在配置中。它们不是整个运行或支出的上限。本地超时并不能证明远端处理或计费已经停止。

### 3. 停止提供方服务

在活动的完整运行中按一次 `Ctrl-C`，即可停止该运行及其提供方容器。若要直接停止本仓库的提供方服务容器：

```bash
./provider-services/bin/provider-services stop
```

停止操作会保留 Docker 卷、提供方/数据库数据、已保存结果、日志和证据。它会在提供方执行结束后清除临时生命周期标记。使用上面的 resume 命令继续中断的完整运行。

## LME-60 结果与故障排查

完整报告写入 `outputs/full-test/<run-label>/`，smoke 报告写入 `outputs/smoke-test/<run-label>/`。自包含 HTML 会自动打开。无界面环境可设置 `OAMB_NO_OPEN=1`；报告仍会生成并输出其路径。

### 从各提供方的独立结果生成报告

当 Hindsight、Mem0 和 OpenViking 已分别完成并作为独立快照保存在同一目录下时，可以在不重新运行提供方的情况下生成报告：

```bash
./run.sh --generate-report --result-dir=outputs/saved-results/lme60-provider-native
```

该目录必须恰好包含每个提供方的一份完整 60 题快照，并且每份快照中都包含 `resolved-plan.json` 和 `results/<provider>.json`。该命令需要根目录 `.env` 提供固定评判模型的连接；如果没有匹配的分析缓存，它会进行一次有界的 LLM 分析调用。当报告输入和分析配置不变时，它会复用 `<result-dir>/report-analysis-cache` 中按内容寻址的缓存，不再调用模型。该命令不需要预检或提供方服务，不会修改源快照，并会创建一次写入的 `comparison/` 目录（如果该名称已存在，则创建带时间戳的后继目录）。它会记录每个源计划/结果的哈希；如果保存的计划不能证明使用了相同的比较控制项，则抑制受影响的领先者声明。

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
