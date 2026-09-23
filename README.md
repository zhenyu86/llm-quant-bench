# llm-quant-bench

一个通用的大模型效果与性能对比工具。只要模型提供 OpenAI 兼容的 Chat Completions 接口，就可以参与测试。可以比较任意两个或多个模型、推理框架或远程服务，例如：

- 不同厂商的模型；
- 同一模型的不同版本；
- SGLang 与 vLLM 服务；
- 本地服务与官方 API；
- 原始模型与不同精度版本。

客户端只负责发送请求、评分和生成报告，不需要模型权重或 GPU。

## 1. 安装

推荐使用 Conda 和 Python 3.10：

```bash
conda create -n llm-quant-bench python=3.10 -y
conda activate llm-quant-bench
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

以后重新打开终端，进入项目目录后执行：

```bash
conda activate llm-quant-bench
```

## 2. 用什么命令启动

项目的所有操作统一使用：

```bash
python main.py <功能> <参数>
```

例如检查模型 A 的接口：

```bash
python main.py doctor --model model_a
```

这里：

- `python`：使用当前 Conda 环境中的 Python；
- `main.py`：本项目的启动文件；
- `doctor`：选择“检查接口”功能；
- `--model model_a`：使用配置文件中的 `model_a`。


查看所有功能：

```bash
python main.py --help
python main.py doctor --help
python main.py run --help
```

## 3. 在哪里填写模型参数

编辑 [configs/models.yaml](configs/models.yaml)。

```yaml
models:
  model_a:
    provider: sglang
    base_url: http://127.0.0.1:30000/v1
    model: YOUR_MODEL_A

  model_b:
    provider: vllm
    base_url: http://127.0.0.1:8000/v1
    model: YOUR_MODEL_B
```

`model_a` 和 `model_b` 是自己起的配置名称，可以改成 `qwen`、`deepseek` 等容易识别的名字。命令中 `--model` 后面填写的是这个配置名称。

每个模型最重要的参数：

|参数|作用|
|---|---|
|`provider`|服务类型：`sglang`、`vllm`、`deepseek` 或 `openai_compatible`|
|`base_url`|OpenAI 兼容服务地址|
|`model`|服务实际暴露的模型名|
|`api_key_env`|保存 API 密钥的环境变量名称|
|`variant`|可选说明，例如 instruct、fp16、int8|
|`hardware`|运行服务的硬件，用于解释性能差异|
|`server_parameters`|服务启动参数记录，用于判断测试条件是否一致|
|`extra_body`|需要随请求发送的服务专有参数|

`base_model` 是可选的模型家族信息，只用于记录，两个模型无需具有相同的 `base_model` 才能对比。

有鉴权时，只在 YAML 中写环境变量名称：

```yaml
api_key_env: MODEL_A_API_KEY
```

真正的密钥放到环境变量中。

Windows PowerShell：

```powershell
$env:MODEL_A_API_KEY="你的密钥"
```

Linux/macOS：

```bash
export MODEL_A_API_KEY="你的密钥"
```

如果本机存在 `configs/models.local.yaml`，程序会优先读取它。这个文件不会提交到 Git，适合保存内网地址和本机模型路径。也可以显式指定配置：

```bash
python main.py doctor --config configs/models.local.yaml --model model_a
```

## 4. 效果和性能参数

同一个 [configs/models.yaml](configs/models.yaml) 文件下方还有两组参数。

### `quality`：效果测试参数

|参数|作用|
|---|---|
|`output_budget`|每道题最多生成多少 token|
|`temperature`|生成随机性；对比时建议使用 0|
|`timeout_seconds`|单个请求超时时间|
|`concurrency`|同时发送多少道能力测试题|
|`thinking_mode`|是否启用模型思考模式|
|`retries`|能力请求失败后的重试次数|

### `performance`：性能测试参数

性能测试中最容易混淆的是“并发数”和“总请求数”：

- **并发数**：同一时刻最多有多少个请求正在等待模型返回；
- **每轮请求数**：在一个并发场景的一轮中，累计要完成多少个正式请求；
- **轮数**：同一个并发场景重复几次；
- **预热请求**：每轮正式计时前先发送的请求，不进入最终指标。

配置示例：

```yaml
performance:
  concurrency: [1, 4, 8]
  requests_per_round: 128
  rounds: 3
  warmup_requests: 4
  output_budget: 256
  timeout_seconds: 120
```

这组参数会按顺序执行：

1. 并发 1：每轮累计发送 128 个正式请求，重复 3 轮；
2. 并发 4：每轮累计发送 128 个正式请求，最多同时处理 4 个，重复 3 轮；
3. 并发 8：每轮累计发送 128 个正式请求，最多同时处理 8 个，重复 3 轮。

`concurrency: 4` 不表示总共只发 4 个请求。它表示程序会尽量保持最多 4 个请求同时进行，完成一个后再补充下一个，直到这一轮累计完成 `requests_per_round` 个正式请求。

|参数|含义|如何选择|
|---|---|---|
|`concurrency`|要依次测试的并发级别。例如 `[1, 4, 8]` 会产生三个独立场景|从 1 开始，逐步增加，观察吞吐何时不再增长、延迟何时明显上升|
|`requests_per_round`|每个并发场景、每一轮累计发送的正式请求数|快速检查可用 16；正式测试建议 128 或更多，并且应明显大于并发数|
|`rounds`|每个并发场景重复几轮|通常使用 3；多轮可以发现偶然波动|
|`warmup_requests`|每个场景每轮正式计时前发送的请求数|通常至少 4；用于预热连接、缓存和服务|
|`output_budget`|每个请求允许生成的最大 token 数|模型之间必须保持一致；它是上限，不保证一定生成这么多|
|`timeout_seconds`|单个请求允许等待的最长秒数|模型输出较长或并发较高时需要适当增加|
|`mode`|`api` 使用固定文本；`fixed_tokens` 固定输入和输出 token 长度|`api` 更容易使用；`fixed_tokens` 更适合严格性能实验|
|`temperature`|生成随机性|性能对比通常设为 0|
|`thinking_mode`|是否启用思考模式|所有参与对比的模型应使用相同设置|

### 怎样计算总请求数

单个模型的正式请求总数：

```text
并发场景数量 × requests_per_round × rounds
```

单个模型的预热请求总数：

```text
并发场景数量 × warmup_requests × rounds
```

以上面默认参数为例：

```text
并发场景数量 = 3（并发 1、4、8）
正式请求 = 3 × 128 × 3 = 1152
预热请求 = 3 × 4 × 3 = 36
实际请求合计 = 1152 + 36 = 1188
```

如果一次测试两个模型，两个模型都会执行完整负载，因此实际请求合计是：

```text
1188 × 2 = 2376
```

命令行可以临时覆盖两个最常改的参数：

```bash
python main.py perf --model model_a --concurrency 1 4 --requests 16
```

这条命令表示测试并发 1 和并发 4；每个并发场景、每轮累计发送 16 个正式请求。`rounds` 和 `warmup_requests` 仍读取配置文件。

注意：`--profile smoke`、`quick`、`full` 主要控制效果测试的题量，不会自动减少性能请求数。性能请求量由本节参数控制。

`quality.concurrency` 是效果评测同时发送多少道题，与 `performance.concurrency` 是两套独立参数。

## 5. 完整使用流程

```text
填写 model_a、model_b
        ↓
doctor：检查接口是否正常
        ↓
prepare：下载并固定测试题
        ↓
run：测试每个模型的效果和性能
        ↓
summarize：选择一个参考模型并生成对比报告
```

### 第一步：检查接口

```bash
python main.py doctor --model model_a
python main.py doctor --model model_b
```

目的：提前确认服务地址、模型名、鉴权、普通请求和流式请求是否正确。这个步骤不评分。

### 第二步：准备测试题

```bash
python main.py prepare --profile smoke
```

目的：下载 MMLU、C-Eval、GSM8K、ARC、HellaSwag，并固定抽样题目。所有模型使用相同题目，结果才可以直接比较。

`run` 会自动执行这一步，所以也可以不单独运行 `prepare`。首次下载数据时需要联网。

|规模|用途|题量|
|---|---|---|
|`smoke`|快速确认流程|每项 5 题|
|`quick`|日常对比|MMLU/C-Eval 每学科 10 题，其他各 300 题|
|`full`|正式完整测试|数据集指定分割的全部题目|

### 第三步：测试模型

先用小规模测试：

```bash
python main.py run --model model_a --profile smoke
python main.py run --model model_b --profile smoke
```

也可以在一条命令中依次测试：

```bash
python main.py run --model model_a --model model_b --profile smoke
```

`run` 会完成两类测试：

1. 效果测试：计算准确率、失败题数、未评分题数和学科宏平均；
2. 性能测试：测量吞吐、首 token 时间、生成速度、总延迟和成功率。

只测效果：

```bash
python main.py quality --model model_a --profile smoke
```

只测性能：

```bash
python main.py perf --model model_a --profile smoke
```

临时减少性能请求数量：

```bash
python main.py perf --model model_a --profile smoke --concurrency 1 4 --requests 16
```

只测部分能力数据集：

```bash
python main.py quality --model model_a --profile smoke --datasets mmlu gsm8k
```

### 第四步：生成对比报告

选择任意一个已经测试过的模型作为参考：

```bash
python main.py summarize --input outputs --reference model_a --output reports
```

`model_a` 只是报告中的参考模型。其他模型会与它计算效果差值并并排展示性能数据。模型可以来自不同厂商、不同框架、不同硬件或不同 API。

当题目、生成参数、硬件或服务参数不一致时，报告仍展示结果，同时将 `controlled_comparison` 标为 `False` 并写明原因。这时结果表示两个完整服务的端到端体验差异，不应把全部差异只归因于模型本身。

旧版的 `--baseline` 仍然可用，但新文档统一使用更通用的 `--reference`。

## 6. 终端会显示哪些进度

数据准备、效果测试和性能测试都会持续在终端显示当前进度。首次下载数据时，EvalScope 自己输出的下载进度也会直接显示，不需要到模型部署端判断任务是否仍在运行。

常见提示如下：

```text
[配置] 使用配置文件：configs/models.local.yaml
[数据准备 2/5] gsm8k：正在下载和准备数据……
[效果测试 3/5] gsm8k：开始测试，预计 5 道题，并发 4
[性能测试计划] 共 3 个场景；每轮 128 个正式请求，共 3 轮
[性能测试 2/3] 并发=4，第 1/1 轮：开始
```

各类提示的含义：

- `数据准备 i/N`：当前正在处理第几个数据集。缓存已经存在时也会明确提示复用缓存；
- `效果测试 i/N`：当前数据集、预计题数和请求并发数；
- `性能测试 i/N`：当前并发档位和轮次；开始前会汇总正式请求数、预热请求数和总请求数；
- `模型 i/N`：一次测试多个模型时，表示当前正在测试哪个模型；
- EvalScope 的下载、请求和统计输出：原样实时转发到终端，敏感鉴权信息会被隐藏。

底层输出同时保存在相应结果目录的 `evalscope.log` 或 `prepare.log` 中，方便测试结束后排查问题。个别请求生成时间较长时，终端可能短暂停顿；场景开始提示可以确认程序已经进入该项测试。

## 7. 性能指标是什么意思

|指标|含义|
|---|---|
|请求吞吐|每秒成功完成的请求数|
|输出 token/s|测试窗口内每秒生成的输出 token 数|
|TTFT|从发出请求到收到首段内容的时间|
|首答案时间|从发出请求到收到最终回答内容的时间，可与思考内容区分|
|TPOT|首个 token 之后，每个输出 token 的平均生成时间|
|延迟|一个请求从开始到结束的总耗时|
|成功率|正式请求中成功请求的比例|
|轮间 CV|多轮吞吐的相对波动，越小通常越稳定|

远程 API 的指标会同时受到网络、排队、缓存和限流影响。缺少可信 usage 时，token 吞吐和 TPOT 显示 `N/A`。

## 8. 结果保存在哪里

每次测试都会创建独立目录，不会覆盖以前的结果：

```text
outputs/<时间_模型名_随机ID>/
├── run.json
├── quality/
│   ├── quality_status.json
│   └── <数据集>/
│       ├── questions.jsonl
│       ├── effective_config.json
│       └── evalscope.log
└── performance/
    ├── performance_status.json
    ├── workload.txt
    └── c<并发>_r<轮次>/
        ├── requests.jsonl
        └── effective_config.json
```

最终报告位于 `reports/`：

|文件|用途|
|---|---|
|`report.html`|浏览器查看的离线报告|
|`report.md`|文字报告|
|`quality_summary.csv`|Excel 可打开的效果明细|
|`performance_summary.csv`|Excel 可打开的性能明细|
|`comparison.json`|完整结构化结果|

状态含义：

- `complete`：测试和评分完整；
- `incomplete`：请求、样本或评分存在缺失；
- `not_run`：该数据集没有执行。

## 9. 本地模拟验证

不连接真实模型也可以验证项目是否安装正确：

```bash
python scripts/verify_mock.py
```

它会启动临时模拟服务，完成五项 smoke 效果测试、三轮小规模性能测试，并生成 `reports/mock/`。模拟模型固定回答 `42`，分数只用于验证流程。

运行单元测试：

```bash
python -m pytest -q
```

## 10. 常见问题

- 401/403：检查 `api_key_env` 和对应环境变量。
- 404：检查 `base_url` 和服务暴露的模型名。
- 429：降低并发和每轮请求数。
- 超时：增加 `timeout_seconds` 或降低并发。
- 数据下载失败：检查网络和 `quality.dataset_hub`。
- 结果为 `incomplete`：查看对应目录中的 `evalscope.log` 和状态文件。
