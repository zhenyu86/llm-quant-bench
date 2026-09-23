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

普通用户统一使用：

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

项目安装后也可以使用 `qbench doctor ...`，它与 `python main.py doctor ...` 功能相同。本文统一采用更容易理解的 `python main.py` 写法。

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

|参数|作用|
|---|---|
|`mode`|`api` 使用固定文本；`fixed_tokens` 使用固定 token 长度|
|`concurrency`|依次测试的并发数，例如 `[1, 4, 8]`|
|`requests_per_round`|每个并发、每轮发送多少个正式请求|
|`rounds`|每个并发重复测试几轮|
|`warmup_requests`|正式计时前的预热请求数|
|`output_budget`|每个性能请求最多生成多少 token|
|`timeout_seconds`|单个请求超时时间|

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

## 6. 性能指标是什么意思

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

## 7. 结果保存在哪里

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

## 8. 本地模拟验证

不连接真实模型也可以验证项目是否安装正确：

```bash
python scripts/verify_mock.py
```

它会启动临时模拟服务，完成五项 smoke 效果测试、三轮小规模性能测试，并生成 `reports/mock/`。模拟模型固定回答 `42`，分数只用于验证流程。

运行单元测试：

```bash
python -m pytest -q
```

## 9. 常见问题

- 401/403：检查 `api_key_env` 和对应环境变量。
- 404：检查 `base_url` 和服务暴露的模型名。
- 429：降低并发和每轮请求数。
- 超时：增加 `timeout_seconds` 或降低并发。
- 数据下载失败：检查网络和 `quality.dataset_hub`。
- 结果为 `incomplete`：查看对应目录中的 `evalscope.log` 和状态文件。
