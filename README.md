# llm-quant-bench

用 EvalScope 做五项能力评测，用 EvalScope perf 测 OpenAI 兼容 Chat Completions 服务的客户端观察性能。支持 SGLang、vLLM、DeepSeek 官方 API 和其他兼容服务；评测客户端单独安装，不需要模型权重或 GPU。

## 安装与首次运行

推荐用 Conda 创建独立的 Python 3.10 环境。先确认 Conda 已安装：

```bash
conda --version
```

在 Windows PowerShell、Linux 或 macOS 终端中进入项目目录，然后执行：

```bash
conda create -n llm-quant-bench python=3.10 -y
conda activate llm-quant-bench
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

安装完成后运行小规模评测：

```bash
qbench doctor --model sglang_baseline
qbench prepare --profile smoke
qbench run --model sglang_baseline --profile smoke
qbench summarize --input outputs --baseline sglang_baseline --output reports
```

以后重新打开终端，只需进入项目目录并激活已有环境：

```bash
conda activate llm-quant-bench
```

依赖已固定在 `pyproject.toml`。不要把评测客户端安装到 SGLang 或 vLLM 推理服务使用的环境中。若安装 EvalScope 依赖时提示编译 `editdistance` 失败，请确认当前是上述 Python 3.10 Conda 环境。首次 `prepare` 需联网下载数据集，之后可用 `quality.dataset_ids` 指向本地数据集目录。EvalScope 的缓存目录由 `quality.dataset_dir` 指定。

## 参数在哪里修改

主要参数都在 [`configs/models.yaml`](configs/models.yaml)，文件中已经给每个参数加了中文注释。命令默认读取它；如果需要保存多套方案，可复制该文件并通过 `--config` 指定：

```bash
qbench doctor --config configs/my_test.yaml --model my_model
```

配置分成三部分：

|位置|控制内容|常改参数|
|---|---|---|
|`models.<名称>`|连接哪个模型服务，以及记录量化实验条件|`provider`、`base_url`、`model`、`base_model`、`variant`、`hardware`|
|`quality`|能力评测怎样生成答案|`output_budget`、`temperature`、`timeout_seconds`、`concurrency`、`thinking_mode`|
|`performance`|性能测试施加多大负载|`concurrency`、`requests_per_round`、`rounds`、`warmup_requests`、`output_budget`|

命令中的 `--model sglang_baseline` 指的是 YAML 中 `models.sglang_baseline` 这一段，并不是模型文件路径。最少需要修改：

```yaml
models:
  sglang_baseline:                  # 自己起的配置名称
    provider: sglang
    base_url: http://服务器IP:30000/v1
    model: 服务实际暴露的模型名
```

`base_url` 可写到主机、`/v1` 或 `/v1/chat/completions`，程序会自动规范化。有鉴权时，`api_key_env` 填环境变量名称，真正的密钥不能写入 YAML：

```powershell
# Windows PowerShell
$env:DEEPSEEK_API_KEY="你的密钥"
```

```bash
# Linux/macOS
export DEEPSEEK_API_KEY="你的密钥"
```

`base_model` 表示量化前后的共同基础模型，`variant` 区分 `baseline`、`int8`、`int4` 等版本。`hardware` 和 `server_parameters` 用于确认两次性能测试的实验条件是否一致；它们只记录信息，不负责启动推理服务。量化对比时应保持硬件、服务参数、缓存状态、思考模式和负载相同。

少数参数可以在单次命令中临时覆盖：

|命令参数|作用|
|---|---|
|`--profile smoke`|选择评测规模|
|`--datasets mmlu gsm8k`|只测指定能力数据集|
|`--concurrency 1 4`|临时覆盖性能并发级别|
|`--requests 16`|临时覆盖每轮性能请求数|
|`--output 路径`|修改原始结果目录|

## 一次完整测试的主要流程

```text
填写模型配置
    ↓
doctor 检查接口
    ↓
prepare 准备并锁定测试题
    ↓
quality 测回答正确率  +  perf 测服务速度与稳定性
    ↓
summarize 汇总基线和量化模型，生成报告
```

### 第 1 步：`doctor` 检查模型接口

```bash
qbench doctor --model sglang_baseline
```

它会检查 `/models`、普通 Chat Completions 请求和流式请求，确认地址、模型名、鉴权、最终回答、思考内容和 usage 是否可用。目的是在正式评测前发现 401、404、模型名错误或流式协议不兼容，避免跑很久后才失败。这个步骤不计算能力分数和性能结论。

### 第 2 步：`prepare` 准备能力测试题

```bash
qbench prepare --profile smoke
```

它会通过 EvalScope 下载或读取 MMLU、C-Eval、GSM8K、ARC、HellaSwag，按固定随机种子选择题目，并保存题目清单及哈希。目的是确保基线模型和量化模型回答完全相同的题，否则准确率差异没有可比性。首次运行通常需要联网；之后使用本地缓存。

`quality` 和 `run` 会自动调用 `prepare`，所以该步骤可以省略。单独运行它适合提前下载数据、检查数据是否齐全。

|profile|用途|样本规模|
|---|---|---|
|`smoke`|先确认全流程能运行|每项 5 题|
|`quick`|开发阶段的中等规模比较|MMLU 每学科 10 题、C-Eval 每学科 10 题、其余各 300 题|
|`full`|正式完整评测|使用指定数据集分割的全部样本|

### 第 3 步：`quality` 测试模型效果

```bash
qbench quality --model sglang_baseline --profile smoke
```

它把选定题目发送给模型，由 EvalScope 提取答案并评分。主要结果包括准确率、有效样本数、失败数、未评分数，以及 MMLU/C-Eval 的学科宏平均。目的在于回答“量化后模型能力下降了多少”。

只测部分数据集：

```bash
qbench quality --model sglang_baseline --profile smoke --datasets mmlu gsm8k
```

如果中途失败，可以使用输出中的具体 run 目录续跑同一次能力测试：

```bash
qbench quality --model sglang_baseline --profile smoke --resume outputs/原来的run目录
```

### 第 4 步：`perf` 测试服务性能

```bash
qbench perf --model sglang_baseline --profile smoke
```

默认 `api` 模式使用固定的中文文本请求清单，在多个并发级别下先预热，再执行多轮正式请求。目的在于回答“这个服务能承受多少并发、响应多快、输出速度如何、是否稳定”。

主要指标：

|指标|含义|
|---|---|
|请求吞吐|每秒成功完成的请求数|
|输出 token/s|整个测试窗口每秒生成的输出 token 数|
|TTFT|从发出请求到收到首段内容的时间|
|首答案时间|从发出请求到收到首段最终回答内容的时间；可与思考内容区分|
|TPOT|首个 token 之后，每个输出 token 的平均生成时间|
|延迟|一个请求从开始到结束的总耗时|
|成功率|正式请求中成功请求的比例|
|轮间 CV|多轮 token 吞吐的相对波动，越小通常越稳定|

快速试跑时可以减少负载：

```bash
qbench perf --model sglang_baseline --profile smoke --concurrency 1 4 --requests 16
```

`performance.mode: fixed_tokens` 用于严格控制输入和输出 token 长度，需要配置 `tokenizer_path`、`input_tokens`、`output_tokens`，只支持适配的本地 SGLang/vLLM 服务。无可信 usage 时，token 吞吐和 TPOT 显示 `N/A`，程序不会用字符数冒充 token 数。

### 第 5 步：`run` 一次执行效果和性能测试

```bash
qbench run --model sglang_baseline --profile smoke
```

`run` 等于先执行 `quality`，再执行 `perf`，适合日常使用。也可以一次依次测试多个配置：

```bash
qbench run --model sglang_baseline --model vllm_quantized --profile smoke
```

任一能力数据集或性能场景为 `incomplete` 时，命令返回非零退出码，同时保留已经产生的结果和日志。

### 第 6 步：`summarize` 生成对比报告

```bash
qbench summarize --input outputs --baseline sglang_baseline --output reports
```

它只读取已经保存的结果，不再访问模型服务。程序选择每个模型配置最新的一次运行，将其他模型与 `sglang_baseline` 对比，计算能力差值、配对置信区间、性能指标和失败原因。只有基础模型、硬件、服务参数、负载等条件满足要求时，报告才允许给出量化性能结论。

`reports/` 中会生成：

|文件|用途|
|---|---|
|`report.html`|浏览器查看的离线报告|
|`report.md`|便于阅读和纳入文档|
|`quality_summary.csv`|用 Excel 分析能力结果|
|`performance_summary.csv`|用 Excel 分析性能结果|
|`comparison.json`|供程序继续处理的完整结构化结果|

## 原始结果放在哪里

每次 `quality`、`perf` 或 `run` 都会创建独立的 run 目录，不会覆盖以前的测试：

```text
outputs/<时间_模型名_随机ID>/
├── run.json                         # 本次模型、版本和参数快照
├── quality/
│   ├── quality_status.json          # 各数据集 complete/incomplete 状态
│   └── <数据集>/
│       ├── questions.jsonl          # 每题答案、正确性和错误信息
│       ├── effective_config.json    # 本次实际使用的能力参数
│       └── evalscope.log            # EvalScope 日志
└── performance/
    ├── performance_status.json      # 每个并发和轮次的状态
    ├── workload.txt                 # 本次固定请求清单
    └── c<并发>_r<轮次>/
        ├── requests.jsonl           # 每个请求的时间、usage 和错误
        └── effective_config.json    # 本次实际使用的性能参数
```

状态含义：`complete` 表示样本和评分完整，`incomplete` 表示请求、样本或评分存在缺失，`not_run` 表示该数据集没有选择执行。正式分析前应先确认状态为 `complete`。

## 本地模拟验证

安装测试依赖后运行 `python scripts/verify_mock.py`，会启动临时 OpenAI 兼容服务，执行 `doctor`、五项数据集的 smoke 能力评测、三轮小规模性能测试，并生成 `reports/mock/`。模拟服务对每题固定回答 `\boxed{42}`；分数只用于确认评分链路工作，不能代表真实模型能力。模拟运行在 JSON、CSV、Markdown 和 HTML 报告中均有标记；未选择的数据集显示为 `not_run`，失败或未评分的数据集显示为 `incomplete`。

## 排障

- `doctor` 先查询 `/models`，若不支持模型列表，继续通过最小普通和流式请求校验模型名。401/403 检查 `api_key_env` 和密钥；404 检查模型名和 URL。
- 429 为限流。降低 `performance.concurrency`、`requests_per_round`，并记录正式失败数；性能测试不自动重试。
- 超时可调整 `quality.timeout_seconds`、`performance.timeout_seconds`；并发过大时先降并发。
- 缺少 usage 时 token 指标为 N/A，不能把 SSE chunk 或字符数当成 token。
- 能力数据下载失败时检查 `quality.dataset_hub`、网络或用 `quality.dataset_ids` 指向本地副本。

运行测试：`python -m pytest -q`。项目自身无需数据库、任务队列或前端构建环境；EvalScope perf 可能使用其内部临时结果存储。
