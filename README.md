<div align="center">

<div align="center">
  <img src="assets/weco.svg" alt="Weco Logo" width="120" height="120" style="margin-bottom: 20px;">
  <h1>Weco: The Code Optimization Agent</h1>
</div>

[![Python](https://img.shields.io/badge/Python-3.8.0+-blue)](https://www.python.org)
[![PyPI version](https://img.shields.io/pypi/v/weco?label=PyPI%20version&color=f05138&labelColor=555555)](https://badge.fury.io/py/weco)
[![docs](https://img.shields.io/website?url=https://docs.weco.ai/&label=docs)](https://docs.weco.ai/)
[![PyPI Downloads](https://static.pepy.tech/badge/weco?color=4c1)](https://pepy.tech/projects/weco)
[![arXiv on AIDE](https://img.shields.io/badge/arXiv-AIDE-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.13138)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg?labelColor=ffffff&color=F17E01)](https://colab.research.google.com/github/WecoAI/weco-cli/blob/main/examples/hello-world/colab_notebook_walkthrough.ipynb)

`pip install weco`

</div>

---

Weco systematically optimizes your code, guided directly by your evaluation metrics.

Example applications include:

- **GPU Kernel Optimization**: Reimplement PyTorch functions using [CUDA](/examples/cuda/README.md) or [Triton](/examples/triton/README.md), optimizing for `latency`, `throughput`, or `memory_bandwidth`.
- **Model Development**: Tune feature transformations, architectures or [the whole training pipeline](/examples/spaceship-titanic/README.md), optimizing for `validation_accuracy`, `AUC`, or `Sharpe Ratio`.
- **Prompt Engineering**: Refine prompts for LLMs (e.g., for [math problems](/examples/prompt/README.md)), optimizing for `win_rate`, `relevance`, or `format_adherence`

![image](assets/example-optimization.gif)

---

## Overview

The `weco` CLI leverages a tree search approach guided by LLMs to iteratively explore and refine your code. It automatically applies changes, runs your evaluation script, parses the results, and proposes further improvements based on the specified goal.


## Install the Package

```bash
pip install weco
```

## Getting Started

### Quickstart with an example project

**Configure optimization parameters yourself** - If you need precise control over the optimization parameters, you can use the direct `weco run` command:

**Example: Optimizing Simple PyTorch Operations**

```bash
git clone https://github.com/WecoAI/weco-cli.git
cd weco-cli/examples/hello-world/
pip install -r requirements.txt

# Run Weco with configuration
weco run --source module.py \
     --eval-command "python evaluate.py --path module.py" \
     --metric speedup \
     --goal maximize \
     --steps 10 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

**Note:** If you have an NVIDIA GPU, change the device in the `--eval-command` to `cuda`. If you are running this on Apple Silicon, set it to `mps`.

For more advanced examples, including [Triton](/examples/triton/README.md), [CUDA kernel optimization](/examples/cuda/README.md), [ML model optimization](/examples/spaceship-titanic/README.md), and [prompt engineering for math problems](examples/prompt/README.md), please see the `README.md` files within the corresponding subdirectories under the [`examples/`](examples/) folder.

> Note: When recommend removing any backticks from your code if any are present. We currently don't support backticks but will support this in the future.

---

### Arguments for `weco run`

**Required:**

| Argument            | Description                                                                                                                                                                                  | Example               |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------- |
| `-s, --source`      | Path to the source code file that will be optimized.                                                                                                                   | `-s model.py`      |
| `-c, --eval-command`| Command to run for evaluating the code in `--source`. This command should print the target `--metric` and its value to the terminal (stdout/stderr). See note below.                        | `-c "python eval.py"` |
| `-m, --metric`      | The name of the metric you want to optimize (e.g., 'accuracy', 'speedup', 'loss'). This metric name does not need to match what's printed by your `--eval-command` exactly (e.g., its okay to use "speedup" instead of "Speedup:").                                    | `-m speedup`          |
| `-g, --goal`        | `maximize`/`max` to maximize the `--metric` or `minimize`/`min` to minimize it.                                                                                                              | `-g maximize`         |

<br>

**Optional:**

| Argument                       | Description                                                                                                                                                                                                                | Default                                                                                                                                                | Example             |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------ |
| `-n, --steps`                  | Number of optimization steps (LLM iterations) to run.                                                                                                                                                                      | 100                                                                                                                                                     | `-n 50`             |
| `-M, --model`                  | Model identifier for the LLM to use (e.g., `o4-mini`, `claude-sonnet-4-5`, `gpt-5`).                                                                                                        | `o4-mini` | `-M o4-mini`         |
| `-i, --additional-instructions`| Natural language description of specific instructions **or** path to a file containing detailed instructions to guide the LLM. Supported file formats include - `.txt`, `.md`, and `.rst`.                                                                                             | `None`                                                                                                                                                  | `-i instructions.md` or `-i "Optimize the model for faster inference"`|
| `-l, --log-dir`                | Path to the directory to log intermediate steps and final optimization result.                                                                                                                                             | `.runs/`                                                                                                                                               | `-l ./logs/`        |
| `--eval-timeout`       | Timeout in seconds for each step in evaluation.                                                                                                                                                                             | No timeout (unlimited)                                                                                                                                                  | `--eval-timeout 3600`             |
| `--save-logs`          | Save execution output from each optimization step to disk. Creates timestamped directories with raw output files and a JSONL index for tracking execution history.                                                        | `False`                                                                                                                                                 | `--save-logs`       |
| `--apply-change`       | Automatically apply the best solution to the source file without prompting.                                                                                                                                                | `False`                                                                                                                                                 | `--apply-change`       |
| `--api-key`            | API keys for LLM providers (BYOK). Format: `provider=key`. Can specify multiple providers.                                                                                                                                  | `None`                                                                                                                                                  | `--api-key openai=sk-xxx` |

---

## Command Reference

### Basic Usage Patterns

| Command | Description | When to Use |
|---------|-------------|-------------|
| `weco run [options]` | Direct optimization execution | **For advanced users** - When you know exactly what to optimize and how |
| `weco resume <run-id>` | Resume an interrupted run | Continue from the last completed step |
| `weco logout` | Clear authentication credentials | To switch accounts or troubleshoot authentication issues |

### Model Selection

You can specify which LLM model to use with the `-M` or `--model` flag:

```bash
weco run --model gpt-5 --source optimize.py [other options...]
```

**Available models (30 total):**

**OpenAI Models:**
- GPT-5 Series: `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5-codex`, `gpt-5-pro`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- O-Series Reasoning: `o3-pro`, `o3`, `o3-mini`, `o4-mini`, `o1-pro`, `o1`, `codex-mini-latest`
- GPT-4 Series: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`

**Anthropic Claude (via Vertex AI):**
- `claude-opus-4-5`, `claude-opus-4-1`, `claude-opus-4`, `claude-sonnet-4-5`, `claude-sonnet-4`, `claude-haiku-4-5`

**Google Gemini:**
- `gemini-3-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`

All models are available through Weco. If no model is specified, Weco automatically selects the best model for your optimization task.

---

### Resuming Interrupted Runs

If your optimization run is interrupted (network issues, restart, etc.), resume from the most recent node:

```bash
# Resume an interrupted run
weco resume 0002e071-1b67-411f-a514-36947f0c4b31

```

Arguments for `weco resume`:

| Argument | Description | Example |
|----------|-------------|---------|
| `run-id` | The UUID of the run to resume (shown at the start of each run) | `0002e071-1b67-411f-a514-36947f0c4b31` |
| `--apply-change` | Automatically apply the best solution to the source file without prompting | `--apply-change` |
| `--api-key` | (Optional) API keys for LLM providers (BYOK). Format: `provider=key` | `--api-key openai=sk-xxx` |

Notes:
- Works only for interrupted runs (status: `error`, `terminated`, etc.).
- You’ll be prompted to confirm that your evaluation environment (source file + evaluation command) hasn’t changed.
- The source file is restored to the most recent solution before continuing.
- All progress and metrics from the original run are preserved.
- Log directory, save-logs behavior, and evaluation timeout are reused from the original run.

### Performance & Expectations

Weco, powered by the AIDE algorithm, optimizes code iteratively based on your evaluation results. Achieving significant improvements, especially on complex research-level tasks, often requires substantial exploration time.

The following plot from the independent [Research Engineering Benchmark (RE-Bench)](https://metr.org/AI_R_D_Evaluation_Report.pdf) report shows the performance of AIDE (the algorithm behind Weco) on challenging ML research engineering tasks over different time budgets.

<p align="center">
<img src="https://github.com/user-attachments/assets/ff0e471d-2f50-4e2d-b718-874862f533df" alt="RE-Bench Performance Across Time" width="60%"/>
</p>

As shown, AIDE demonstrates strong performance gains over time, surpassing lower human expert percentiles within hours and continuing to improve. This highlights the potential of evaluation-driven optimization but also indicates that reaching high levels of performance comparable to human experts on difficult benchmarks can take considerable time (tens of hours in this specific benchmark, corresponding to many `--steps` in the Weco CLI). Factor this into your planning when setting the number of `--steps` for your optimization runs.

---

### Saving Execution Logs

When using the `--save-logs` flag, Weco saves the execution output from each optimization step to help with debugging and analysis. The logs are organized as follows:

```
.runs/
└── <source-file-name>/
    └── <run-uuid>/
        ├── exec_output.jsonl      # Index file with metadata for each step
        ├── outputs/
        │   ├── step_0.out.txt      # Raw output from initial evaluation
        │   ├── step_1.out.txt      # Raw output from step 1
        │   ├── step_2.out.txt      # Raw output from step 2
        │   └── ...
        ├── step_0.py               # Code snapshot from initial evaluation
        ├── step_1.py               # Code snapshot from step 1
        ├── step_2.py               # Code snapshot from step 2
        └── ...
```

Each run is organized under the source file name (e.g., `spaceship-titanic` for `spaceship-titanic.py`) and a unique UUID. The `outputs/` directory and `exec_output.jsonl` file are only created when the `--save-logs` flag is used.

The `exec_output.jsonl` file contains one JSON object per line with:
- `step`: The optimization step number
- `timestamp`: When the execution occurred
- `output_file`: Relative path to the full output file
- `output_length`: Total length of the output

This is particularly useful for:
- Debugging why certain optimizations fail
- Analyzing patterns in evaluation results
- Keeping records of long-running optimization sessions
- Troubleshooting evaluation script issues

---

### Important Note on Evaluation

The command specified by `--eval-command` is crucial. It's responsible for executing the potentially modified code from `--source` and assessing its performance. **This command MUST print the metric you specified with `--metric` along with its numerical value to the terminal (standard output or standard error).** Weco reads this output to understand how well each code version performs and guide the optimization process.

For example, if you set `--metric speedup`, your evaluation script (`eval.py` in the examples) should output a line like:

```
speedup: 1.5
```

or

```
Final speedup value = 1.5
```

Weco will parse this output to extract the numerical value (1.5 in this case) associated with the metric name ('speedup').

**Note on Output Truncation:** When evaluation output exceeds 51,000 characters, Weco truncates it to show the first 25,000 and last 25,000 characters. For best results, ensure your evaluation script prints the metric value near the end of its output.

## Supported Models

A list of models we support can be found in our documentation [here](https://docs.weco.ai/cli/supported-models).

---

## Contributing

We welcome contributions! Please see [contributing.md](contributing.md) for detailed guidelines on how to contribute to this project.

---
