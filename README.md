<div align="center">

<div align="center">
  <img src="assets/weco.svg" alt="Weco Logo" width="120" height="120" style="margin-bottom: 20px;">
  <h1>Weco: The Platform for Self-Improving Code</h1>
</div>

[![Python](https://img.shields.io/badge/Python-3.8.0+-blue)](https://www.python.org)
[![PyPI version](https://img.shields.io/pypi/v/weco?label=PyPI%20version&color=f05138&labelColor=555555)](https://badge.fury.io/py/weco)
[![docs](https://img.shields.io/website?url=https://docs.weco.ai/&label=docs)](https://docs.weco.ai/)
[![PyPI Downloads](https://static.pepy.tech/badge/weco?color=4c1)](https://pepy.tech/projects/weco)
[![arXiv on AIDE](https://img.shields.io/badge/arXiv-AIDE-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.13138)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg?labelColor=ffffff&color=F17E01)](https://colab.research.google.com/github/WecoAI/weco-cli/blob/main/examples/hello-kernel-world/colab_notebook_walkthrough.ipynb)

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

![image](https://github.com/user-attachments/assets/a6ed63fa-9c40-498e-aa98-a873e5786509)

---

## Setup

1.  **Install the Package:**

    ```bash
    pip install weco>=0.2.18
    ```

2.  **Set Up LLM API Keys (Required):**

    `weco` requires API keys for the LLMs it uses internally. You **must** provide these keys via environment variables:

    - **OpenAI:** `export OPENAI_API_KEY="your_key_here"` (Create your OpenAI API key [here](https://platform.openai.com/api-keys))
    - **Anthropic:** `export ANTHROPIC_API_KEY="your_key_here"` (Create your Anthropic API key [here](https://console.anthropic.com/settings/keys))
    - **Google:** `export GEMINI_API_KEY="your_key_here"` (Google AI Studio has a free API usage quota. Create your Gemini API key [here](https://aistudio.google.com/apikey) to use `weco` for free.)

---

## Get Started

### Quick Start (Recommended for New Users)

The easiest way to get started with Weco is to use the **interactive copilot**. Simply navigate to your project directory and run:

```bash
weco
```

Or specify a project path:

```bash
weco /path/to/your/project
```

This launches Weco's interactive copilot that will:

1. **Analyze your codebase** using AI to understand your project structure and identify optimization opportunities
2. **Suggest specific optimizations** tailored to your code (e.g., GPU kernel optimization, model improvements, prompt engineering)
3. **Generate evaluation scripts** automatically or help you configure existing ones
4. **Set up the complete optimization pipeline** with appropriate metrics and commands
5. **Run the optimization** or provide you with the exact command to execute

<div style="background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
  <strong>⚠️ Warning: Code Modification</strong><br>
  <code>weco</code> directly modifies the file specified by <code>--source</code> during the optimization process. It is <strong>strongly recommended</strong> to use version control (like Git) to track changes and revert if needed. Alternatively, ensure you have a backup of your original file before running the command. Upon completion, the file will contain the best-performing version of the code found during the run.
</div>

### Manual Setup

**Configure optimization parameters yourself** - If you need precise control over the optimization parameters, you can use the direct `weco run` command:

**Example: Optimizing Simple PyTorch Operations**

```bash
# Navigate to the example directory
cd examples/hello-kernel-world

# Install dependencies
pip install torch

# Run Weco with manual configuration
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py --device cpu" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."

# Run with execution logs saved to disk for debugging
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py --device cpu" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --save-logs \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

**Note:** If you have an NVIDIA GPU, change the device in the `--eval-command` to `cuda`. If you are running this on Apple Silicon, set it to `mps`.

For more advanced examples, including [Triton](/examples/triton/README.md), [CUDA kernel optimization](/examples/cuda/README.md), [ML model optimization](/examples/spaceship-titanic/README.md), and [prompt engineering for math problems](examples/prompt/README.md), please see the `README.md` files within the corresponding subdirectories under the [`examples/`](examples/) folder.

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
| `-M, --model`                  | Model identifier for the LLM to use (e.g., `o4-mini`, `claude-sonnet-4-0`).                                                                                                        | `o4-mini` when `OPENAI_API_KEY` is set; `claude-sonnet-4-0` when `ANTHROPIC_API_KEY` is set; `gemini-2.5-pro` when `GEMINI_API_KEY` is set. | `-M o4-mini`         |
| `-i, --additional-instructions`| Natural language description of specific instructions **or** path to a file containing detailed instructions to guide the LLM.                                                                                             | `None`                                                                                                                                                  | `-i instructions.md` or `-i "Optimize the model for faster inference"`|
| `-l, --log-dir`                | Path to the directory to log intermediate steps and final optimization result.                                                                                                                                             | `.runs/`                                                                                                                                               | `-l ./logs/`        |
| `--eval-timeout`       | Timeout in seconds for each step in evaluation.                                                                                                                                                                             | No timeout (unlimited)                                                                                                                                                  | `--eval-timeout 3600`             |
| `--save-logs`          | Save execution output from each optimization step to disk. Creates timestamped directories with raw output files and a JSONL index for tracking execution history.                                                        | `False`                                                                                                                                                 | `--save-logs`       |

---

### Authentication & Dashboard

Weco offers both **anonymous** and **authenticated** usage:

#### Anonymous Usage
You can use Weco without creating an account by providing LLM API keys via environment variables. This is perfect for trying out Weco or for users who prefer not to create accounts.

#### Authenticated Usage (Recommended)
To save your optimization runs and view them on the Weco dashboard, you can log in using Weco's secure device authentication flow:

1. **During onboarding**: When you run `weco` for the first time, you'll be prompted to log in or skip
2. **Manual login**: Use `weco logout` to clear credentials, then run `weco` again to re-authenticate
3. **Device flow**: Weco will open your browser automatically and guide you through a secure OAuth-style authentication

![image (16)](https://github.com/user-attachments/assets/8a0a285b-4894-46fa-b6a2-4990017ca0c6)

**Benefits of authenticated usage:**
- **Run history**: View all your optimization runs on the Weco dashboard
- **Progress tracking**: Monitor long-running optimizations remotely
- **Enhanced support**: Get better assistance with your optimization challenges

---

## Command Reference

### Basic Usage Patterns

| Command | Description | When to Use |
|---------|-------------|-------------|
| `weco` | Launch interactive onboarding | **Recommended for beginners** - Analyzes your codebase and guides you through setup |
| `weco /path/to/project` | Launch onboarding for specific project | When working with a project in a different directory |
| `weco run [options]` | Direct optimization execution | **For advanced users** - When you know exactly what to optimize and how |
| `weco resume <run-id> [options]` | Resume an interrupted optimization | When a run was interrupted and you want to continue from the last completed step |
| `weco extend <run-id> <steps>` | Extend a completed optimization | When a run completed but you want to continue optimizing with more steps |
| `weco logout` | Clear authentication credentials | To switch accounts or troubleshoot authentication issues |

### Resuming Interrupted Runs

If your optimization run is interrupted (due to network issues, system restart, etc.), you can resume it from the last completed step using the `weco resume` command:

```bash
# Resume an interrupted run from where it left off
weco resume 0002e071-1b67-411f-a514-36947f0c4b31

# Skip validation prompts and resume immediately
weco resume 0002e071-1b67-411f-a514-36947f0c4b31 --skip-validation
```

**Arguments for `weco resume`:**

| Argument | Description | Example |
|----------|-------------|---------|
| `run-id` | The UUID of the run to resume (shown at the start of each run) | `0002e071-1b67-411f-a514-36947f0c4b31` |
| `--skip-validation` | Skip environment validation checks (confirmation that evaluation script, target optimization file, and additional instructions haven't changed) | `--skip-validation` |

**Important notes:**
- The resume feature only works for interrupted runs (status: error, terminated, etc.)
- For completed runs, use the `weco extend` command instead
- You'll be prompted to confirm that your evaluation environment hasn't changed since the original run
- The source file will be restored to the last completed solution before continuing
- All progress and metrics from the original run are preserved

### Extending Completed Runs

If your optimization run has completed but you want to continue optimizing with more steps:

```bash
# Extend a completed run with 20 additional steps
weco extend 0002e071-1b67-411f-a514-36947f0c4b31 20

# Extend with 50 more steps
weco extend 0002e071-1b67-411f-a514-36947f0c4b31 50

# Skip validation prompts and extend immediately
weco extend 0002e071-1b67-411f-a514-36947f0c4b31 30 --skip-validation
```

**Arguments for `weco extend`:**

| Argument | Description | Example |
|----------|-------------|---------|
| `run-id` | The UUID of the completed run to extend | `0002e071-1b67-411f-a514-36947f0c4b31` |
| `steps` | Number of additional steps to add | `20` |
| `--skip-validation` | Skip environment validation checks and extend immediately | `--skip-validation` |

**Important notes:**
- The extend feature only works for completed runs
- For interrupted runs, use the `weco resume` command instead
- You'll be prompted to confirm that your evaluation environment hasn't changed since the original run
- The optimization will continue from the last completed step
- The extended run will have a new total step count (original + additional)

### Model Selection

You can specify which LLM model to use with the `-M` or `--model` flag:

```bash
# Use with onboarding
weco --model gpt-4o

# Use with direct execution
weco run --model claude-3.5-sonnet --source optimize.py [other options...]
```

**Available models:**
- `gpt-4o`, `o4-mini` (requires `OPENAI_API_KEY`)
- `claude-3.5-sonnet`, `claude-sonnet-4-20250514` (requires `ANTHROPIC_API_KEY`)
- `gemini-2.5-pro` (requires `GEMINI_API_KEY`)

If no model is specified, Weco automatically selects the best available model based on your API keys.

---

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
- `output_preview`: A truncated preview of the output

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

## Supported Models

Weco supports the following LLM models:

### OpenAI Models
- `gpt-5` (recommended)
- `gpt-5-mini`
- `gpt-5-nano`
- `o3-pro` (recommended)
- `o3` (recommended)
- `o4-mini` (recommended)
- `o3-mini`
- `o1-pro`
- `o1`
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-4o`
- `gpt-4o-mini`
- `codex-mini-latest`

### Anthropic Models
- `claude-opus-4-1`
- `claude-opus-4-0`
- `claude-sonnet-4-0`
- `claude-3-7-sonnet-latest`

### Gemini Models
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

You can specify any of these models using the `-M` or `--model` flag. Ensure you have the corresponding API key set as an environment variable for the model provider you wish to use.

---

## Contributing

We welcome contributions! Please see [contributing.md](contributing.md) for detailed guidelines on how to contribute to this project.

---
