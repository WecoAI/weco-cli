# Weco CLI – Code Optimizer for Machine Learning Engineers

[![Python](https://img.shields.io/badge/Python-3.12.0-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/weco.svg)](https://badge.fury.io/py/weco)

`weco` is a command-line interface for interacting with Weco AI's code optimizer, powered by [AI-Driven Exploration](https://arxiv.org/abs/2502.13138). It helps you automate the improvement of your code for tasks like GPU kernel optimization, feature engineering, model development, and prompt engineering.

https://github.com/user-attachments/assets/cb724ef1-bff6-4757-b457-d3b2201ede81

---

## Overview

The `weco` CLI leverages a tree search approach guided by Large Language Models (LLMs) to iteratively explore and refine your code. It automatically applies changes, runs your evaluation script, parses the results, and proposes further improvements based on the specified goal.

![image](https://github.com/user-attachments/assets/a6ed63fa-9c40-498e-aa98-a873e5786509)

---

## Example Use Cases

Here's how `weco` can be applied to common ML engineering tasks:

*   **GPU Kernel Optimization:**
    *   **Goal:** Improve the speed or efficiency of low-level GPU code.
    *   **How:** `weco` iteratively refines CUDA, Triton, Metal, or other kernel code specified in your `--source` file.
    *   **`--eval-command`:** Typically runs a script that compiles the kernel, executes it, and benchmarks performance (e.g., latency, throughput).
    *   **`--metric`:** Examples include `latency`, `throughput`, `TFLOPS`, `memory_bandwidth`. Optimize to `minimize` latency or `maximize` throughput.

*   **Feature Engineering:**
    *   **Goal:** Discover better data transformations or feature combinations for your machine learning models.
    *   **How:** `weco` explores different processing steps or parameters within your feature transformation code (`--source`).
    *   **`--eval-command`:** Executes a script that applies the features, trains/validates a model using those features, and prints a performance score.
    *   **`--metric`:** Examples include `accuracy`, `AUC`, `F1-score`, `validation_loss`. Usually optimized to `maximize` accuracy/AUC/F1 or `minimize` loss.

*   **Model Development:**
    *   **Goal:** Tune hyperparameters or experiment with small architectural changes directly within your model's code.
    *   **How:** `weco` modifies hyperparameter values (like learning rate, layer sizes if defined in the code) or structural elements in your model definition (`--source`).
    *   **`--eval-command`:** Runs your model training and evaluation script, printing the key performance indicator.
    *   **`--metric`:** Examples include `validation_accuracy`, `test_loss`, `inference_time`, `perplexity`. Optimize according to the metric's nature (e.g., `maximize` accuracy, `minimize` loss).

*   **Prompt Engineering:**
    *   **Goal:** Refine prompts used within larger systems (e.g., for LLM interactions) to achieve better or more consistent outputs.
    *   **How:** `weco` modifies prompt templates, examples, or instructions stored in the `--source` file.
    *   **`--eval-command`:** Executes a script that uses the prompt, generates an output, evaluates that output against desired criteria (e.g., using another LLM, checking for keywords, format validation), and prints a score.
    *   **`--metric`:** Examples include `quality_score`, `relevance`, `task_success_rate`, `format_adherence`. Usually optimized to `maximize`.

---


## Setup

1.  **Install the Package:**

    ```bash
    pip install weco
    ```

2.  **Configure API Keys:**

    Set the appropriate environment variables for your desired language model provider:

    -   **OpenAI:** `export OPENAI_API_KEY="your_key_here"`
    -   **Anthropic:** `export ANTHROPIC_API_KEY="your_key_here"`
    -   **Google DeepMind:** `export GEMINI_API_KEY="your_key_here"` (Google AI Studio has a free API usage quota. Create a key [here](https://aistudio.google.com/apikey) to use weco for free.)

---

## Usage
<div style="background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
  <strong>⚠️ Warning: Code Modification</strong><br>
  <code>weco</code> directly modifies the file specified by <code>--source</code> during the optimization process. It is <strong>strongly recommended</strong> to use version control (like Git) to track changes and revert if needed. Alternatively, ensure you have a backup of your original file before running the command. Upon completion, the file will contain the best-performing version of the code found during the run.
</div>

---

### Examples

**Example 1: Optimizing PyTorch simple operations**

```bash
cd examples/hello-kernel-world
pip install torch 
weco --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py --device cpu" \
     --metric speedup \
     --maximize true \
     --steps 15 \
     --model gemini-2.5-pro-exp-03-25 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

Note that if you have an NVIDIA gpu, change the device to `cuda`. If you are running this on Apple Silicon, set it to `mps`.

**Example 2: Optimizing MLX operations with instructions from a file**

Lets optimize a 2D convolution operation in [`mlx`](https://github.com/ml-explore/mlx) using [Metal](https://developer.apple.com/documentation/metal/). Sometimes, additional context or instructions are too complex for a single command-line string. You can provide a path to a file containing these instructions.

```bash
cd examples/metal
pip install mlx
weco --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py" \
     --metric speedup \
     --maximize true \
     --steps 30 \
     --model gemini-2.5-pro-exp-03-25 \
     --additional-instructions examples.rst
```

**Example 3: Level Agnostic Optimization: Causal Self Attention with Triton & CUDA**

Given how useful causal multihead self attention is to transformers, we've seen its wide adoption across ML engineering and AI research. Its great to keep things at a high-level (in PyTorch) when doing research, but when moving to production you often need to write highly customized low-level kernels to make things run as fast as they can. The `weco` CLI can optimize kernels across a variety of different abstraction levels and frameworks. Example 2 uses Metal but lets explore two more frameworks:

1. [Triton](https://github.com/triton-lang/triton)
    ```bash
   cd examples/triton
   pip install torch triton
   weco --source optimize.py \
        --eval-command "python evaluate.py --solution-path optimize.py" \
        --metric speedup \
        --maximize true \
        --steps 30 \
        --model gemini-2.5-pro-exp-03-25 \
        --additional-instructions "Use triton to optimize the code while ensuring a small max float diff. Maintain the same code format."
   ```

2. [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
   ```bash
   cd examples/cuda
   pip install torch
   weco --source optimize.py \
        --eval-command "python evaluate.py --solution-path optimize.py" \
        --metric speedup \
        --maximize true \
        --steps 30 \
        --model gemini-2.5-pro-exp-03-25 \
        --additional-instructions guide.md
   ```

**Example 4: Optimizing a Classification Model**

This example demonstrates optimizing a script for a Kaggle competition ([Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview)) to improve classification accuracy. The additional instructions are provided via a separate file (`examples/spaceship-titanic/README.md`).

First, install the requirements for the example environment:
```bash
pip install -r examples/spaceship-titanic/requirements-test.txt
```
And run utility function once to prepare the dataset
```bash
python examples/spaceship-titanic/utils.py
```

You should see the following structure at `examples/spaceship-titanic`. You need to prepare the kaggle credentials for downloading the dataset.
```
.
├── baseline.py
├── evaluate.py
├── optimize.py
├── private
│   └── test.csv
├── public
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── README.md
├── requirements-test.txt
└── utils.py
```

Then, execute the optimization command:
```bash
weco --source examples/spaceship-titanic/optimize.py \
     --eval-command "python examples/spaceship-titanic/optimize.py && python examples/spaceship-titanic/evaluate.py" \
     --metric accuracy \
     --maximize true \
     --steps 10 \
     --model gemini-2.5-pro-exp-03-25 \
     --additional-instructions examples/spaceship-titanic/README.md
```

*The [baseline.py](examples/spaceship-titanic/baseline.py) is provided as a start point for optimization*

---

### Command Line Arguments

| Argument                    | Description                                                                                                                                                              | Required |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------- |
| `--source`                  | Path to the source code file that will be optimized (e.g., `optimize.py`).                                                                                               | Yes      |
| `--eval-command`            | Command to run for evaluating the code in `--source`. This command should print the target `--metric` and its value to the terminal (stdout/stderr). See note below. | Yes      |
| `--metric`                  | The name of the metric you want to optimize (e.g., 'accuracy', 'speedup', 'loss'). This metric name should match what's printed by your `--eval-command`.            | Yes      |
| `--maximize`                | Whether to maximize (`true`) or minimize (`false`) the metric.                                                                                                           | Yes      |
| `--steps`                   | Number of optimization steps (LLM iterations) to run.                                                                                                                    | Yes      |
| `--model`                   | Model identifier for the LLM to use (e.g., `gpt-4o`, `claude-3.5-sonnet`). Recommended models to try include `o3-mini`, `claude-3-haiku`, and `gemini-2.5-pro-exp-03-25`.        | Yes      |
| `--additional-instructions` | (Optional) Natural language description of specific instructions OR path to a file containing detailed instructions to guide the LLM.                                       | No       |

---

### Performance & Expectations

Weco, powered by the AIDE algorithm, optimizes code iteratively based on your evaluation results. Achieving significant improvements, especially on complex research-level tasks, often requires substantial exploration time.

The following plot from the independent [Research Engineering Benchmark (RE-Bench)](https://metr.org/AI_R_D_Evaluation_Report.pdf) report shows the performance of AIDE (the algorithm behind Weco) on challenging ML research engineering tasks over different time budgets.
<p align="center">
<img src="https://github.com/user-attachments/assets/ff0e471d-2f50-4e2d-b718-874862f533df" alt="RE-Bench Performance Across Time" width="60%"/>
</p>

As shown, AIDE demonstrates strong performance gains over time, surpassing lower human expert percentiles within hours and continuing to improve. This highlights the potential of evaluation-driven optimization but also indicates that reaching high levels of performance comparable to human experts on difficult benchmarks can take considerable time (tens of hours in this specific benchmark, corresponding to many `--steps` in the Weco CLI). Factor this into your planning when setting the number of `--steps` for your optimization runs.

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


## Contributing

We welcome contributions! To get started:

1.  **Fork and Clone the Repository:**
    ```bash
    git clone https://github.com/WecoAI/weco-cli.git
    cd weco-cli
    ```

2.  **Install Development Dependencies:**
    ```bash
    pip install -e ".[dev]"
    ```

3.  **Create a Feature Branch:**
    ```bash
    git checkout -b feature/your-feature-name
    ```

4.  **Make Your Changes:** Ensure your code adheres to our style guidelines and includes relevant tests.

5.  **Commit and Push** your changes, then open a pull request with a clear description of your enhancements.

---
