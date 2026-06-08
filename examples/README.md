## Weco Examples

Explore runnable examples that show how to use Weco to optimize ML models, prompts, and GPU kernels. Pick an example and get going in minutes.

### Table of Contents

- [Weco Examples](#weco-examples)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Examples at a glance](#examples-at-a-glance)
- [Quick starts](#quick-starts)
  - [🧭 Hello World](#-hello-world)
  - [📋 LangSmith ZephHR QA](#-langsmith-zephhr-qa)
  - [⚡ Triton Optimization](#-triton-optimization)
  - [🚀 CUDA Optimization](#-cuda-optimization)
  - [🧠 Prompt Engineering](#-prompt-engineering)
  - [📊 Extract Line Plot — Chart to CSV](#-extract-line-plot--chart-to-csv)
  - [🛰️ Model Development — Spaceship Titanic](#️-model-development--spaceship-titanic)
  - [🕵️ Fraud Detection — IEEE-CIS](#️-fraud-detection--ieee-cis)

### Prerequisites

- **Install the CLI**
```bash
pip install weco
```

### Examples at a glance

| Example | Focus | Dependencies | Docs |
| :-- | :-- | :-- | :-- |
| 🧭 Hello World | Learn the Weco workflow on a small PyTorch model | `torch` | [README](hello-world/README.md) • [Colab](hello-world/colab_notebook_walkthrough.ipynb) |
| 📋 LangSmith ZephHR QA | LLM-judge prompt optimization on HR policy QA | `openai`, `langsmith`, OpenAI + LangSmith API keys | [README](langsmith-zephhr-qa/README.md) |
| ⚡ Triton Optimization | Speed up attention with Triton kernels | `numpy`, `torch`, `triton`, NVIDIA GPU | [README](triton/README.md) |
| 🚀 CUDA Optimization | Generate low-level CUDA kernels for max speed | `ninja`, `numpy`, `torch`, `triton`, NVIDIA GPU, CUDA Toolkit | [README](cuda/README.md) |
| 🧠 Prompt Engineering | Iteratively refine LLM prompts to improve accuracy | `openai`, `datasets`, OpenAI API key | [README](prompt/README.md) |
| 📊 Agentic Scaffolding | Optimize agentic scaffolding for chart-to-CSV extraction | `openai`, `huggingface_hub`, `uv`, OpenAI API key | [README](extract-line-plot/README.md) |
| 🛰️ Spaceship Titanic | Improve a Kaggle model training pipeline | `pandas`, `numpy`, `scikit-learn`, `torch`, `xgboost`, `lightgbm`, `catboost` | [README](spaceship-titanic/README.md) |
| 🕵️ Fraud Detection | Optimize a fraud pipeline on IEEE-CIS (real Vesta transactions) | `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pyarrow`, `kaggle` | [README](fraud-detection/README.md) |

---

## Quick starts

Minimal commands to run each example. For full context and explanations, see the linked READMEs.

> **Tip**: Add `--apply-change` to any command below to automatically apply the best solution to your source file without prompting.

### 🧭 Hello World

```bash
cd examples/hello-world
pip install -r requirements.txt
weco run --source module.py \
     --eval-command "python evaluate.py --path module.py" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```
- **Tip**: Use `--device cuda` (NVIDIA GPU) or `--device mps` (Apple Silicon).

### 📋 LangSmith ZephHR QA

- **Requirements**: OpenAI + LangSmith API keys
- **Setup**: Configure `helpfulness` (1–5) and `correctness` (binary) online evaluators in LangSmith dashboard
- **Run**:
```bash
cd examples/langsmith-zephhr-qa
python setup_dataset.py
weco run --source agent.py \
  --eval-backend langsmith \
  --langsmith-dataset zephhr-qa-opt \
  --langsmith-target agent:answer_hr_question \
  --langsmith-evaluators evaluators:json_schema_validity evaluators:conciseness \
  --langsmith-dashboard-evaluators helpfulness correctness \
  --langsmith-metric-function evaluators:qa_score \
  --additional-instructions optimizer_exemplars.md \
  --metric qa_score --goal maximize --steps 30
```

### ⚡ Triton Optimization

- **Requirements**: NVIDIA GPU

```bash
cd examples/triton
pip install -r requirements.txt
weco run --source module.py \
     --eval-command "python evaluate.py --path module.py" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --model o4-mini \
     --additional-instructions "Use a combination of triton and pytorch to optimize the forward pass while ensuring a small max float diff. Maintain the same code interface. Do not use any fallbacks. Assume any required dependencies are installed and data is already on the gpu." \
     --eval-timeout 120
```

### 🚀 CUDA Optimization

- **Requirements**: NVIDIA GPU and CUDA Toolkit
- **Optional**: If compatible, install [flash attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) (`pip install flash-attn --no-build-isolation`)

```bash
cd examples/cuda
pip install -r requirements.txt
weco run --source module.py \
     --eval-command "python evaluate.py --path module.py" \
     --metric speedup \
     --goal maximize \
     --steps 50 \
     --model gpt-5 \
     --additional-instructions "Write in-line CUDA using pytorch's load_inline() to optimize the code while ensuring a small max float diff. Maintain the same code interface. Do not use any fallbacks and never use the build_directory arg for load_inline(). Assume any required dependencies are installed and data is already on the gpu." \
     --eval-timeout 600
```

### 🧠 Prompt Engineering

- **Requirements**: OpenAI API key (create [here](https://platform.openai.com/api-keys))
- **Install Dependencies**: `pip install openai datasets`
- **Run**:
```bash
cd examples/prompt
export OPENAI_API_KEY="your_key_here"
weco run --source optimize.py \
     --eval-command "python eval.py" \
     --metric score \
     --goal maximize \
     --steps 20 \
     --model o4-mini \
     --additional-instructions "Improve the prompt to get better scores. Focus on clarity, specificity, and effective prompt engineering techniques."
```

### 📊 Extract Line Plot — Chart to CSV

- **Requirements**: OpenAI API key (create [here](https://platform.openai.com/api-keys))
- **Install Dependencies**: `pip install uv openai huggingface_hub`
- **Run**:
```bash
cd examples/extract-line-plot
export OPENAI_API_KEY="your_key_here"
uv run --with huggingface_hub python prepare_data.py  # prepare dataset
weco run --source optimize.py \
         --eval-command 'uv run --with openai python eval.py --max-samples 100 --num-workers 50' \
         --metric accuracy \
         --goal maximize \
         --steps 20 \
         --model gpt-5
```

### 🛰️ Model Development — Spaceship Titanic

- **Install Dependencies**: `pip install pandas numpy scikit-learn torch xgboost lightgbm catboost`
- **Run**:
```bash
cd examples/spaceship-titanic
weco run --source train.py \
     --eval-command "python evaluate.py --data-dir ./data --seed 0" \
     --metric accuracy \
     --goal maximize \
     --steps 10 \
     --model o4-mini \
     --additional-instructions "Improve feature engineering, model choice and hyper-parameters." \
     --log-dir .runs/spaceship-titanic
```

### 🕵️ Fraud Detection — IEEE-CIS

Optimize a tabular fraud-detection pipeline on real Vesta payment data.
Reproduces Weco's
[fraud-detection case study](https://weco.ai/blog/framing-the-problem)
(baseline AUC 0.914 → pooled 6-seed mean 0.9305 ± 0.0035 with full
instructions at 200 steps).

- **Prereqs**: Kaggle API token + [join the competition](https://www.kaggle.com/c/ieee-fraud-detection)
- **Install Dependencies**: `pip install -r requirements.txt`
- **Prepare data** (once, ~2-3 min): `python prepare_data.py`
- **Run**:
```bash
cd examples/fraud-detection
weco run --source train.py \
     --eval-command "python evaluate.py" \
     --metric auc_roc \
     --goal maximize \
     --steps 50 \
     --model gemini-3.1-pro-preview \
     --additional-instructions instructions.md \
     --eval-timeout 300 \
     --log-dir .runs/fraud-detection
```

---

If you're new to Weco, start with **Hello World**, then try **LangSmith ZephHR QA** for a realistic LangSmith optimization workflow, explore **Triton** and **CUDA** for kernel engineering, **Prompt Engineering** for optimzing an LLM's prompt, **Extract Line Plot** for optimzing agentic scaffolds, **Spaceship Titanic** for model development, or **Fraud Detection** for a production-scale tabular ML case study.


