## Weco Examples

Explore runnable examples that show how to use Weco to optimize kernels, prompts, and ML pipelines. Pick an example and get going in minutes.

### Table of Contents

- [Weco Examples](#weco-examples)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Examples at a glance](#examples-at-a-glance)
- [Quick starts](#quick-starts)
  - [🧭 Hello Kernel World](#-hello-kernel-world)
  - [⚡ Triton Optimization](#-triton-optimization)
  - [🚀 CUDA Optimization](#-cuda-optimization)
  - [🧠 Prompt Engineering](#-prompt-engineering)
  - [📊 Extract Line Plot — Chart to CSV](#-extract-line-plot--chart-to-csv)
  - [🛰️ Model Development — Spaceship Titanic](#️-model-development--spaceship-titanic)

### Prerequisites

- **Install the CLI**
```bash
pip install weco
```

### Examples at a glance

| Example | Focus | Extras | Docs |
| :-- | :-- | :-- | :-- |
| 🧭 Hello Kernel World | Learn the Weco workflow on a small PyTorch model | `torch` | [README](hello-kernel-world/README.md) • [Colab](hello-kernel-world/colab_notebook_walkthrough.ipynb) |
| ⚡ Triton Optimization | Speed up attention with Triton kernels | `torch`, `triton` | [README](triton/README.md) |
| 🚀 CUDA Optimization | Generate low-level CUDA kernels for max speed | `torch`, `ninja`, `triton`, NVIDIA GPU + CUDA Toolkit | [README](cuda/README.md) |
| 🧠 Prompt Engineering | Iteratively refine LLM prompts to improve accuracy | `openai`, `datasets` | [README](prompt/README.md) |
| 📊 Agentic Scaffolding | Optimize agentic scaffolding for chart-to-CSV extraction | `openai`, `huggingface_hub`, `uv` | [README](extract-line-plot/README.md) |
| 🛰️ Spaceship Titanic | Improve a Kaggle model training pipeline | `pandas`, `numpy`, `scikit-learn`, `torch`, `xgboost`, `lightgbm`, `catboost` | [README](spaceship-titanic/README.md) |

---

## Quick starts

Minimal commands to run each example. For full context and explanations, see the linked READMEs.

### 🧭 Hello Kernel World

- **Install extra deps**: `pip install torch`
- **Run**:
```bash
cd examples/hello-kernel-world
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py --device cpu" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```
- **Tip**: Use `--device cuda` (NVIDIA GPU) or `--device mps` (Apple Silicon).

### ⚡ Triton Optimization

- **Install extra deps**: `pip install numpy torch triton`
- **Requires**: NVIDIA GPU
- **Run**:
```bash
cd examples/triton
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --model o4-mini \
     --additional-instructions "Use a combination of triton and pytorch to optimize the forward pass while ensuring a small max float diff. Maintain the same code interface. Do not use any fallbacks. Assume any required dependencies are installed and data is already on the gpu." \
     --eval-timeout 120
```

### 🚀 CUDA Optimization

- **Install extra deps**: `pip install ninja numpy torch triton`
- **Requires**: NVIDIA GPU and CUDA Toolkit
- **Optional**: If compatible, install [flash attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) (`pip install flash-attn --no-build-isolation`).
- **Run**:
```bash
cd examples/cuda
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py" \
     --metric speedup \
     --goal maximize \
     --steps 50 \
     --model gpt-5 \
     --additional-instructions "Write in-line CUDA using pytorch's load_inline() to optimize the code while ensuring a small max float diff. Maintain the same code interface. Do not use any fallbacks and never use the build_directory arg for load_inline(). Assume any required dependencies are installed and data is already on the gpu." \
     --eval-timeout 600
```

### 🧠 Prompt Engineering

- **Install extra deps**: `pip install openai datasets`
- **Configure environment**: Create your OpenAI API key [here](https://platform.openai.com/api-keys) and run `export OPENAI_API_KEY="your_key_here"`.
- **Run**:
```bash
cd examples/prompt
weco run --source optimize.py \
     --eval-command "python eval.py" \
     --metric score \
     --goal maximize \
     --steps 20 \
     --model o4-mini \
     --additional-instructions "Improve the prompt to get better scores. Focus on clarity, specificity, and effective prompt engineering techniques."
```

### 📊 Extract Line Plot — Chart to CSV

- **Install extra deps**:
  - Install `uv` (see `https://docs.astral.sh/uv/`)
  - `pip install openai huggingface_hub`
- **Configure environment**: Create your OpenAI API key [here](https://platform.openai.com/api-keys) and run `export OPENAI_API_KEY="your_key_here"`.
- **Setup**: Prepare the dataset first:
```bash
cd examples/extract-line-plot
uv run --with huggingface_hub python prepare_data.py
```
- **Run**:
```bash
weco run --source optimize.py --eval-command 'uv run --with openai python eval.py --max-samples 100 --num-workers 50' --metric accuracy --goal maximize --steps 20 --model gpt-5
```

### 🛰️ Model Development — Spaceship Titanic

- **Install extra deps**: `pip install pandas numpy scikit-learn torch xgboost lightgbm catboost`
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

---

If you're new to Weco, start with **Hello Kernel World**, then explore **Triton** and **CUDA** for kernel engineering, **Prompt Engineering** for optimzing an LLM's prompt, **Extract Line Plot** for optimzing agentic scaffolds, or try **Spaceship Titanic** for model development.


