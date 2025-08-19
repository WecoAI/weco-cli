# Hello Kernel World

This example demonstrates the basics of using Weco to optimize a simple PyTorch model. The model performs a series of basic operations: matrix multiplication, division, summation, and scaling. It's designed as an introductory tutorial to help you understand how Weco works before moving on to more advanced optimization tasks.

## Setup

Install the CLI using `pip`:
```bash
pip install weco>=0.2.18
```

Create your API key from one of the supported providers:
- **OpenAI:** Create your API key [here](https://platform.openai.com/api-keys), then run: `export OPENAI_API_KEY="your_key_here"`
- **Anthropic:** Create your API key [here](https://console.anthropic.com/settings/keys), then run: `export ANTHROPIC_API_KEY="your_key_here"`  
- **Google:** Create your API key [here](https://aistudio.google.com/apikey), then run: `export GEMINI_API_KEY="your_key_here"`

Install the required dependencies:
```bash
pip install torch
```

## Run Weco

Now run Weco to optimize your code:
```bash
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py --device cpu" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

**Note:** If you have an NVIDIA GPU, change the device in the `--eval-command` to `cuda`. If you are running this on Apple Silicon, set it to `mps`.

### Explanation

*   `--source optimize.py`: The simple PyTorch model to be optimized.
*   `--eval-command "python evaluate.py --solution-path optimize.py --device cpu"`: Runs the evaluation script, which benchmarks the optimized code against a baseline and prints the `speedup`.
*   `--metric speedup`: The optimization target metric.
*   `--goal maximize`: To increase the speedup.
*   `--steps 15`: The number of optimization iterations.
*   `--additional-instructions "..."`: Provides specific guidance to focus on operation fusion while maintaining correctness.

Weco will iteratively modify `optimize.py`, attempting to fuse and optimize the operations in the forward method, guided by the performance feedback from the evaluation script.

## Interactive Tutorial
****
For a hands-on walkthrough of this example, check out the [Colab notebook](colab_notebook_walkthrough.ipynb) that provides step-by-step guidance through the optimization process.

## Next Steps

Once you've mastered the basics with this example, explore more advanced optimization techniques:
- [Triton Optimization](/examples/triton/README.md) for GPU kernel programming
- [CUDA Optimization](/examples/cuda/README.md) for low-level GPU optimization
- [Model Development](/examples/spaceship-titanic/README.md) for ML model optimization
- [Prompt Engineering](/examples/prompt/README.md) for LLM prompt optimization

You can also check out our [CLI Reference](https://docs.weco.ai/cli/cli-reference) to learn more about what you can do with the tool.