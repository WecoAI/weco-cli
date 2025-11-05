# Hello World

This example demonstrates the basics of using Weco to optimize a simple PyTorch model. The model performs a series of basic operations: matrix multiplication, division, summation, and scaling. It's designed as an introductory tutorial to help you understand how Weco works before moving on to more advanced optimization tasks.

## Setup

Install the CLI and dependencies for the example:
```bash
pip install weco
pip install -r requirements.txt
```

## Run Weco

Now run Weco to optimize your code:
```bash
weco run --source module.py \
     --eval-command "python evaluate.py --path module.py --device cpu" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

**Note:** If you have an NVIDIA GPU, change the device in the `--eval-command` to `cuda`. If you are running this on Apple Silicon, set it to `mps`.

### Explanation

*   `--source module.py`: The simple PyTorch model to be optimized.
*   `--eval-command "python evaluate.py --path module.py --device cpu"`: Runs the evaluation script, which benchmarks the optimized code against a baseline and prints the `speedup`.
*   `--metric speedup`: The optimization target metric.
*   `--goal maximize`: To increase the speedup.
*   `--steps 15`: The number of optimization iterations.
*   `--additional-instructions "..."`: Provides specific guidance to focus on operation fusion while maintaining correctness.

Weco will iteratively modify `module.py`, attempting to fuse and optimize the operations in the forward method, guided by the performance feedback from the evaluation script.

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