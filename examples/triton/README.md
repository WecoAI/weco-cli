# Triton Optimization

This example demonstrates using Weco to optimize a simple activation function implemented in PyTorch. In this example, we'll ask Weco to leverage [Triton](https://github.com/triton-lang/triton) to accelerate our code.

## Setup

Install the CLI and dependencies for the example:
```bash
pip install weco
pip install -r requirements.txt
```
> **Note:** This example requires an NVIDIA GPU.

## Run Weco

Now run Weco to optimize your code using Triton:
```bash
weco run --source module.py \
     --eval-command "python evaluate.py --path module.py" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --model o4-mini \
     --additional-instructions "Use a combination of triton and pytorch to optimize the forward pass while ensuring a small max float diff. Maintain the same code interface. Do not use any fallbacks. Assume any required dependencies are installed and data is already on the gpu." \
     --eval-timeout 120
```

### Explanation

*   `--source module.py`: Specifies the PyTorch Swish activation implementation (`module.py`) that Weco will optimize.
*   `--eval-command "python evaluate.py --path module.py"`: Defines the command to execute the evaluation script. This script benchmarks the generated solution in `module.py` against a baseline and outputs the `speedup`.
*   `--metric speedup`: Sets the metric Weco should focus on improving during optimization.
*   `--goal maximize`: Instructs Weco to aim for the highest possible speedup value.
*   `--steps 15`: Determines the number of optimization iterations Weco will perform.
*   `--model o4-mini`: Specifies the large language model to drive the optimization process.
*   `--additional-instructions "..."`: Provides specific guidance to the LLM.

Weco will iteratively modify `module.py`, incorporating Triton kernels, guided by the performance feedback (`speedup`) from the evaluation script and the instructions provided.

## Next Steps

After mastering Triton kernels, explore [CUDA Optimization](/examples/cuda/README.md) for even lower-level GPU programming, or check the [CLI Reference](https://docs.weco.ai/cli/cli-reference) to improve the results you get with Weco.
