# CUDA Optimization

This example showcases using Weco to optimize a PyTorch causal multi-head self-attention implementation by generating custom [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) kernels. 
This approach aims for low-level optimization beyond standard PyTorch or even Triton for potentially higher performance on NVIDIA GPUs.

## Setup

Install the CLI and dependencies for the example:
```bash
pip install weco
pip install -r requirements.txt
```
> **Note:**
> 1. This example requires a compatible NVIDIA GPU and the CUDA Toolkit installed on your system for compiling and running the generated CUDA code.
> 2. If compatible, install [flash attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) (`pip install flash-attn --no-build-isolation`).

## Run Weco

Now run Weco to optimize your code:
```bash
weco run --source module.py \
     --eval-command "python evaluate.py --path module.py" \
     --metric speedup \
     --goal maximize \
     --steps 50 \
     --model gpt-5 \
     --additional-instructions "Write in-line CUDA using pytorch's load_inline() to optimize the code while ensuring a small max float diff. Maintain the same code interface. Do not use any fallbacks and never use the build_directory arg for load_inline(). Assume any required dependencies are installed and data is already on the gpu." \
     --eval-timeout 600
```

### Explanation

*   `--source module.py`: The initial PyTorch self-attention code to be optimized with CUDA.
*   `--eval-command "python evaluate.py --path module.py"`: Runs the evaluation script, which compiles (if necessary) and benchmarks the CUDA-enhanced code in `module.py` against a baseline, printing the `speedup`.
*   `--metric speedup`: The optimization target metric.
*   `--goal maximize`: Weco aims to increase the speedup.
*   `--steps 50`: The number of optimization iterations.
*   `--model gpt-5`: The LLM used for code generation.
*   `--additional-instructions "..."`: Provides guidance to the LLM on the optimization approach.
*   `--eval-timeout 600`: Stop runnning the evaluation script if it does not complete in 600 seconds.

Weco will iteratively modify `module.py`, generating and integrating CUDA C++ code, guided by the evaluation results and the additional instructions provided.

## Next Steps

Now that you've optimized your code with CUDA kernels, try [Triton Optimization](/examples/triton/README.md) for a higher-level GPU programming approach. If you're more interested in [Model Development](/examples/spaceship-titanic/README.md) or [Prompt Engineering](/examples/prompt/README.md), we've got you covered! 

You can check out our [CLI Reference](https://docs.weco.ai/cli/cli-reference) to learn more about what you can do with the tool.
