# CUDA Optimization

This example showcases using Weco to optimize a PyTorch causal multi-head self-attention implementation by generating custom [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) kernels. 
This approach aims for low-level optimization beyond standard PyTorch or even Triton for potentially higher performance on NVIDIA GPUs.
This example uses a separate Markdown file (`guide.md`) to provide detailed instructions and context to the LLM.

## Setup

Install the CLI using `pip`:
```bash
pip install weco>=0.2.18
```

Create your OpenAI API key [here](https://platform.openai.com/api-keys), then run:
```bash
export OPENAI_API_KEY="your_key_here"
```

Install the required dependencies:
```bash
pip install torch ninja
```
> **Note:** This example requires a compatible NVIDIA GPU and the CUDA Toolkit installed on your system for compiling and running the generated CUDA code.

## Run Weco

Now run Weco to optimize your code:
```bash
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py" \
     --metric speedup \
     --goal maximize \
     --steps 15 \
     --model o4-mini \
     --additional-instructions guide.md
```

### Explanation

*   `--source optimize.py`: The initial PyTorch self-attention code to be optimized with CUDA.
*   `--eval-command "python evaluate.py --solution-path optimize.py"`: Runs the evaluation script, which compiles (if necessary) and benchmarks the CUDA-enhanced code in `optimize.py` against a baseline, printing the `speedup`.
*   `--metric speedup`: The optimization target metric.
*   `--goal maximize`: Weco aims to increase the speedup.
*   `--steps 15`: The number of optimization iterations.
*   `--model o4-mini`: The LLM used for code generation.
*   `--additional-instructions guide.md`: Provides guidance to the LLM on the optimization approach.

Weco will iteratively modify `optimize.py`, generating and integrating CUDA C++ code, guided by the evaluation results and the instructions in `guide.md`.

## Next Steps

Now that you've optimized your code with CUDA kernels, try [Triton Optimization](/examples/triton/README.md) for a higher-level GPU programming approach. If you're more interested in [Model Development](/examples/spaceship-titanic/README.md) or [Prompt Engineering](/examples/prompt/README.md), we've got you covered! 

You can check out our [CLI Reference](https://docs.weco.ai/cli/cli-reference) to learn more about what you can do with the tool.
