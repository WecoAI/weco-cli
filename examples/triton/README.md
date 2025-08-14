# Triton Optimization

This example demonstrates using Weco to optimize a causal multi-head self-attention mechanism, 
a core component of Transformer models, implemented in PyTorch. 
The optimization target is to leverage [Triton](https://github.com/triton-lang/triton) 
for writing highly efficient GPU code, to accelerate the operation.

## Setup

Install the CLI using `pip`:
```bash
pip install weco>=0.2.18
```

Create your OpenAI API key [here](https://platform.openai.com/api-keys), then run:
```bash
export OPENAI_API_KEY="your_key_here"
```

Install the dependencies of the scripts shown in subsequent sections:
```bash
pip install torch triton
```
*(Note: Triton installation might require specific CUDA versions. Refer to the official Triton documentation if you encounter issues.)*

## Run Weco

Now run Weco to optimize your code using Triton:
```bash
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py" \
     --metric speedup \
     --goal maximize \
     --steps 30 \
     --model o4-mini \
     --additional-instructions "Use triton to optimize the code while ensuring a small max float diff. Maintain the same code format."
```

### Explanation

*   `--source optimize.py`: Specifies the PyTorch self-attention implementation (`optimize.py`) that Weco will optimize.
*   `--eval-command "python evaluate.py --solution-path optimize.py"`: Defines the command to execute the evaluation script. This script benchmarks the generated solution in `optimize.py` against a baseline and outputs the `speedup`.
*   `--metric speedup`: Sets the metric Weco should focus on improving during optimization.
*   `--goal maximize`: Instructs Weco to aim for the highest possible speedup value.
*   `--steps 30`: Determines the number of optimization iterations Weco will perform.
*   `--model o4-mini`: Specifies the large language model to drive the optimization process.
*   `--additional-instructions "..."`: Provides specific guidance to the LLM. In this case, it directs the model to use Triton for optimization, ensure the numerical difference ("max float diff") between the original and optimized code remains small, and keep the overall code structure consistent.

Weco will iteratively modify `optimize.py`, incorporating Triton kernels, guided by the performance feedback (`speedup`) from the evaluation script and the instructions provided.

## Next Steps

After mastering Triton kernels, explore [CUDA Optimization](/examples/cuda/README.md) for even lower-level GPU programming, or check the [CLI Reference](https://docs.weco.ai/cli/cli-reference) to improve the results you get with Weco.
