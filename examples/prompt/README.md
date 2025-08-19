# Prompt Engineering

This example shows how **Weco** can iteratively improve a prompt for solving American Invitational Mathematics Examination (AIME) problems. 
The experiment runs locally, requires only two short Python files and a prompt guide, and aims to improve the accuracy metric.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/WecoAI/weco-cli.git
   cd examples/prompt
   ```

2. Install the CLI and dependencies for the example:
   ```bash
   pip install weco>=0.2.18 openai datasets
   ```

3. This example uses `o4-mini` via the OpenAI API by default. Create your OpenAI API key [here](https://platform.openai.com/api-keys), then run:
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```


## Files in this folder

| File          | Purpose                                                                                                                                                           |
| :------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `optimize.py` | Holds the prompt template (instructing the LLM to reason step-by-step and use `\\boxed{}` for the final answer) and the mutable `EXTRA_INSTRUCTIONS` string. Weco edits **only** this file during the search. |
| `eval.py`     | Downloads a small slice of the 2024 AIME dataset, calls `optimize.solve` in parallel, parses the LLM output (looking for `\\boxed{}`), compares it to the ground truth, prints progress logs, and finally prints an `accuracy:` line that Weco reads. |


Now run Weco to optimize your prompt:
```bash
weco run --source optimize.py \
     --eval-command "python eval.py" \
     --metric score \
     --goal maximize \
     --steps 15 \
     --model o4-mini \
     --additional-instructions "Improve the prompt to get better scores. Focus on clarity, specificity, and effective prompt engineering techniques."
```

*Note: You can replace `--model o4-mini` with another powerful model like `o3` or others, provided you have the respective API keys set.*

During each evaluation round, you will see log lines similar to the following:

```text
[setup] loading 20 problems from AIME 2024 …
[progress] 5/20 completed, accuracy: 0.0000, elapsed 7.3 s
[progress] 10/20 completed, accuracy: 0.1000, elapsed 14.6 s
[progress] 15/20 completed, accuracy: 0.0667, elapsed 21.8 s
[progress] 20/20 completed, accuracy: 0.0500, elapsed 28.9 s
accuracy: 0.0500
```

Weco then mutates the prompt instructions in `optimize.py`, tries again, and gradually pushes the accuracy higher.

## How it works

*   `eval.py` slices the **Maxwell-Jia/AIME_2024** dataset to twenty problems for fast feedback. You can change the slice in one line within the script.
*   The script sends model calls in parallel via `ThreadPoolExecutor`, so network latency is hidden.
*   Every five completed items, the script logs progress and elapsed time.
*   The final line `accuracy: value` is the only part Weco needs for guidance.

## Next Steps

Now that you've automated prompt engineering for yourself, check out our guide on [Model Development](/examples/spaceship-titanic/README.md) or [CUDA Kernel Engineering](/examples/cuda/README.md).

You can check out our [CLI Reference](https://docs.weco.ai/cli/cli-reference) to learn more about what you can do with the tool.

