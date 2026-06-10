# Weco Prompt Optimization Guidelines for AIME (Targeting GPT-4.1)

## 1. Goal

Your objective is to modify the the `optimize.py` file to improve the `accuracy` metric when solving AIME math problems. The modifications should leverage the capabilities of the target model, **GPT-4.1**.

## 2. Files and Workflow

*   **Target File for Modification:** `optimize.py`. *   **Evaluation Script:** `eval.py`. This script:
    *   Defines the actual LLM used for solving (`MODEL_TO_USE`, which is set to `gpt-4.1` in this context).
    *   Calls `optimize.solve(problem, model_name="gpt-4.1")`.
    *   Parses the output from `optimize.solve`. **Crucially, it expects the final 3-digit answer (000-999) to be enclosed in `\boxed{XXX}`.** For example: `\boxed{042}`. Your prompt modifications *must* ensure the model consistently produces this format for the final answer.
    *   Compares the extracted answer to the ground truth and prints the `accuracy:` metric, which Weco uses for guidance.

## 3. Target Model: GPT-4.1

You are optimizing the prompt for `gpt-4.1`. Based on its characteristics, consider the following:

*   **Strengths:**
    *   **Significantly Improved Instruction Following:** GPT-4.1 is better at adhering to complex instructions, formats, and constraints compared to previous models. This is key for AIME where precision is vital. It excels on hard instruction-following tasks.
    *   **Stronger Coding & Reasoning:** Its improved coding performance (e.g., SWE-bench) suggests enhanced logical reasoning capabilities applicable to mathematical problem-solving.
    *   **Refreshed Knowledge:** Knowledge cutoff is June 2024.
*   **Considerations:**
    *   **Literal Interpretation:** GPT-4.1 can be more literal. Prompts should be explicit and specific about the desired reasoning process and output format. Avoid ambiguity.

## 4. Optimization Strategies (Focus on `PROMPT_TEMPLATE` in `optimize.py`)

The primary goal is to enhance the model's reasoning process for these challenging math problems. Focus on Chain-of-Thought (CoT) designs within the `PROMPT_TEMPLATE`.

**Ideas to Explore:**
You don't have to implement all of them, but the following ideas might be helpful:
*   **Workflow Patterns** try to use some of the following patterns:
    *  **Linear**: Linear workflow, standarded CoT E.g. considering the following thinking steps (you don't have to include all of them), "1. Understand the problem constraints. 2. Identify relevant theorems/formulas. 3. Formulate a plan. 4. Execute calculations step-by-step. 5. Verify intermediate results. 6. State the final answer in the required format."
    *  **List Candidates**: You can ask the model to propose a few solutions in a particular step and pick the best solution. You can potentially also set the criterias in the prompt.
    *  **Code** Use pesudo code to define even more complex workflows with loops, conditional statement, or go to statement.
*   **Other CoT Techniques:**
    *   Self-Correction/Reflection
    *   Plan Generation
    *   Debate, simulating multiple characters
    *   Tree of thought
*   **Few-Shot Examples:** You *could* experiment with adding 1-2 high-quality AIME problem/solution examples directly into the `PROMPT_TEMPLATE` string (similar to how Weco attempted in one of the runs). Ensure the examples clearly show the desired reasoning style and the final `\boxed{XXX}` format.
*   **Play with format:** The way you format the prompt. Markdown, xml, json, code or natural language. Similarly for the thinking tokens themselves you can also try out different formats.

## 5. Constraints
*   **Ensure the final output reliably contains `\boxed{XXX}` as the evaluation script depends on it.**
