# weco-cli/examples/prompt/optimize.py
"""
optimize.py
This module holds the prompt template and the LLM call.
Weco modifies this file to optimize the prompt instructions.
The model used for the LLM call is passed in from eval.py.
"""

from openai import OpenAI

client = OpenAI()  # API key must be in OPENAI_API_KEY
# MODEL constant removed from here

PROMPT_TEMPLATE = """You are an expert competition mathematician tasked with solving an AIME problem.
The final answer must be a three-digit integer between 000 and 999, inclusive.
Please reason step-by-step towards the solution. Keep your reasoning concise.
Conclude your response with the final answer enclosed in \\boxed{{}}. For example: The final answer is \\boxed{{042}}.

Problem:
{problem}

Solution:
"""


def solve(problem: str, model_name: str) -> str:
    """Return the model's raw text answer for one problem using the specified model."""
    prompt = PROMPT_TEMPLATE.format(problem=problem)

    response = client.chat.completions.create(
        model=model_name,  # Use the passed-in model name
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
