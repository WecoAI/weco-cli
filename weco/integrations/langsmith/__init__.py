"""Weco + LangSmith integration.

Bridge module that lets Weco optimize code using LangSmith datasets and evaluators.

Usage:
    python -m weco.integrations.langsmith \\
        --dataset my-dataset \\
        --target agent:run_chain \\
        --evaluators correctness relevance \\
        --metric correctness
"""

from .bridge import import_target, resolve_evaluators, run_langsmith_eval

__all__ = ["import_target", "resolve_evaluators", "run_langsmith_eval"]
