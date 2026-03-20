"""Weco + LangFuse integration.

Bridge module that lets Weco optimize code using LangFuse datasets and evaluators.

Usage:
    python -m weco.integrations.langfuse \\
        --dataset my-dataset \\
        --target agent:run_chain \\
        --evaluators correctness relevance \\
        --metric correctness
"""

from .bridge import import_target, resolve_evaluators, run_langfuse_eval

__all__ = ["import_target", "resolve_evaluators", "run_langfuse_eval"]
