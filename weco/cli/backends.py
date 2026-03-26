import argparse
import importlib
from types import ModuleType
from typing import Protocol, runtime_checkable


# Eval backend registry. Each entry maps a backend name to its module path.
# To add a new backend, create weco/integrations/<name>/backend.py with
# register_args(), validate_args(), and build_eval_command(), then add it here.
EVAL_BACKENDS = {"langsmith": "weco.integrations.langsmith.backend", "langfuse": "weco.integrations.langfuse.backend"}


@runtime_checkable
class EvalBackendProtocol(Protocol):
    def register_args(self, parser: argparse.ArgumentParser) -> None: ...
    def validate_args(self, args: argparse.Namespace) -> None: ...
    def build_eval_command(self, args: argparse.Namespace) -> str: ...


def _validate_backend_contract(name: str, backend: ModuleType) -> None:
    required = ("register_args", "validate_args", "build_eval_command")
    missing = [attr for attr in required if not callable(getattr(backend, attr, None))]
    if missing:
        joined = ", ".join(missing)
        raise TypeError(f"Eval backend '{name}' is missing required callable(s): {joined}")


def load_backend(name: str) -> ModuleType:
    """Lazily import an eval backend module by name."""
    backend = importlib.import_module(EVAL_BACKENDS[name])
    _validate_backend_contract(name, backend)
    return backend


def register_backend_args(parser: argparse.ArgumentParser) -> None:
    """Register all backend-specific CLI flags on a parser."""
    for backend_name in EVAL_BACKENDS:
        load_backend(backend_name).register_args(parser)
