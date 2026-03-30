"""Model selection logic for provider-based defaults."""

from .constants import DEFAULT_MODELS
from .errors import DefaultModelNotFoundError, UnrecognizedAPIKeysError


def get_default_model(api_keys: dict[str, str] | None = None) -> str:
    """Determine the default model to use based on provided API keys."""
    supported_providers = {provider for provider, _ in DEFAULT_MODELS}

    if api_keys and not all(provider in supported_providers for provider in api_keys):
        raise UnrecognizedAPIKeysError(api_keys, supported_providers=supported_providers)

    if api_keys:
        for provider, model in DEFAULT_MODELS:
            if provider in api_keys:
                return model
        raise DefaultModelNotFoundError(api_keys)

    return DEFAULT_MODELS[0][1]
