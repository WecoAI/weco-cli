# weco/constants.py
"""
Constants for the Weco CLI package.
"""

# Output truncation configuration
TRUNCATION_THRESHOLD = 51000  # Maximum length before truncation
TRUNCATION_KEEP_LENGTH = 25000  # Characters to keep from beginning and end

# Supported file extensions for additional instructions
SUPPORTED_FILE_EXTENSIONS = [".md", ".txt", ".rst"]

# Default models for each provider in order of preference
DEFAULT_MODELS = [("gemini", "gemini-3.1-pro-preview"), ("openai", "o4-mini"), ("anthropic", "claude-opus-4-5")]

_SUPPORTED_PROVIDERS = {provider for provider, _ in DEFAULT_MODELS}


class UnrecognizedAPIKeysError(Exception):
    """Exception raised when unrecognized API keys are provided."""

    def __init__(self, api_keys: dict[str, str]):
        self.api_keys = api_keys
        super().__init__(
            f"Unrecognized API key provider in {set(api_keys.keys())}. Supported providers: {', '.join(_SUPPORTED_PROVIDERS)}"
        )


class DefaultModelNotFoundError(Exception):
    """Exception raised when no default model is found for the API keys."""

    def __init__(self, api_keys: dict[str, str]):
        self.api_keys = api_keys
        super().__init__(f"No default model found for any of the provided API keys: {set(api_keys.keys())}")


def get_default_model(api_keys: dict[str, str] | None = None) -> str:
    """Determine the default model to use based on the API keys."""
    if api_keys and not all(provider in _SUPPORTED_PROVIDERS for provider in api_keys):
        raise UnrecognizedAPIKeysError(api_keys)

    if api_keys:
        for provider, model in DEFAULT_MODELS:
            if provider in api_keys:
                return model
        raise DefaultModelNotFoundError(api_keys)
    return DEFAULT_MODELS[0][1]
