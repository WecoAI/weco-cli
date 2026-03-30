"""Custom exceptions shared across core modules."""


class UnrecognizedAPIKeysError(Exception):
    """Raised when unrecognized API key providers are provided."""

    def __init__(self, api_keys: dict[str, str], supported_providers: set[str]):
        self.api_keys = api_keys
        providers = ", ".join(sorted(supported_providers))
        super().__init__(f"Unrecognized API key provider in {set(api_keys.keys())}. Supported providers: {providers}")


class DefaultModelNotFoundError(Exception):
    """Raised when no default model is found for the provided API keys."""

    def __init__(self, api_keys: dict[str, str]):
        self.api_keys = api_keys
        super().__init__(f"No default model found for any of the provided API keys: {set(api_keys.keys())}")
