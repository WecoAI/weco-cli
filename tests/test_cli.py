"""Tests for CLI functions, particularly parse_api_keys."""

import pytest
from weco.cli import parse_api_keys


class TestParseApiKeys:
    """Test cases for parse_api_keys function."""

    def test_parse_api_keys_none(self):
        """Test that None input returns empty dict."""
        result = parse_api_keys(None)
        assert result == {}
        assert isinstance(result, dict)

    def test_parse_api_keys_empty_list(self):
        """Test that empty list returns empty dict."""
        result = parse_api_keys([])
        assert result == {}
        assert isinstance(result, dict)

    def test_parse_api_keys_single_key(self):
        """Test parsing a single API key."""
        result = parse_api_keys(["openai=sk-xxx"])
        assert result == {"openai": "sk-xxx"}

    def test_parse_api_keys_multiple_keys(self):
        """Test parsing multiple API keys."""
        result = parse_api_keys(["openai=sk-xxx", "anthropic=sk-ant-yyy"])
        assert result == {"openai": "sk-xxx", "anthropic": "sk-ant-yyy"}

    def test_parse_api_keys_whitespace_handling(self):
        """Test that whitespace is stripped from provider and key."""
        result = parse_api_keys([" openai = sk-xxx ", "  anthropic  =  sk-ant-yyy  "])
        assert result == {"openai": "sk-xxx", "anthropic": "sk-ant-yyy"}

    def test_parse_api_keys_key_contains_equals(self):
        """Test that keys containing '=' are handled correctly (split on first '=' only)."""
        result = parse_api_keys(["openai=sk-xxx=extra=more"])
        assert result == {"openai": "sk-xxx=extra=more"}

    def test_parse_api_keys_no_equals(self):
        """Test that missing '=' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid API key format.*Expected format: 'provider=key'"):
            parse_api_keys(["openai"])

    def test_parse_api_keys_empty_provider(self):
        """Test that empty provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider and key must be non-empty"):
            parse_api_keys(["=sk-xxx"])

    def test_parse_api_keys_empty_key(self):
        """Test that empty key raises ValueError."""
        with pytest.raises(ValueError, match="Provider and key must be non-empty"):
            parse_api_keys(["openai="])

    def test_parse_api_keys_both_empty(self):
        """Test that both empty provider and key raises ValueError."""
        with pytest.raises(ValueError, match="Provider and key must be non-empty"):
            parse_api_keys(["="])

    def test_parse_api_keys_duplicate_provider(self):
        """Test that duplicate providers overwrite previous value."""
        result = parse_api_keys(["openai=sk-xxx", "openai=sk-yyy"])
        assert result == {"openai": "sk-yyy"}

    def test_parse_api_keys_mixed_case_provider(self):
        """Test that mixed case providers are normalized correctly."""
        result = parse_api_keys(["OpenAI=sk-xxx", "ANTHROPIC=sk-ant-yyy"])
        assert result == {"openai": "sk-xxx", "anthropic": "sk-ant-yyy"}
