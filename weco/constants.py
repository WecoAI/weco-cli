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
DEFAULT_MODELS = [("gemini", "gemini-3-pro-preview"), ("openai", "o4-mini"), ("anthropic", "claude-opus-4-5")]
