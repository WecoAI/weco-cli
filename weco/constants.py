# weco/constants.py
"""
Constants for the Weco CLI package.
"""

# Output truncation configuration
TRUNCATION_THRESHOLD = 51000  # Maximum length before truncation
TRUNCATION_KEEP_LENGTH = 25000  # Characters to keep from beginning and end

# Default model configuration
DEFAULT_MODEL = "o4-mini"

# Supported file extensions for additional instructions
SUPPORTED_FILE_EXTENSIONS = [".md", ".txt", ".rst"]
