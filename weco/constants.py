# weco/constants.py
"""
Constants for the Weco CLI package.
"""

# API timeout configuration (connect_timeout, read_timeout) in seconds
CODEGEN_API_TIMEOUT = (30, 3650)
STATUS_API_TIMEOUT = (10, 180)

# Output truncation configuration
TRUNCATION_THRESHOLD = 51000  # Maximum length before truncation
TRUNCATION_KEEP_LENGTH = 25000  # Characters to keep from beginning and end

# Default model configuration
DEFAULT_MODEL = "o4-mini"

# Supported file extensions for additional instructions
SUPPORTED_FILE_EXTENSIONS = [".md", ".txt", ".rst"]
