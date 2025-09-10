# weco/constants.py
"""
Constants for the Weco CLI package.
"""

# API timeout configuration (connect_timeout, read_timeout) in seconds
DEFAULT_API_TIMEOUT = (10, 3650)

# Output truncation configuration
TRUNCATION_THRESHOLD = 51000  # Maximum length before truncation
TRUNCATION_KEEP_LENGTH = 25000  # Characters to keep from beginning and end
