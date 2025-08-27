# weco/constants.py
"""
Constants for the Weco CLI package.
"""

# API timeout configuration (connect_timeout, read_timeout) in seconds
DEFAULT_API_TIMEOUT = (10, 800)

# Output truncation configuration
DEFAULT_MAX_LINES = 100
DEFAULT_MAX_CHARS = 10000
MAX_PRESERVED_METRIC_LINES = 20  # cap preserved metric lines
