# weco/constants.py
"""
Constants for the Weco CLI package.
"""

# API timeout configuration (connect_timeout, read_timeout) in seconds
DEFAULT_API_TIMEOUT = (10, 800)

# Output truncation configuration
TRUNCATION_THRESHOLD = 51000  # Maximum length before truncation
TRUNCATION_KEEP_LENGTH = 25000  # Characters to keep from beginning and end

# Optimization process timeouts and delays
HEARTBEAT_JOIN_TIMEOUT = 2  # Seconds to wait for heartbeat thread to stop
HEARTBEAT_STARTUP_DELAY = 2  # Seconds to wait before starting heartbeat (for DB sync)
SIGNAL_HANDLER_TIMEOUT = 3  # Seconds for report_termination in signal handlers

# Heartbeat tuning (keep server-aware liveness under 90s backend cutoff)
HEARTBEAT_INTERVAL = 25  # Seconds between heartbeats
HEARTBEAT_REQUEST_TIMEOUT = (5, 5)  # (connect, read) timeouts in seconds
HEARTBEAT_RETRY_DELAY = 3  # Seconds to wait before a quick retry on failure
HEARTBEAT_RETRY_ATTEMPTS = 1  # Number of quick retries before waiting full interval
