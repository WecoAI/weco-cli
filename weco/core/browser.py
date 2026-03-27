"""Cross-platform browser utilities."""

import webbrowser


def open_browser(url: str) -> bool:
    """
    Open a URL in the user's default web browser.

    This function is cross-platform compatible (Windows, macOS, Linux).
    It uses Python's built-in webbrowser module which handles platform
    detection automatically.

    Args:
        url: The URL to open in the browser.

    Returns:
        True if the browser was opened successfully, False otherwise.
    """
    try:
        # webbrowser.open() is cross-platform and uses the default browser
        # - On macOS: uses 'open' command
        # - On Windows: uses 'start' command
        # - On Linux: tries common browsers (xdg-open, gnome-open, etc.)
        return webbrowser.open(url)
    except Exception:
        # Silently fail - browser opening is a convenience feature
        # and should not interrupt the optimization flow
        return False
