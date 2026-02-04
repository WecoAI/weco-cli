# weco/config.py
"""Configuration and credential management for Weco CLI.

This module handles:
- API key storage and retrieval
- Installation ID management for anonymous event tracking
- Config directory management
"""

import os
import pathlib
import json
import stat
import uuid


CONFIG_DIR = pathlib.Path.home() / ".config" / "weco"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
INSTALLATION_FILE = CONFIG_DIR / "installation.json"


def ensure_config_dir():
    """Ensures the configuration directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure directory permissions are secure (optional but good practice)
    try:
        os.chmod(CONFIG_DIR, stat.S_IRWXU)  # Read/Write/Execute for owner only
    except OSError as e:
        print(f"Warning: Could not set permissions on {CONFIG_DIR}: {e}")


def save_api_key(api_key: str):
    """Saves the Weco API key securely."""
    ensure_config_dir()
    credentials = {"api_key": api_key}
    try:
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump(credentials, f)
        # Set file permissions to read/write for owner only (600)
        os.chmod(CREDENTIALS_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as e:
        print(f"Error: Unable to save credentials file or set permissions on {CREDENTIALS_FILE}: {e}")


def load_weco_api_key() -> str | None:
    """Loads the Weco API key."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        # Check permissions before reading (optional but safer)
        file_stat = os.stat(CREDENTIALS_FILE)
        if file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):  # Check if group/other have permissions
            print(f"Warning: Credentials file {CREDENTIALS_FILE} has insecure permissions. Please set to 600.")
            # Optionally, refuse to load or try to fix permissions

        with open(CREDENTIALS_FILE, "r") as f:
            credentials = json.load(f)
            return credentials.get("api_key")
    except (IOError, json.JSONDecodeError, OSError) as e:
        print(f"Warning: Unable to read credentials file at {CREDENTIALS_FILE}: {e}")
        return None


def clear_api_key():
    """Removes the stored API key."""
    if CREDENTIALS_FILE.exists():
        try:
            os.remove(CREDENTIALS_FILE)
            print("Logged out successfully.")
        except OSError as e:
            print(f"Error: Unable to remove credentials file at {CREDENTIALS_FILE}: {e}")
    else:
        print("Already logged out.")


def get_or_create_installation_id() -> str:
    """Gets or creates a persistent installation ID for anonymous event reporting.

    The installation ID is stored in ~/.config/weco/installation.json and
    persists across CLI invocations. It is used to link anonymous events
    to the user once they authenticate.

    Returns:
        The installation ID (a UUID string prefixed with 'inst_').
    """
    ensure_config_dir()

    # Try to load existing installation ID
    try:
        with open(INSTALLATION_FILE, "r") as f:
            data = json.load(f)
            installation_id = data.get("installation_id")
            if installation_id:
                return installation_id
    except (IOError, json.JSONDecodeError, FileNotFoundError):
        pass  # Will create a new one

    # Generate a new installation ID
    installation_id = f"inst_{uuid.uuid4().hex}"

    # Save it
    try:
        with open(INSTALLATION_FILE, "w") as f:
            json.dump({"installation_id": installation_id}, f)
        # Set secure permissions
        os.chmod(INSTALLATION_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # Non-fatal - we can still use the ID for this session

    return installation_id


def load_installation_id() -> str | None:
    """Loads the installation ID if it exists.

    Returns:
        The installation ID or None if not found.
    """
    try:
        with open(INSTALLATION_FILE, "r") as f:
            data = json.load(f)
            return data.get("installation_id")
    except (IOError, json.JSONDecodeError, FileNotFoundError):
        return None
