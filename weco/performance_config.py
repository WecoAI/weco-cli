"""
Performance configuration settings for Weco CLI.

This module centralizes performance-related settings to allow for easy tuning
and optimization based on system capabilities and user preferences.
"""

import os
from typing import Dict, Any


class PerformanceConfig:
    """Configuration class for performance settings."""
    
    # Network settings
    REQUEST_TIMEOUT = int(os.getenv("WECO_REQUEST_TIMEOUT", "800"))
    CONNECTION_POOL_SIZE = int(os.getenv("WECO_POOL_SIZE", "10"))
    CONNECTION_POOL_MAXSIZE = int(os.getenv("WECO_POOL_MAXSIZE", "20"))
    RETRY_BACKOFF_JITTER = float(os.getenv("WECO_RETRY_JITTER", "0.1"))
    
    # UI refresh settings
    UI_REFRESH_RATE = int(os.getenv("WECO_UI_REFRESH_RATE", "2"))
    UI_TRANSITION_DELAY = float(os.getenv("WECO_UI_DELAY", "0.02"))
    UI_EVAL_DELAY = float(os.getenv("WECO_EVAL_DELAY", "0.01"))
    
    # File processing settings
    MAX_FILE_SIZE = int(os.getenv("WECO_MAX_FILE_SIZE", str(1024 * 1024)))  # 1MB
    LARGE_CODEBASE_THRESHOLD = int(os.getenv("WECO_LARGE_THRESHOLD", "500000"))  # 500KB
    
    # Subprocess settings
    EVAL_COMMAND_TIMEOUT = int(os.getenv("WECO_EVAL_TIMEOUT", "300"))  # 5 minutes
    
    # Caching settings
    CACHE_SIZE_API_KEYS = int(os.getenv("WECO_CACHE_API_KEYS", "1"))
    CACHE_SIZE_FILE_HASH = int(os.getenv("WECO_CACHE_FILE_HASH", "128"))
    CACHE_SIZE_FILE_CONTENT = int(os.getenv("WECO_CACHE_FILE_CONTENT", "64"))
    
    # Heartbeat settings
    HEARTBEAT_INTERVAL = int(os.getenv("WECO_HEARTBEAT_INTERVAL", "30"))
    HEARTBEAT_TIMEOUT = int(os.getenv("WECO_HEARTBEAT_TIMEOUT", "10"))
    
    @classmethod
    def get_session_config(cls) -> Dict[str, Any]:
        """Get HTTP session configuration."""
        return {
            "pool_connections": cls.CONNECTION_POOL_SIZE,
            "pool_maxsize": cls.CONNECTION_POOL_MAXSIZE,
            "timeout": cls.REQUEST_TIMEOUT,
            "backoff_jitter": cls.RETRY_BACKOFF_JITTER,
        }
    
    @classmethod
    def get_ui_config(cls) -> Dict[str, Any]:
        """Get UI performance configuration."""
        return {
            "refresh_rate": cls.UI_REFRESH_RATE,
            "transition_delay": cls.UI_TRANSITION_DELAY,
            "eval_delay": cls.UI_EVAL_DELAY,
        }
    
    @classmethod
    def get_file_config(cls) -> Dict[str, Any]:
        """Get file processing configuration."""
        return {
            "max_file_size": cls.MAX_FILE_SIZE,
            "large_threshold": cls.LARGE_CODEBASE_THRESHOLD,
        }
    
    @classmethod
    def get_cache_config(cls) -> Dict[str, Any]:
        """Get caching configuration."""
        return {
            "api_keys": cls.CACHE_SIZE_API_KEYS,
            "file_hash": cls.CACHE_SIZE_FILE_HASH,
            "file_content": cls.CACHE_SIZE_FILE_CONTENT,
        }
    
    @classmethod
    def print_current_config(cls, console=None):
        """Print current performance configuration."""
        if console is None:
            from rich.console import Console
            console = Console()
        
        console.print("[bold cyan]Current Performance Configuration:[/]")
        console.print(f"  Request timeout: {cls.REQUEST_TIMEOUT}s")
        console.print(f"  Connection pool size: {cls.CONNECTION_POOL_SIZE}")
        console.print(f"  UI refresh rate: {cls.UI_REFRESH_RATE} Hz")
        console.print(f"  Evaluation timeout: {cls.EVAL_COMMAND_TIMEOUT}s")
        console.print(f"  Max file size: {cls.MAX_FILE_SIZE // 1024}KB")
        console.print(f"  Heartbeat interval: {cls.HEARTBEAT_INTERVAL}s")


# Global performance config instance
perf_config = PerformanceConfig()