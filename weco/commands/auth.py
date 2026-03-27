"""``weco login`` and ``weco logout``"""

import sys

from rich.console import Console


def handle_login(console: Console) -> None:
    from ..core.auth import perform_login
    from ..core.config import load_weco_api_key

    if load_weco_api_key():
        console.print("[bold green]You are already logged in.[/]")
        console.print("[dim]Use 'weco logout' to log out first if you want to switch accounts.[/]")
        sys.exit(0)

    sys.exit(0 if perform_login(console) else 1)


def handle_logout() -> None:
    from ..core.config import clear_api_key

    clear_api_key()
