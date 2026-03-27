"""Share command handler for the Weco CLI."""

import sys
from rich.console import Console

from .. import __dashboard_url__
from ..core.api import WecoClient
from ..core.auth import handle_authentication


def handle_share_command(run_id: str, output_mode: str, console: Console) -> None:
    """Handle the `weco share <run_id>` command."""
    _, auth_headers = handle_authentication(console)
    if not auth_headers:
        sys.exit(1)

    client = WecoClient(auth_headers)
    share_id = client.share(run_id)

    if share_id is None:
        console.print("[bold red]Error creating share link.[/]")
        sys.exit(1)

    share_url = f"{__dashboard_url__}/share/{share_id}"

    if output_mode == "plain":
        console.print(share_url)
    else:
        console.print()
        console.print("[bold green]Share link created![/]")
        console.print(f"[bold]{share_url}[/]")
        console.print()
        console.print("[dim]Anyone with this link can view the run's optimization progress and results.[/]")
