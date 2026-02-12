"""Share command handler for the Weco CLI."""

import sys
from rich.console import Console

from . import __dashboard_url__
from .api import create_share_link
from .auth import handle_authentication


def handle_share_command(run_id: str, output_mode: str, console: Console) -> None:
    """Handle the `weco share <run_id>` command.

    Creates a public share link for the given run. Requires CLI sharing
    to be enabled in the user's account settings on the dashboard.

    Args:
        run_id: UUID of the run to share.
        output_mode: 'rich' or 'plain'.
        console: Rich console instance for output.
    """
    # Authenticate
    _, auth_headers = handle_authentication(console)
    if not auth_headers:
        sys.exit(1)

    # Attempt to create the share link
    share_id = create_share_link(
        console=console,
        run_id=run_id,
        auth_headers=auth_headers,
    )

    if share_id is None:
        # Error message was already printed by create_share_link / handle_api_error
        sys.exit(1)

    share_url = f"{__dashboard_url__}/share/{share_id}"

    if output_mode == "plain":
        console.print(share_url)
    else:
        console.print()
        console.print(f"[bold green]Share link created![/]")
        console.print(f"[bold]{share_url}[/]")
        console.print()
        console.print("[dim]Anyone with this link can view the run's optimization progress and results.[/]")
