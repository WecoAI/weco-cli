# weco/auth.py
import time
import requests
import webbrowser
from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt
from . import __base_url__
from .config import save_api_key, load_weco_api_key
from .events import send_event, get_event_context, AuthStartedEvent, AuthCompletedEvent, AuthFailedEvent


def perform_login(console: Console):
    """Handles the device login flow."""
    ctx = get_event_context()

    # Track auth started
    send_event(AuthStartedEvent(), ctx)

    try:
        # 1. Initiate device login
        console.print("Initiating login...")
        init_response = requests.post(f"{__base_url__}/auth/device/initiate")
        init_response.raise_for_status()
        init_data = init_response.json()

        device_code = init_data["device_code"]
        verification_uri = init_data["verification_uri"]
        expires_in = init_data["expires_in"]
        interval = init_data["interval"]

        # 2. Display instructions
        console.print("\n[bold yellow]Action Required:[/]")
        console.print("Please open the following URL in your browser to authenticate:")
        console.print(f"[link={verification_uri}]{verification_uri}[/link]")
        console.print(f"This request will expire in {expires_in // 60} minutes.")
        console.print("Attempting to open the authentication page in your default browser...")  # Notify user

        # Automatically open the browser
        try:
            if not webbrowser.open(verification_uri):
                console.print("[yellow]Could not automatically open the browser. Please open the link manually.[/]")
        except Exception as browser_err:
            console.print(
                f"[yellow]Could not automatically open the browser ({browser_err}). Please open the link manually.[/]"
            )

        console.print("Waiting for authentication...", end="")

        # 3. Poll for token
        start_time = time.time()
        # Use a simple text update instead of Spinner within Live for potentially better compatibility
        polling_status = "Waiting..."
        with Live(polling_status, refresh_per_second=1, transient=True, console=console) as live_status:
            while True:
                # Check for timeout
                if time.time() - start_time > expires_in:
                    send_event(AuthFailedEvent(reason="timeout"), ctx)
                    console.print("\n[bold red]Error:[/] Login request timed out.")
                    return False

                time.sleep(interval)
                live_status.update("Waiting... (checking status)")

                try:
                    token_response = requests.post(
                        f"{__base_url__}/auth/device/token",
                        json={
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                            "device_code": device_code,
                            "installation_id": ctx.installation_id,
                        },
                    )

                    # Check for 202 Accepted - Authorization Pending
                    if token_response.status_code == 202:
                        token_data = token_response.json()
                        if token_data.get("error") == "authorization_pending":
                            live_status.update("Waiting... (authorization pending)")
                            continue  # Continue polling
                        else:
                            # Unexpected 202 response format
                            send_event(AuthFailedEvent(reason="unexpected_response"), ctx)
                            console.print(
                                f"\n[bold red]Error:[/] Received unexpected response from authentication server: {token_data}"
                            )
                            return False
                    # Check for standard OAuth2 errors (often 400 Bad Request)
                    elif token_response.status_code == 400:
                        token_data = token_response.json()
                        error_code = token_data.get("error", "unknown_error")
                        if error_code == "slow_down":
                            interval += 5  # Increase polling interval if instructed
                            live_status.update(f"Waiting... (slowing down polling to {interval}s)")
                            continue
                        elif error_code == "expired_token":
                            send_event(AuthFailedEvent(reason="expired"), ctx)
                            console.print("\n[bold red]Error:[/] Login request expired.")
                            return False
                        elif error_code == "access_denied":
                            send_event(AuthFailedEvent(reason="denied"), ctx)
                            console.print("\n[bold red]Error:[/] Authorization denied by user.")
                            return False
                        else:  # invalid_grant, etc.
                            send_event(AuthFailedEvent(reason=error_code), ctx)
                            error_desc = token_data.get("error_description", "Unknown authentication error occurred.")
                            console.print(f"\n[bold red]Error:[/] {error_desc} ({error_code})")
                            return False

                    # Check for other non-200/non-202/non-400 HTTP errors
                    token_response.raise_for_status()
                    # If successful (200 OK and no 'error' field)
                    token_data = token_response.json()
                    if "access_token" in token_data:
                        api_key = token_data["access_token"]
                        save_api_key(api_key)
                        send_event(AuthCompletedEvent(), ctx)
                        console.print("\n[bold green]Login successful![/]")
                        return True
                    else:
                        # Unexpected successful response format
                        send_event(AuthFailedEvent(reason="unexpected_response"), ctx)
                        console.print("\n[bold red]Error:[/] Received unexpected response from server during polling.")
                        print(token_data)
                        return False
                except requests.exceptions.RequestException as e:
                    # Handle network errors during polling gracefully
                    live_status.update("Waiting... (network error, retrying)")
                    console.print(f"\n[bold yellow]Warning:[/] Network error during polling: {e}. Retrying...")
                    time.sleep(interval * 2)  # Simple backoff
    except requests.exceptions.HTTPError as e:
        from .api import handle_api_error  # Import here to avoid circular imports

        send_event(AuthFailedEvent(reason="http_error"), ctx)
        handle_api_error(e, console)
        return False
    except requests.exceptions.RequestException as e:
        # Catch other request errors
        send_event(AuthFailedEvent(reason="network_error"), ctx)
        console.print(f"\n[bold red]Network Error:[/] {e}")
        return False
    except Exception as e:
        send_event(AuthFailedEvent(reason="error"), ctx)
        console.print(f"\n[bold red]An unexpected error occurred during login:[/] {e}")
        return False


def handle_authentication(console: Console) -> tuple[str | None, dict]:
    """
    Handle the complete authentication flow. Authentication is now mandatory.

    Returns:
        tuple: (weco_api_key, auth_headers)
    """
    weco_api_key = load_weco_api_key()

    if not weco_api_key:
        console.print("[bold yellow]Authentication Required[/]")
        console.print("With our new credit-based billing system, authentication is required to use Weco.")
        console.print("You'll receive free credits to get started!")
        console.print("")

        login_choice = Prompt.ask(
            "Would you like to log in now? ([bold]Y[/]es / [bold]N[/]o)", choices=["y", "n"], default="y"
        ).lower()

        if login_choice == "y":
            console.print("[cyan]Starting login process...[/]")
            if not perform_login(console):
                console.print("[bold red]Login process failed or was cancelled.[/]")
                return None, {}

            weco_api_key = load_weco_api_key()
            if not weco_api_key:
                console.print("[bold red]Error: Login completed but failed to retrieve API key.[/]")
                return None, {}
        else:
            console.print("[yellow]Authentication is required to use Weco. Please run 'weco' again when ready to log in.[/]")
            return None, {}

    # Build auth headers
    auth_headers = {}
    if weco_api_key:
        auth_headers["Authorization"] = f"Bearer {weco_api_key}"

    return weco_api_key, auth_headers
