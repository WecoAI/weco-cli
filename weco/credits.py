"""Credit management commands for the Weco CLI."""

import webbrowser
import requests
from rich.console import Console
from rich.table import Table
from . import __base_url__
from .api import handle_api_error
from .auth import load_weco_api_key, handle_authentication


def handle_credits_command(args, console: Console) -> None:
    """Handle the credits command and its subcommands."""
    # Ensure user is authenticated
    weco_api_key = load_weco_api_key()
    if not weco_api_key:
        console.print("[bold yellow]Authentication Required[/]")
        console.print("You need to be logged in to manage credits.")
        weco_api_key, _ = handle_authentication(console)
        if not weco_api_key:
            console.print("[bold red]Authentication failed. Please run 'weco' to log in.[/]")
            return

    auth_headers = {"Authorization": f"Bearer {weco_api_key}"}

    if args.credits_command == "balance" or args.credits_command is None:
        check_balance(console, auth_headers)
    elif args.credits_command == "topup":
        topup_credits(console, auth_headers, args.amount)
    elif args.credits_command == "autotopup":
        configure_autotopup(console, auth_headers, args)
    else:
        console.print(f"[bold red]Unknown credits command: {args.credits_command}[/]")


def check_balance(console: Console, auth_headers: dict) -> None:
    """Check and display the current credit balance."""
    try:
        response = requests.get(f"{__base_url__}/billing/balance", headers=auth_headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        balance = data.get("balance_credits", 0)

        # Create a nice table display
        table = Table(title="Weco Credit Balance", show_header=True, header_style="bold cyan")
        table.add_column("Balance", style="green", justify="right")
        table.add_column("Status", justify="left")

        status = "✅ Good" if balance > 10 else "⚠️ Low" if balance > 0 else "❌ Empty"
        table.add_row(f"{balance:.2f} credits", status)

        console.print(table)

        if balance < 10:
            console.print("\n[yellow]💡 Tip: You're running low on credits. Run 'weco credits topup' to add more.[/]")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            console.print("[bold red]Authentication failed. Please log in again with 'weco'.[/]")
        else:
            console.print(f"[bold red]Error checking balance: {e}[/]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/]")


def topup_credits(console: Console, auth_headers: dict, amount: float) -> None:
    """Initiate a credit top-up via Stripe."""
    try:
        console.print(f"[cyan]Preparing to purchase {amount:.2f} credits...[/]")

        response = requests.post(
            f"{__base_url__}/billing/topup/checkout", headers=auth_headers, json={"amount": amount}, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        checkout_url = data.get("checkout_url")
        if checkout_url:
            console.print("\n[bold green]✅ Checkout session created![/]")
            console.print(f"Total: {amount:.2f} credits")
            console.print("\n[yellow]Opening checkout page in your browser...[/]")

            # Try to open the browser
            try:
                if webbrowser.open(checkout_url):
                    console.print("[green]Browser opened successfully![/]")
                else:
                    console.print(f"[yellow]Please open this URL manually:[/]\n{checkout_url}")
            except Exception:
                console.print(f"[yellow]Please open this URL in your browser:[/]\n{checkout_url}")

            console.print("\n[dim]Complete the payment in your browser. Your credits will be added automatically.[/]")
        else:
            console.print("[bold red]Error: No checkout URL received.[/]")

    except requests.exceptions.HTTPError as e:
        response = getattr(e, "response", None)

        if response is not None and response.status_code == 401:
            console.print("[bold red]Authentication failed. Please log in again with 'weco'.[/]")
        else:
            console.print("[bold red]Error creating checkout session[/]")
            handle_api_error(e, console)
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/]")


def configure_autotopup(console: Console, auth_headers: dict, args) -> None:
    """Configure automatic top-up settings."""
    try:
        # Handle conflicting flags
        if args.enable and args.disable:
            console.print("[bold red]Error: Cannot use both --enable and --disable flags.[/]")
            return

        # If neither flag is set, show current settings
        if not args.enable and not args.disable:
            # Get current auto top-up settings from API
            response = requests.get(f"{__base_url__}/billing/auto-topup", headers=auth_headers, timeout=10)
            response.raise_for_status()
            settings_data = response.json()

            # Also check if user has a payment method
            balance_response = requests.get(f"{__base_url__}/billing/balance", headers=auth_headers, timeout=10)
            balance_response.raise_for_status()
            balance_data = balance_response.json()

            has_payment_method = bool(balance_data.get("stripe_customer_id"))

            console.print("[cyan]Current Auto Top-Up Settings:[/]")

            if settings_data.get("enabled"):
                if has_payment_method:
                    console.print("Status: [green]✅ Enabled[/]")
                else:
                    console.print("Status: [yellow]⚠️ Enabled but no payment method saved[/]")
            else:
                console.print("Status: [red]❌ Disabled[/]")

            console.print(f"Threshold: {settings_data.get('threshold_credits', 4.0)} credits")
            console.print(f"Top-up Amount: {settings_data.get('topup_amount_credits', 50.0)} credits")

            if not has_payment_method:
                console.print("\n[yellow]💡 Note: Auto top-up requires a saved payment method.[/]")
                console.print("Complete a manual top-up first to save your payment details.")

            console.print("\nUse --enable or --disable to change settings.")
            return

        # Configure auto top-up
        settings = {"enabled": args.enable, "threshold_credits": args.threshold, "topup_amount_credits": args.amount}

        response = requests.post(f"{__base_url__}/billing/auto-topup", headers=auth_headers, json=settings, timeout=10)
        response.raise_for_status()

        if args.enable:
            console.print("[bold green]✅ Auto top-up enabled![/]")
            console.print(f"When your balance falls below {args.threshold} credits,")
            console.print(f"we'll automatically add {args.amount} credits to your account.")
            console.print("\n[yellow]Note: Requires a saved payment method.[/]")
        else:
            console.print("[bold green]✅ Auto top-up disabled![/]")
            console.print("You'll need to manually top up your credits when running low.")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            console.print("[bold red]Authentication failed. Please log in again with 'weco'.[/]")
        else:
            console.print(f"[bold red]Error configuring auto top-up: {e}[/]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/]")
