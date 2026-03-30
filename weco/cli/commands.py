import argparse
import sys

from rich.console import Console

from .backends import load_backend


def parse_api_keys(api_key_args: list[str] | None) -> dict[str, str]:
    """Parse API key arguments from CLI into a dictionary."""
    if not api_key_args:
        return {}

    api_keys: dict[str, str] = {}
    for arg in api_key_args:
        if "=" not in arg:
            raise ValueError(f"Invalid API key format: '{arg}'. Expected format: 'provider=key'")

        provider, key = (s.strip() for s in arg.split("=", 1))
        if not provider or not key:
            raise ValueError(f"Invalid API key format: '{arg}'. Provider and key must be non-empty.")

        api_keys[provider.lower()] = key

    return api_keys


def _collect_source_paths(args: argparse.Namespace) -> list[str] | None:
    if getattr(args, "sources", None):
        return args.sources
    if getattr(args, "source", None):
        return [args.source]
    return None


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


def cmd_login(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.auth import handle_login

    handle_login(console)


def cmd_logout(args: argparse.Namespace, *, console: Console) -> None:
    del args
    from ..commands.auth import handle_logout

    handle_logout()


def cmd_run(args: argparse.Namespace, *, console: Console) -> None:
    """Start a new optimization run."""
    from ..commands.credits import check_promotional_credits
    from ..core.events import RunStartAttemptedEvent, get_event_context, send_event
    from ..commands.run.optimize import optimize
    from ..core.errors import DefaultModelNotFoundError, UnrecognizedAPIKeysError
    from ..core.model_selection import get_default_model
    from ..core.validation import ValidationError, print_validation_error, validate_log_directory, validate_sources

    ctx = get_event_context()

    backend_name = getattr(args, "eval_backend", "shell")
    if backend_name != "shell":
        backend = load_backend(backend_name)
        backend.validate_args(args)
        args.eval_command = backend.build_eval_command(args)
    elif not args.eval_command:
        console.print("[bold red]Error: --eval-command is required (or use --eval-backend <backend>)[/]")
        sys.exit(1)

    missing = []
    if not args.source and not args.sources:
        missing.append("--source / --sources")
    if not args.metric:
        missing.append("--metric")
    if not args.goal:
        missing.append("--goal")
    if missing:
        console.print(f"[bold red]Error: missing required arguments: {', '.join(missing)}[/]")
        sys.exit(1)

    source_arg = args.sources if args.sources is not None else [args.source]

    try:
        validate_sources(source_arg)
        validate_log_directory(args.log_dir)
    except ValidationError as exc:
        print_validation_error(exc, console)
        sys.exit(1)

    try:
        api_keys = parse_api_keys(args.api_key)
    except ValueError as exc:
        console.print(f"[bold red]Error parsing API keys: {exc}[/]")
        sys.exit(1)

    model = args.model
    if not model:
        try:
            model = get_default_model(api_keys=api_keys)
        except (UnrecognizedAPIKeysError, DefaultModelNotFoundError) as exc:
            console.print(f"[bold red]Error: {exc}[/]")
            sys.exit(1)
        if api_keys:
            console.print(f"[bold yellow]Custom API keys provided. Using default model: {model} for the run.[/]")

    model = check_promotional_credits(model, api_keys, console)

    send_event(
        RunStartAttemptedEvent(
            output_mode=args.output,
            require_review=args.require_review,
            save_logs=args.save_logs,
            steps=args.steps,
            model=model,
        ),
        ctx,
    )

    success = optimize(
        source=source_arg,
        eval_command=args.eval_command,
        metric=args.metric,
        goal=args.goal,
        model=model,
        steps=args.steps,
        log_dir=args.log_dir,
        additional_instructions=args.additional_instructions,
        eval_timeout=args.eval_timeout,
        save_logs=args.save_logs,
        api_keys=api_keys,
        apply_change=args.apply_change,
        require_review=args.require_review,
        output_mode=args.output,
    )
    sys.exit(0 if success else 1)


def cmd_resume(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.optimize import resume_optimization

    try:
        api_keys = parse_api_keys(args.api_key)
    except ValueError as exc:
        console.print(f"[bold red]Error parsing API keys: {exc}[/]")
        sys.exit(1)

    success = resume_optimization(
        run_id=args.run_id, api_keys=api_keys, apply_change=args.apply_change, output_mode=args.output
    )
    sys.exit(0 if success else 1)


def cmd_credits(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.credits import handle_credits_command

    handle_credits_command(args, console)


def cmd_share(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.share import handle_share_command

    handle_share_command(run_id=args.run_id, output_mode=args.output, console=console)


def cmd_setup(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.setup import handle_setup_command

    handle_setup_command(args, console)


def cmd_observe(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.observe import handle

    handle(args, console)


# ---------------------------------------------------------------------------
# `weco run` subcommands — thin dispatch to commands.run.*
# ---------------------------------------------------------------------------


def cmd_run_status(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.status import handle

    handle(run_id=args.run_id, console=console)


def cmd_run_results(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.results import handle

    handle(
        run_id=args.run_id, top=args.top, format=args.format, plot=args.plot, include_code=args.include_code, console=console
    )


def cmd_run_show(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.show import handle

    handle(run_id=args.run_id, step=args.step, console=console)


def cmd_run_diff(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.diff import handle

    handle(run_id=args.run_id, step=args.step, against=args.against, console=console)


def cmd_run_stop(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.stop import handle

    handle(run_id=args.run_id, console=console)


def cmd_run_instruct(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.instruct import handle

    handle(run_id=args.run_id, instructions=args.instructions, console=console)


def cmd_run_review(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.review import handle

    handle(run_id=args.run_id, console=console)


def cmd_run_revise(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.revise import handle

    handle(run_id=args.run_id, node_id=args.node, source_paths=_collect_source_paths(args), console=console)


def cmd_run_submit(args: argparse.Namespace, *, console: Console) -> None:
    from ..commands.run.submit import handle

    handle(
        run_id=args.run_id,
        node_id=args.node,
        source_paths=_collect_source_paths(args),
        eval_command_override=getattr(args, "eval_command", None),
        console=console,
    )
