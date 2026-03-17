"""CLI commands for weco observe.

All commands follow the fire-and-forget pattern: they print warnings to
stderr on failure but always exit 0 so they never crash an agent's loop.
"""

import argparse
import json
import sys
import warnings

from weco.auth import handle_authentication
from weco.observe import api


def configure_observe_parser(observe_parser: argparse.ArgumentParser) -> None:
    """Configure the observe command parser and all its subcommands."""
    subparsers = observe_parser.add_subparsers(dest="observe_command", help="Observe commands")

    # --- init ---
    init_parser = subparsers.add_parser("init", help="Initialize an external run for tracking")
    init_parser.add_argument("--name", type=str, default=None, help="Run name")
    init_parser.add_argument("--metric", type=str, required=True, help="Primary metric name (e.g. val_bpb)")
    init_parser.add_argument(
        "-g",
        "--goal",
        type=str,
        choices=["maximize", "max", "minimize", "min"],
        default="minimize",
        help="Specify 'maximize'/'max' or 'minimize'/'min' (default: minimize)",
    )
    init_source_group = init_parser.add_mutually_exclusive_group(required=True)
    init_source_group.add_argument(
        "-s", "--source", type=str, help="Path to a single source code file to track (e.g. train.py)"
    )
    init_source_group.add_argument(
        "--sources", nargs="+", type=str, help="Paths to multiple source code files to track (e.g. train.py prepare.py)"
    )
    init_parser.add_argument(
        "-i", "--additional-instructions", type=str, default=None, help="Additional instructions for the run"
    )

    # --- log ---
    log_parser = subparsers.add_parser("log", help="Log a step for an external run")
    log_parser.add_argument("--run-id", type=str, required=True, help="Run ID (from weco observe init)")
    log_parser.add_argument("--step", type=int, required=True, help="Step number")
    log_parser.add_argument(
        "--status", type=str, default="completed", choices=["completed", "failed"], help="Step status (default: completed)"
    )
    log_parser.add_argument("--description", type=str, default=None, help="Description of what was tried")
    log_parser.add_argument("--metrics", type=str, default=None, help="Metrics as JSON (e.g. '{\"val_bpb\": 1.03}')")
    log_source_group = log_parser.add_mutually_exclusive_group()
    log_source_group.add_argument("-s", "--source", type=str, default=None, help="Single source code file to snapshot")
    log_source_group.add_argument(
        "--sources", nargs="+", type=str, default=None, help="Multiple source code files to snapshot"
    )
    log_parser.add_argument("--parent-step", type=int, default=None, help="Parent step number for tree lineage")

    # --- complete/fail are no longer needed ---
    # External run lifecycle is managed by the dashboard, not the CLI.
    # Logging a step to a closed run will silently reopen it.


def _read_code_files(paths: list[str]) -> dict[str, str]:
    """Read source code files from disk."""
    source_code = {}
    for path in paths:
        try:
            with open(path) as f:
                source_code[path] = f.read()
        except FileNotFoundError:
            warnings.warn(f"weco observe: file not found: {path}", stacklevel=2)
        except Exception as e:
            warnings.warn(f"weco observe: error reading {path}: {e}", stacklevel=2)
    return source_code


def execute_observe_command(args: argparse.Namespace) -> None:
    """Execute an observe subcommand. Always exits 0."""
    if not args.observe_command:
        print("Usage: weco observe {init,log,complete,fail}", file=sys.stderr)
        sys.exit(0)

    # Authenticate
    try:
        _, auth_headers = handle_authentication(None)
        if not auth_headers:
            print("weco observe: not logged in. Run `weco login` first.", file=sys.stderr)
            sys.exit(0)
    except Exception as e:
        print(f"weco observe: authentication failed: {e}", file=sys.stderr)
        sys.exit(0)

    if args.observe_command == "init":
        _handle_init(args, auth_headers)
    elif args.observe_command == "log":
        _handle_log(args, auth_headers)


def _handle_init(args: argparse.Namespace, auth_headers: dict) -> None:
    """Handle `weco observe init`."""
    source_arg = args.sources if args.sources is not None else [args.source]
    source_code = _read_code_files(source_arg)
    if not source_code:
        print("weco observe: no source files could be read", file=sys.stderr)
        sys.exit(0)

    maximize = args.goal in ("maximize", "max")

    result = api.create_run(
        source_code=source_code,
        metric_name=args.metric,
        maximize=maximize,
        name=args.name,
        additional_instructions=args.additional_instructions,
        auth_headers=auth_headers,
    )

    if result and result.get("run_id"):
        # Print only the run_id to stdout so it can be captured by $(...)
        print(result["run_id"])
    else:
        print("weco observe: failed to create run", file=sys.stderr)


def _handle_log(args: argparse.Namespace, auth_headers: dict) -> None:
    """Handle `weco observe log`."""
    # Parse metrics JSON
    metrics = {}
    if args.metrics:
        try:
            metrics = json.loads(args.metrics)
        except json.JSONDecodeError as e:
            print(f"weco observe: invalid metrics JSON: {e}", file=sys.stderr)
            sys.exit(0)

    # Read source files if specified
    code = None
    source_arg = args.sources if args.sources is not None else ([args.source] if args.source else None)
    if source_arg:
        code = _read_code_files(source_arg)

    api.log_step(
        run_id=args.run_id,
        step=args.step,
        status=args.status,
        description=args.description,
        metrics=metrics,
        code=code,
        parent_step=args.parent_step,
        auth_headers=auth_headers,
    )
