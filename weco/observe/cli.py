"""CLI commands for weco observe.

Exit-code policy follows each command's role in the caller's workflow, not a
fault taxonomy:

- ``init`` runs once, before the loop being tracked exists. Any failure exits
  non-zero: nothing is at risk yet, and ``RUN_ID=$(weco observe init ...)``
  must never capture an empty string from a "polite" exit 0.
- ``log`` runs inside the tracked loop (often under ``set -e``). Errors the
  caller must fix (bad ``--metrics`` JSON, unreadable ``--source``, not logged
  in, a 4xx from the API) exit non-zero; weco-side failures print to stderr
  and exit 0 so a blip never crashes the tracked loop. ``--strict`` opts into
  making those fatal as well.
"""

import argparse
import json
import sys

from weco.browser import open_browser
from weco.config import load_weco_api_key
from weco.events import send_event, ObserveInitEvent, ObserveLogEvent
from weco.observe import api
from weco import __dashboard_url__


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
    log_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when weco itself fails (default: warn on stderr and exit 0 so the tracked loop survives)",
    )

    # --- complete/fail are no longer needed ---
    # External run lifecycle is managed by the dashboard, not the CLI.
    # Logging a step to a closed run will silently reopen it.


def _read_code_files(paths: list[str]) -> dict[str, str]:
    """Read source code files from disk. Exits 1 if any file cannot be read.

    All-or-nothing: silently logging a partial snapshot would show wrong
    code in the dashboard, so a single unreadable file fails the command.
    """
    source_code = {}
    for path in paths:
        try:
            with open(path) as f:
                source_code[path] = f.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"weco observe: cannot read {path}: {e}", file=sys.stderr)
            sys.exit(1)
    return source_code


def execute_observe_command(args: argparse.Namespace) -> None:
    """Execute an observe subcommand."""
    if not args.observe_command:
        print("Usage: weco observe {init,log}", file=sys.stderr)
        sys.exit(2)

    # Build headers straight from the stored key: a sidecar embedded in a
    # scripted loop must never open handle_authentication's interactive prompt.
    api_key = load_weco_api_key()
    if not api_key:
        print("weco observe: not logged in. Run `weco login` first.", file=sys.stderr)
        sys.exit(1)
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    if args.observe_command == "init":
        _handle_init(args, auth_headers)
    elif args.observe_command == "log":
        _handle_log(args, auth_headers)


def _handle_init(args: argparse.Namespace, auth_headers: dict) -> None:
    """Handle `weco observe init`. Any failure exits non-zero (see module docstring)."""
    source_arg = args.sources if args.sources is not None else [args.source]
    source_code = _read_code_files(source_arg)

    maximize = args.goal in ("maximize", "max")

    send_event(
        ObserveInitEvent(metric=args.metric, goal="maximize" if maximize else "minimize", source_count=len(source_code))
    )

    try:
        result = api.create_run(
            source_code=source_code,
            metric_name=args.metric,
            maximize=maximize,
            name=args.name,
            additional_instructions=args.additional_instructions,
            auth_headers=auth_headers,
        )
    except api.ObserveError as e:
        print(f"weco observe: {e}", file=sys.stderr)
        sys.exit(1)

    run_id = result.get("run_id")
    if not run_id:
        print("weco observe: create run response carried no run_id", file=sys.stderr)
        sys.exit(1)

    # Print only the run_id to stdout so it can be captured by $(...)
    print(run_id)
    # Open the dashboard in the user's browser
    dashboard_url = f"{__dashboard_url__}/runs/{run_id}"
    open_browser(dashboard_url)


def _handle_log(args: argparse.Namespace, auth_headers: dict) -> None:
    """Handle `weco observe log`."""
    # Parse metrics JSON
    metrics = {}
    if args.metrics:
        try:
            metrics = json.loads(args.metrics)
        except json.JSONDecodeError as e:
            print(f"weco observe: invalid metrics JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # Read source files if specified
    code = None
    source_arg = args.sources if args.sources is not None else ([args.source] if args.source else None)
    if source_arg:
        code = _read_code_files(source_arg)

    send_event(ObserveLogEvent(status=args.status))

    try:
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
    except api.CallerError as e:
        print(f"weco observe: {e}", file=sys.stderr)
        sys.exit(1)
    except api.TransientError as e:
        print(f"weco observe: {e} (step {args.step} dropped)", file=sys.stderr)
        if args.strict:
            sys.exit(1)
        # Default: a weco-side blip must not crash the loop being tracked.
