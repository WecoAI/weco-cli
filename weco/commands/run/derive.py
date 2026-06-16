"""``weco run derive <run-id>`` — create a derived run from an existing run's step."""

import json
import os
import pathlib
import sys

import requests
from rich.console import Console
from rich.prompt import Confirm

from ...auth import handle_authentication
from ...browser import open_browser
from ...consumer_lock import consumer_lock
from ...core.api import WecoClient, handle_api_error
from ...optimizer import _daemonize_to_log, _should_auto_open_browser, run_lineage_loop
from ...utils import read_additional_instructions, read_from_path
from ... import __dashboard_url__


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeriveError(Exception):
    """User-correctable error before the optimization loop starts.

    Helpers raise this to describe a failure they can't recover from. The
    handler catches it once near the top and routes the message through the
    right channel for the active output mode.
    """


# ---------------------------------------------------------------------------
# from-step resolution
# ---------------------------------------------------------------------------


# Aliases the CLI accepts for the two "best node" keywords. The dict is also
# the documentation: any key here is a valid --from-step value.
_FROM_STEP_KEYWORDS: dict[str, str] = {
    "best": "lineage_best",
    "lineage-best": "lineage_best",
    "lineage_best": "lineage_best",
    "run-best": "run_best",
    "run_best": "run_best",
}


def _resolve_step_to_node_id(client: WecoClient, run_id: str, step: int) -> str:
    """Look up the node UUID at a given step in ``run_id``.

    One API call. Raises :class:`DeriveError` if no node exists at that step.
    HTTP/network errors propagate unchanged for the caller to format.
    """
    response = client.list_nodes(run_id, step=step, include_code=False)
    nodes = response.get("nodes", [])
    if not nodes:
        raise DeriveError(f"No node found at step {step} in run {run_id}")
    # Backend's NodeListItem schema (api/models/legacy.py) guarantees `node_id`.
    # Multiple nodes per step are possible in branched lineages; we take the
    # first as a deliberate convention.
    return nodes[0]["node_id"]


def _resolve_derive_from(client: WecoClient, run_id: str, from_step: str) -> str:
    """Translate the user-facing ``--from-step`` value into the backend's
    ``derive_from`` value.

    Three forms are accepted:

    * a keyword from :data:`_FROM_STEP_KEYWORDS` (``best``, ``run-best``, …)
      → returns the backend keyword (``"lineage_best"`` / ``"run_best"``)
    * an integer step number → looks up the node at that step (one API call)
    * anything else → presumed node UUID, passed through for the backend to
      validate (and 404 on if bogus)
    """
    if from_step in _FROM_STEP_KEYWORDS:
        return _FROM_STEP_KEYWORDS[from_step]
    try:
        step = int(from_step)
    except ValueError:
        return from_step  # presumed node UUID
    return _resolve_step_to_node_id(client, run_id, step)


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


def _report_error(console: Console, output_mode: str, message: str) -> None:
    """Print an error message in the format appropriate for ``output_mode``.

    Always writes to stderr — errors should never pollute stdout where
    machine consumers (jq, NDJSON parsers) expect normal output.
    """
    if output_mode == "plain":
        print(json.dumps({"error": message}), file=sys.stderr, flush=True)
    else:
        console.print(f"[bold red]{message}[/]")


def _resolve_originals(inherited_source: dict[str, str]) -> dict[str, str]:
    """Build the file-restoration baseline for the optimization loop.

    Prefer the user's local files (they may have edits since the parent run);
    fall back to the inherited baseline source from the source node — never
    the candidate, which would pollute the working directory with generated
    code on every eval cycle.
    """
    originals: dict[str, str] = {}
    for rel_path, inherited_content in inherited_source.items():
        fp = pathlib.Path(rel_path)
        originals[rel_path] = read_from_path(fp=fp, is_json=False) if fp.exists() else inherited_content
    return originals


def _attach_or_consume(
    *,
    console: Console,
    auth_headers: dict[str, str],
    run_info: dict,
    lineage_id: str,
    originals: dict[str, str],
    output_mode: str,
    api_keys: dict[str, str] | None,
) -> bool:
    """Feed the new derived run to the single lineage consumer.

    Deriving never spawns its own optimization loop — that's what caused
    parallel derived runs to evaluate concurrently in one working tree and
    clobber each other. Instead we try to acquire the working-tree consumer
    lock:

    * lock **held** — a consumer is already draining this lineage in this tree;
      it will pick up the new run on its next poll. Print the run id and return.
    * lock **free** — become the lineage consumer and drain the lineage (this
      run plus any other active members), one eval at a time.
    """
    new_run_id = run_info["id"]

    with consumer_lock() as acquired:
        if not acquired:
            if output_mode == "plain":
                print(json.dumps({"run_id": new_run_id, "lineage_id": lineage_id, "attached": True}), flush=True)
            else:
                console.print(
                    f"[green]Derived run {new_run_id} created.[/] An active consumer in this directory will evaluate it."
                )
            return True

        return run_lineage_loop(
            lineage_id=lineage_id,
            auth_headers=auth_headers,
            originals=originals,
            eval_command=run_info["evaluation_command"],
            eval_timeout=run_info["eval_timeout"],
            save_logs=run_info["save_logs"],
            log_dir=run_info["log_dir"],
            dashboard_base=__dashboard_url__,
            api_keys=api_keys,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handle(
    run_id: str,
    from_step: str,
    steps: int | None,
    additional_instructions: str | None,
    api_keys: dict[str, str] | None,
    output_mode: str,
    console: Console,
    daemon: bool = False,
    no_open: bool = False,
) -> bool:
    """Create a derived run and feed it to the single lineage consumer.

    Never spawns its own optimization loop — if a consumer is already draining
    this lineage in the working tree, the new run is picked up by it; otherwise
    this process becomes that consumer. Returns ``True`` on success, ``False``
    on failure. The dispatcher uses this to set the process exit code.
    """
    weco_api_key, auth_headers = handle_authentication(console)
    if weco_api_key is None:
        sys.exit(1)

    client = WecoClient(auth_headers)
    additional_instructions = read_additional_instructions(additional_instructions)

    # Single try block for everything that talks to the backend before the
    # optimization loop. Each exception type maps to a distinct user-visible
    # error format; the loop's own errors are reported separately by the loop.
    try:
        derive_from = _resolve_derive_from(client, run_id, from_step)
        response = client.derive_run(
            run_id, derive_from=derive_from, additional_instructions=additional_instructions, steps=steps, api_keys=api_keys
        )
    except DeriveError as e:
        _report_error(console, output_mode, str(e))
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if output_mode == "plain":
            _report_error(console, output_mode, f"HTTP {e.response.status_code}: {e.response.text}")
        else:
            handle_api_error(e, console)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        _report_error(console, output_mode, f"Network error contacting Weco API: {e}")
        sys.exit(1)

    run_info = response["run"]
    new_run_id = run_info["id"]
    # A derived run is never its own dashboard page — it lives inside its
    # root run's lineage tree. Point the dashboard link at the lineage root
    # (the derived run's `lineage_id`) so we don't open a dead-end tab for
    # the sub-run; it surfaces under the root instead. Falls back to the
    # run's own id for the (non-derived) safety case where lineage_id is
    # absent.
    lineage_id = run_info.get("lineage_id") or new_run_id
    dashboard_url = f"{__dashboard_url__}/runs/{lineage_id}"

    originals = _resolve_originals(run_info["source_code"])

    # Skip the interactive confirm when there's no human at the terminal:
    #   * daemon mode — we've forked, no stdin;
    #   * plain output — non-interactive caller;
    #   * agent session — the derive was driven by the agent in `weco start`
    #     (signalled by WECO_CC_SESSION_ID), where the dashboard already
    #     confirmed intent and blocking on stdin would hang the agent.
    in_agent_session = bool(os.environ.get("WECO_CC_SESSION_ID"))
    if (
        not daemon
        and output_mode != "plain"
        and not in_agent_session
        and not Confirm.ask(
            "Have the source files in your working directory been kept consistent with "
            "the parent run? Local edits will override the inherited baseline.",
            default=True,
        )
    ):
        console.print("[yellow]Derive cancelled. Adjust your working directory and re-run.[/]")
        return False

    if daemon:
        # Print everything stdout-watchers (claude, cursor, the wrapper's
        # find_run_ids) care about BEFORE forking — once we're detached,
        # stdout is the log file.
        print(f"Run ID: {new_run_id}", flush=True)
        print(f"Run name: {run_info['name']}", flush=True)
        print(f"Dashboard: {dashboard_url}", flush=True)
        log_path = _daemonize_to_log(new_run_id)
        if log_path is None:
            console.print("[yellow]--daemon not supported on this platform; running in foreground.[/]")
        else:
            # Force plain UI in the daemonized child — Rich's ANSI escapes
            # would pollute the log file and serve no purpose without a TTY.
            output_mode = "plain"
            console = Console(force_terminal=False)
            print(f"weco run derive daemon started; log: {log_path}", flush=True)
    elif _should_auto_open_browser(no_open):
        open_browser(dashboard_url)

    return _attach_or_consume(
        console=console,
        auth_headers=auth_headers,
        run_info=run_info,
        lineage_id=lineage_id,
        originals=originals,
        output_mode=output_mode,
        api_keys=api_keys,
    )
