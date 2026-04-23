"""``weco run derive <run-id>`` — create a derived run from an existing run's step."""

import json
import pathlib
import sys

import requests
from rich.console import Console
from rich.prompt import Confirm

from ...api import report_termination
from ...artifacts import RunArtifacts
from ...auth import handle_authentication
from ...core.api import WecoClient, handle_api_error
from ...heartbeat import heartbeat
from ...optimizer import OptimizationResult, offer_apply_best_solution, run_optimization_loop
from ...ui import LiveOptimizationUI, PlainOptimizationUI
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


def _print_resume_hint(console: Console, output_mode: str, run_id: str) -> None:
    """Tell the user how to resume an interrupted run."""
    hint = f"To resume this run, use: weco resume {run_id}"
    if output_mode == "plain":
        print(f"\n{hint}\n", flush=True)
    else:
        console.print(f"\n[cyan]{hint}[/]\n")


def _drive_optimization(
    *,
    console: Console,
    auth_headers: dict[str, str],
    run_info: dict,
    artifacts: RunArtifacts,
    originals: dict[str, str],
    dashboard_url: str,
    output_mode: str,
    api_keys: dict[str, str] | None,
) -> OptimizationResult:
    """Run the optimization loop within heartbeat + UI contexts.

    Always returns an :class:`OptimizationResult` — the loop catches its own
    exceptions and reports failure rather than raising. Termination is
    reported to the backend on the way out.
    """
    new_run_id = run_info["id"]
    ui_kwargs = dict(
        run_id=new_run_id,
        run_name=run_info["name"],
        total_steps=run_info["steps"],
        dashboard_url=dashboard_url,
        model=run_info["model"],
        metric_name=run_info["metric_name"],
        maximize=run_info["maximize"],
    )
    if output_mode == "plain":
        ui_instance = PlainOptimizationUI(**ui_kwargs)
    else:
        ui_instance = LiveOptimizationUI(console, **ui_kwargs)

    result: OptimizationResult | None = None
    try:
        with heartbeat(new_run_id, auth_headers), ui_instance as ui:
            # The UI's standard run header (printed by on_init) is the single
            # source of truth for "what run we just created". The parent
            # reference is surfaced via the derived_from payload.
            ui.on_init(derived_from=run_info["derived_from"])

            # The backend has already created the inherited step-0 baseline
            # and generated the step-1 candidate; the loop picks up step 1.
            result = run_optimization_loop(
                ui=ui,
                run_id=new_run_id,
                auth_headers=auth_headers,
                source_code=originals,
                eval_command=run_info["evaluation_command"],
                eval_timeout=run_info["eval_timeout"],
                artifacts=artifacts,
                save_logs=run_info["save_logs"],
                start_step=1,
                api_keys=api_keys,
            )
        return result
    finally:
        if result is not None:
            try:
                report_termination(
                    run_id=new_run_id,
                    status_update=result.status,
                    reason=result.reason,
                    details=result.details,
                    auth_headers=auth_headers,
                )
            except Exception:
                pass


def _run_derived_loop(
    *,
    console: Console,
    auth_headers: dict[str, str],
    run_info: dict,
    originals: dict[str, str],
    dashboard_url: str,
    output_mode: str,
    api_keys: dict[str, str] | None,
) -> bool:
    """Drive the optimization loop and react to its outcome.

    Returns ``True`` iff the loop completed successfully.
    """
    new_run_id = run_info["id"]
    artifacts = RunArtifacts(log_dir=run_info["log_dir"], run_id=new_run_id)

    result = _drive_optimization(
        console=console,
        auth_headers=auth_headers,
        run_info=run_info,
        artifacts=artifacts,
        originals=originals,
        dashboard_url=dashboard_url,
        output_mode=output_mode,
        api_keys=api_keys,
    )

    if result.status == "terminated":
        _print_resume_hint(console, output_mode, new_run_id)

    if result.success:
        offer_apply_best_solution(
            console=console,
            run_id=new_run_id,
            source_code=originals,
            artifacts=artifacts,
            auth_headers=auth_headers,
            apply_change=False,
        )

    return result.success


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
) -> bool:
    """Create a derived run and run the local optimization loop against it.

    Returns ``True`` on successful completion, ``False`` if the loop failed,
    was terminated, or errored out. The dispatcher uses this to set the
    process exit code.
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
    dashboard_url = f"{__dashboard_url__}/runs/{new_run_id}"

    originals = _resolve_originals(run_info["source_code"])

    if output_mode != "plain" and not Confirm.ask(
        "Have the source files in your working directory been kept consistent with "
        "the parent run? Local edits will override the inherited baseline.",
        default=True,
    ):
        console.print("[yellow]Derive cancelled. Adjust your working directory and re-run.[/]")
        return False

    return _run_derived_loop(
        console=console,
        auth_headers=auth_headers,
        run_info=run_info,
        originals=originals,
        dashboard_url=dashboard_url,
        output_mode=output_mode,
        api_keys=api_keys,
    )
