"""``weco slots`` — verify and clean isolated parallel evaluation slots.

``verify`` is the M4 setup skill's proof hook: provision the project's slots
and run the given (fast, smoke-sized) eval command in every slot at the same
time. Both/all must succeed and genuinely overlap in wall-clock time — no
score-parity check, since many evals are stochastic. ``clean`` removes slot
directories left behind by crashed consumers.
"""

from __future__ import annotations

import pathlib
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console

from ..slots import SlotProvisionError, create_slot_provider, find_stale_slot_dirs, clean_stale_slots


def handle_verify(*, eval_command: str, parallel: int, eval_timeout: int, console: Console) -> bool:
    """Provision slots, run the eval concurrently in each, report pass/fail."""
    if parallel < 2:
        console.print("[bold red]--parallel must be at least 2 for a meaningful concurrency check.[/]")
        return False

    try:
        provider = create_slot_provider(pathlib.Path.cwd(), parallel)
    except SlotProvisionError as e:
        console.print(f"[bold red]{e}[/]")
        return False

    console.print(f"[cyan]Provisioning {parallel} slots ({provider.provider_name} provider)...[/]")
    try:
        slots = provider.provision()
    except SlotProvisionError as e:
        console.print(f"[bold red]Slot provisioning failed: {e}[/]")
        return False

    try:
        if len(slots) < 2:
            console.print(f"[bold red]Only {len(slots)} slot(s) could be provisioned; cannot verify concurrency.[/]")
            return False
        if len(slots) < parallel:
            console.print(f"[yellow]Only {len(slots)} of {parallel} slots provisioned; verifying those.[/]")

        def _run(slot):
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    eval_command,
                    shell=True,
                    cwd=slot.cwd,
                    env=slot.full_env(),
                    capture_output=True,
                    text=True,
                    timeout=eval_timeout,
                )
                return slot, started, time.monotonic(), proc.returncode, (proc.stdout or "") + (proc.stderr or "")
            except subprocess.TimeoutExpired:
                return slot, started, time.monotonic(), None, f"timed out after {eval_timeout}s"

        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            results = list(pool.map(_run, slots))

        all_ok = True
        for slot, started, ended, returncode, output in results:
            ok = returncode == 0
            all_ok = all_ok and ok
            status = "[green]PASS[/]" if ok else f"[red]FAIL (exit {returncode})[/]"
            console.print(f"slot {slot.index}: {status}  ({ended - started:.1f}s)  cwd={slot.cwd}")
            if not ok:
                tail = "\n".join(output.strip().splitlines()[-10:])
                console.print(f"[dim]{tail}[/]")

        # Concurrency proof: at least two eval intervals overlapped.
        intervals = sorted((s, e) for _, s, e, _, _ in results)
        overlapped = any(intervals[i + 1][0] < intervals[i][1] for i in range(len(intervals) - 1))
        if not overlapped:
            console.print("[red]Evaluations did not overlap in time — concurrency could not be demonstrated.[/]")
        elif all_ok:
            console.print(f"[bold green]Slots verified: {len(slots)} concurrent evaluations succeeded.[/]")

        return all_ok and overlapped
    finally:
        provider.cleanup()


def handle_clean(*, console: Console, dry_run: bool = False) -> bool:
    """Remove stale slot base dirs whose owning consumer process is gone."""
    stale = find_stale_slot_dirs()
    if not stale:
        console.print("[green]No stale slot directories found.[/]")
        return True
    if dry_run:
        for base, meta in stale:
            console.print(f"would remove {base}  (project: {meta.get('project', 'unknown')})")
        return True
    removed = clean_stale_slots()
    for path in removed:
        console.print(f"removed {path}")
    leftover = len(stale) - len(removed)
    if leftover:
        console.print(f"[yellow]{leftover} directorie(s) could not be removed; check permissions.[/]")
    return leftover == 0
