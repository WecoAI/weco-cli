"""TUI + Claude Agent SDK orchestration for `weco start claude`.

Architecture:

    user prompt (TUI input)        dashboard inject_prompt
                │                              │
                └──────────┬───────────────────┘
                           ▼
                    pending_prompts queue
                           │
                           ▼
              ClaudeSDKClient (persistent)
                           │            ▲
        ┌──────────────────┼────────────┘
        │                  │
        │              can_use_tool ─► ApprovalRouter ─► outbound (dashboard)
        │                  ▲                                        │
        │                  │                                        ▼
        │              resolve(id, …) ◄────── on_inbound ◄── channel relay
        │                                                           │
        │  ┌─────────────────────────────────────────── inline ApprovalCard
        │  ▼                                                       (local TUI)
        │  app.mount_inline_card(...)   ◄── same picker as AskUserQuestion
        │
        ▼
   async for message ─► envelope + render to TUI (Renderer)

Companion modules:
  * ``relay``           — session create / Realtime channel / heartbeat.
  * ``sdk_config``      — claude_args parsing, billing env, system prompt.
  * ``envelopes``       — SDK message → dashboard transcript conversion.
  * ``rendering``       — SDK message → local TUI post_* calls.
  * ``approval_router`` — can_use_tool ↔ dashboard + local-modal routing.

Billing: when ``--billing weco`` is passed, the SDK's HTTP traffic is
pointed at the Weco LLM proxy (see ``sdk_config``).
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
from typing import Any, Callable, Optional

from rich.console import Console

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, CLINotFoundError, HookMatcher
from claude_agent_sdk.types import ResultMessage, ToolResultBlock, UserMessage

from weco.ui.tui import WecoTUI
from weco.ui.tui.approval import ApprovalCard
from weco.ui.tui.question import QuestionCard

from .approval_router import ApprovalRouter
from .session import DashboardSession, SetupError
from .envelopes import envelope_for, is_synthetic_interrupt_message, render_ui_context_preamble, stringify_tool_result
from .rendering import Renderer
from .run_watcher import RunWatcher, find_run_ids
from .sdk_config import (
    VALID_EFFORTS,
    WECO_SYSTEM_PROMPT_APPEND,
    build_sdk_env,
    keep_stream_open_hook,
    parse_claude_args,
    resolve_model,
)


# Wire protocol -------------------------------------------------------------
META_TYPE = "_weco_meta"  # envelope `type` for Weco's out-of-band control traffic
AGENT_TYPE = "claude-code"  # session kind registered with the Weco API
# Keys the dashboard may bundle as per-turn UI context on `inject_prompt`.
UI_CONTEXT_KEYS = ("run_id", "lineage_id", "step", "best_metric", "summary", "url")
# Tool-approval decisions the dashboard is allowed to send.
VALID_DECISIONS = ("approve", "deny", "ask")

# Teardown / buffering knobs ------------------------------------------------
STDERR_BUFFER_LINES = 64  # tail of SDK-subprocess stderr kept for crash diagnostics
TASK_SHUTDOWN_TIMEOUT_S = 2.0  # grace for relay tasks to drain on shutdown
RUN_STOP_TIMEOUT_S = 5.0  # grace for a `weco run stop` subprocess on exit

# System prompt handed to the SDK: claude's own Claude Code preset plus our
# Weco-bridge delta (append, don't replace — that keeps native behaviour).
SYSTEM_PROMPT = {"type": "preset", "preset": "claude_code", "append": WECO_SYSTEM_PROMPT_APPEND}


# ---------------------------------------------------------------------------
# Public entry point — called from cli.py.
# ---------------------------------------------------------------------------


def run_tui_bridge(
    *,
    claude_args: list[str],
    api_key: str,
    console: Console,
    billing: str = "claude",
    weco_api_base: Optional[str] = None,
    effort: Optional[str] = None,
    seed_prompt: Optional[str] = None,
) -> int:
    """Boot the TUI and the SDK-driven orchestrator."""
    try:
        session = DashboardSession.create(api_key=api_key, agent_type=AGENT_TYPE)
    except SetupError as e:
        console.print(f"[red]Could not create dashboard session:[/] {e}")
        console.print("[yellow]Falling back to a plain local Claude Code session.[/]")
        session = DashboardSession.offline()

    app = WecoTUI()
    orchestrator = Orchestrator(
        app=app,
        claude_args=claude_args,
        api_key=api_key,
        session=session,
        billing=billing,
        weco_api_base=weco_api_base,
        effort=effort,
        seed_prompt=seed_prompt,
    )

    app.set_submit_callback(orchestrator.on_user_submit)
    app.set_startup_callback(lambda _app: orchestrator.run())
    app.set_preempt_callback(orchestrator.on_user_preempt)
    app.set_exit_with_stop_callback(orchestrator.on_exit_stop_runs)

    app.run()
    return orchestrator.exit_code


def run_headless_bridge(
    *,
    claude_args: list[str],
    api_key: str,
    console: Console,
    billing: str = "claude",
    weco_api_base: Optional[str] = None,
    effort: Optional[str] = None,
    seed_prompt: Optional[str] = None,
) -> int:
    """Run the SDK-driven orchestrator with NO local TUI — the bridge streams to
    the dashboard and key lifecycle lines print to the console. This is the mode
    an agent launches in the background (`weco start claude --headless`): there's
    no terminal to draw a Textual app into, so the dashboard is the only
    interactive surface. Pair with `--allow-tools` (no local approval modal) and
    `--prompt` to seed the first turn."""
    try:
        session = DashboardSession.create(api_key=api_key, agent_type=AGENT_TYPE)
    except SetupError as e:
        console.print(f"[red]Could not create dashboard session:[/] {e}")
        console.print(
            "[yellow]Headless mode relies on the dashboard relay for interactivity — continuing offline; "
            "the session will only run the seeded prompt.[/]"
        )
        session = DashboardSession.offline()

    app = HeadlessUI(console)
    orchestrator = Orchestrator(
        app=app,
        claude_args=claude_args,
        api_key=api_key,
        session=session,
        billing=billing,
        weco_api_base=weco_api_base,
        effort=effort,
        seed_prompt=seed_prompt,
    )

    try:
        asyncio.run(orchestrator.run_until_stopped())
    except KeyboardInterrupt:
        pass
    return orchestrator.exit_code


# ---------------------------------------------------------------------------
# HeadlessUI — a no-TTY stand-in for WecoTUI used by `--headless`.
# ---------------------------------------------------------------------------


class HeadlessUI:
    """Duck-typed replacement for ``WecoTUI`` when there's no terminal to draw
    into. The dashboard relay carries the full interactive transcript; here we
    just print a readable lifecycle log to the console (banner, system notices,
    tool calls, run updates) so a backgrounded session leaves a useful trail.

    Everything visual — token-stream deltas, the thinking spinner, inline
    approval/question cards — is a no-op, supplied by ``__getattr__`` so the
    Orchestrator and Renderer can call the full WecoTUI surface unchanged. With
    ``--allow-tools`` the SDK bypasses approvals, so the cards never fire; without
    it, ``mount_inline_card`` no-ops locally and the dashboard resolves them."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def post_banner(
        self,
        *,
        agent: str = "Claude Code",
        model: Optional[str] = None,
        billing: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        bits = [agent]
        if model:
            bits.append(f"model={model}")
        if billing:
            bits.append(f"billing={billing}")
        if session_id:
            bits.append(f"session={session_id}")
        self._console.print(f"[bold cyan]{' · '.join(bits)}[/]")

    def post_system_notice(self, text: str, *, style: str = "dim italic") -> None:
        self._console.print(text, style=style, markup=False, highlight=False)

    def post_user_message(self, text: str) -> None:
        self._console.print(f"› {text}", style="dim", markup=False, highlight=False)

    def post_tool_use(self, tool_id: str, name: str, summary: str, tool_input: dict) -> None:
        line = f"[tool] {name}" + (f" — {summary}" if summary else "")
        self._console.print(line, style="dim", markup=False, highlight=False)

    def post_run_update(self, update: dict) -> None:
        if not isinstance(update, dict):
            return
        rid = str(update.get("run_id", "") or "")
        parts = [f"run {rid[:8]}"] if rid else ["run"]
        status = update.get("status")
        if status:
            parts.append(str(status))
        step = update.get("step", update.get("current_step"))
        if step is not None:
            total = update.get("total_steps")
            parts.append(f"step {step}" + (f"/{total}" if total else ""))
        best = update.get("best_metric")
        if best is not None:
            parts.append(f"best={best}")
        self._console.print("● " + " · ".join(parts), style="cyan", markup=False, highlight=False)

    def post_turn_end(self, subtype: str, *, cost: Optional[float] = None, duration_ms: Optional[int] = None) -> None:
        suffix = f" (${cost:.4f} api equiv)" if isinstance(cost, (int, float)) else ""
        style = "dim" if subtype == "success" else "bold yellow"
        self._console.print(f"— {subtype}{suffix} —", style=style, markup=False, highlight=False)

    def mount_inline_card(self, card: Any) -> None:
        # No local surface — leave the approval/question card unresolved for the
        # dashboard to answer. (With --allow-tools the SDK never asks.)
        return

    def exit(self) -> None:
        return

    def __getattr__(self, _name: str) -> Callable[..., None]:
        # Everything else WecoTUI exposes (post_assistant_delta, show_thinking,
        # hide_thinking, end_assistant_block, post_tool_result, …) is a visual
        # no-op without a terminal.
        return lambda *a, **k: None


# ---------------------------------------------------------------------------
# Orchestrator — owns the SDK client + WS relay + run watcher + approvals.
# ---------------------------------------------------------------------------


class Orchestrator:
    def __init__(
        self,
        *,
        app: WecoTUI,
        claude_args: list[str],
        api_key: str,
        session: DashboardSession,
        billing: str,
        weco_api_base: Optional[str],
        effort: Optional[str],
        seed_prompt: Optional[str] = None,
    ) -> None:
        self.app = app
        self.claude_args = claude_args
        self.api_key = api_key
        self._session = session
        self.billing = billing
        self.weco_api_base = weco_api_base
        self.effort = effort
        # Optional first-turn prompt (from `--prompt`). Enqueued once the SDK
        # client is connected so it drives the session without a typed input —
        # required for headless launches, handy for the TUI.
        self._seed_prompt = seed_prompt

        self.session_id: Optional[str] = session.id
        self.dashboard_url: Optional[str] = session.dashboard_url

        self.exit_code: int = 0
        self._renderer = Renderer(app)
        self._stop_event = asyncio.Event()
        # (prompt_text, ui_context_or_none). Dashboard-submitted prompts ride
        # with the dashboard's current view snapshot bundled in the
        # inject_prompt payload; TUI prompts carry None.
        # (prompt, ui_context, echo_user). `echo_user=False` suppresses the
        # user-bubble mirror to the dashboard transcript — used for structured
        # requests (e.g. derive_request) that surface as their own meta card.
        self._pending_prompts: asyncio.Queue[tuple[str, Optional[dict], bool]] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._run_watcher: Optional[RunWatcher] = None
        self._approval_router: Optional[ApprovalRouter] = None
        # Persistent SDK client. One long-lived consumer drains the SDK message
        # stream (see `_consume_messages`); prompts are fed in separately via
        # `_prompt_pump`/query(). Cancel paths (SIGINT, dashboard 'interrupt',
        # mid-turn inject_prompt preemption, local Esc) send client.interrupt()
        # and let the consumer drain the interrupted turn's tail in order.
        self._sdk_client: Optional[ClaudeSDKClient] = None
        # Count of submitted prompt-turns not yet terminated by a ResultMessage.
        # > 0 means a user-prompted turn is in flight (so a preempt should
        # interrupt). Incremented on a successful query(), decremented by the
        # consumer on each ResultMessage.
        self._inflight_turns: int = 0
        # Set while an SDK interrupt is mid-flight and we're letting the
        # in-flight turn drain naturally. Lets repeated cancel attempts
        # early-out so a second SIGINT escalates to "exit" instead of
        # re-firing a no-op interrupt. Cleared by the consumer when it sees the
        # interrupted turn's terminal ResultMessage.
        self._interrupting: bool = False
        # request_id → inline card, so the bridge can dismiss the local card
        # when the dashboard answers first OR when the SDK turn is cancelled.
        # The `_inflight` sets de-dup a single approval bouncing through
        # can_use_tool more than once.
        self._approval_card_inflight: set[str] = set()
        self._active_approval_cards: dict[str, Any] = {}
        self._question_card_inflight: set[str] = set()
        self._active_question_cards: dict[str, Any] = {}
        # SDK subprocess stderr tail — the TUI owns the terminal so the raw
        # `claude` child's stderr is invisible; we surface it via system
        # notices on connect/run failures.
        self._sdk_stderr_buf: list[str] = []

    # --- Meta-envelope helper -------------------------------------------

    def _emit_meta(self, event: str, **fields: Any) -> None:
        """Publish a Weco control/meta envelope to the dashboard."""
        self._session.publish(json.dumps({"type": META_TYPE, "event": event, **fields}))

    # --- App startup callback -------------------------------------------

    def _post_banner(self) -> None:
        self.app.post_banner(
            agent="Claude Code", model=resolve_model(self.claude_args), billing=self.billing, session_id=self.session_id
        )
        if not self.dashboard_url:
            self.app.post_system_notice("Running without dashboard relay.", style="dim yellow")

    def run(self) -> None:
        """Top-level startup callback. Scheduled by WecoTUI once the app
        is mounted. Post the banner first so it's visible before any async
        lifecycle work, then kick off the orchestrator."""
        self._post_banner()
        asyncio.create_task(self._run())

    async def run_until_stopped(self) -> None:
        """Headless entry point. Posts the banner, then runs the full lifecycle
        to completion (blocks until the session stops). The TUI path uses the
        sync `run()` startup callback instead, which schedules `_run()` as a task
        inside Textual's own event loop."""
        self._post_banner()
        await self._run()

    async def _run(self) -> None:
        try:
            await self._setup_and_loop()
        finally:
            await self._shutdown()

    async def _setup_and_loop(self) -> None:
        self._emit_meta("claude_session_started")

        self._run_watcher = RunWatcher(
            weco_bin=shutil.which("weco"), notify=self._notify_run_update, stop_event=self._stop_event
        )

        self._approval_router = ApprovalRouter(
            publish=self._session.publish,
            on_approval_request=self._on_approval_request,
            on_question_request=self._on_question_request,
        )

        # No-op for an offline session (returns immediately).
        self._tasks.append(
            asyncio.create_task(self._session.run(on_inbound=self._handle_inbound, stop_event=self._stop_event))
        )

        # SIGINT handling: 1st press cancels the in-flight turn; 2nd press
        # within the window exits and stops watched runs. The TUI's ctrl_c
        # handler owns the primary UX (set_exit_with_stop_callback); this
        # only fires if the user is somehow outside the TUI capture.
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGINT, self._on_sigint)
        except (NotImplementedError, RuntimeError):
            pass

        self._sdk_client = ClaudeSDKClient(options=self._build_options())
        try:
            await self._sdk_client.connect()
        except CLINotFoundError:
            # Backstop for the pre-flight `which claude` check (start/cli.py):
            # covers the binary vanishing between check and launch.
            self.app.post_system_notice(
                "Claude Code CLI not found — install it "
                "(https://code.claude.com/docs/en/quickstart), then re-run `weco start claude`.",
                style="bold red",
            )
            self._stop_event.set()
            return
        except Exception as e:
            self.app.post_system_notice(f"Could not connect to Claude SDK: {e}", style="bold red")
            # The SDK's wrapped error usually says "Check stderr output for
            # details" — surface the captured stderr so the user can diagnose
            # without --debug-stderr.
            stderr_blob = "".join(self._sdk_stderr_buf).strip()
            if stderr_blob:
                self.app.post_system_notice(f"claude stderr:\n{stderr_blob}", style="dim red")
            self._stop_event.set()
            return

        # Start the single long-lived stream consumer, then feed prompts. The
        # consumer fans out every message (prompted AND between-turn) the instant
        # the SDK surfaces it; the pump only submits query()s. This is the SDK's
        # blessed streaming pattern and is what keeps autonomous output (e.g.
        # background-task completions, scheduled wakeups) from being stranded in
        # the transport until the next prompt drains it.
        self._tasks.append(asyncio.create_task(self._consume_messages()))
        # Seed the first turn (from `--prompt`) now that the consumer is live and
        # the SDK is connected. The pump drains it on its first iteration.
        if self._seed_prompt:
            await self._pending_prompts.put((self._seed_prompt, None, True))
        await self._prompt_pump()

    def _build_options(self) -> ClaudeAgentOptions:
        parsed_args, extra_args = parse_claude_args(self.claude_args)
        effort = self.effort or parsed_args.get("effort")
        permission_mode = "bypassPermissions" if parsed_args.get("dangerously_skip_permissions") else None
        sdk_env = build_sdk_env(
            billing=self.billing, api_key=self.api_key, weco_api_base=self.weco_api_base, session_id=self.session_id
        )
        return ClaudeAgentOptions(
            can_use_tool=self._approval_router.can_use_tool,
            # PreToolUse no-op hook keeps the SDK's input channel open long
            # enough for can_use_tool to fire — documented SDK workaround.
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[keep_stream_open_hook])]},
            env=sdk_env,
            model=resolve_model(self.claude_args),
            effort=effort if effort in VALID_EFFORTS else None,  # type: ignore[arg-type]
            permission_mode=permission_mode,  # type: ignore[arg-type]
            extra_args=extra_args,
            # Token-level streaming. The SDK yields `StreamEvent` objects
            # alongside the usual message stream; the Renderer consumes their
            # `content_block_delta` text deltas in real time.
            include_partial_messages=True,
            stderr=self._capture_sdk_stderr,
            # Anchor the agent at the directory the user launched from —
            # without this the SDK's default cwd grounds claude's project-root
            # detection above the user's actual project.
            cwd=os.getcwd(),
            system_prompt=SYSTEM_PROMPT,  # type: ignore[arg-type]
        )

    async def _shutdown(self) -> None:
        self._stop_event.set()
        if self._sdk_client is not None:
            try:
                await self._sdk_client.disconnect()
            except Exception:
                pass
            self._sdk_client = None
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await asyncio.wait_for(task, timeout=TASK_SHUTDOWN_TIMEOUT_S)
            except (asyncio.CancelledError, Exception):
                pass
        if self._run_watcher is not None:
            await self._run_watcher.stop()

    # --- User → bridge ---------------------------------------------------

    async def on_user_submit(self, text: str) -> None:
        if not text:
            return
        if self._cancel_current_turn():
            self.app.post_system_notice("(interrupting current turn — your prompt will run next)", style="dim yellow")
        # TUI prompt — no dashboard context (the user is typing into the
        # terminal, not looking at the dashboard).
        await self._pending_prompts.put((text, None, True))

    def on_user_preempt(self) -> None:
        if self._cancel_current_turn():
            self.app.hide_thinking()
            self.app.post_system_notice("(interrupted)", style="dim yellow")

    async def on_exit_stop_runs(self) -> None:
        """Second Ctrl-C — stop every active `weco run` before tearing down."""
        run_ids = self._run_watcher.watching() if self._run_watcher is not None else []
        if not run_ids:
            return
        weco_bin = shutil.which("weco")
        if not weco_bin:
            return
        self.app.post_system_notice(f"Stopping {len(run_ids)} active run(s)…", style="dim yellow")
        await asyncio.gather(*(self._stop_one_run(weco_bin, rid) for rid in run_ids), return_exceptions=True)

    async def _stop_one_run(self, weco_bin: str, run_id: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                weco_bin, "run", "stop", run_id, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(proc.wait(), timeout=RUN_STOP_TIMEOUT_S)
        except Exception:
            pass

    def _on_sigint(self) -> None:
        # First-press fallback path. The TUI's two-tier handler owns the
        # primary UX; this only fires outside the TUI capture.
        if not self._cancel_current_turn():
            self._stop_event.set()
            self.app.exit()

    def _cancel_current_turn(self) -> bool:
        """Interrupt the in-flight prompt-turn (if any). Returns True if one was
        running and we kicked off an interrupt, False otherwise."""
        if self._inflight_turns <= 0:
            return False
        if self._interrupting:
            # Already draining a prior interrupt — let the in-flight SDK
            # interrupt finish. Returning False lets the caller escalate.
            return False
        client = self._sdk_client
        if client is not None:
            # Send the SDK an interrupt control message; the single consumer
            # drains the interrupted turn's tail in order — the SDK's synthetic
            # `[Request interrupted]` user-message (skipped) and its terminal
            # ResultMessage (swallowed in `_fan_out`, which also clears
            # `_interrupting`). A queued follow-up prompt then submits cleanly:
            # because one in-order consumer owns the stream, there's no stale
            # buffered ResultMessage for a new receive loop to trip over.
            self._interrupting = True
            asyncio.create_task(_safe_interrupt(client))
        # Tell consumers we're interrupting now — the user already moved on.
        self._emit_meta("turn_interrupted")
        self.app.hide_thinking()
        # Tear down pending cards — the turn they gated on is gone.
        self._dismiss_active_cards()
        return True

    def _dismiss_active_cards(self) -> None:
        """Freeze every pending approval/question card. `resolve_remotely({})`
        is a no-op on an already-answered card; ApprovalCard reads it as
        deny-once."""
        for cards in (self._active_approval_cards, self._active_question_cards):
            for card in list(cards.values()):
                try:
                    card.resolve_remotely({})
                except Exception:
                    pass
            cards.clear()

    # --- Dashboard → bridge ---------------------------------------------

    async def _handle_inbound(self, msg: str) -> None:
        try:
            event = json.loads(msg)
        except Exception:
            return
        if not isinstance(event, dict):
            return
        handler = {
            "inject_prompt": self._inbound_inject_prompt,
            "approval_response": self._inbound_approval_response,
            "question_response": self._inbound_question_response,
            "interrupt": self._inbound_interrupt,
            "derive_request": self._inbound_derive_request,
        }.get(event.get("type"))
        if handler is not None:
            await handler(event)

    async def _inbound_inject_prompt(self, event: dict) -> None:
        text = event.get("text")
        if not isinstance(text, str) or not text:
            return
        self._cancel_current_turn()
        self.app.post_user_message(text)
        # Dashboard-initiated actions (e.g. "Explore a new path" derives) list
        # new runs as `Run ID: <uuid>` lines — watch them mechanically so
        # terminal/dashboard pings don't depend on the agent acting.
        if self._run_watcher is not None:
            for run_id in find_run_ids(text):
                self._run_watcher.watch(run_id)
        # Optional `context` rides with the prompt — the dashboard bundles its
        # current view snapshot so the agent has context for THIS turn without
        # a continuous `ui_context` broadcast.
        raw_ctx = event.get("context")
        ctx = {k: v for k, v in raw_ctx.items() if k in UI_CONTEXT_KEYS} if isinstance(raw_ctx, dict) else None
        await self._pending_prompts.put((text, ctx, True))

    async def _inbound_derive_request(self, event: dict) -> None:
        """Dashboard "Explore a new path" — a structured request to launch
        derived runs. Surfaces as a `_weco_meta:derive_request` transcript card
        (not a user chat bubble); the agent receives full launch instructions.
        """
        paths = parse_derive_paths(event)
        run_id = event.get("run_id")
        if not isinstance(run_id, str) or not run_id or not paths:
            return
        self._cancel_current_turn()
        # Structured transcript card for the dashboard (+ scrollback replay).
        self._emit_meta("derive_request", run_id=run_id, paths=paths)
        noun = "path" if len(paths) == 1 else "paths"
        self.app.post_system_notice(f"Dashboard derive request: {len(paths)} {noun} — launching…", style="dim cyan")
        await self._pending_prompts.put((build_derive_prompt(run_id, paths), None, False))

    async def _inbound_approval_response(self, event: dict) -> None:
        request_id = event.get("id")
        decision = event.get("decision")
        scope = str(event.get("scope", "once"))
        if not (isinstance(request_id, str) and decision in VALID_DECISIONS and self._approval_router is not None):
            return
        # If the local TUI also has a card for this id, tell it the answer
        # landed on the dashboard so it stops awaiting. Do this *before*
        # resolving the router so the card transitions cleanly even if the
        # router resolution triggers a fast SDK callback.
        card = self._active_approval_cards.pop(request_id, None)
        if card is not None:
            try:
                card.resolve_remotely_decision(decision, scope)
            except Exception:
                pass
        self._approval_router.resolve(request_id, decision, scope)

    async def _inbound_question_response(self, event: dict) -> None:
        request_id = event.get("id")
        answers = event.get("answers")
        if not (isinstance(request_id, str) and isinstance(answers, dict) and self._approval_router is not None):
            return
        card = self._active_question_cards.pop(request_id, None)
        if card is not None:
            try:
                card.resolve_remotely(answers)
            except Exception:
                pass
        self._approval_router.resolve_question(request_id, answers)

    async def _inbound_interrupt(self, event: dict) -> None:
        if self._cancel_current_turn():
            self.app.hide_thinking()
            self.app.post_system_notice("(interrupted from dashboard)", style="dim yellow")

    # --- Approval router → local modal ----------------------------------
    #
    # An approval and an AskUserQuestion ride the same inline-picker component
    # and the same cross-surface race: whichever surface (local TUI card or
    # dashboard) answers first resolves the SDK call; the loser is de-duped.
    # `_drive_inline_card` owns that shared lifecycle; the two `_on_*_request`
    # entry points just supply the card, the fallback, and the wire shapes.

    def _on_approval_request(self, request_id: str, tool_name: str, summary: str, tool_input: dict[str, Any]) -> None:
        if request_id in self._approval_card_inflight:
            return
        self._approval_card_inflight.add(request_id)

        async def task() -> None:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            await self._drive_inline_card(
                request_id=request_id,
                inflight=self._approval_card_inflight,
                active=self._active_approval_cards,
                card=ApprovalCard(tool_name, summary, future),
                future=future,
                fallback=("deny", "once"),
                resolve_router=lambda result: self._approval_router.resolve(request_id, *result),
                response_event="approval_response",
                response_fields=lambda result: {"decision": result[0], "scope": result[1]},
            )

        asyncio.create_task(task())

    def _on_question_request(self, request_id: str, questions: list[dict]) -> None:
        if request_id in self._question_card_inflight:
            return
        self._question_card_inflight.add(request_id)

        async def task() -> None:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            await self._drive_inline_card(
                request_id=request_id,
                inflight=self._question_card_inflight,
                active=self._active_question_cards,
                card=QuestionCard(questions, future),
                future=future,
                fallback={},
                resolve_router=lambda result: self._approval_router.resolve_question(request_id, result),
                response_event="question_response",
                response_fields=lambda result: {"answers": result},
            )

        asyncio.create_task(task())

    async def _drive_inline_card(
        self,
        *,
        request_id: str,
        inflight: set[str],
        active: dict[str, Any],
        card: Any,
        future: asyncio.Future,
        fallback: Any,
        resolve_router: Callable[[Any], None],
        response_event: str,
        response_fields: Callable[[Any], dict],
    ) -> None:
        """Mount an inline picker, await an answer from any surface, then
        resolve the SDK call and broadcast the answer to the dashboard."""
        active[request_id] = card
        try:
            self.app.mount_inline_card(card)
        except Exception:
            # Mount failed → resolve with the fallback so the await below
            # returns instead of hanging the SDK call on a card that never
            # appeared.
            if not future.done():
                future.set_result(fallback)

        try:
            result = await future
        except asyncio.CancelledError:
            result = fallback
        finally:
            active.pop(request_id, None)
            inflight.discard(request_id)

        if self._approval_router is None:
            return
        # Resolve the SDK side first so claude can continue, then broadcast so
        # the dashboard's card dismisses. If the dashboard answered first both
        # are no-ops (router de-dupes; the dashboard already dropped its card).
        resolve_router(result)
        self._emit_meta(response_event, id=request_id, source="cli", **response_fields(result))

    def _capture_sdk_stderr(self, line: str) -> None:
        """Sink for the SDK subprocess's stderr. The TUI owns the terminal so
        raw stderr is invisible; we keep the tail for the connect/run error
        paths to flush into a system notice."""
        if not isinstance(line, str):
            return
        self._sdk_stderr_buf.append(line)
        if len(self._sdk_stderr_buf) > STDERR_BUFFER_LINES:
            del self._sdk_stderr_buf[:-STDERR_BUFFER_LINES]

    # --- Run watcher → app + dashboard ----------------------------------

    def _notify_run_update(self, update: dict) -> None:
        if not isinstance(update, dict):
            return
        # Both surfaces are best-effort and independently guarded: a failure
        # publishing to the dashboard must not stop the local render, and a
        # render error must not bubble out of the run-watcher's notify
        # callback — that would kill the polling task and silently freeze
        # *all* further updates (the watcher is shared across surfaces).
        try:
            self._emit_meta("run_update", **update)
        except Exception:
            pass
        try:
            self.app.post_run_update(update)
        except Exception:
            pass

    # --- Stream consumer + prompt pump -----------------------------------

    async def _consume_messages(self) -> None:
        """Single long-lived consumer of the SDK message stream — the SDK's
        blessed streaming pattern (one `receive_messages()` owner; prompts fed
        via `query()` from `_prompt_pump`). Fans out EVERY message the instant
        the SDK surfaces it, whether it belongs to a user-prompted turn or to a
        between-turn autonomous one (background-task completions, scheduled
        wakeups). The old per-turn `receive_response()` drained only while a
        prompt was in flight, so messages produced between turns sat buffered in
        the transport until the next prompt flushed them — appearing late, in a
        burst, on both the TUI and dashboard at once."""
        client = self._sdk_client
        if client is None:
            return
        try:
            async for message in client.receive_messages():
                if self._stop_event.is_set():
                    return
                self._fan_out(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The receive stream died (transport/SDK error). Nothing can be
            # consumed after this — surface it (unless we're already shutting
            # down, where disconnect() ends the stream by design) and bring the
            # session down so the prompt pump unblocks and `_shutdown` runs.
            if not self._stop_event.is_set():
                self.app.post_system_notice(f"Claude SDK error: {e}", style="bold red")
                stderr_blob = "".join(self._sdk_stderr_buf).strip()
                if stderr_blob:
                    self.app.post_system_notice(f"claude stderr:\n{stderr_blob}", style="dim red")
                self._emit_meta("claude_error", message=str(e), stderr=stderr_blob or None)
        finally:
            self._stop_event.set()

    def _fan_out(self, message: Any) -> None:
        """Render one SDK message to every surface — dashboard mirror, run
        watcher, local TUI. Each side effect is independently guarded so one
        failure can't abandon the consumer loop (which would strand every later
        message). Also owns turn-boundary bookkeeping (`_inflight_turns`,
        `_interrupting`), now that the boundary is observed in the stream rather
        than by a per-turn receive loop ending."""
        # While interrupting, the SDK emits a synthetic UserMessage
        # ("[Request interrupted by user]") then a final ResultMessage. Drop the
        # synthetic message; the ResultMessage is handled just below.
        if is_synthetic_interrupt_message(message):
            return
        if isinstance(message, ResultMessage):
            # Terminal message of a turn: settle the in-flight count, and if this
            # closes an interrupted turn, swallow it — `turn_interrupted` already
            # signalled the early end and its `error_during_execution` would read
            # like a bug the user caused themselves.
            self._inflight_turns = max(0, self._inflight_turns - 1)
            if self._interrupting:
                self._interrupting = False
                return

        try:
            envelope = envelope_for(message)
            if envelope is not None:
                self._session.publish(json.dumps(envelope))
        except Exception:
            pass

        try:
            if self._run_watcher is not None:
                scan_for_run_ids(message, self._run_watcher)
        except Exception:
            pass

        try:
            self._renderer.render(message)
        except Exception:
            # Don't let a render bug derail the session.
            pass

    async def _prompt_pump(self) -> None:
        """Feed queued prompts into the SDK. Response *consumption* is owned by
        `_consume_messages`; this only submits query()s. Returns when the session
        is stopping."""
        while not self._stop_event.is_set():
            entry = await self._next_prompt()
            if entry is None:
                return
            await self._submit_prompt(entry)

    async def _next_prompt(self) -> Optional[tuple[str, Optional[dict], bool]]:
        stop_task = asyncio.create_task(self._stop_event.wait())
        get_task = asyncio.create_task(self._pending_prompts.get())
        done, _ = await asyncio.wait({stop_task, get_task}, return_when=asyncio.FIRST_COMPLETED)
        if get_task in done:
            stop_task.cancel()
            return get_task.result()
        get_task.cancel()
        return None

    async def _submit_prompt(self, prompt_entry: tuple[str, Optional[dict], bool]) -> None:
        prompt, ctx, echo_user = prompt_entry
        client = self._sdk_client
        if client is None:
            return

        # Mirror the user's prompt to the dashboard transcript first so both
        # surfaces see it before the agent starts thinking. Structured requests
        # (echo_user=False) already surfaced as their own meta card.
        if echo_user:
            self._session.publish(json.dumps({"type": "user", "message": {"role": "user", "content": prompt}}))

        # A dashboard prompt may carry a UI-view snapshot; prepend it as a
        # hidden system reminder so the agent can answer "what am I looking
        # at?" without the user spelling it out. TUI prompts carry ctx=None.
        outgoing = render_ui_context_preamble(ctx) + prompt

        self.app.show_thinking()
        # Count the turn in flight BEFORE query() returns, so a fast first
        # message can never decrement past it. The consumer settles it on the
        # turn's ResultMessage.
        self._inflight_turns += 1
        try:
            await client.query(outgoing)
        except Exception as e:
            self._inflight_turns = max(0, self._inflight_turns - 1)
            self.app.hide_thinking()
            self.app.post_system_notice(f"Claude SDK error: {e}", style="bold red")
            stderr_blob = "".join(self._sdk_stderr_buf).strip()
            if stderr_blob:
                self.app.post_system_notice(f"claude stderr:\n{stderr_blob}", style="dim red")
            self._emit_meta("claude_error", message=str(e), stderr=stderr_blob or None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _safe_interrupt(client: ClaudeSDKClient) -> None:
    try:
        await client.interrupt()
    except Exception:
        pass


def parse_derive_paths(event: dict) -> list[dict]:
    """Validate a derive_request's `paths` into `{node_id, step?, instructions?, steps?}` dicts."""
    raw = event.get("paths")
    if not isinstance(raw, list):
        return []
    paths: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        path: dict = {"node_id": node_id}
        if isinstance(item.get("step"), (int, float)):
            path["step"] = item["step"]
        instructions = item.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            path["instructions"] = instructions.strip()
        # Per-path step-count override (`weco run derive -n`); inherit when absent.
        steps = item.get("steps")
        if isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1:
            path["steps"] = steps
        paths.append(path)
    return paths


def build_derive_prompt(run_id: str, paths: list[dict]) -> str:
    """Agent instructions for a dashboard derive request.

    One `weco run derive` per path: the first command becomes the lineage's
    evaluation consumer (working-tree lock); the rest attach and queue behind
    it — so this works whether or not a weco process is already running.
    """
    commands = []
    for path in paths:
        cmd = f"weco run derive {run_id} --from-step {path['node_id']}"
        instructions = path.get("instructions")
        if instructions:
            cmd += f" -i {shlex.quote(instructions)}"
        steps = path.get("steps")
        if steps:
            cmd += f" -n {steps}"
        cmd += " --output plain"
        commands.append(cmd)
    return (
        f"[Dashboard derive request] The user composed {len(paths)} derived run(s) from run {run_id} "
        f'in the dashboard ("Explore a new path"). Launch each command below now, each as a background '
        f"task (run_in_background: true) — the first becomes the evaluation consumer for the lineage and "
        f"the rest attach behind it, so do not wait for one to finish before starting the next:\n\n"
        + "\n".join(commands)
        + "\n\nThen briefly confirm, report the new run IDs, and add them ALL to your monitoring loop "
        "(poll `weco run status <run-id>` for each, non-blocking; report new bests and completions)."
    )


def scan_for_run_ids(message: Any, run_watcher: RunWatcher) -> None:
    """Spot `Run ID: <uuid>` in tool_result text and start polling.

    The run watcher's notify callback surfaces status updates to the UI via
    `Orchestrator._notify_run_update`, so no app reference is needed here.
    """
    if not isinstance(message, UserMessage):
        return
    blocks = message.content if isinstance(message.content, list) else []
    for block in blocks:
        if isinstance(block, ToolResultBlock):
            text = stringify_tool_result(block.content)
            if text:
                for run_id in find_run_ids(text):
                    run_watcher.watch(run_id)
