"""
orchestrator_client.py — Orchestrator WebSocket client for the avatar agent.

Registers the `talk_to_avatar` capability so other agents (e.g. the planner)
can push messages to all active avatar sessions.

Also handles `settings_push` to forward common_settings to the session manager,
and `prompt_push` to hot-reload the conversation system prompt.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

AGENT_NAME = "avatar-agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Human-like animated avatar interface for conversational AI interactions"
HEARTBEAT_INTERVAL = 15

_ID_FILE = Path(".agent_id")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_agent_id() -> str:
    if _ID_FILE.exists():
        return _ID_FILE.read_text().strip()
    agent_id = str(uuid.uuid4())
    _ID_FILE.write_text(agent_id)
    logger.info("Generated new stable agent ID: %s (saved to %s)", agent_id, _ID_FILE)
    return agent_id


class OrchestratorClient:
    """Manages registration + WebSocket lifecycle with the orchestrator."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base = base_url.rstrip("/")
        self.agent_id: Optional[str] = None
        self.ws_url: Optional[str] = None
        self._ws: Optional[Any] = None
        self._running = False
        self._stop_event = asyncio.Event()
        self._pending: dict[str, asyncio.Future] = {}
        self._active_tasks = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._start_time = time.monotonic()
        self._common_settings: dict = {}

        # Callbacks
        self._on_settings_push: Optional[Callable[[dict], None]] = None
        self._on_avatar_message: Optional[Callable[[str], Any]] = None
        self._on_prompt_push: Optional[Callable[[str], None]] = None

    def on_settings_push(self, cb: Callable[[dict], None]) -> None:
        self._on_settings_push = cb

    def on_avatar_message(self, cb: Callable[[str], Any]) -> None:
        """Called when talk_to_avatar task_request arrives."""
        self._on_avatar_message = cb

    def on_prompt_push(self, cb: Callable[[str], None]) -> None:
        """Called when a prompt_push message arrives with new system prompt content."""
        self._on_prompt_push = cb

    # ── Registration ─────────────────────────────────────────────────────────

    async def register(self) -> str:
        _prompt_file = Path(__file__).parent / "prompts" / "system_prompt.md"
        default_prompt = _prompt_file.read_text(encoding="utf-8").strip() if _prompt_file.exists() else ""

        payload = {
            "id": _stable_agent_id(),
            "name": AGENT_NAME,
            "description": AGENT_DESCRIPTION,
            "version": AGENT_VERSION,
            "default_prompt": default_prompt,
            "capabilities": [
                {
                    "name": "talk_to_avatar",
                    "description": (
                        "Send a message to the avatar that will be spoken aloud and "
                        "displayed on the animated avatar UI. Use this to proactively "
                        "inform the user about results or request input."
                    ),
                    "tags": [
                        "avatar", "speak", "notify", "voice", "display",
                        "announce", "user", "interface", "ui", "talk",
                    ],
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to speak on the avatar.",
                            },
                        },
                        "required": ["message"],
                    },
                }
            ],
            "tags": ["avatar", "ui", "voice", "conversation", "interface"],
        }
        payload["required_settings"] = [
            {
                "key": "avatar_persona_name",
                "label": "Avatar Name",
                "type": "string",
                "required": False,
                "default": "Aria",
                "description": (
                    "Display name / persona name of the avatar. "
                    "Used in greetings and the system prompt sent to the LLM."
                ),
            },
            {
                "key": "avatar_provider",
                "label": "LLM Provider",
                "type": "string",
                "options": ["anthropic", "openai", "gemini"],
                "required": False,
                "default": "",
                "description": (
                    "LLM provider for avatar conversations. "
                    "Leave blank to inherit the orchestrator default."
                ),
            },
            {
                "key": "avatar_model",
                "label": "Conversation Model",
                "type": "string",
                "required": False,
                "default": "",
                "description": (
                    "Model name for conversational responses "
                    "(e.g. claude-sonnet-4-6, gpt-4o, gemini-1.5-pro). "
                    "Leave blank to use the provider default."
                ),
            },
            {
                "key": "avatar_intent_model",
                "label": "Intent Detection Model",
                "type": "string",
                "required": False,
                "default": "",
                "description": (
                    "Optional separate (cheaper/faster) model for classifying user intent. "
                    "Leave blank to reuse the conversation model."
                ),
            },
            {
                "key": "avatar_max_tokens",
                "label": "Max Tokens",
                "type": "integer",
                "required": False,
                "default": 512,
                "description": "Maximum tokens per conversational response.",
            },
            {
                "key": "avatar_temperature",
                "label": "Temperature",
                "type": "float",
                "required": False,
                "default": 0.8,
                "description": (
                    "Sampling temperature for conversational responses (0.0–1.0). "
                    "Higher = more creative and varied."
                ),
            },
        ]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/api/v1/agents/register",
                json=payload,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

        self.agent_id = data["agent_id"]
        self.ws_url = data["ws_url"]
        # Merge both orchestrator-wide common_settings and this agent's own
        # required_settings values (agent_settings). agent_settings takes
        # precedence so avatar_persona_name / avatar_model etc. override globals.
        self._common_settings = {
            **data.get("common_settings", {}),
            **data.get("agent_settings", {}),
        }
        logger.info("Registered as %s  ws_url=%s", self.agent_id, self.ws_url)

        if self._on_settings_push and self._common_settings:
            self._on_settings_push(self._common_settings)

        system_prompt = data.get("system_prompt", "")
        if system_prompt and self._on_prompt_push:
            self._on_prompt_push(system_prompt)

        return self.agent_id

    # ── Run loop ─────────────────────────────────────────────────────────────

    async def connect_and_run(self) -> None:
        self._running = True
        retry_delay = 1.0

        while self._running:
            try:
                logger.info("Connecting to %s", self.ws_url)
                async with websockets.connect(self.ws_url) as ws:
                    self._ws = ws
                    retry_delay = 1.0

                    sender = asyncio.create_task(self._heartbeat_loop(), name="heartbeat")
                    receiver = asyncio.create_task(self._recv_loop(), name="recv")
                    stopper = asyncio.create_task(self._stop_event.wait(), name="stop")

                    done, pending = await asyncio.wait(
                        [sender, receiver, stopper],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    for t in done:
                        if t != stopper and not t.cancelled() and t.exception():
                            raise t.exception()

            except ConnectionClosed as exc:
                if exc.code == 4004:
                    logger.warning("Orchestrator doesn't recognise agent_id (4004) — re-registering")
                    await self.register()
                elif exc.code == 4003:
                    logger.info("Agent disabled by orchestrator (4003) — retrying")
                    retry_delay = max(retry_delay, 10.0)
                else:
                    logger.warning("WS closed (code=%s): %s", exc.code, exc.reason)
            except Exception as exc:
                logger.warning("WS error: %s", exc)
            finally:
                self._ws = None

            if not self._running:
                break
            logger.info("Reconnecting in %.1f s …", retry_delay)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60.0)

    # ── Task dispatch ─────────────────────────────────────────────────────────

    async def send_task(
        self,
        target_agent_id: str,
        capability: str,
        input_data: dict,
        timeout_ms: float = 120_000,
    ) -> dict:
        """Send a task_request over WS and await the task_response payload."""
        if not self._ws:
            raise RuntimeError("WebSocket is not connected to orchestrator")

        msg = self._make_envelope(
            "task_request",
            {"capability": capability, "input_data": input_data, "timeout_ms": timeout_ms},
            recipient_id=target_agent_id,
        )
        req_id = msg["id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        await self._ws.send(json.dumps(msg))
        logger.debug("Sent task_request id=%s capability=%s to %s", req_id, capability, target_agent_id)

        try:
            return await asyncio.wait_for(fut, timeout=timeout_ms / 1000)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"task_request timed out after {timeout_ms} ms")

    async def dispatch_to_planner(self, goal: str, session_id: str = "") -> dict:
        """Discover the task-planner-agent and send it a plan_task request."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self._base}/api/v1/discover",
                params={"status": "available,busy"},
            )
            resp.raise_for_status()
            agents: list[dict] = resp.json()

        planners = [
            a for a in agents
            if a.get("name") == "task-planner-agent"
            or "plan_task" in a.get("capabilities", [])
        ]
        if not planners:
            raise RuntimeError("task-planner-agent is not available on the network")

        planners.sort(key=lambda a: (
            {"available": 0, "busy": 1}.get(a.get("status", ""), 2),
            a.get("score", 1.0),
        ))
        planner_id = planners[0]["agent_id"]
        logger.info("Dispatching to planner %s: %r", planner_id, goal)

        return await self.send_task(
            target_agent_id=planner_id,
            capability="plan_task",
            input_data={
                "goal": goal,
                "auto_execute": True,
                "source": "avatar",
                "session_id": session_id,
            },
            timeout_ms=120_000,
        )

    async def get_network_context(self) -> str:
        """
        Fetch the live agent/capability list from the orchestrator and return a
        compact, factual summary for injection into the system prompt.
        Excludes avatar-agent itself and meta-agents (planner, executor).
        """
        _HIDDEN = {"avatar-agent", "task-planner-agent", "task-executor-agent"}
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"{self._base}/api/v1/discover",
                    params={"status": "available,busy"},
                )
                resp.raise_for_status()
                agents: list[dict] = resp.json()
        except Exception as exc:
            logger.debug("Could not fetch network context: %s", exc)
            return ""

        lines: list[str] = []
        for a in agents:
            name = a.get("name", "")
            if name in _HIDDEN:
                continue
            caps = a.get("capabilities") or []
            status = a.get("status", "unknown")
            cap_str = ", ".join(caps) if caps else "no listed capabilities"
            lines.append(f"• {name} [{status}] — {cap_str}")

        if not lines:
            return ""

        return (
            "LIVE NETWORK CONTEXT (fetched from orchestrator right now):\n"
            "The following specialised agents are currently connected and available "
            "to handle tasks on the user's behalf:\n"
            + "\n".join(lines)
            + "\n\nWhen the user asks what you can do, always base your answer strictly "
            "on this list. Do not invent or guess capabilities not listed above."
        )

    async def get_user_name_from_cortex(self) -> str:
        """Read the global cortex memory and extract the user's name if stored."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base}/api/v1/cortex/__global__")
                if resp.status_code == 404:
                    return ""
                resp.raise_for_status()
                data = resp.json()

            content = data.get("content") or data.get("memory") or ""
            for line in content.splitlines():
                stripped = line.strip().lstrip("-•* ").strip()
                if stripped.lower().startswith("name:"):
                    name = stripped.split(":", 1)[1].strip().strip("*").strip()
                    if name:
                        logger.info("Found user name in cortex: %s", name)
                        return name
            return ""
        except Exception as exc:
            logger.debug("Could not read user name from cortex: %s", exc)
            return ""

    async def shutdown(self) -> None:
        self._running = False
        self._stop_event.set()
        if self.agent_id:
            async with httpx.AsyncClient() as client:
                try:
                    await client.delete(f"{self._base}/api/v1/agents/{self.agent_id}", timeout=5.0)
                except Exception:
                    pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ── Internal ─────────────────────────────────────────────────────────────

    def _make_envelope(self, msg_type: str, payload: dict,
                       recipient_id: Optional[str] = None,
                       correlation_id: Optional[str] = None) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "type": msg_type,
            "sender_id": self.agent_id,
            "recipient_id": recipient_id,
            "payload": payload,
            "timestamp": _now_iso(),
            "correlation_id": correlation_id,
        }

    async def _heartbeat_loop(self) -> None:
        while True:
            uptime = time.monotonic() - self._start_time
            hb = self._make_envelope("heartbeat", {
                "status": "busy" if self._active_tasks > 0 else "available",
                "current_load": min(self._active_tasks / 10.0, 1.0),
                "active_tasks": self._active_tasks,
                "metrics": {
                    "tasks_completed": self._tasks_completed,
                    "tasks_failed": self._tasks_failed,
                    "uptime_seconds": round(uptime, 1),
                },
            })
            await self._ws.send(json.dumps(hb))
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _recv_loop(self) -> None:
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "task_request":
                asyncio.create_task(self._dispatch_task(msg))

            elif msg_type == "task_response":
                corr_id = msg.get("correlation_id")
                if corr_id and corr_id in self._pending:
                    fut = self._pending.pop(corr_id)
                    if not fut.done():
                        fut.set_result(msg["payload"])

            elif msg_type == "settings_push":
                settings = msg.get("payload", {}).get("settings", {})
                self._common_settings.update(settings)
                if self._on_settings_push:
                    self._on_settings_push(self._common_settings)
                logger.info("Settings push received (%d keys)", len(settings))

            elif msg_type == "prompt_push":
                content = msg.get("payload", {}).get("content", "")
                if content and self._on_prompt_push:
                    self._on_prompt_push(content)
                    logger.info("Prompt push received (%d chars)", len(content))

            elif msg_type == "error":
                err = msg.get("payload", {})
                original_id = err.get("original_message_id")
                if original_id and original_id in self._pending:
                    fut = self._pending.pop(original_id)
                    if not fut.done():
                        fut.set_exception(RuntimeError(f"{err.get('code')}: {err.get('detail')}"))

            elif msg_type == "agent_restart":
                logger.info("Restart requested by orchestrator — shutting down for restart")
                import sys
                asyncio.get_event_loop().call_later(1.0, lambda: sys.exit(0))

    async def _dispatch_task(self, msg: dict) -> None:
        self._active_tasks += 1
        start = time.monotonic()
        capability = msg.get("payload", {}).get("capability")

        try:
            if capability == "talk_to_avatar":
                input_data = msg.get("payload", {}).get("input_data", {})
                message = input_data.get("message", "")
                if self._on_avatar_message and message:
                    await self._on_avatar_message(message)
                output_data = {"delivered": True, "message": message}
            else:
                raise ValueError(f"Unknown capability: {capability!r}")

            duration_ms = (time.monotonic() - start) * 1000
            resp = self._make_envelope(
                "task_response",
                {"success": True, "output_data": output_data, "duration_ms": round(duration_ms, 2)},
                recipient_id=msg.get("sender_id"),
                correlation_id=msg.get("id"),
            )
            await self._ws.send(json.dumps(resp))
            self._tasks_completed += 1
            logger.info("Handled task (capability=%s, %.1f ms)", capability, duration_ms)

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error("Task error (capability=%s): %s", capability, exc)
            err_resp = self._make_envelope(
                "task_response",
                {"success": False, "error": str(exc), "duration_ms": round(duration_ms, 2)},
                recipient_id=msg.get("sender_id"),
                correlation_id=msg.get("id"),
            )
            try:
                await self._ws.send(json.dumps(err_resp))
            except Exception:
                pass
            self._tasks_failed += 1
        finally:
            self._active_tasks = max(0, self._active_tasks - 1)
