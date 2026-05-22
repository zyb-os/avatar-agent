"""
session_manager.py — Manages active avatar WebSocket sessions.

Each connected browser tab is a Session with its own conversation history,
per-session settings, and an intent state machine.

Intent state machine (per session):
  idle       — normal conversation, intent detection runs after every LLM reply
  confirming — a solid intent was detected; waiting for user yes/no
  executing  — task dispatched to task-planner-agent; awaiting result

On first connect the avatar greets the user by name (fetched from cortex global
memory via the orchestrator client).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from fastapi import WebSocket

import models as ev
from conversation_engine import ConversationEngine

if TYPE_CHECKING:
    from orchestrator_client import OrchestratorClient

logger = logging.getLogger(__name__)

_MAX_HISTORY = 40       # prune oldest messages beyond this
_INTENT_THRESHOLD = 0.75  # minimum confidence to enter confirming state

# Words/phrases that count as "yes, go ahead"
_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead", "do it",
    "correct", "right", "please", "absolutely", "definitely", "of course",
    "sounds good", "go for it", "that's right", "affirmative", "aye",
}

# Words/phrases that count as "no, stop"
_NEGATIVE = {
    "no", "nope", "nah", "don't", "do not", "stop", "cancel", "never mind",
    "nevermind", "not really", "no thanks", "forget it", "skip it", "ignore",
}


def _is_affirmative(text: str) -> bool:
    t = text.lower().strip().rstrip(".")
    return t in _AFFIRMATIVE or any(t.startswith(a) for a in _AFFIRMATIVE)


def _is_negative(text: str) -> bool:
    t = text.lower().strip().rstrip(".")
    return t in _NEGATIVE or any(t.startswith(n) for n in _NEGATIVE)


# Phrases that indicate the user is asking about the avatar's own capabilities
# or the connected agent network — triggers a live orchestrator lookup.
_SELF_QUERY_PHRASES = (
    "what can you do", "what can you help", "what are you able",
    "what are your capabilities", "what capabilities", "what can you access",
    "what agents", "which agents", "what tools", "what services",
    "can you search", "can you send", "can you browse", "can you read",
    "can you write", "can you check", "can you create", "can you schedule",
    "are you connected", "what are you connected", "tell me what you can",
    "show me what you can", "help me understand what you",
    "what do you have access", "what do you support",
    "what can you actually", "what exactly can you",
)


def _is_self_query(text: str) -> bool:
    """Return True if the user is asking about the avatar's own capabilities."""
    t = text.lower()
    return any(phrase in t for phrase in _SELF_QUERY_PHRASES)


# ── Intent state ──────────────────────────────────────────────────────────────

@dataclass
class IntentState:
    status: str = "idle"          # idle | confirming | executing
    intent_summary: str = ""      # one-line description of the action
    confirmation_prompt: str = "" # question posed to the user


# ── Session ───────────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str
    ws: WebSocket
    history: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    intent_state: IntentState = field(default_factory=IntentState)
    user_name: str = ""
    _response_task: Optional[asyncio.Task] = field(default=None, repr=False)

    async def send(self, event: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(event))
        except Exception as exc:
            logger.debug("send failed for session %s: %s", self.session_id[:8], exc)

    def add_user_message(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_message(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self) -> None:
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_MAX_HISTORY:]


# ── Session manager ───────────────────────────────────────────────────────────

class SessionManager:
    """Lifecycle + message routing for all avatar WebSocket sessions."""

    def __init__(
        self,
        engine: ConversationEngine,
        orchestrator_client: "OrchestratorClient",
        common_settings: Optional[dict] = None,
    ) -> None:
        self._engine = engine
        self._oc = orchestrator_client
        self._common_settings: dict = common_settings or {}
        self._sessions: dict[str, Session] = {}

    def update_common_settings(self, settings: dict) -> None:
        self._common_settings = settings

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def connect(self, ws: WebSocket) -> Session:
        await ws.accept()
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, ws=ws)
        self._sessions[session_id] = session
        await session.send(ev.session_created(session_id))
        logger.info("Avatar session connected: %s", session_id[:8])
        # Greet the user in background so the WS handshake completes first
        asyncio.create_task(
            self._greet_user(session),
            name=f"avatar-greet-{session_id[:8]}",
        )
        return session

    async def disconnect(self, session: Session) -> None:
        if session._response_task and not session._response_task.done():
            session._response_task.cancel()
        self._sessions.pop(session.session_id, None)
        logger.info("Avatar session disconnected: %s", session.session_id[:8])

    # ── Message dispatch ──────────────────────────────────────────────────────

    async def handle(self, session: Session, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON frame from session %s", session.session_id[:8])
            return

        event_type = msg.get("type", "")
        logger.debug("Session %s received: %s", session.session_id[:8], event_type)

        if event_type in ("input.text_message", "input.speech_transcription"):
            text = (msg.get("text") or "").strip()
            is_final = msg.get("is_final", True)
            if text and is_final:
                await self._handle_user_input(session, text)

        elif event_type == "input.speech_started":
            await session.send(ev.avatar_state("listening"))

        elif event_type == "input.interrupt":
            await self._interrupt(session)

        elif event_type == "input.reset_conversation":
            await self._reset(session)

        elif event_type == "session.update":
            session.settings.update(msg.get("settings") or {})

        elif event_type == "session.ping":
            await session.send(ev.session_ping(msg.get("ts", time.time())))

        else:
            logger.debug("Unhandled event type: %s", event_type)

    # ── Broadcast (used by orchestrator task handler) ─────────────────────────

    async def broadcast_message(self, text: str) -> None:
        """Push *text* to all active sessions as an assistant notification."""
        for session in list(self._sessions.values()):
            session.add_assistant_message(text)
            await session.send(ev.task_result(text))
            await self._stream_and_speak(session, text)

    # ── Greeting ──────────────────────────────────────────────────────────────

    async def _greet_user(self, session: Session) -> None:
        """Fetch the user name from cortex and deliver a personalised greeting."""
        await asyncio.sleep(0.6)   # let the WS settle before speaking
        try:
            user_name = await self._oc.get_user_name_from_cortex()
        except Exception:
            user_name = ""

        session.user_name = user_name or ""
        persona_name = self._common_settings.get("avatar_persona_name") or "Aria"

        if user_name:
            intro = (
                f"Hi {user_name}! I'm {persona_name}, your personal AI assistant. "
                f"I'm connected to a network of specialised agents that can search the web, "
                f"manage documents, send messages, and much more. "
                f"What can I help you with today?"
            )
        else:
            intro = (
                f"Hi! I'm {persona_name}, your personal AI assistant. "
                "I'm connected to a network of specialised agents that can search the web, "
                "manage documents, send messages, and much more. "
                "What can I help you with today?"
            )

        session.add_assistant_message(intro)
        await self._stream_and_speak(session, intro)

    # ── User input handling ───────────────────────────────────────────────────

    async def _handle_user_input(self, session: Session, text: str) -> None:
        # Cancel any in-progress response first
        if session._response_task and not session._response_task.done():
            session._response_task.cancel()
            try:
                await session._response_task
            except asyncio.CancelledError:
                pass

        # ── Confirmation gate ────────────────────────────────────────────────
        if session.intent_state.status == "confirming":
            if _is_affirmative(text):
                session._response_task = asyncio.create_task(
                    self._execute_intent(session),
                    name=f"avatar-execute-{session.session_id[:8]}",
                )
                return
            elif _is_negative(text):
                # User said no — reset intent and fall through to normal chat
                session.intent_state = IntentState()
            else:
                # Ambiguous reply — reset and treat as new input
                session.intent_state = IntentState()

        # ── Normal conversational turn with parallel intent detection ─────────
        session._response_task = asyncio.create_task(
            self._converse_and_detect(session, text),
            name=f"avatar-resp-{session.session_id[:8]}",
        )

    # ── Conversational turn ───────────────────────────────────────────────────

    async def _converse_and_detect(self, session: Session, user_text: str) -> None:
        """
        1. Add user message to history.
        2. Call LLM for conversational response AND intent detection concurrently.
        3. If a solid intent is detected, append a confirmation question.
        4. Stream the final text.
        """
        session.add_user_message(user_text)
        await session.send(ev.avatar_state("thinking"))

        merged = {**self._common_settings, **session.settings}

        # If the user is asking about capabilities, fetch live data from the
        # orchestrator so the LLM answers from facts, not hallucination.
        injected_context = ""
        if _is_self_query(user_text):
            try:
                injected_context = await self._oc.get_network_context()
                logger.info("Session %s: self-query detected — injecting network context (%d chars)",
                            session.session_id[:8], len(injected_context))
            except Exception as exc:
                logger.debug("Session %s: network context fetch failed: %s", session.session_id[:8], exc)

        try:
            response_text, intent_data = await asyncio.gather(
                self._engine.respond(session.history, merged, injected_context=injected_context),
                self._engine.detect_intent(session.history, merged),
            )
        except asyncio.CancelledError:
            await session.send(ev.avatar_state("idle"))
            raise
        except Exception as exc:
            logger.error("LLM error for session %s: %s", session.session_id[:8], exc)
            await session.send(ev.output_error("LLM_ERROR", str(exc)))
            await session.send(ev.avatar_state("idle"))
            return

        # ── Intent confirmation injection ────────────────────────────────────
        if (
            intent_data.get("has_intent")
            and intent_data.get("confidence", 0.0) >= _INTENT_THRESHOLD
            and session.intent_state.status == "idle"
        ):
            confirmation_q = intent_data.get(
                "suggested_confirmation",
                f"Just to confirm — you'd like me to {intent_data['intent']}?",
            )
            session.intent_state = IntentState(
                status="confirming",
                intent_summary=intent_data["intent"],
                confirmation_prompt=confirmation_q,
            )
            # Append confirmation question to the LLM reply
            final_text = response_text.rstrip() + " " + confirmation_q
            logger.info(
                "Session %s: intent detected (confidence=%.2f): %r",
                session.session_id[:8],
                intent_data["confidence"],
                intent_data["intent"],
            )
        else:
            final_text = response_text

        if not final_text:
            await session.send(ev.avatar_state("idle"))
            return

        session.add_assistant_message(final_text)
        await self._stream_and_speak(session, final_text)

    # ── Intent execution ──────────────────────────────────────────────────────

    async def _execute_intent(self, session: Session) -> None:
        """
        User confirmed the intent. Dispatch to task-planner-agent, stream
        a live acknowledgement, then stream the result when it comes back.
        """
        intent = session.intent_state.intent_summary
        session.intent_state.status = "executing"
        await session.send(ev.avatar_state("thinking"))

        # Immediate acknowledgement while the planner works
        ack = f"On it! Let me take care of that for you — {intent}."
        session.add_assistant_message(ack)
        await self._stream_and_speak(session, ack)

        # Tell the UI a task has been delegated
        await session.send(ev.task_delegated(intent, "task-planner-agent"))

        try:
            await self._oc.dispatch_to_planner(
                goal=intent,
                session_id=session.session_id,
            )
            # plan_task returns immediately with {task_id, status: "running"}.
            # The actual result arrives asynchronously via talk_to_avatar →
            # broadcast_message, which will stream it to the session automatically.
            response = (
                "I've kicked that off. I'll let you know as soon as it's done!"
            )
        except TimeoutError:
            response = (
                "That's taking longer than expected. "
                "The task is still running in the background — I'll update you when it's done."
            )
        except RuntimeError as exc:
            response = f"I couldn't reach the task network right now: {exc}. Please try again in a moment."
        except Exception as exc:
            logger.error("Planner dispatch error for session %s: %s", session.session_id[:8], exc)
            response = f"Something unexpected happened: {exc}. Shall I try again?"
        finally:
            session.intent_state = IntentState()   # always reset after execution

        session.add_assistant_message(response)
        await self._stream_and_speak(session, response)

    # ── Streaming helper ──────────────────────────────────────────────────────

    async def _stream_and_speak(self, session: Session, text: str) -> None:
        """Stream *text* word-by-word then simulate speaking duration."""
        try:
            words = text.split()
            for idx, word in enumerate(words):
                chunk = word if idx == 0 else " " + word
                await session.send(ev.text_delta(chunk, idx))
                await asyncio.sleep(0.04)  # ~25 words/sec visual speed

            await session.send(ev.text_done(text, len(words)))
            await session.send(ev.avatar_state("speaking"))
            speak_duration = min(max(len(words) * 0.35, 1.5), 30.0)
            await asyncio.sleep(speak_duration)
            await session.send(ev.speaking_done())
            await session.send(ev.avatar_state("idle"))
        except asyncio.CancelledError:
            await session.send(ev.avatar_state("idle"))
            raise

    # ── Control ───────────────────────────────────────────────────────────────

    async def _interrupt(self, session: Session) -> None:
        if session._response_task and not session._response_task.done():
            session._response_task.cancel()
            try:
                await session._response_task
            except asyncio.CancelledError:
                pass
        # Preserve intent state through interrupts so the user can still confirm
        await session.send(ev.avatar_state("idle"))

    async def _reset(self, session: Session) -> None:
        await self._interrupt(session)
        session.history.clear()
        session.intent_state = IntentState()
        await session.send(ev.session_reset())
        logger.info("Conversation reset for session %s", session.session_id[:8])
