"""
models.py — AvatarStream Protocol v1 event models.

Protocol events are plain JSON dicts transmitted over WebSocket.

UI → Agent (input events):
  input.text_message       {text}
  input.speech_started     {}
  input.speech_stopped     {}
  input.speech_transcription  {text, is_final}
  input.interrupt          {}
  input.reset_conversation {}
  session.update           {settings: {model?, temperature?, max_tokens?}}
  session.ping             {ts}

Agent → UI (output events):
  session.created          {session_id, version}
  session.ping             {ts}
  session.reset            {}
  output.avatar_state      {state}  — idle|listening|thinking|speaking|happy|error
  output.text_delta        {delta, index}
  output.text_done         {text, index}
  output.speaking_done     {}
  output.task_delegated    {task, agent}
  output.task_result       {result}
  output.error             {code, message}
"""
from __future__ import annotations

from typing import Literal

AvatarState = Literal["idle", "listening", "thinking", "speaking", "happy", "error"]

PROTOCOL_VERSION = "1.0"


def make_event(event_type: str, **kwargs) -> dict:
    return {"type": event_type, **kwargs}


# ── Outbound helpers ──────────────────────────────────────────────────────────

def session_created(session_id: str) -> dict:
    return make_event("session.created", session_id=session_id, version=PROTOCOL_VERSION)

def session_ping(ts: float) -> dict:
    return make_event("session.ping", ts=ts)

def session_reset() -> dict:
    return make_event("session.reset")

def avatar_state(state: AvatarState) -> dict:
    return make_event("output.avatar_state", state=state)

def text_delta(delta: str, index: int) -> dict:
    return make_event("output.text_delta", delta=delta, index=index)

def text_done(text: str, index: int) -> dict:
    return make_event("output.text_done", text=text, index=index)

def speaking_done() -> dict:
    return make_event("output.speaking_done")

def task_delegated(task: str, agent: str) -> dict:
    return make_event("output.task_delegated", task=task, agent=agent)

def task_result(result: str) -> dict:
    return make_event("output.task_result", result=result)

def output_error(code: str, message: str) -> dict:
    return make_event("output.error", code=code, message=message)
