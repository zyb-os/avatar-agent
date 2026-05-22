"""
conversation_engine.py — LLM-backed conversational engine.

Calls the orchestrator LLM proxy at /api/v1/llm/complete.
Maintains no history itself — callers pass the full message list.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "system_prompt.md"

def _load_default_prompt() -> str:
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return (
            "You are an AI assistant embodied as a friendly animated avatar named {name}. "
            "Keep responses natural and conversational — short sentences, no bullet lists. "
            "Be warm, curious, and engaging."
        )

_SYSTEM_PROMPT = _load_default_prompt()

_INTENT_SYSTEM = """\
You are an intent classifier for a voice assistant. Analyse the conversation \
and determine whether the user has expressed a clear, actionable intent that \
requires calling an external service or tool (web search, file operations, \
sending emails or messages, creating documents, checking live data like weather \
or stocks, scheduling tasks, browsing websites, etc.).

A solid intent is ONLY something like:
- "search for X online / find X"
- "send an email / message to Y"
- "create / write / update a document about X"
- "check the weather / stock price / news"
- "schedule a meeting / reminder"
- "browse to / open a website"
- "run code / execute a script"

NOT a solid intent:
- General chit-chat or greetings
- Questions the AI can answer directly from knowledge
- Vague statements with no clear external action required

Respond with ONLY a valid JSON object, no markdown fences:
{
  "has_intent": true,
  "intent": "concise one-line description of the action to perform",
  "confidence": 0.85,
  "suggested_confirmation": "Just to confirm — you'd like me to [action]?"
}

If there is no solid intent:
{
  "has_intent": false,
  "intent": "",
  "confidence": 0.0,
  "suggested_confirmation": ""
}
"""


class ConversationEngine:
    """Calls the orchestrator LLM proxy to generate avatar responses."""

    def __init__(self, orchestrator_base: str, agent_id: str) -> None:
        self._base = orchestrator_base.rstrip("/")
        self._agent_id = agent_id
        self._system_prompt: str = _SYSTEM_PROMPT

    def update_prompt(self, content: str) -> None:
        """Hot-reload the system prompt (called on prompt_push from orchestrator)."""
        if content and content.strip():
            self._system_prompt = content.strip()
            logger.info("System prompt updated (%d chars)", len(self._system_prompt))

    def _build_payload(
        self,
        system: str,
        messages: list[dict],
        common_settings: dict,
        max_tokens: int,
        temperature: float,
        model_override: str = "",
    ) -> dict:
        cs = common_settings
        # Avatar-specific settings take precedence over orchestrator-wide defaults
        provider = cs.get("avatar_provider") or cs.get("default_provider", "anthropic")
        avatar_model = model_override or cs.get("avatar_model") or ""
        if provider == "anthropic":
            model = avatar_model or cs.get("default_model", "claude-haiku-4-5-20251001")
        elif provider == "openai":
            model = avatar_model or cs.get("default_model", "gpt-4o-mini")
        elif provider == "gemini":
            model = avatar_model or cs.get("default_model", "gemini-1.5-flash")
        else:
            model = avatar_model or cs.get("default_model", "claude-haiku-4-5-20251001")
            provider = "anthropic"
        return {
            "provider": provider,
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    async def _call_proxy(self, payload: dict) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base}/api/v1/llm/complete",
                json=payload,
                headers={"X-Agent-Id": self._agent_id},
            )
            resp.raise_for_status()
            data = resp.json()
        raw = data.get("content", "")
        if isinstance(raw, list):
            return next((b["text"] for b in raw if b.get("type") == "text"), "").strip()
        return str(raw).strip()

    async def respond(
        self,
        history: list[dict],
        common_settings: Optional[dict] = None,
        injected_context: str = "",
    ) -> str:
        """Return the assistant's next conversational reply given *history*.

        If *injected_context* is provided it is appended to the system prompt
        so the LLM has live, factual data (e.g. connected agents) instead of
        guessing.
        """
        cs = common_settings or {}
        persona_name = cs.get("avatar_persona_name") or "Aria"
        system = self._system_prompt.replace("{name}", persona_name)
        if injected_context:
            system = f"{system}\n\n{injected_context}"
        payload = self._build_payload(
            system=system,
            messages=history,
            common_settings=cs,
            max_tokens=int(cs.get("avatar_max_tokens") or cs.get("default_max_tokens", 512)),
            temperature=float(cs.get("avatar_temperature") or cs.get("default_temperature", 0.8)),
        )
        logger.debug("LLM respond: provider=%s injected_context=%s", payload["provider"], bool(injected_context))
        return await self._call_proxy(payload)

    async def detect_intent(
        self,
        history: list[dict],
        common_settings: Optional[dict] = None,
    ) -> dict:
        """
        Analyse *history* and return a structured intent dict:
        {has_intent, intent, confidence, suggested_confirmation}

        Runs a separate, low-temperature LLM call so it does not interfere
        with the conversational response.  Returns a safe default on error.
        """
        cs = common_settings or {}
        # Use a dedicated intent model if configured, otherwise reuse the conversation model
        intent_model = cs.get("avatar_intent_model") or ""
        payload = self._build_payload(
            system=_INTENT_SYSTEM,
            messages=history,
            common_settings=cs,
            max_tokens=200,
            temperature=0.1,  # deterministic classification
            model_override=intent_model,
        )
        _default = {"has_intent": False, "intent": "", "confidence": 0.0, "suggested_confirmation": ""}
        try:
            raw = await self._call_proxy(payload)
            # Strip possible markdown fences just in case
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            return {**_default, **data}
        except Exception as exc:
            logger.debug("Intent detection failed: %s", exc)
            return _default
