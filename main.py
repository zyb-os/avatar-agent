"""
main.py — Entry point for the Avatar Agent.

Runs two concurrent tasks:
  1. AvatarServer (FastAPI, port 8010) — serves the Avatar UI and accepts
     WebSocket connections from browsers via AvatarStream Protocol v1.
  2. OrchestratorClient — registers with the orchestrator, maintains heartbeat,
     and handles incoming task_request messages (e.g. talk_to_avatar).

The avatar agent is NOT a proxy — it owns the conversation state.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

import config
from avatar_server import AvatarServer
from conversation_engine import ConversationEngine
from orchestrator_client import OrchestratorClient
from session_manager import SessionManager


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.getLevelName(config.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def main(orchestrator_url: str) -> None:
    logger = logging.getLogger("avatar-agent")

    # ── Wiring ────────────────────────────────────────────────────────────────
    engine = ConversationEngine(orchestrator_base=orchestrator_url, agent_id="pending")
    oc = OrchestratorClient(base_url=orchestrator_url)
    session_mgr = SessionManager(engine=engine, orchestrator_client=oc)

    # After registration, the orchestrator client knows its agent_id
    # — patch the engine so LLM proxy calls carry the correct X-Agent-Id
    def _after_register(settings: dict) -> None:
        engine._agent_id = oc.agent_id or "avatar-agent"
        session_mgr.update_common_settings(settings)

    oc.on_settings_push(_after_register)
    oc.on_avatar_message(session_mgr.broadcast_message)
    oc.on_prompt_push(engine.update_prompt)

    avatar_server = AvatarServer(
        session_manager=session_mgr,
        ui_dir=config.AVATAR_UI_DIR,
        host=config.HOST,
        port=config.PORT,
    )

    # ── Register with orchestrator (with retry) ───────────────────────────────
    logger.info("Registering with orchestrator at %s …", orchestrator_url)
    for attempt in range(1, 6):
        try:
            await oc.register()
            # Patch engine agent_id now that we know it
            engine._agent_id = oc.agent_id
            break
        except Exception as exc:
            if attempt == 5:
                logger.error("Could not register after 5 attempts: %s", exc)
                return
            wait = 2 ** attempt
            logger.warning("Attempt %d failed (%s) — retrying in %d s", attempt, exc, wait)
            await asyncio.sleep(wait)

    # ── Shutdown wiring ───────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _on_signal(*_):
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    logger.info(
        "Avatar Agent running. UI at http://%s:%d | Press Ctrl+C to stop.",
        config.HOST if config.HOST != "0.0.0.0" else "localhost",
        config.PORT,
    )

    # ── Run server + orchestrator WS concurrently ─────────────────────────────
    server_task = asyncio.create_task(avatar_server.serve(), name="avatar-server")
    oc_task = asyncio.create_task(oc.connect_and_run(), name="orchestrator-ws")
    stop_task = asyncio.create_task(shutdown_event.wait(), name="stop")

    done, pending = await asyncio.wait(
        [server_task, oc_task, stop_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    # ── Cleanup ───────────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    await avatar_server.stop()
    await oc.shutdown()
    logger.info("Avatar Agent stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avatar Agent")
    parser.add_argument(
        "--orchestrator-url",
        default=os.getenv("ORCHESTRATOR_URL", "http://localhost:8000"),
    )
    args = parser.parse_args()

    _setup_logging()
    asyncio.run(main(args.orchestrator_url))
