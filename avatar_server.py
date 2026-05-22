"""
avatar_server.py — FastAPI server for the Avatar UI.

Serves:
  GET  /            → avatar-ui/index.html
  GET  /static/*    → avatar-ui static assets
  WS   /avatar-ws   → AvatarStream Protocol v1 WebSocket endpoint

The server does NOT connect to the orchestrator — that is handled by
orchestrator_client.py running concurrently in the same process.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from session_manager import SessionManager

logger = logging.getLogger(__name__)


def create_app(session_manager: SessionManager, ui_dir: Path) -> FastAPI:
    app = FastAPI(title="Avatar Agent", docs_url=None, redoc_url=None)

    # ── Static assets (CSS, JS, images) ──────────────────────────────────────
    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")
    else:
        logger.warning("Avatar UI directory not found: %s", ui_dir)

    # ── Root → serve index.html ───────────────────────────────────────────────
    @app.get("/")
    async def index():
        html = ui_dir / "index.html"
        if html.exists():
            return FileResponse(str(html))
        return {"error": "avatar-ui/index.html not found"}

    # ── WebSocket endpoint ────────────────────────────────────────────────────
    @app.websocket("/avatar-ws")
    async def avatar_ws(ws: WebSocket):
        session = await session_manager.connect(ws)
        try:
            while True:
                raw = await ws.receive_text()
                await session_manager.handle(session, raw)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Avatar WS error (session %s): %s", session.session_id[:8], exc)
        finally:
            await session_manager.disconnect(session)

    return app


class AvatarServer:
    """Wrapper that runs the uvicorn server as an asyncio task."""

    def __init__(
        self,
        session_manager: SessionManager,
        ui_dir: Path,
        host: str = "0.0.0.0",
        port: int = 8010,
    ) -> None:
        self._app = create_app(session_manager, ui_dir)
        self._host = host
        self._port = port
        self._server: Optional[uvicorn.Server] = None

    async def serve(self) -> None:
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",  # suppress uvicorn noise; our own logging handles it
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        logger.info("Avatar UI serving on http://%s:%d", self._host, self._port)
        await self._server.serve()

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
