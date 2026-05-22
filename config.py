"""config.py — Avatar agent configuration via environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

# Server config
HOST: str = os.getenv("AVATAR_HOST", "0.0.0.0")
PORT: int = int(os.getenv("AVATAR_PORT", "8010"))

# Orchestrator
ORCHESTRATOR_URL: str = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# Static files — avatar-ui directory sits beside avatar-agent
_HERE = Path(__file__).parent
AVATAR_UI_DIR: Path = _HERE.parent / "avatar-ui"
