from __future__ import annotations

import os
from pathlib import Path


API_SCRIPTS_ROOT = Path(
    os.environ.get("API_SCRIPTS_ROOT", Path(__file__).resolve().parents[2])
)
RUNTIME_ROOT = API_SCRIPTS_ROOT / "runtime"
DATA_ROOT = RUNTIME_ROOT / "data"
SYSTEM_TOOLS_ROOT = Path(os.environ.get("J_SYSTEM_TOOLS_ROOT", r"J:\system_tools"))


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def system_tool_path(*parts: str) -> Path:
    return SYSTEM_TOOLS_ROOT.joinpath(*parts)
