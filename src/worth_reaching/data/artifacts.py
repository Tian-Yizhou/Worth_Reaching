"""Small shared helpers for experiment manifests."""

import json
import subprocess
from pathlib import Path


def code_revision() -> dict[str, str | bool | None]:
    repo = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": revision, "dirty": bool(status)}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
