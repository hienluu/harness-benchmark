"""Sandboxed bash execution scoped to a scratch directory."""

from __future__ import annotations

import subprocess
from pathlib import Path


def bash(command: str, scratch_dir: Path, timeout_s: int = 30) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(scratch_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"[exit=124 TIMEOUT after {timeout_s}s]"
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > 8000:
        out = out[:8000] + f"\n[truncated, total {len(out)} chars]"
    return f"[exit={proc.returncode}]\n{out}"
