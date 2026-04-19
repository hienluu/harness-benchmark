"""Filesystem tools scoped to a per-task scratch directory."""

from __future__ import annotations

from pathlib import Path


def _resolve(path: str, scratch_dir: Path) -> Path:
    """Resolve a relative path inside the scratch directory.

    Ensures the resolved path does not escape the provided scratch directory.

    Args:
        path: The relative path to resolve.
        scratch_dir: The base scratch directory.

    Returns:
        The resolved absolute Path inside the scratch directory.

    Raises:
        ValueError: If the resolved path escapes the scratch directory.
    """
    scratch_dir = scratch_dir.resolve()
    p = (scratch_dir / path).resolve()
    if scratch_dir not in p.parents and p != scratch_dir:
        raise ValueError(f"path {path!r} escapes scratch dir")
    return p


def read_file(path: str, scratch_dir: Path) -> str:
    """Read text from a file within the scratch directory.

    Args:
        path: The file path relative to the scratch directory.
        scratch_dir: The base scratch directory.

    Returns:
        The file text, truncated to 16,000 characters if needed, or an error string if missing.
    """
    p = _resolve(path, scratch_dir)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    text = p.read_text(errors="replace")
    if len(text) > 16000:
        return text[:16000] + f"\n[truncated, total {len(text)} chars]"
    return text


def write_file(path: str, content: str, scratch_dir: Path) -> str:
    """Write text to a file within the scratch directory.

    Creates parent directories if necessary.

    Args:
        path: The file path relative to the scratch directory.
        content: The text content to write.
        scratch_dir: The base scratch directory.

    Returns:
        A confirmation string describing the number of characters written.
    """
    p = _resolve(path, scratch_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def edit_file(path: str, old: str, new: str, scratch_dir: Path) -> str:
    """Replace a unique substring in a file within the scratch directory.

    Args:
        path: The file path relative to the scratch directory.
        old: The text to replace.
        new: The replacement text.
        scratch_dir: The base scratch directory.

    Returns:
        A status string indicating success or a descriptive error.
    """
    p = _resolve(path, scratch_dir)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    text = p.read_text()
    count = text.count(old)
    if count == 0:
        return f"ERROR: old text not found in {path}"
    if count > 1:
        return f"ERROR: old text appears {count} times in {path}; make it unique"
    p.write_text(text.replace(old, new, 1))
    return f"edited {path}"


def list_dir(path: str, scratch_dir: Path) -> str:
    """List directory entries under a path inside the scratch directory.

    Skips hidden files and directories, and truncates the listing after 200 entries.

    Args:
        path: The directory path relative to the scratch directory.
        scratch_dir: The base scratch directory.

    Returns:
        A newline-separated string of entries prefixed with "d" for directories and "f" for files.
        Returns an error string if the path does not exist, or "[empty]" if no entries are found.
    """
    p = _resolve(path, scratch_dir)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    entries = []
    for item in sorted(p.rglob("*")):
        if any(part.startswith(".") for part in item.relative_to(p).parts):
            continue
        rel = item.relative_to(p)
        kind = "d" if item.is_dir() else "f"
        entries.append(f"{kind} {rel}")
        if len(entries) >= 200:
            entries.append("[truncated at 200 entries]")
            break
    return "\n".join(entries) if entries else "[empty]"
