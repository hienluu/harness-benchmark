"""Shared tool surface used by every harness.

The fairness of the benchmark depends on both harnesses using an identical
tool list with identical implementations. Exposes:
  - TOOL_SCHEMAS: list[dict] in Anthropic tool-use format
  - execute(tool_name, tool_input, scratch_dir) -> str
"""

from __future__ import annotations

from pathlib import Path

from .bash import bash
from .fs import edit_file, list_dir, read_file, write_file

TOOL_SCHEMAS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command inside the task scratch directory. "
            "Returns combined stdout+stderr and the exit code. "
            "30s default timeout."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout_s": {"type": "integer", "description": "Seconds before kill.", "default": 30},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file relative to the scratch directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Overwrite a text file relative to the scratch directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace the first exact occurrence of `old` with `new` in the file. "
            "Fails if `old` is not present or appears more than once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories under the given path (recursive, max 200 entries).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "finish",
        "description": (
            "Terminal tool. Call exactly once when the task is complete. "
            "Pass the final answer (for research tasks) or a brief success note."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
]


def execute(name: str, tool_input: dict, scratch_dir: Path) -> str:
    """Dispatch a tool call and return a string result."""
    try:
        if name == "bash":
            return bash(tool_input["command"], scratch_dir, tool_input.get("timeout_s", 30))
        if name == "read_file":
            return read_file(tool_input["path"], scratch_dir)
        if name == "write_file":
            return write_file(tool_input["path"], tool_input["content"], scratch_dir)
        if name == "edit_file":
            return edit_file(tool_input["path"], tool_input["old"], tool_input["new"], scratch_dir)
        if name == "list_dir":
            return list_dir(tool_input.get("path", "."), scratch_dir)
        if name == "finish":
            return f"FINISHED: {tool_input['answer']}"
        return f"ERROR: unknown tool {name!r}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
