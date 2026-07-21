"""Memory — Jarvis remembers things across sessions.

Notes are stored in a local JSON file. They're injected into the system
prompt at the start of every conversation, and Claude gets two local
tools (save_memory / forget_memory) to manage them mid-conversation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "name": "save_memory",
        "description": (
            "Save a durable note about the user or the world to Jarvis's "
            "long-term memory, so it persists across sessions. Call this when "
            "the user shares a preference, fact, or instruction worth "
            "remembering (e.g. their name, how they like things done, an "
            "ongoing project). Keep notes short and self-contained."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to remember, phrased as a standalone fact.",
                }
            },
            "required": ["note"],
        },
    },
    {
        "name": "forget_memory",
        "description": (
            "Delete a note from long-term memory by its id (ids are shown in "
            "the memory section of your context). Use when the user asks you "
            "to forget something or a note is outdated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "The id of the note to delete.",
                }
            },
            "required": ["memory_id"],
        },
    },
]

_TOOL_NAMES = {t["name"] for t in TOOLS}


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.notes: list[dict[str, Any]] = []
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.notes = data.get("notes", [])
                self._next_id = data.get("next_id", len(self.notes) + 1)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[memory] could not read {self.path}: {e} — starting fresh.")

    def _save(self) -> None:
        self.path.write_text(
            json.dumps({"notes": self.notes, "next_id": self._next_id}, indent=2)
        )

    # ---- operations ----

    def add(self, note: str) -> int:
        note_id = self._next_id
        self._next_id += 1
        self.notes.append(
            {
                "id": note_id,
                "note": note.strip(),
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
            }
        )
        self._save()
        return note_id

    def remove(self, note_id: int) -> bool:
        before = len(self.notes)
        self.notes = [n for n in self.notes if n["id"] != note_id]
        if len(self.notes) != before:
            self._save()
            return True
        return False

    # ---- integration with the brain ----

    def render(self) -> str:
        """Memory section for the system prompt (empty string if no notes)."""
        if not self.notes:
            return ""
        lines = [
            "Long-term memory — things you've been asked to remember from "
            "previous sessions (each with its id):"
        ]
        for n in self.notes:
            lines.append(f"  [{n['id']}] ({n['created_at']}) {n['note']}")
        return "\n".join(lines)

    def is_memory_tool(self, name: str) -> bool:
        return name in _TOOL_NAMES

    def handle(self, name: str, tool_input: dict[str, Any]) -> str:
        if name == "save_memory":
            note = str(tool_input.get("note", "")).strip()
            if not note:
                return "Error: empty note."
            note_id = self.add(note)
            return f"Saved to memory with id {note_id}."
        if name == "forget_memory":
            note_id = int(tool_input.get("memory_id", -1))
            if self.remove(note_id):
                return f"Memory {note_id} deleted."
            return f"No memory with id {note_id} found."
        return f"Unknown memory tool: {name}"
