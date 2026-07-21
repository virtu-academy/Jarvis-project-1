"""Configuration — loads keys and settings from .env."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

JARVIS_SYSTEM_PROMPT = """\
You are JARVIS — a capable, dryly witty personal AI assistant, inspired by
Tony Stark's JARVIS. You are speaking out loud through a text-to-speech
voice, so:

- Keep responses conversational and concise: this is a spoken dialogue,
  not an essay. A few sentences is usually right; go longer only when the
  user asks for depth.
- Never use markdown, bullet lists, headings, or code blocks in your
  replies — they will be read aloud verbatim. Speak in plain prose.
- Address the user as "sir" occasionally, in the classic JARVIS style,
  but don't overdo it.
- You may have tools available from connected MCP servers (for example
  media generation services). Use them when they genuinely help, and
  briefly narrate what you're doing while you work.
- If you don't know something or a tool fails, say so plainly and suggest
  a next step.
- You have long-term memory: use the save_memory tool when the user shares
  something worth remembering across sessions (their name, preferences,
  ongoing projects), and forget_memory when asked to forget. Anything in
  your memory section was saved in a previous session — use it naturally.
"""


@dataclass
class Config:
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    fish_api_key: str = field(default_factory=lambda: os.environ.get("FISH_AUDIO_API_KEY", ""))
    fish_voice_id: str = field(default_factory=lambda: os.environ.get("FISH_AUDIO_VOICE_ID", ""))
    whisper_model: str = field(default_factory=lambda: os.environ.get("WHISPER_MODEL", "small"))

    model: str = "claude-opus-4-8"
    max_tokens: int = 2048

    # Audio settings
    mic_sample_rate: int = 16_000     # what Whisper expects
    tts_sample_rate: int = 44_100     # Fish Audio PCM output rate

    # Wake word & barge-in
    wake_words: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            w.strip().lower()
            for w in os.environ.get("WAKE_WORDS", "jarvis").split(",")
            if w.strip()
        )
    )
    # After Jarvis replies, follow-ups within this window don't need the wake word.
    follow_up_window_s: float = field(
        default_factory=lambda: float(os.environ.get("FOLLOW_UP_WINDOW", "25"))
    )
    # While Jarvis is talking, require this much louder input to register
    # (crude echo rejection for barge-in).
    barge_in_threshold_multiplier: float = 4.0

    # Long-term memory file
    memory_path: Path = field(default_factory=lambda: PROJECT_ROOT / "jarvis_memory.json")

    def validate(self, need_voice: bool = True) -> list[str]:
        def unset(value: str) -> bool:
            v = value.strip()
            return not v or v.endswith("...") or v == "REPLACE-ME"

        problems = []
        if unset(self.anthropic_api_key):
            problems.append(
                "ANTHROPIC_API_KEY is not set (or still the placeholder) — "
                "Jarvis has no brain. Get a key at platform.claude.com."
            )
        if need_voice and unset(self.fish_api_key):
            problems.append(
                "FISH_AUDIO_API_KEY is not set (or still the placeholder) — "
                "Jarvis has no voice. Get a key at fish.audio."
            )
        return problems
