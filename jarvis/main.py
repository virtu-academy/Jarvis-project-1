"""Jarvis — main loop.

Usage:
    python -m jarvis.main            # full voice mode: always listening for "Jarvis"
    python -m jarvis.main --text     # type instead of speaking (voice out)
    python -m jarvis.main --silent   # type in, printed text out (no audio)

Voice mode behavior:
  - Jarvis is always listening but only responds when you address him by
    a wake word ("Jarvis" by default — configurable via WAKE_WORDS).
  - After he replies, you have a follow-up window (default 25s) where you
    can keep talking without repeating the wake word.
  - Barge-in: say his name while he's talking and he stops to listen.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time

from .brain import Brain
from .config import PROJECT_ROOT, Config
from .mcp_client import MCPManager
from .memory import MemoryStore

EXIT_PHRASES = {"exit", "quit", "goodbye", "goodbye jarvis", "shutdown", "shut down"}


def has_wake_word(text: str, wake_words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in wake_words)


def strip_wake_word(text: str, wake_words: tuple[str, ...]) -> str:
    """Remove a leading 'hey jarvis' / 'jarvis' so only the command remains.
    A wake word mid-sentence is left alone."""
    pattern = r"^\s*(?:(?:hey|hi|ok|okay|yo)[\s,]+)?(?:%s)[\s,.!?]*" % "|".join(
        re.escape(w) for w in wake_words
    )
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


def is_exit(text: str) -> bool:
    cleaned = re.sub(r"[.,!?]", "", text.lower()).strip()
    return cleaned in EXIT_PHRASES


async def run(text_mode: bool, silent: bool) -> None:
    config = Config()
    problems = config.validate(need_voice=not silent)
    if problems:
        for p in problems:
            print(f"[config] {p}")
        print("[config] copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    # Voice out
    if silent:
        from .voice import ConsoleSpeaker

        speaker = ConsoleSpeaker()
    else:
        from .voice import Speaker

        speaker = Speaker(
            api_key=config.fish_api_key,
            voice_id=config.fish_voice_id,
            sample_rate=config.tts_sample_rate,
        )

    # Ears in
    listener = None
    if not text_mode and not silent:
        from .ears import Listener

        listener = Listener(
            whisper_model=config.whisper_model,
            sample_rate=config.mic_sample_rate,
        )

    # MCP servers (Higgsfield etc.)
    mcp = MCPManager(PROJECT_ROOT / "mcp_servers.json")
    await mcp.connect_all()
    if mcp.tool_index:
        print(f"[mcp] tools available: {', '.join(mcp.tool_index)}")
    else:
        print("[mcp] no MCP servers enabled (edit mcp_servers.json to add some).")

    # Long-term memory
    memory = MemoryStore(config.memory_path)
    if memory.notes:
        print(f"[memory] {len(memory.notes)} note(s) loaded from previous sessions.")

    brain = Brain(config, mcp, memory)

    greeting = "Online and at your service, sir."
    print(f"\nJARVIS: {greeting}")
    if listener is not None:
        print(
            f"[jarvis] say '{config.wake_words[0]}' to address me — "
            f"or interrupt me mid-sentence the same way."
        )
    speaker.say(greeting)

    last_exchange = 0.0  # for the follow-up window

    def respond_deadline_open() -> bool:
        return (time.time() - last_exchange) < config.follow_up_window_s

    try:
        while True:
            # ---------- get user input ----------
            if listener is not None:
                jarvis_talking = speaker.is_speaking
                threshold = listener.silence_threshold * (
                    config.barge_in_threshold_multiplier if jarvis_talking else 1.0
                )
                user_text = await asyncio.to_thread(
                    listener.listen, threshold, jarvis_talking
                )
                if not user_text:
                    continue

                woke = has_wake_word(user_text, config.wake_words)

                if speaker.is_speaking:
                    # Barge-in: only accept if addressed by name, so Jarvis's
                    # own voice coming out of the speakers doesn't trigger it.
                    if not woke:
                        continue
                    speaker.stop()
                    print("\n[jarvis] (interrupted)")
                elif not woke and not respond_deadline_open():
                    continue  # not addressed and outside follow-up window

                command = strip_wake_word(user_text, config.wake_words) if woke else user_text
                print(f"\nYou: {user_text}")

                if not command:
                    # Just the wake word — acknowledge and open the window.
                    speaker.say("Yes, sir?")
                    print("JARVIS: Yes, sir?")
                    last_exchange = time.time()
                    continue
            else:
                user_text = input("\nYou: ").strip()
                if not user_text:
                    continue
                command = user_text

            # ---------- exit? ----------
            if is_exit(command) or is_exit(user_text):
                farewell = "Powering down. Goodbye, sir."
                print(f"JARVIS: {farewell}")
                speaker.say(farewell)
                speaker.wait_until_done()
                break

            # ---------- respond, speaking as text streams in ----------
            print("JARVIS: ", end="", flush=True)

            def on_text(delta: str) -> None:
                print(delta, end="", flush=True)
                speaker.feed(delta)

            await brain.respond(command, on_text)
            speaker.flush()
            print()
            last_exchange = time.time()
    except (KeyboardInterrupt, EOFError):
        print("\n[jarvis] interrupted — shutting down.")
    finally:
        await mcp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis voice assistant")
    parser.add_argument("--text", action="store_true", help="type input instead of using the mic")
    parser.add_argument("--silent", action="store_true", help="no audio at all (text in, text out)")
    args = parser.parse_args()
    asyncio.run(run(text_mode=args.text, silent=args.silent))


if __name__ == "__main__":
    main()
