"""Jarvis — main loop.

Usage:
    python -m jarvis.main            # full voice mode: always listening for "Jarvis"
    python -m jarvis.main --text     # type instead of speaking (voice out)
    python -m jarvis.main --silent   # type in, printed text out (no audio)
    python -m jarvis.main --no-hud   # skip the browser HUD

Voice mode behavior:
  - Jarvis is always listening but only responds when you address him by
    a wake word ("Jarvis" by default — configurable via WAKE_WORDS).
  - After he replies, you have a follow-up window (default 25s) where you
    can keep talking without repeating the wake word.
  - Barge-in: say his name while he's talking and he stops to listen.

Unless --no-hud is passed, a visual HUD opens in your browser showing
Jarvis's state (listening / thinking / speaking) and the live transcript.
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


async def run(text_mode: bool, silent: bool, no_hud: bool) -> None:
    config = Config()
    problems = config.validate(need_voice=not silent)
    if problems:
        for p in problems:
            print(f"[config] {p}")
        print("[config] copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    # Visual HUD
    if no_hud:
        from .hud import NullHUD

        hud = NullHUD()
    else:
        from .hud import HUD

        hud = HUD()
    await hud.start(open_browser=not no_hud)

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

    brain = Brain(config, mcp, memory, notify=hud.emit)

    greeting = "Online and at your service, sir."
    print(f"\nJARVIS: {greeting}")
    if listener is not None:
        print(
            f"[jarvis] say '{config.wake_words[0]}' to address me — "
            f"or interrupt me mid-sentence the same way."
        )
    hud.emit("jarvis", text=greeting)
    hud.emit("state", state="speaking")
    speaker.say(greeting)

    last_exchange = 0.0  # for the follow-up window

    def follow_up_open() -> bool:
        return (time.time() - last_exchange) < config.follow_up_window_s

    try:
        while True:
            # ---------- get user input ----------
            if listener is not None:
                jarvis_talking = speaker.is_speaking
                hud.emit("state", state="speaking" if jarvis_talking else "listening")
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
                    hud.emit("state", state="listening")
                elif not woke and not follow_up_open():
                    continue  # not addressed and outside follow-up window

                command = strip_wake_word(user_text, config.wake_words) if woke else user_text
                print(f"\nYou: {user_text}")

                if not command:
                    # Just the wake word — acknowledge and open the window.
                    hud.emit("user", text=user_text)
                    hud.emit("jarvis", text="Yes, sir?")
                    speaker.say("Yes, sir?")
                    print("JARVIS: Yes, sir?")
                    last_exchange = time.time()
                    continue
            else:
                hud.emit("state", state="idle")
                user_text = input("\nYou: ").strip()
                if not user_text:
                    continue
                command = user_text

            # ---------- exit? ----------
            if is_exit(command) or is_exit(user_text):
                farewell = "Powering down. Goodbye, sir."
                print(f"JARVIS: {farewell}")
                hud.emit("user", text=user_text)
                hud.emit("jarvis", text=farewell)
                hud.emit("state", state="idle")
                speaker.say(farewell)
                speaker.wait_until_done()
                break

            # ---------- respond, speaking as text streams in ----------
            hud.emit("user", text=command)
            hud.emit("state", state="thinking")
            print("JARVIS: ", end="", flush=True)

            first_delta = True

            def on_text(delta: str) -> None:
                nonlocal first_delta
                if first_delta:
                    hud.emit("state", state="speaking")
                    first_delta = False
                print(delta, end="", flush=True)
                speaker.feed(delta)
                hud.emit("delta", text=delta)

            full_text = await brain.respond(command, on_text)
            speaker.flush()
            print()
            hud.emit("jarvis_done", text=full_text)
            last_exchange = time.time()
    except (KeyboardInterrupt, EOFError):
        print("\n[jarvis] interrupted — shutting down.")
    finally:
        await mcp.close()
        await hud.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis voice assistant")
    parser.add_argument("--text", action="store_true", help="type input instead of using the mic")
    parser.add_argument("--silent", action="store_true", help="no audio at all (text in, text out)")
    parser.add_argument("--no-hud", action="store_true", help="don't open the browser HUD")
    args = parser.parse_args()
    asyncio.run(run(text_mode=args.text, silent=args.silent, no_hud=args.no_hud))


if __name__ == "__main__":
    main()
