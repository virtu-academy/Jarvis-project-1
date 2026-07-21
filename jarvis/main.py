"""Jarvis — main loop.

Usage:
    python -m jarvis.main            # full voice mode (mic in, voice out)
    python -m jarvis.main --text     # type instead of speaking (voice out)
    python -m jarvis.main --silent   # type in, printed text out (no audio)
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .brain import Brain
from .config import PROJECT_ROOT, Config
from .mcp_client import MCPManager


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

    brain = Brain(config, mcp)

    greeting = "Online and at your service, sir."
    print(f"\nJARVIS: {greeting}")
    speaker.say(greeting)

    try:
        while True:
            # --- get user input ---
            if listener is not None:
                speaker.wait_until_done()  # don't listen while Jarvis is talking
                user_text = await asyncio.to_thread(listener.listen)
                if not user_text:
                    continue
                print(f"\nYou: {user_text}")
            else:
                user_text = input("\nYou: ").strip()
                if not user_text:
                    continue

            if user_text.lower().rstrip(".!") in {"exit", "quit", "goodbye jarvis", "shutdown"}:
                farewell = "Powering down. Goodbye, sir."
                print(f"JARVIS: {farewell}")
                speaker.say(farewell)
                speaker.wait_until_done()
                break

            # --- respond, speaking as text streams in ---
            print("JARVIS: ", end="", flush=True)

            def on_text(delta: str) -> None:
                print(delta, end="", flush=True)
                speaker.feed(delta)

            await brain.respond(user_text, on_text)
            speaker.flush()
            print()
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
