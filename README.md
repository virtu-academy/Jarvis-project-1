# Jarvis

A local, voice-driven AI assistant in the spirit of Tony Stark's JARVIS.

- **Brain** — [Claude](https://platform.claude.com) (Opus 4.8) with adaptive thinking and tool use
- **Ears** — your microphone, transcribed locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Voice** — [Fish Audio](https://fish.audio) streaming text-to-speech, spoken sentence-by-sentence while the response is still generating
- **Hands** — a generic [MCP](https://modelcontextprotocol.io) client: plug in any MCP server (Higgsfield, filesystem, whatever) and Jarvis can use its tools
- **Wake word** — always listening, but only answers when you say **"Jarvis"** (configurable via `WAKE_WORDS`); after he replies you get a follow-up window (default 25s) where no wake word is needed
- **Barge-in** — say his name while he's talking and he stops mid-sentence to listen
- **Memory** — long-term memory across sessions (`jarvis_memory.json`, kept local): tell him your preferences once and he remembers; ask him to forget and he does

## Setup

```bash
# 1. Create a virtualenv and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure keys
cp .env.example .env
#    then edit .env: add ANTHROPIC_API_KEY and FISH_AUDIO_API_KEY
#    (optionally FISH_AUDIO_VOICE_ID to pick a specific Fish voice)
```

> **Linux note:** `sounddevice` needs PortAudio — `sudo apt install libportaudio2` (Debian/Ubuntu).

## Run

```bash
python -m jarvis.main            # full voice: always listening for "Jarvis"
python -m jarvis.main --text     # type your side, he still speaks
python -m jarvis.main --silent   # pure text mode, no audio (good for testing)
```

In voice mode: say **"Jarvis, ..."** to give a command, just **"Jarvis?"** to get
his attention, keep talking within ~25s of his reply without the wake word,
and say his name mid-reply to cut him off. Say **"goodbye Jarvis"** (or type
`exit`) to shut down.

## Plugging in MCP servers (Higgsfield etc.)

Edit `mcp_servers.json`. Each entry is either:

- **HTTP** — `{"transport": "http", "url": "...", "headers": {...}}`
- **stdio** — `{"transport": "stdio", "command": "...", "args": [...]}`

Set `"enabled": true` and restart. Jarvis discovers the server's tools
automatically and Claude decides when to use them mid-conversation.

There's a placeholder entry for **Higgsfield** — once you have a Higgsfield
MCP endpoint and API key, drop them in and enable it.

## Project layout

```
jarvis/
  main.py        # the conversation loop (wake word, barge-in, follow-up window)
  brain.py       # Claude agentic loop (streaming + memory/MCP tool use)
  ears.py        # mic capture + VAD + whisper STT
  voice.py       # Fish Audio TTS with sentence-streaming, interruptible playback
  memory.py      # long-term memory store + save/forget tools
  mcp_client.py  # generic MCP server manager
  config.py      # .env loading + Jarvis system prompt
mcp_servers.json # which MCP servers to connect to
```
