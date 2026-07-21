"""Voice — Fish Audio streaming text-to-speech played through the speakers.

Text is fed in as it streams from the model; it's chunked into sentences
and synthesized/played by a background worker so Jarvis starts talking
while the response is still being generated.
"""

from __future__ import annotations

import queue
import re
import threading

import numpy as np
import sounddevice as sd
from fish_audio_sdk import Session, TTSRequest

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])$")


class Speaker:
    def __init__(self, api_key: str, voice_id: str = "", sample_rate: int = 44_100):
        self.session = Session(api_key)
        self.voice_id = voice_id or None
        self.sample_rate = sample_rate
        self._buffer = ""
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ---- feeding text in ----

    def feed(self, delta: str) -> None:
        """Feed streaming text; complete sentences are spoken as they form."""
        self._buffer += delta
        parts = _SENTENCE_END.split(self._buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                if sentence.strip():
                    self._queue.put(sentence.strip())
            self._buffer = parts[-1]

    def flush(self) -> None:
        """Speak whatever remains in the buffer."""
        if self._buffer.strip():
            self._queue.put(self._buffer.strip())
        self._buffer = ""

    def say(self, text: str) -> None:
        """Speak a complete piece of text."""
        if text.strip():
            self._queue.put(text.strip())

    def wait_until_done(self) -> None:
        """Block until everything queued has been spoken."""
        self._queue.join()

    # ---- background synthesis + playback ----

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                self._speak(text)
            except Exception as e:  # keep the loop alive on TTS/audio errors
                print(f"[voice] error: {e}")
            finally:
                self._queue.task_done()

    def _speak(self, text: str) -> None:
        request = TTSRequest(
            text=text,
            reference_id=self.voice_id,
            format="pcm",  # raw 16-bit mono PCM @ 44.1kHz
        )
        pcm = b"".join(self.session.tts(request))
        if not pcm:
            return
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=self.sample_rate, blocking=True)


class ConsoleSpeaker:
    """Drop-in replacement when no Fish Audio key is set — prints instead."""

    def feed(self, delta: str) -> None:
        print(delta, end="", flush=True)

    def flush(self) -> None:
        print()

    def say(self, text: str) -> None:
        print(text)

    def wait_until_done(self) -> None:
        pass
