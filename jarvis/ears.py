"""Ears — microphone capture with simple voice-activity detection,
transcribed locally by faster-whisper."""

from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class Listener:
    """Blocks on the mic until the user finishes speaking, returns the text.

    Uses a lightweight energy-based VAD: recording starts when the input
    level crosses a threshold and stops after ~1 second of silence.
    """

    def __init__(
        self,
        whisper_model: str = "small",
        sample_rate: int = 16_000,
        silence_threshold: float = 0.01,
        silence_duration_s: float = 1.0,
        max_utterance_s: float = 60.0,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration_s = silence_duration_s
        self.max_utterance_s = max_utterance_s
        print(f"[ears] loading whisper model '{whisper_model}' (first run downloads it)...")
        self.model = WhisperModel(whisper_model, device="auto", compute_type="auto")
        print("[ears] ready.")

    def listen(self) -> str:
        """Record one utterance from the mic and return its transcription."""
        audio = self._record_utterance()
        if audio is None or len(audio) < self.sample_rate // 4:  # < 0.25s — noise
            return ""
        segments, _info = self.model.transcribe(audio, language=None, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _record_utterance(self) -> np.ndarray | None:
        block_s = 0.05
        block_frames = int(self.sample_rate * block_s)
        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        chunks: list[np.ndarray] = []
        speaking = False
        silent_blocks = 0
        silence_limit = int(self.silence_duration_s / block_s)
        max_blocks = int(self.max_utterance_s / block_s)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_frames,
            callback=callback,
        ):
            print("\n[ears] listening... (speak now)")
            total = 0
            while total < max_blocks:
                block = q.get()
                total += 1
                level = float(np.sqrt(np.mean(block**2)))

                if not speaking:
                    if level >= self.silence_threshold:
                        speaking = True
                        chunks.append(block)
                else:
                    chunks.append(block)
                    if level < self.silence_threshold:
                        silent_blocks += 1
                        if silent_blocks >= silence_limit:
                            break
                    else:
                        silent_blocks = 0

        if not chunks:
            return None
        print("[ears] transcribing...")
        return np.concatenate(chunks)
