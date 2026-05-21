"""Voice control for Mac mini. Always listening. Wake word: 'computer'.

Flow:
  - Mic streams continuously.
  - WebRTC VAD segments speech into utterances.
  - Each utterance is transcribed by local Whisper.
  - In command mode (default), utterances must start with 'computer'.
      'computer, pull up DaVinci Resolve'  ->  open the app
      'computer, dictate'                  ->  enter dictation mode
  - In dictation mode, every utterance is pasted at the cursor.
      'stop dictating'                     ->  return to command mode
"""

from __future__ import annotations

import collections
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np
import pyperclip
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from pynput import keyboard

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000        # 480 samples per frame
FRAME_BYTES = FRAME_SAMPLES * 2                       # int16 -> 960 bytes
MODEL_NAME = "small.en"
WAKE_WORD = "computer"
EXIT_DICTATION = ("stop dictating", "end dictation", "done dictating")
VAD_AGGRESSIVENESS = 2                                # 0..3, higher = stricter
VOICE_TRIGGER_FRAMES = 5                              # ~150ms voiced to start
SILENCE_END_FRAMES = 25                               # ~750ms silence to end
PREROLL_FRAMES = 10                                   # ~300ms preroll buffer
APPS_PATH = os.path.join(os.path.dirname(__file__), "apps.json")


def load_aliases() -> dict[str, str]:
    with open(APPS_PATH) as f:
        return json.load(f)


class VADSegmenter:
    """Frame-fed VAD. Emits a completed utterance as raw int16 bytes."""

    def __init__(self) -> None:
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.preroll: collections.deque[bytes] = collections.deque(maxlen=PREROLL_FRAMES)
        self.recording: list[bytes] | None = None
        self.voiced = 0
        self.silent = 0

    def feed(self, frame: bytes) -> bytes | None:
        is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
        if self.recording is None:
            self.preroll.append(frame)
            self.voiced = self.voiced + 1 if is_speech else 0
            if self.voiced >= VOICE_TRIGGER_FRAMES:
                self.recording = list(self.preroll)
                self.preroll.clear()
                self.silent = 0
            return None
        self.recording.append(frame)
        if is_speech:
            self.silent = 0
            return None
        self.silent += 1
        if self.silent < SILENCE_END_FRAMES:
            return None
        utterance = b"".join(self.recording)
        self.recording = None
        self.voiced = 0
        self.silent = 0
        return utterance


class App:
    def __init__(self) -> None:
        print(f"loading whisper ({MODEL_NAME})...", flush=True)
        self.model = WhisperModel(MODEL_NAME, device="auto", compute_type="int8")
        self.aliases = load_aliases()
        self.segmenter = VADSegmenter()
        self.mode = "command"
        self.kbd = keyboard.Controller()
        self.jobs: queue.Queue[bytes] = queue.Queue()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        while True:
            pcm = self.jobs.get()
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                segments, _ = self.model.transcribe(audio, language="en", vad_filter=False)
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as exc:
                print(f"[transcribe error] {exc}", file=sys.stderr)
                continue
            if text:
                self._handle(text)

    def _handle(self, text: str) -> None:
        print(f"[{self.mode}] {text!r}", flush=True)
        if self.mode == "dictation":
            self._handle_dictation(text)
        else:
            self._handle_command(text)

    def _handle_dictation(self, text: str) -> None:
        lower = text.lower()
        if any(phrase in lower for phrase in EXIT_DICTATION):
            self.mode = "command"
            print("-> exited dictation", flush=True)
            return
        self._paste(text.strip() + " ")

    def _handle_command(self, text: str) -> None:
        cleaned = re.sub(r"^[^\w]+", "", text.lower())
        if not cleaned.startswith(WAKE_WORD):
            return
        remainder = cleaned[len(WAKE_WORD):].lstrip(" ,.:-")
        if not remainder:
            return
        self._dispatch(remainder.rstrip(".!?,"))

    def _dispatch(self, command: str) -> None:
        if command in ("dictate", "start dictating", "start dictation", "begin dictation"):
            self.mode = "dictation"
            print("-> entered dictation", flush=True)
            return
        for verb in ("pull up ", "open ", "launch ", "start ", "switch to "):
            if command.startswith(verb):
                self._open_app(command[len(verb):].strip())
                return
        print(f"-> no match: {command!r}", flush=True)

    def _open_app(self, target: str) -> None:
        app_name = self.aliases.get(target.lower(), target.title())
        result = subprocess.run(["open", "-a", app_name], capture_output=True)
        if result.returncode == 0:
            print(f"-> opened {app_name}", flush=True)
        else:
            print(f"-> couldn't open {app_name!r}: {result.stderr.decode().strip()}", file=sys.stderr)

    def _paste(self, text: str) -> None:
        pyperclip.copy(text)
        with self.kbd.pressed(keyboard.Key.cmd):
            self.kbd.press("v")
            self.kbd.release("v")

    def _on_audio(self, indata, frames, _t, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        pcm = (indata.flatten() * 32767).astype(np.int16).tobytes()
        for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
            done = self.segmenter.feed(pcm[i:i + FRAME_BYTES])
            if done is not None:
                self.jobs.put(done)

    def run(self) -> None:
        print(f"listening. say '{WAKE_WORD}, ...' to issue a command.", flush=True)
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._on_audio,
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nbye.")


if __name__ == "__main__":
    App().run()
