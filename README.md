# Voice — hands-free Mac control

Always-on voice assistant for a Mac mini. No keyboard, no mouse, no clicks.

- Wake word: **"computer, ..."**
- **`computer, pull up DaVinci Resolve`** — launches the app.
- **`computer, open Claude`** — launches Claude.
- **`computer, dictate`** — enters dictation mode; everything you say gets pasted at the cursor.
- **`stop dictating`** — returns to command mode.

Whisper runs on-device — audio never leaves the machine. Novel commands ("fire up Resolve", "quit Slack", "the video editor") can optionally fall back to Claude Haiku for intent extraction.

## Setup (Mac mini, Apple Silicon)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python dictate.py
```

First run downloads the `small.en` Whisper model (~150 MB) to `~/.cache/huggingface/`.

## macOS permissions

Grant the terminal (or whatever runs `dictate.py`) all three:

1. **Microphone** — *System Settings → Privacy & Security → Microphone*
2. **Input Monitoring** — required for `pynput` to read modifier state when pasting
3. **Accessibility** — required to simulate Cmd+V into other apps

You'll get prompted on first use of each.

## How it works

1. Mic streams continuously into a WebRTC VAD.
2. VAD segments speech into utterances (~150ms voiced → start, ~750ms silence → end).
3. Each utterance is transcribed locally by `faster-whisper` (small.en, int8 on CPU).
4. **Command mode** (default): transcript must start with `computer`. Verbs handled locally: `pull up`, `open`, `launch`, `start`, `switch to`, plus `dictate` to switch modes.
5. **Claude fallback** (optional): if the local pattern doesn't match and `ANTHROPIC_API_KEY` is set, the transcript is sent to Claude Haiku 4.5 with a closed schema (`open_app`, `close_app`, `focus_app`, `start_dictation`, `stop_dictation`, `none`). The result is dispatched the same way as a local match.
6. **Dictation mode**: each utterance is copied to clipboard and pasted via Cmd+V into the focused field. Exits on `stop dictating` / `end dictation` / `done dictating`.

## Enabling the Claude fallback

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python dictate.py
```

Without the env var, fallback is disabled and unmatched commands log `no match`. Cost is a few cents per day at typical use — Haiku 4.5 is $1/$5 per 1M tokens, and each call is ~500 tokens.

## Adding apps

Edit `apps.json` to map a spoken phrase → an installed app name (the same name `open -a` accepts). Unknown phrases fall back to title-casing the spoken text and trying `open -a` directly, so most apps work without an alias.

## Roadmap

- v1.1: menu-bar indicator (rumps) showing current mode.
- v1.2: voice-driven alias learning ("computer, learn app Figma").
- v2: free-form Claude actions beyond the closed schema ("draft a reply to the last email").
- v2: context-aware dictation (code formatting in terminal, prose in Notion).
- v2: custom vocabulary biasing via Whisper's `initial_prompt` (Offframe, Janelle, Sebastian, TPx).

## Known limitations

- Whisper sometimes hallucinates `"Thank you."` on noisy silence. Rare in practice because VAD filters non-speech first.
- The wake word is a string match on Whisper output, not a dedicated wake-word model — robust enough for personal use, not robust enough for a room full of people. Upgrade path: Picovoice Porcupine.
- No streaming transcription; latency = utterance length + ~0.5–1s. Fine for short commands, noticeable for long dictation.
