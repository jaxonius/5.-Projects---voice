"""Claude Haiku fallback for voice commands the local patterns didn't match.

Opt-in: only runs if ANTHROPIC_API_KEY is set. If absent, interpret() returns None
and the caller falls through to a "no match" log.
"""

from __future__ import annotations

import json
import os
import sys

import anthropic
from anthropic import Anthropic

MODEL = "claude-haiku-4-5"

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "open_app",
                "close_app",
                "focus_app",
                "start_dictation",
                "stop_dictation",
                "none",
            ],
        },
        "app": {
            "type": "string",
            "description": "Mac app name when action is open_app, close_app, or focus_app.",
        },
        "reason": {
            "type": "string",
            "description": "Brief reason when action is 'none'.",
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}


class ClaudeFallback:
    """Map an unrecognized command to a structured action using Claude Haiku."""

    def __init__(self, app_aliases: dict[str, str]) -> None:
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if not self.enabled:
            print(
                "[fallback] ANTHROPIC_API_KEY not set; Claude fallback disabled.",
                flush=True,
            )
            return
        self.client = Anthropic()
        known = "\n".join(
            f"- {alias} -> {name}" for alias, name in sorted(app_aliases.items())
        )
        self.system = (
            "You map a short Mac voice command to a structured action. "
            "The user has already said the wake word 'computer'; the input you receive "
            "is whatever followed it.\n\n"
            "Actions:\n"
            "- open_app: launch or focus a Mac application. Set 'app' to the Mac app name.\n"
            "- close_app: quit a running app. Set 'app'.\n"
            "- focus_app: bring a running app to the foreground. Set 'app'.\n"
            "- start_dictation: enter dictation mode (speech is typed at the cursor).\n"
            "- stop_dictation: exit dictation mode.\n"
            "- none: command is unclear or out of scope. Set 'reason'.\n\n"
            "Known apps (alias -> Mac app name):\n"
            f"{known}\n\n"
            "If the user mentions an app outside this list, return its name in title case. "
            "Be permissive about phrasing ('fire up', 'gimme', 'bring up' all mean open). "
            "If the request is genuinely ambiguous or unrelated to Mac control, return action=none."
        )

    def interpret(self, command: str) -> dict | None:
        if not self.enabled:
            return None
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=self.system,
                messages=[{"role": "user", "content": command}],
                output_config={
                    "format": {"type": "json_schema", "schema": ACTION_SCHEMA}
                },
            )
            text = next(b.text for b in response.content if b.type == "text")
            return json.loads(text)
        except anthropic.APIError as exc:
            print(f"[fallback] api error: {exc}", file=sys.stderr)
            return None
        except (json.JSONDecodeError, StopIteration) as exc:
            print(f"[fallback] bad response: {exc}", file=sys.stderr)
            return None
