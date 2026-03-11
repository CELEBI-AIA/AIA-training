"""Emoji decorations for debug-focused console output."""

from __future__ import annotations

import builtins
import re
import sys
from typing import MutableMapping, Any

# Replace bracketed debug tags first so they stay explicit and searchable.
_TAG_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("[CACHE]", "📦 [CACHE]"),
    ("[SETUP]", "🧰 [SETUP]"),
    ("[VERIFY]", "🧪 [VERIFY]"),
    ("[AMP]", "⚙️ [AMP]"),
    ("[RESUME]", "🔄 [RESUME]"),
    ("[LEAKAGE]", "🔍 [LEAKAGE]"),
    ("[BBOX AUDIT]", "🧪 [BBOX AUDIT]"),
    ("[EXCLUDED LABELS]", "📄 [EXCLUDED LABELS]"),
    ("[BEST_MAP50]", "📈 [BEST_MAP50]"),
    ("[CLEANUP]", "🧹 [CLEANUP]"),
)

# Status words remain in output for grepability.
_WORD_EMOJIS: tuple[tuple[str, str], ...] = (
    (r"\bCLOUD\b", "☁️"),
    (r"\bERROR\b", "🚫"),
    (r"\bWARN\b", "⚠️"),
    (r"\bINFO\b", "🔍"),
    (r"\bFAILED\b", "❌"),
    (r"\bFAIL\b", "❌"),
    (r"\bPASSED\b", "✅"),
    (r"\bOK\b", "✅"),
)

# Optional context phrases used heavily in training/debug output.
_PHRASE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("GPU Status", "🖥️ GPU Status"),
    ("TRAINING CONFIGURATION", "⚙️ TRAINING CONFIGURATION"),
    ("FINAL RESULTS", "📊 FINAL RESULTS"),
    ("PER-CLASS mAP50", "📊 PER-CLASS mAP50"),
    ("Running tests", "🧪 Running tests"),
    ("Running Pipeline Audit", "🔍 Running Pipeline Audit"),
    ("Dataset built successfully", "✅ Dataset built successfully"),
    ("Training completed successfully", "✅ Training completed successfully"),
    ("Starting training", "🤖 Starting training"),
)


def _stdout_supports_emoji() -> bool:
    """Return True when current stdout encoding can render emoji safely."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "✅".encode(encoding)
        return True
    except Exception:
        return False


def add_debug_emojis(message: str) -> str:
    """Decorate known status/tags with emojis while keeping original tokens."""
    if not _stdout_supports_emoji():
        return message

    line = message

    for token, replacement in _TAG_REPLACEMENTS:
        if token in line and replacement not in line:
            line = line.replace(token, replacement)

    for pattern, emoji in _WORD_EMOJIS:
        line = re.sub(pattern, lambda m, e=emoji: f"{e} {m.group(0)}", line)

    for token, replacement in _PHRASE_REPLACEMENTS:
        if token in line and replacement not in line:
            line = line.replace(token, replacement)

    return line


def install_emoji_print(namespace: MutableMapping[str, Any]) -> None:
    """Patch module-level print to apply emoji decorations to string arguments."""
    current_print = namespace.get("print", builtins.print)
    if getattr(current_print, "_emoji_wrapped", False):
        return

    def _emoji_print(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        formatted_args = tuple(
            add_debug_emojis(arg) if isinstance(arg, str) else arg
            for arg in args
        )
        current_print(*formatted_args, **kwargs)

    setattr(_emoji_print, "_emoji_wrapped", True)
    namespace["print"] = _emoji_print
