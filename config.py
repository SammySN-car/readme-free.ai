"""
Configuration — reads everything from environment variables, never hardcodes keys.

API KEYS (auto-detected by prefix, mixed pools allowed):
    - Groq keys      start with "gsk_"    (https://console.groq.com/keys)
    - OpenRouter keys start with "sk-or-" (https://openrouter.ai/keys)

Set them via the GROQ_API_KEYS environment variable as a comma-separated list.
The pipeline rotates across all keys and picks the correct provider per key.

    PowerShell:  $env:GROQ_API_KEYS = "gsk_...,sk-or-..."
    Bash:        export GROQ_API_KEYS="gsk_...,sk-or-..."

You may also put them in a `.env` file (one KEY=VALUE per line); it is loaded
if present. Never commit `.env` or real keys — see `.gitignore`.
"""

import os

# ------------------------------------------------------------
#  API KEYS — put your Groq and/or OpenRouter keys here
# ------------------------------------------------------------
if os.path.exists(".env"):
    with open(".env", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

raw_keys = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
GROQ_API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

# Primary key (used for Whisper transcription fallback)
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""

# ------------------------------------------------------------
#  Models
# ------------------------------------------------------------
WHISPER_MODEL = "whisper-large-v3"

# ------------------------------------------------------------
#  Directories (runtime data is gitignored)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "transcripts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
BROWSER_PROFILE_DIR = os.path.join(BASE_DIR, "browser_profile")

for d in [DOWNLOADS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


def google_profile_ready() -> bool:
    """True if the user has run `python analyzer.py --login` (a Chrome profile
    with a saved Google session exists). Single source of truth shared by the
    CLI, transcript fetcher, and browser scraper."""
    return os.path.isdir(BROWSER_PROFILE_DIR) and os.path.exists(
        os.path.join(BROWSER_PROFILE_DIR, "Default")
    )