# Viva AI Analyzer

Turns Google Drive / Google Meet viva (oral exam) recordings into graded, TA-ready
**Markdown** and **HTML** reports.

It fetches the recording's captions/transcript, chunks it into 15-minute segments,
digests each segment with an LLM, then synthesizes a full assessment: executive
summary, action items, performance scorecard, topic competency matrix, key technical
questions, audit notes, and a chronological chapter log.

## Setup (one-time)

### 1. Log in to Google (required)

```
python analyzer.py --login
```

A Chrome window opens. Sign in with the Google account that can access your
Drive recordings, then close the window. The authenticated session is saved to
`browser_profile/` and reused on later runs. This step is required before
processing any recording.

### 2. Add your API keys

The pipeline uses either **Groq** and/or **OpenRouter** models. Keys are read from
the `GROQ_API_KEYS` environment variable as a comma-separated list, and the code
**automatically detects** which provider each key belongs to by its prefix:

| Provider    | Key prefix | Where to get one          |
|-------------|------------|---------------------------|
| Groq        | `gsk_...`  | https://console.groq.com/keys |
| OpenRouter  | `sk-or-...`| https://openrouter.ai/keys    |

You may mix both providers in the same list — the router round-robins across all
keys and routes each to the right endpoint.

PowerShell:
```powershell
$env:GROQ_API_KEYS = "gsk_xxx,sk-or-yyy"
```

Bash:
```bash
export GROQ_API_KEYS="gsk_xxx,sk-or-yyy"
```

Alternatively, create a `.env` file in this folder (see `.env` keys below);
it is loaded automatically. Never commit `.env` or your real keys.

### 3. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium   # optional, only if system Chrome is unavailable
```

`ffmpeg` must be on your PATH (used only for audio transcription fallback).

## Usage

```bash
python analyzer.py "https://drive.google.com/file/d/FILE_ID/view"
python analyzer.py "LINK_1" "LINK_2" ...        # multiple links
python analyzer.py --file links.txt             # one link per line, # comments allowed
```

### Outputs

- `reports/{file_id}_report.md` — the full graded report
- `reports/{file_id}_report.html` — interactive visual dashboard
- `transcripts/{file_id}_transcript.txt` — cached transcript (fast re-runs)

## How it works

1. **Transcript fetch** — reads captions from the Drive timedtext endpoint via the
   authenticated session; falls back to downloading the video + Whisper transcription
   if no captions exist.
2. **Report generation (2-pass)** — digests each 15-min segment (Pass 1), then
   synthesizes the full scorecard with token-aware grounding (Pass 2). Chapters and
   metrics are checked against actual dialogue coverage so silent stretches don't
   skew the report.
3. **HTML dashboard** — renders the report as an interactive page.

## Project layout

```
analyzer.py            CLI entry point
transcript_fetcher.py  transcript acquisition (cache → timedtext → browser → whisper)
report_generator.py    2-pass LLM report generation + key/model router
html_generator.py      interactive HTML dashboard
browser_scraper.py     authenticated browser session (--login) + caption scraping
gdrive_downloader.py   video download (public or authenticated)
audio_extractor.py     ffmpeg audio extraction for whisper fallback
config.py              configuration + API key loading (env / .env)
```

## Notes

- Reports and transcripts are cached; re-running a link skips network/Whisper work.
- Never commit `browser_profile/`, `cookies.txt`, `.env`, or anything under
  `transcripts/`, `audio/`, `downloads/`, `reports/` (all gitignored).