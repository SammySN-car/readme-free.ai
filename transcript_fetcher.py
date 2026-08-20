import os
import json
import urllib.request
import xml.etree.ElementTree as ET
from config import TRANSCRIPTS_DIR, BROWSER_PROFILE_DIR
from gdrive_downloader import download_from_gdrive, extract_file_id
from audio_extractor import extract_audio_from_video


def _auth_profile_ready() -> bool:
    """True if the user has run `python analyzer.py --login` (browser auth exists)."""
    return os.path.isdir(BROWSER_PROFILE_DIR) and os.path.exists(
        os.path.join(BROWSER_PROFILE_DIR, "Default")
    )


def fetch_transcript_from_gdrive(file_id_or_url: str) -> dict:
    """
    Get the transcript for a Google Drive / YouTube video using a multi-tiered
    strategy. Publicly accessible files need NO login at all:

    1. Local Cache Check (reads existing .txt or auto-parses raw TimedText JSON).
    2. Google Drive TimedText API endpoint (public captions).
    3. Authenticated Browser Scraper — ONLY if `python analyzer.py --login` has
       been run (needed for domain/org-restricted files).
    4. Fallback: Download video -> ffmpeg audio -> Groq Whisper.

    Returns:
        dict: {
            "text": str (full transcript text),
            "segments": list (optional timestamp segments),
            "method": str (which strategy succeeded),
            "file_id": str
        }
    """
    file_id = extract_file_id(file_id_or_url)
    print(f"\n{'='*60}")
    print(f"[Transcript] Processing GDrive file: {file_id}")
    print(f"{'='*60}")

    # -------------------------------------------------------------
    # 1. Local Cache Check (.txt or .json)
    # -------------------------------------------------------------
    cache_txt = os.path.join(TRANSCRIPTS_DIR, f"{file_id}_transcript.txt")
    cache_json = os.path.join(TRANSCRIPTS_DIR, f"{file_id}_transcript.json")

    for path in [cache_txt, cache_json]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # If it's a raw JSON file, parse it automatically into clean formatted text
            if content.startswith("{") and "events" in content:
                print(f"[Cache] Found raw Google TimedText JSON in cache: parsing dialogue...")
                parsed = _parse_timedtext_json_content(content)
                if parsed:
                    content = parsed
                    with open(cache_txt, "w", encoding="utf-8") as f:
                        f.write(content)

            if len(content) > 50:
                print(f"[Cache] Loaded validated transcript: {path}")
                return {
                    "text": content,
                    "segments": [],
                    "method": "local_cache",
                    "file_id": file_id
                }

    # -------------------------------------------------------------
    # 2. TimedText Direct API Endpoint (public captions, no login)
    # -------------------------------------------------------------
    print(f"[Attempt] Trying Google Drive timedtext endpoint...")
    timedtext_result = _try_timedtext_endpoint(file_id)
    if timedtext_result:
        _save_transcript(file_id, timedtext_result)
        return {
            "text": timedtext_result,
            "segments": [],
            "method": "timedtext_api",
            "file_id": file_id
        }

    # -------------------------------------------------------------
    # 3. Browser Scraper (authenticated session, only if --login was run)
    # -------------------------------------------------------------
    if _auth_profile_ready():
        try:
            from browser_scraper import scrape_gdrive_transcript
            scraped_text = scrape_gdrive_transcript(file_id)
            if scraped_text and len(scraped_text) > 100:
                print(f"[Browser Scraper] Successfully extracted full transcript!")
                return {
                    "text": scraped_text,
                    "segments": [],
                    "method": "browser_scraper",
                    "file_id": file_id
                }
        except Exception as e:
            print(f"[Browser Scraper] Scraper notice: {e}")

    # -------------------------------------------------------------
    # 4. Fallback: Download Video -> Compress Audio -> Groq Whisper
    # -------------------------------------------------------------
    print(f"\n[Fallback] Downloading video and transcribing with Groq Whisper...")
    try:
        video_path = download_from_gdrive(file_id)
        audio_path = extract_audio_from_video(video_path, file_id)

        from groq import Groq
        from config import GROQ_API_KEY, WHISPER_MODEL

        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model=WHISPER_MODEL,
                response_format="verbose_json",
                language="en"
            )

        transcript_text = transcription.text
        segments = getattr(transcription, "segments", [])

        # Format segments with timestamps
        if segments:
            formatted_lines = []
            for seg in segments:
                start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
                text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
                mins = int(start // 60)
                secs = int(start % 60)
                formatted_lines.append(f"[{mins:02d}:{secs:02d}] {text.strip()}")
            transcript_text = "\n".join(formatted_lines)

        _save_transcript(file_id, transcript_text)

        return {
            "text": transcript_text,
            "segments": segments,
            "method": "whisper_fallback",
            "file_id": file_id
        }
    except Exception as e:
        print(f"[Error] Audio extraction fallback failed: {e}")

    # -------------------------------------------------------------
    # All tiers failed — give the user actionable guidance
    # -------------------------------------------------------------
    hint = (
        "\n[Hint] No public transcript/captions were found for this file.\n"
        "    - If the file is PUBLIC, it may have no auto-generated captions "
        "(only a video download is possible, which already failed).\n"
        "    - If the file is RESTRICTED to your org/domain, authenticate once with:\n"
        "          python analyzer.py --login\n"
        "      then re-run this same command."
    )
    print(hint)
    return {"text": "", "segments": [], "method": "failed", "file_id": file_id}


# Alias for backward compatibility
get_transcript = fetch_transcript_from_gdrive


def _try_timedtext_endpoint(file_id: str) -> str:
    """Try fetching captions from Google Drive's timedtext endpoint."""
    urls = [
        f"https://drive.google.com/timedtext?v={file_id}&lang=en&fmt=srv3",
        f"https://drive.google.com/timedtext?v={file_id}&lang=en&fmt=vtt",
        f"https://drive.google.com/timedtext?v={file_id}&lang=en",
        f"https://video.google.com/timedtext?v={file_id}&lang=en&fmt=vtt",
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8")
                if "events" in content:
                    parsed = _parse_timedtext_json_content(content)
                    if parsed:
                        return parsed
                elif "<transcript>" in content or "<text" in content:
                    root = ET.fromstring(content)
                    lines = []
                    for elem in root.findall(".//text"):
                        start = float(elem.attrib.get("start", 0))
                        mins = int(start // 60)
                        secs = int(start % 60)
                        text = elem.text or ""
                        text = text.replace("&#39;", "'").replace("&quot;", '"').replace("&amp;", "&")
                        lines.append(f"[{mins:02d}:{secs:02d}] {text.strip()}")
                    if lines:
                        return "\n".join(lines)
        except Exception:
            continue
    return None


def _parse_timedtext_json_content(raw_json_str: str) -> str:
    """Parse Google TimedText JSON structure into clean chronological dialogue turns."""
    try:
        data = json.loads(raw_json_str)
        events = data.get("events", [])
        dialogue_turns = []
        current_time = "00:00"
        current_words = []

        for ev in events:
            t_ms = ev.get("tStartMs", 0)
            mins = int(t_ms // 60000)
            secs = int((t_ms % 60000) // 1000)
            timestamp = f"{mins:02d}:{secs:02d}"

            segs = ev.get("segs", [])
            text = "".join(s.get("utf8", "") for s in segs)
            text = text.replace("\n", " ").strip()
            if not text:
                continue

            if text.startswith(">>"):
                if current_words:
                    dialogue_turns.append(f"[{current_time}] {' '.join(current_words)}")
                    current_words = []
                current_time = timestamp
                text = text[2:].strip()
                current_words.append(text)
            else:
                if not current_words:
                    current_time = timestamp
                current_words.append(text)

        if current_words:
            dialogue_turns.append(f"[{current_time}] {' '.join(current_words)}")

        return "\n\n".join(dialogue_turns)
    except Exception:
        return None


def _save_transcript(file_id: str, text: str):
    """Save transcript text to local file."""
    path = os.path.join(TRANSCRIPTS_DIR, f"{file_id}_transcript.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Saved] Transcript cached: {path} ({len(text)} chars)")
