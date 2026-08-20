import os
import time
import json
import urllib.parse
from config import BROWSER_PROFILE_DIR, TRANSCRIPTS_DIR


def _profile_ready() -> bool:
    """True if an authenticated browser profile already exists (user ran --login)."""
    return os.path.isdir(BROWSER_PROFILE_DIR) and any(
        os.path.exists(os.path.join(BROWSER_PROFILE_DIR, name))
        for name in ("Default", "Local State", "Cookies")
    )


def login_to_google():
    """One-time authentication for Google Drive access (required first step).

    Opens a headed Chrome window bound to ``browser_profile/``. Log in with the
    Google account that has access to your files, then CLOSE the browser window.
    The authenticated session is persisted locally and reused by the
    browser-based transcript downloader/scraper on later runs.
    """
    from playwright.sync_api import sync_playwright

    os.makedirs(BROWSER_PROFILE_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("  Google Login (one-time setup)")
    print("=" * 60)
    print(f"  Profile: {BROWSER_PROFILE_DIR}")
    print("  This is the required first step before processing any recording.")
    print("  A Chrome window will open. Sign in with the Google account that")
    print("  can access your Drive recordings, then CLOSE the browser window")
    print("  to finish. The session is saved locally for later runs.\n")

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=BROWSER_PROFILE_DIR,
            headless=False,
            channel="chrome",
            viewport={"width": 1400, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto("https://drive.google.com", wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            print(f"    Navigation notice: {e}")

        print("  [i] Waiting for you to log in and close the browser window...")
        try:
            ctx.wait_for_event("close", timeout=600000)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass

    print("\n  Profile saved. Restricted files will now work with the normal command.\n")


def scrape_gdrive_transcript(file_id: str) -> str:
    """
    Extract a Google Drive / Google Meet transcript using an authenticated
    browser session (requires ``python analyzer.py --login`` to have been run).

    1. Mounts the authenticated persistent browser session in Playwright.
    2. Opens the Google Drive recording URL and activates the video player.
    3. Intercepts Google's signed TimedText API URL and requests the
       WEB_EMBEDDED_PLAYER json3 track.
    4. Parses the full dialogue into timestamped turns.

    Returns the transcript text, or None if captions are unavailable.
    """
    from playwright.sync_api import sync_playwright

    if not _profile_ready():
        print("\n[ℹ️ Browser Scraper] Skipped — no authenticated browser profile found.")
        print("    Run `python analyzer.py --login` once to sign in, then retry this file.")
        return None

    url = f"https://drive.google.com/file/d/{file_id}/view"

    print(f"\n[Browser Scraper] Native transcript extractor (authenticated session)")
    print(f"    Target File ID: {file_id}")
    print(f"    Connecting to Google Drive...")

    base_urls = []

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=BROWSER_PROFILE_DIR,
            headless=False,
            channel="chrome",
            viewport={"width": 1400, "height": 900},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--autoplay-policy=no-user-gesture-required"
            ]
        )

        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        def on_response(response):
            if "timedtext" in response.url:
                base_urls.append(response.url)

        page.on("response", on_response)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            time.sleep(3)
        except Exception as e:
            print(f"    Navigation notice: {e}")

        print("    [1/3] Activating video player and starting playback...")

        # Drive only requests the caption (timedtext) stream while the video is
        # actually playing, so we must get real playback going. Wait for the
        # <video> element to appear, then force muted autoplay.
        try:
            started = page.evaluate("""() => new Promise((resolve) => {
                let tries = 12;
                const attempt = () => {
                    const v = document.querySelector('video');
                    if (v) {
                        v.muted = true;
                        const p = v.play();
                        if (p && p.catch) p.catch(() => {});
                        resolve(true);
                    } else if (tries-- > 0) {
                        setTimeout(attempt, 500);
                    } else {
                        resolve(false);
                    }
                };
                attempt();
            })""")
            print(f"    [1/3] Video element present: {started}")
        except Exception as e:
            print(f"    [1/3] play() notice: {e}")

        time.sleep(3)

        # If playback is still paused (player sometimes needs a click to
        # activate), toggle the video WITHOUT pausing it if already playing.
        try:
            was_paused = page.evaluate("() => { const v = document.querySelector('video'); return v ? v.paused : true; }")
            if was_paused:
                page.mouse.click(700, 450)
                time.sleep(2)
                page.evaluate("""() => {
                    const v = document.querySelector('video');
                    if (v) {
                        v.muted = true;
                        const p = v.play();
                        if (p && p.catch) p.catch(() => {});
                    }
                }""")
                time.sleep(2)
        except Exception:
            pass

        # Nudge playback a few seconds forward — this reliably forces Drive to
        # request the caption track if it hasn't fired yet.
        try:
            page.evaluate("""() => {
                const v = document.querySelector('video');
                if (v && v.readyState > 0 && v.duration > 0) {
                    v.currentTime = Math.min(v.currentTime + 3, v.duration - 1);
                }
            }""")
            time.sleep(2)
        except Exception:
            pass

        try:
            page.keyboard.press("c")       # keyboard CC toggle
            time.sleep(1)
        except Exception:
            pass

        # Click the on-screen captions (CC) button if present.
        try:
            page.click(
                '[aria-label*="captions"], [aria-label*="Captions"], '
                '[title*="captions"], [title*="Subtitles"]',
                timeout=4000,
            )
        except Exception:
            pass

        # Keep watching for the TimedText request — Google may fire it a few
        # seconds into playback.
        deadline = time.time() + 30
        while not base_urls and time.time() < deadline:
            print("    [2/3] Waiting for video playback to load captions...")
            time.sleep(2)

        if not base_urls:
            print("    [!] Could not locate an active TimedText stream. Video may not have captions.")
            ctx.close()
            return None

        print("    [2/3] Constructing authenticated track handshake...")
        u = base_urls[-1]
        parsed_u = urllib.parse.urlparse(u)
        params = urllib.parse.parse_qs(parsed_u.query)

        p_copy = {k: v[0] for k, v in params.items() if k not in ["type", "tlangs", "vssids"]}
        p_copy["type"] = "track"
        p_copy["lang"] = "en"
        p_copy["name"] = ""
        p_copy["kind"] = "asr"
        p_copy["fmt"] = "json3"
        p_copy["c"] = "WEB_EMBEDDED_PLAYER"
        p_copy["cplayer"] = "UNIPLAYER"
        p_copy["cplatform"] = "DESKTOP"

        track_url = urllib.parse.urlunparse((
            parsed_u.scheme,
            parsed_u.netloc,
            parsed_u.path,
            "",
            urllib.parse.urlencode(p_copy),
            ""
        ))

        print("    [3/3] Fetching complete dialogue stream...")
        raw_json_text = page.evaluate("""async (target) => {
            try {
                let r = await fetch(target, { credentials: 'include' });
                return await r.text();
            } catch(e) {
                return null;
            }
        }""", track_url)

        ctx.close()

    if not raw_json_text or "events" not in raw_json_text:
        print("    [!] Failed to retrieve caption track content.")
        return None

    # Keep the raw TimedText payload so transcript caching can re-parse with
    # better turn-splitting (speaker changes + pause gaps) without re-opening
    # the browser.
    try:
        raw_path = os.path.join(TRANSCRIPTS_DIR, f"{file_id}_raw_json.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_json_text)
        print(f"    [Raw] Cached raw TimedText JSON: {raw_path}")
    except Exception as e:
        print(f"    [Raw] Could not cache raw JSON: {e}")

    parsed_transcript = _parse_timedtext_json(raw_json_text)
    if parsed_transcript:
        _save_transcript(file_id, parsed_transcript)
        print(f"    [SUCCESS] Authentic Google Meet Transcript Extracted ({len(parsed_transcript)} chars)!")
        return parsed_transcript

    return None


def _ends_sentence(text: str) -> bool:
    s = text.rstrip()
    return bool(s) and s[-1] in ".?!"


def _parse_timedtext_json(raw_json_str: str) -> str:
    """Parse Google TimedText JSON structure into clean chronological dialogue turns.

    Turns are split on three signals so long stretches where Google fails to
    flag a speaker change still produce readable, Meet-style dialogue blocks:
      1. ``isSpeakerChange`` markers (the authoritative signal).
      2. Pauses >= ``TURN_PAUSE_MS`` between caption events (new speaker/turn).
      3. Sentence boundaries once a turn exceeds ``TURN_MAX_CHARS``, so a
         continuous monologue is broken into capped, sentence-aligned blocks
         instead of one giant wall of text.
    """
    TURN_PAUSE_MS = 2500
    TURN_MAX_CHARS = 350
    TURN_MIN_CHARS = 250
    TURN_HARD_MAX = 900

    try:
        data = json.loads(raw_json_str)
        events = data.get("events", [])

        items = []
        for ev in events:
            t_ms = ev.get("tStartMs", 0)
            d_ms = ev.get("dDurationMs", 0)
            segs = ev.get("segs", [])
            text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
            if not text:
                continue
            is_sc = text.startswith(">>") or any(s.get("isSpeakerChange") for s in segs)
            if is_sc:
                text = text.replace(">>", "").strip()
            items.append((t_ms, d_ms, text, is_sc))

        dialogue_turns = []
        cur = []          # pending (t_ms, text) events in the current turn
        start_ms = None
        cur_end_ms = 0

        def _emit(start, texts):
            mins = int(start // 60000)
            secs = int((start % 60000) // 1000)
            dialogue_turns.append(f"[{mins:02d}:{secs:02d}] {' '.join(texts)}")

        for t_ms, d_ms, text, is_sc in items:
            # 1) speaker change  |  2) long pause after a non-empty turn
            if cur and (is_sc or t_ms - cur_end_ms >= TURN_PAUSE_MS):
                _emit(start_ms, [t for _, t in cur])
                cur = []
                start_ms = None

            if not cur:
                cur.append((t_ms, text))
                start_ms = t_ms
            else:
                cur.append((t_ms, text))

            # 3) sentence-boundary cap for unmarked monologues; hard-split at
            #    TURN_HARD_MAX even without punctuation so nothing grows unbounded
            total = sum(len(t) for _, t in cur)
            if total >= TURN_MAX_CHARS:
                cum = 0
                split_i = None
                for i, (_, t) in enumerate(cur):
                    cum += len(t)
                    if cum >= TURN_MIN_CHARS and _ends_sentence(t):
                        split_i = i
                    elif cum >= TURN_HARD_MAX:
                        split_i = i
                        break
                if split_i is not None and split_i < len(cur) - 1:
                    _emit(start_ms, [t for _, t in cur[:split_i + 1]])
                    cur = cur[split_i + 1:]
                    start_ms = cur[0][0]

            cur_end_ms = t_ms + d_ms

        if cur:
            _emit(start_ms, [t for _, t in cur])

        header = [
            f"# Google Meet Official Transcript",
            f"# Total Dialogue Turns: {len(dialogue_turns)}",
            f"============================================================\n"
        ]

        return "\n".join(header) + "\n\n" + "\n\n".join(dialogue_turns)
    except Exception as e:
        print(f"    [JSON Parse Error] {e}")
        return None


def _save_transcript(file_id: str, text: str):
    """Save transcript text to local file."""
    path = os.path.join(TRANSCRIPTS_DIR, f"{file_id}_transcript.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Saved] Transcript cached: {path} ({len(text)} chars)")
