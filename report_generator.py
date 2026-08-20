import os
import re
import time
from groq import Groq
from config import GROQ_API_KEYS, REPORTS_DIR
import tiktoken


# ============================================================
#  PROVIDER SUPPORT
#  Groq keys (default) are used directly. OpenRouter keys
#  (sk-or-...) use the OpenAI-compatible API via the `openai`
#  SDK. Mixed pools work: each key is routed to its provider.
# ============================================================
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _is_openrouter_key(key: str) -> bool:
    return key.strip().startswith("sk-or-")


def _make_client(key: str):
    if _is_openrouter_key(key):
        from openai import OpenAI
        return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
    return Groq(api_key=key)


# ============================================================
#  MODEL POOL — reliability-ordered. qwen3.6-27b is currently the
#  most reliable free-tier Groq model (gpt-oss models have
#  intermittently returned empty content on all keys, so they are
#  fallbacks). OpenRouter keys use their own pool below.
# ============================================================
MODEL_POOL = [
    "qwen/qwen3.6-27b",        # reliable
    "openai/gpt-oss-120b",     # fallback (recently empty on free tier)
    "openai/gpt-oss-20b",      # fallback
]

OPENROUTER_MODEL_POOL = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",   # verified: exact chapter format
    "nvidia/nemotron-3-super-120b-a12b:free",    # verified: exact chapter format
    "google/gemma-4-31b-it:free",                # fallback
]

CALL_COOLDOWN_SECS = 4

# Free-tier hard per-request cap (Groq returns a 413 past this).
TPM_CAP = 8000
# Hard token budget for the synthesis output. The input is dynamically sized so
# that estimated_input_tokens + max_tokens never exceeds TPM_CAP. gpt-oss-120b's
# verbose full report measures ~8.5K chars (~3000 tok), so 3500 gives headroom.
SYNTH_MAX_TOKENS = 3500
SYNTH_TOKEN_SAFETY = 300

_ENC = None

# cl100k under-counts vs Groq's per-model tokenizers (~16% on measured inputs).
# Multiplying by this correction keeps the clamped output budget conservative
# enough that a retry (which resends the same content) still gets a large
# output budget instead of being squeezed to near-zero.
_EST_CORRECTION = 1.2


def _estimate_tokens(text: str) -> int:
    """Best-effort token estimate (cl100k). Only used for request-budget sizing."""
    if not text:
        return 0
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text))


def _budget_tokens(text: str) -> int:
    """Conservative token estimate (cl100k * correction) for budget decisions."""
    return int(_estimate_tokens(text) * _EST_CORRECTION)

# Top-level sections a complete synthesis report must contain.
REQUIRED_SECTIONS = [
    "Executive Summary",
    "Action Items",
    "Performance Scorecard",
    "Topic Competency Matrix",
    "Key Technical Questions",
    "Audit Notes",
]

# Fragments that indicate the model leaked its instructions/thinking into output.
LEAK_MARKERS = [
    "<Title>",
    "Determine Chapter Boundaries",
    "START DIRECTLY",
    "CRITICAL RULES",
    "Zero hallucinations",
    "CRITICAL INSTRUCTIONS",
    "OUTPUT IN THE EXACT FOLLOWING STRUCTURE",
]


def _clean_chapter_digest(text: str) -> str:
    """Extract clean timestamped chapters, discarding thinking traces.

    Returns '' when no valid chapter header exists so the router retries
    instead of leaking reasoning/instructions into the report.
    """
    if not text:
        return ""
    text = re.sub(r'^.*?<response>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</response>\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*```(?:markdown)?\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # First valid chapter header, e.g. "### [00:05 - 00:10] — Title",
    # "### 00:05 - 00:10 Title", "### Chapter 1: [00:05 - 00:10] — Title",
    # or a bolded variant.
    match = re.search(
        r'^#{2,3}\s*(?:\*{1,2}\s*)?(?:chapter\s+\d+\s*[:\-–.]?\s*)?\**\s*'
        r'[`\[]?\d{1,2}:\d{2}',
        text, flags=re.MULTILINE | re.IGNORECASE)
    if match:
        return text[match.start():].strip()

    return ""


def _clean_synthesis_response(text: str) -> str:
    """Extract clean synthesis output, discarding preamble/thinking.

    Returns '' when the output contains no valid report header (e.g. the
    model only echoed its instructions in a 'thinking' block), so the router
    treats it as a failed call and tries the next model/key.
    """
    if not text:
        return ""
    text = re.sub(r'^.*?<response>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</response>\s*$', '', text, flags=re.IGNORECASE)

    match = re.search(r'^#{2}\s*.*?Executive Summary', text, flags=re.MULTILINE | re.IGNORECASE)
    if match:
        return text[match.start():].strip()
    return ""


class ModelRouter:
    """
    Multi-Key & Multi-Model Smart Router:
    - Rotates across multiple Groq API keys (different accounts)
    - Rotates across models to distribute load evenly
    - Auto-blacklists exhausted keys or models with zero downtime
    """

    def __init__(self, api_keys: list[str], models: list[str]):
        self._keys = [k for k in api_keys if k.strip()]
        self._models = list(models)
        self._blacklisted_keys: set[str] = set()
        self._blacklisted_models: set[str] = set()
        self._exhausted_key_models: set[tuple] = set()  # (key, model) pairs at daily TPD
        self._call_count = 0

    def _get_client_and_models(self, skip=None):
        skip = skip or set()
        active_keys = [k for k in self._keys if k not in self._blacklisted_keys and k not in skip]
        if not active_keys:
            raise RuntimeError(
                "All Groq API keys have hit their daily quota. "
                "Add more keys in config.py or wait for daily reset (~midnight UTC)."
            )

        key = active_keys[self._call_count % len(active_keys)]
        client = _make_client(key)

        if _is_openrouter_key(key):
            pool = OPENROUTER_MODEL_POOL
        else:
            pool = self._models

        active_models = [
            m for m in pool
            if m not in self._blacklisted_models and (key, m) not in self._exhausted_key_models
        ]
        if not active_models:
            self._blacklisted_keys.add(key)
            return self._get_client_and_models()

        start = self._call_count % len(active_models)
        ordered_models = active_models[start:] + active_models[:start]

        return client, key, ordered_models

    def blacklist_model(self, model: str):
        self._blacklisted_models.add(model)
        remaining = [m for m in self._models if m not in self._blacklisted_models]
        print(f"    [WARN] [{model.split('/')[-1]}] model exhausted — switching engine. "
              f"Active engines: {len(remaining)}")

    def blacklist_key(self, key: str):
        self._blacklisted_keys.add(key)
        remaining = [k for k in self._keys if k not in self._blacklisted_keys]
        print(f"    [WARN] [API Key ...{key[-6:]}] quota exhausted — switching to next account key. "
              f"Active keys: {len(remaining)}")

    def call(self, messages: list, max_tokens: int,
             temperature: float = 0.0, allow_tpm_wait: bool = False, is_synthesis: bool = False,
             raw: bool = False) -> str:
        """Execute completion across multiple accounts & models, rotating and auto-blacklisting.
        Keys are blacklisted ONLY on genuine daily-quota errors; soft failures
        (empty output) rotate to the next key without burning it.

        raw=True accepts any non-empty response without header validation —
        for calls that legitimately produce output without the standard
        report/chapter headers (e.g. generating only missing sections)."""
        last_error = None
        tried: set = set()

        while True:
            try:
                client, key, models = self._get_client_and_models(skip=tried)
            except RuntimeError as e:
                raise last_error or e

            success = False
            for model in models:
                short = model.split('/')[-1]

                for attempt in range(2):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=90.0,
                        )
                        raw_content = response.choices[0].message.content or ""
                        if raw:
                            content = raw_content.strip()
                        else:
                            content = _clean_synthesis_response(raw_content) if is_synthesis else _clean_chapter_digest(raw_content)

                        if content and len(content.strip()) > 0:
                            print(f"        [OK] [{short} | Key ...{key[-6:]}] Success ({len(content)} chars)")
                            self._call_count += 1
                            return content.strip()
                        else:
                            print(f"        [SKIP] [{short}] Empty cleaned response, trying next...")
                            break

                    except Exception as e:
                        err_str = str(e).lower()
                        is_too_large = ("request too large" in err_str
                                        or "reduce your message size" in err_str)
                        is_tpm = "tokens per minute" in err_str or "tpm" in err_str
                        # Daily-quota is PERMANENT (blacklist). Bare "rate limit"/"quota"
                        # wording is usually a transient 429 (RPM) and must NOT blacklist.
                        is_daily = ("insufficient_quota" in err_str
                                    or "daily rate limit" in err_str
                                    or ("tokens per day" in err_str and "per minute" not in err_str)
                                    or "daily limit" in err_str
                                    or "free model rate limit" in err_str
                                    or "free tier" in err_str
                                    or "quota exceeded" in err_str)
                        is_rate = "429" in err_str or "rate_limit" in err_str or "413" in err_str \
                            or "rate limit" in err_str or "quota" in err_str
                        is_network = "connection" in err_str or "timeout" in err_str or "unavailable" in err_str

                        last_error = e

                        if is_too_large:
                            print(f"        [WARN] [{short}] Request too large for this model — skipping (no wait).")
                            break

                        elif is_tpm and allow_tpm_wait and attempt == 0:
                            print(f"        [WAIT] [{short}] TPM limit — waiting 60s for window reset...")
                            time.sleep(62)
                            continue

                        elif is_rate and allow_tpm_wait and attempt == 0:
                            print(f"        [WAIT] [{short}] Rate limited — waiting 25s and retrying once...")
                            time.sleep(25)
                            continue

                        elif is_daily:
                            self._exhausted_key_models.add((key, model))
                            print(f"        [WARN] [API Key ...{key[-6:]} | {short}] daily TPD exhausted — "
                                  f"skipping this key for that model.")
                            break

                        elif is_rate:
                            print(f"        [WARN] [{short}] Rate limited, switching engine...")
                            break

                        elif is_network and attempt == 0:
                            print(f"        [RETRY] [{short}] Network hiccup, retrying...")
                            time.sleep(3)
                            continue

                        else:
                            print(f"        [ERROR] [{short}] Error: {str(e)[:80]}")
                            break

            if not success:
                tried.add(key)
                if len(tried) >= len(self._keys):
                    raise last_error or RuntimeError("All Groq LLM models failed for this request.")
                continue

        raise last_error or RuntimeError("All Groq LLM models failed for this request.")


# ============================================================
#  SEGMENT PROMPT — 15 min chunk → grounded chronological chapters
# ============================================================
SEGMENT_SYSTEM_PROMPT = (
    "You are an academic meeting intelligence auditor for university Teaching Assistants and Instructors.\n"
    "You will receive a 15-minute segment of a viva recording transcript. Every dialogue line has a real "
    "[MM:SS] timestamp from the actual recording.\n\n"
    "TASK:\n"
    "Break the segment into chronological chapters. Group consecutive dialogue turns into meaningful chapters "
    "(a question plus its answer, a demo walkthrough, a topic change). Use the ACTUAL timestamps from the "
    "transcript for each chapter's time range.\n\n"
    "OUTPUT FORMAT — for each chapter output:\n"
    "### [StartTime - EndTime] — Descriptive Chapter Title\n"
    "- **Discussion & Events:** 2-3 sentence factual narrative of what took place.\n"
    "- **Key Topics & Q&A:** Bullet points with the exact questions asked and the candidate's explanations.\n"
    "- **Auditor Notes:** Accuracy verdict, confidence, hesitations, gaps, screen sharing, or TA flags.\n\n"
    "RULES:\n"
    "1. START DIRECTLY with the first '###' header. Output ONLY the chapter markdown. Any text before the "
    "first '###' header will be discarded, so never include thinking, planning, or instructions.\n"
    "2. Chapter time ranges MUST come from the actual timestamps in the transcript segment. Do not invent "
    "times or content that do not appear there.\n"
    "3. Zero hallucinations — rely 100% on the transcript. If a topic is only mentioned briefly, say so in "
    "the Auditor Notes rather than elaborating.\n"
    "4. COVERAGE: produce a chapter for EVERY distinct question/topic exchange so the log covers the whole "
    "segment without gaps. Do not skip stretches of the segment. If a stretch contains only brief "
    "acknowledgments ('Yes sir.', 'Okay.') and no Q&A, still emit a SHORT chapter for it — describe in "
    "the Auditor Notes that it was acknowledgment-only, never claim 'no substantive discussion' for a "
    "span that actually contains questions or explanations.\n"
    "5. Aim for several short chapters rather than one long one when the segment spans multiple topics."
)


# ============================================================
#  SYNTHESIS PROMPT — Multi-candidate meeting evaluation
# ============================================================
REPORT_SYNTHESIS_PROMPT = """You are an academic meeting intelligence evaluator for university TAs and course instructors.

Inputs:
A) GROUNDING EXCERPT: dialogue turns from the ACTUAL transcript, each with a real [MM:SS] timestamp. Source of truth.
B) CHAPTER DIGESTS: chronological chapter summaries of the entire session.

Use BOTH. Prefer the GROUNDING EXCERPT for exact timestamps, exact questions, and exact quotes.

CRITICAL RULES:
1. Identify candidates only by names explicitly stated (e.g. "my name is ..."); otherwise "Candidate 1", "Candidate 2". Never invent names, colleges, titles, roles, or backgrounds.
2. Action items MUST be grounded: only actions explicitly requested or acknowledged, each with the exact [MM:SS] from the GROUNDING EXCERPT. List fewer rather than fabricate.
3. Start DIRECTLY with '## ⚡ Executive Summary (For TAs & Mentors)'. No preamble, thinking, or instructions.
4. Every markdown table row MUST be on its own line and start/end with |.

GROUNDING EXCERPT (from the raw transcript):
{grounding}

CHAPTER DIGESTS ACROSS ENTIRE SESSION:
{window_summaries}

OUTPUT ONLY THE REPORT MARKDOWN IN THIS EXACT STRUCTURE:

## ⚡ Executive Summary (For TAs & Mentors)
**Session Overview:** 2-3 sentences: full session duration, total candidates evaluated, key technical domains probed, overall verdicts.

### 📌 Participant Breakdown & Core Highlights:
- **Candidate N:** [name if stated] — [what was evaluated, core strengths, identified gaps, examiner feedback]
- **Examiner / Instructor:** [name only if stated]

---

## ⏱️ Assigned Action Items & Next Steps
Markdown table with columns | Timestamp | Assignee | Action |, one row per action. Example row: | [20:20] | Candidate 1 | Show the ADA chart again as requested by the examiner. |. Include EVERY explicit examiner request (e.g. share screen/ID, walk me through your project report, zoom in on the chart, open a blank cell, come back to the ADA), each with its real [MM:SS]. If a request turn is missing from the GROUNDING EXCERPT, reconstruct it from the CHAPTER DIGESTS (label the action "reconstructed"). Never invent requests that were not made.

---

## 📊 Performance Scorecard & Evaluation Rubric
Markdown table | Candidate / Metric | Score / Verdict | Benchmark Status | Examiner Notes | with rows for Overall Rating, Technical Depth, and Live Execution per candidate (X.X/10; Distinction/Merit/Pass/Borderline or 🟢 Strong/🟡 Adequate/🔴 Weak).

---

## 🎯 Topic Competency Matrix
Markdown table | Topic / Concept | Evaluated Candidate | Competency Level | Rating (/10) | Auditor / Examiner Notes |, one row per probed topic.

---

## ❓ Key Technical Questions & Verdicts
1. **"[Exact question from the transcript]?"** (Asked to: [Candidate]) — *Result: [✅ Correct / ⚠️ Partially Correct / ❌ Gaps Identified]*: [analysis of the response]. If verbatim text is missing from the GROUNDING EXCERPT, reconstruct from the CHAPTER DIGESTS (label "reconstructed"). Never invent questions.

---

## 🎯 TA & Instructor Audit Notes
- **Candidate N Audit Notes:** specific flags, hesitations, disputed answers, live-coding observations, gaps, confidence.
- **Overall Recommendation:** recommendation based on the actual performance evidence in the transcript.
"""

SYNTHESIS_SYSTEM_PROMPT = (
    "You are an academic meeting intelligence evaluator. Generate the multi-candidate executive summary, "
    "scorecard, topic matrix, questions, and audit notes in clean Markdown. "
    "Output ONLY the report markdown. Start directly with '## ⚡ Executive Summary (For TAs & Mentors)'. "
    "Ground every timestamp, action item, and attribution in the GROUNDING EXCERPT. "
    "Never invent names, tasks, or facts that are not in the transcript."
)


# ============================================================
#  Deterministic transcript preparation helpers
# ============================================================

def _split_into_segments(transcript_text: str, segment_minutes: int = 15) -> list:
    """Split transcript into chronological 15-minute segments matching actual duration."""
    lines = [l.strip() for l in transcript_text.split('\n')
             if l.strip() and not l.startswith('#') and not l.startswith('===')]

    # Find the maximum timestamp in the transcript
    max_min = 0
    for line in lines:
        t_match = re.match(r'\[?(\d{1,2}):(\d{2})\]?', line)
        if t_match:
            mins = int(t_match.group(1))
            if mins > max_min:
                max_min = mins

    total_duration_mins = max_min + 1

    segments = []
    current_lines = []
    current_start = 0

    for line in lines:
        t_match = re.match(r'\[?(\d{1,2}):(\d{2})\]?', line)
        if t_match:
            mins = int(t_match.group(1))
            seg_start = (mins // segment_minutes) * segment_minutes

            if seg_start != current_start:
                if current_lines:
                    seg_end = min(current_start + segment_minutes, total_duration_mins)
                    segments.append({
                        "start": f"{current_start:02d}:00",
                        "end": f"{seg_end:02d}:00",
                        "text": "\n".join(current_lines)
                    })
                    current_lines = []
                current_start = seg_start

        current_lines.append(line)

    if current_lines:
        seg_end = min(current_start + segment_minutes, total_duration_mins)
        segments.append({
            "start": f"{current_start:02d}:00",
            "end": f"{seg_end:02d}:00",
            "text": "\n".join(current_lines)
        })

    return segments


_FILLER_PREFIX = re.compile(
    r'^(?:uh|um|umm|uhh|uhm|ah|hmm|mm|eh|er|erm|like)\b[\s,.\-–—]*',
    flags=re.IGNORECASE,
)
_TURN_RE = re.compile(r'^(\[[^\[\]]*\d{1,2}:\d{2}[^\[\]]*\])\s*(.*)$')


def _clean_segment_text(text: str) -> str:
    """Normalize a transcript segment: keep timestamped turns, strip filler words,
    collapse whitespace, and drop exact consecutive duplicate turns."""
    out = []
    prev_body = None
    for raw in text.split('\n'):
        line = raw.strip()
        if not line:
            continue
        m = _TURN_RE.match(line)
        if not m:
            continue  # drop non-dialogue meta lines
        ts, body = m.group(1), m.group(2)
        body = re.sub(r'\s+', ' ', body).strip()
        body = _FILLER_PREFIX.sub('', body)
        body = re.sub(r'\s+', ' ', body).strip()
        if len(body) < 3:
            continue
        if prev_body is not None and body.lower() == prev_body.lower():
            continue  # consecutive duplicate turn
        prev_body = body
        out.append(f"{ts} {body}")
    return "\n".join(out)


def _ts_to_seconds(ts: str) -> int:
    """Convert a 'MM:SS' (or 'H:MM:SS') string to total seconds (0 if unparsable)."""
    m = re.match(r'(\d+):(\d{2})', ts)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return 0


def _line_start_seconds(line: str) -> int:
    m = re.match(r'\[(\d{1,2}):(\d{2})', line)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return 0


def _build_grounding_context(cleaned_segments: list, max_tokens: int = 1100) -> str:
    """Assemble a grounded, timestamped excerpt for the synthesis pass.

    Splits a TOKEN budget evenly across segments and prefers the turns that
    matter for grounded action items and questions: examiner questions
    (containing '?') first, then short request/imperative turns (e.g. 'share
    your screen'), then substantive turns. Kept in chronological order so the
    synthesis model can cite real [MM:SS] times.
    """
    seg_count = max(1, len(cleaned_segments))
    per_seg = max(80, max_tokens // seg_count)

    req_re = re.compile(
        r'\b(share|show|open|explain|walk me|come back|go to|write|run|'
        r'demonstrate|start|tell|give|zoom|draw|upload|move|switch|turn|'
        r'repeat|could you|can you|please)\b',
        flags=re.IGNORECASE,
    )

    def priority(body: str) -> int:
        if '?' in body:
            return 2
        if len(body) <= 140 and req_re.search(body):
            return 1
        return 0

    used_total = 0
    parts = []
    for seg in cleaned_segments:
        turns = [ln.strip() for ln in seg.split('\n') if ln.strip()]
        cand = []
        for t in turns:
            body = re.sub(r'^\[[^\]]*\]\s*', '', t)
            if 10 <= len(body) <= 260:
                cand.append((t, body))

        budget = min(per_seg, max_tokens - used_total)
        chosen = []
        # Sort: questions (priority 2) longest-first, then short request turns
        # (priority 1) shortest-first so the bare examiner request survives
        # before longer turns that merely mention a keyword, then others.
        for t, body in sorted(
                cand,
                key=lambda x: (priority(x[1]), -len(x[1]) if priority(x[1]) == 1 else len(x[1])),
                reverse=True):
            if budget <= 0:
                break
            t_tokens = _budget_tokens(t)
            if t_tokens > budget:
                continue
            chosen.append(t)
            budget -= t_tokens
            used_total += t_tokens

        chosen.sort(key=_line_start_seconds)
        parts.append("\n".join(chosen))

    return "\n".join(parts)


# ============================================================
#  Chapter parsing / validation
# ============================================================

_CHAPTER_HEADER_RE = re.compile(
    r'^#{3}\s*(?:\*{1,2}\s*)?(?:chapter\s+\d+\s*[:\-–.]?\s*)?\**\s*'
    r'[`\[]?(\d{1,2}:\d{2})'
    r'(?:\s*[-–]?\s*(\d{1,2}:\d{2}))?[`\]]?\s*[-—–]?\s*(.*)$',
    flags=re.MULTILINE | re.IGNORECASE,
)


def _parse_chapter_range(header: str) -> tuple:
    """Parse a chapter time range into (start_sec, end_sec).

    Accepts 'MM:SS - MM:SS', 'MM:SS', or headers with a stray segment-window
    prefix (e.g. '[00:00 – 15:00] — 00:48] — Title'): the last two times are
    used when ascending, otherwise the final time is treated as the start.
    """
    times = [_ts_to_seconds(x) for x in re.findall(r'\d{1,2}:\d{2}', header)]
    if not times:
        return None, None
    if len(times) == 1:
        return times[0], None
    last, prev = times[-1], times[-2]
    if last > prev:
        return prev, last
    return last, None  # descending → window prefix junk; last time is the real start


def _extract_chapters(text: str, seg_start_sec: int, seg_end_sec: int) -> list:
    """Split a raw LLM digest into validated, in-order chapter dicts.

    Chapters with a start time outside the segment's [start, end) window are
    dropped. Remaining chapters are sorted by start time, and missing/overlapping
    end times are filled from the following chapter so ranges stay contiguous.
    """
    chapters = []
    cur = None
    for line in text.split('\n'):
        m = _CHAPTER_HEADER_RE.match(line.strip())
        if m:
            if cur is not None and cur['body']:
                chapters.append(cur)
            title = m.group(3).strip()
            # strip a stray window-prefix timestamp fragment from the title,
            # e.g. '00:48] — ID Verification' -> 'ID Verification'
            title = re.sub(r'^[\[`\]]?\s*\d{1,2}:\d{2}\s*[\]–—-]?\s*[-—–]?\s*', '', title).strip()
            start, end = _parse_chapter_range(m.group(0))
            cur = {'start': start, 'end': end, 'title': title, 'body': []}
        else:
            if cur is not None:
                cur['body'].append(line)

    if cur is not None and cur['body']:
        chapters.append(cur)

    valid = [ch for ch in chapters
             if ch['start'] is not None and seg_start_sec <= ch['start'] < seg_end_sec]
    valid.sort(key=lambda c: c['start'])

    for i, ch in enumerate(valid):
        nxt = valid[i + 1]['start'] if i + 1 < len(valid) else None
        fallback = nxt if nxt else seg_end_sec
        end = ch['end'] if ch['end'] is not None else fallback
        if end > fallback:
            end = fallback
        if end <= ch['start']:
            end = seg_end_sec
        ch['end'] = end

    return valid


def _fmt_range(start_sec, end_sec) -> str:
    def _fmt(sec):
        return f"{sec // 60:02d}:{sec % 60:02d}"
    if end_sec is not None:
        return f"[{_fmt(start_sec)} – {_fmt(end_sec)}]"
    return f"[{_fmt(start_sec)}]"


def _dialogue_ranges(segment_text: str, gap_sec: int = 15) -> list:
    """Merged (start_sec, end_sec) intervals that actually contain dialogue.

    Transcripts regularly contain silent stretches (no captions) inside the
    segment window. Coverage/gap logic must measure against these real
    dialogue ranges, not the whole window, so empty stretches never trigger
    pointless gap-fill calls. Consecutive turns closer than `gap_sec` seconds
    are merged into one range.
    """
    ts = []
    for ln in segment_text.split('\n'):
        m = re.match(r'^\s*\[?(\d{1,2}):(\d{2})', ln)
        if m:
            ts.append(int(m.group(1)) * 60 + int(m.group(2)))
    ts.sort()
    ranges = []
    for t in ts:
        if ranges and t - ranges[-1][1] <= gap_sec:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], t))
        else:
            ranges.append((t, t))
    return ranges


def _segment_gaps(chapters: list, seg_start_sec: int, seg_end_sec: int, dialogue: list = None) -> list:
    """Uncovered (start_sec, end_sec) ranges, limited to dialogue-bearing time.

    Only stretches that actually contain transcript dialogue are reported as
    gaps; silent windows are ignored so they never trigger gap-fill.
    """
    pts = sorted((c['start'], c['end'] or seg_end_sec) for c in chapters if c['start'] is not None)
    ranges = dialogue if dialogue else [(seg_start_sec, seg_end_sec)]
    gaps = []
    for d_start, d_end in ranges:
        cursor = d_start
        for s, e in pts:
            if e <= cursor:
                continue
            if s > cursor:
                gaps.append((cursor, min(s, d_end)))
            cursor = max(cursor, e)
            if cursor >= d_end:
                break
        if cursor < d_end:
            gaps.append((cursor, d_end))
    return gaps


def _chapter_coverage(chapters: list, seg_start_sec: int, seg_end_sec: int, dialogue: list = None) -> float:
    """Fraction of the segment's DIALOGUE time actually covered (0.0-1.0).

    Silent stretches contribute nothing to the span, so a transcript with long
    gaps does not unfairly penalize otherwise complete chapter coverage.
    """
    ranges = dialogue if dialogue else [(seg_start_sec, seg_end_sec)]
    span = sum(max(1, e - s) for s, e in ranges)
    if span <= 0:
        return 0.0
    covered = 0
    for c in chapters:
        if c['start'] is None:
            continue
        c_end = c['end'] or seg_end_sec
        for d_start, d_end in ranges:
            lo = max(c['start'], d_start)
            hi = min(c_end, d_end)
            if hi > lo:
                covered += hi - lo
    return min(1.0, covered / span)


def _uncovered_ranges(chapters: list, seg_start_sec: int, seg_end_sec: int, dialogue: list = None) -> str:
    """Human-readable list of uncovered dialogue ranges, e.g. '[15:35 – 16:04]'."""
    def _f(x):
        return f"{x // 60:02d}:{x % 60:02d}"
    return ", ".join(f"[{_f(s)} – {_f(e)}]"
                     for s, e in _segment_gaps(chapters, seg_start_sec, seg_end_sec, dialogue))


def _extract_range_text(segment_text: str, start_sec: int, end_sec: int) -> str:
    """Return transcript lines whose leading '[MM:SS' timestamp falls within
    [start_sec, end_sec), so a gap can be re-digested in isolation."""
    out = []
    for ln in segment_text.split('\n'):
        m = re.match(r'^\s*\[(\d{1,2}):(\d{2})', ln)
        if not m:
            continue
        t = int(m.group(1)) * 60 + int(m.group(2))
        if start_sec <= t < end_sec:
            out.append(ln.strip())
    return "\n".join(out)


def _merge_chapters(primary: list, extra: list, seg_start_sec: int, seg_end_sec: int) -> list:
    """Merge gap-filled chapters into the primary list, dropping overlaps and
    re-validating order/range exactly like _extract_chapters."""
    merged = list(primary) + list(extra)
    seen = set()
    uniq = []
    for c in merged:
        key = (c['start'], c['title'])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    valid = [c for c in uniq
             if c['start'] is not None and seg_start_sec <= c['start'] < seg_end_sec]
    valid.sort(key=lambda c: c['start'])
    for i, ch in enumerate(valid):
        nxt = valid[i + 1]['start'] if i + 1 < len(valid) else None
        fallback = nxt if nxt else seg_end_sec
        end = ch['end'] if ch['end'] is not None else fallback
        if end > fallback:
            end = fallback
        if end <= ch['start']:
            end = seg_end_sec
        ch['end'] = end
    return valid


def _serialize_chapters(chapters: list) -> str:
    blocks = []
    for c in chapters:
        body = "\n".join(c['body']).strip()
        rng = _fmt_range(c['start'], c['end'])
        block = f"### {rng} — {c['title']}"
        if body:
            block += "\n" + body
        blocks.append(block)
    return "\n\n".join(blocks)


def _condense_chapters_for_synthesis(chapters_text: str, max_tokens: int = 900) -> str:
    """Extract dense core points from chapters for the synthesis prompt,
    capped by an estimated TOKEN budget so the request stays under the TPM cap."""
    if _budget_tokens(chapters_text) <= max_tokens:
        return chapters_text

    lines = []
    for line in chapters_text.split('\n'):
        l = line.strip()
        if (l.startswith('### ') or l.startswith('- **Discussion') or
                l.startswith('- **Key Topics') or l.startswith('- Q:') or
                l.startswith('*Q:*') or l.startswith('Q:') or l.startswith('- **Auditor Notes') or
                l.startswith('1.') or l.startswith('2.') or l.startswith('3.') or l.startswith('4.')):
            if _budget_tokens('\n'.join(lines + [line])) <= max_tokens:
                lines.append(line)
        elif len(lines) > 0 and _budget_tokens('\n'.join(lines + [line])) <= max_tokens:
            lines.append(line)

    condensed = '\n'.join(lines)
    if _budget_tokens(condensed) > 150:
        return condensed
    return chapters_text[:max_tokens * 4]


def _has_required_sections(text: str) -> list:
    """Return the list of required section names missing from a synthesis output."""
    missing = []
    for name in REQUIRED_SECTIONS:
        if name.lower() not in text.lower():
            missing.append(name)
    return missing


def _splice_missing_sections(report_text: str, sections_text: str) -> str:
    """Insert generated missing sections before the chronological-chapters anchor
    (or append at the end if that anchor is absent), preserving the doc structure."""
    if not sections_text or not sections_text.strip():
        return report_text
    anchor = re.search(r'^##\s*.*Chronological', report_text, flags=re.MULTILINE | re.IGNORECASE)
    block = sections_text.strip()
    if anchor:
        return (report_text[:anchor.start()].rstrip()
                + "\n\n---\n\n" + block
                + "\n\n" + report_text[anchor.start():])
    return report_text.rstrip() + "\n\n---\n\n" + block + "\n"


def _extract_section_block(text: str, section_name: str) -> str:
    """Return the markdown block for a named '##' section (its header line plus
    everything up to the next '##' header), or '' if the header is absent."""
    pattern = re.compile(
        rf'^##\s*[^\n]*{re.escape(section_name)}[^\n]*\n.*?(?=^##\s|\Z)',
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(0).strip() if m else ""


# ============================================================
#  Report generation
# ============================================================

def _build_synthesis_call(system_prompt: str, template: str, grounding: str,
                          synthesis_context: str, output_budget: int = SYNTH_MAX_TOKENS):
    """Format a synthesis prompt and clamp max_tokens so input+output stays
    under TPM_CAP (measured by estimate; the router still hard-skips 413s)."""
    prompt = template.format(grounding=grounding, window_summaries=synthesis_context)
    input_tokens = _budget_tokens(system_prompt) + _budget_tokens(prompt)
    max_tokens = max(400, min(output_budget, TPM_CAP - input_tokens - SYNTH_TOKEN_SAFETY))
    return prompt, max_tokens


def generate_report(transcript_text: str, file_id: str) -> str:
    """
    Generate a comprehensive multi-candidate Viva Intelligence Report.
    Pass 1: grounded chronological chapters per 15-min segment (LLM).
    Pass 2: synthesis of scorecard, action items, and rubrics, grounded in a
    cleaned transcript excerpt (LLM). Python validates, orders, and stitches.
    """
    if not GROQ_API_KEYS:
        raise ValueError("No GROQ_API_KEYS configured in config.py!")

    print(f"\n[Report Generator] Generating TA/Mentor-Grade Viva Intelligence Report...")
    print(f"    Total Transcript: {len(transcript_text)} characters")

    router = ModelRouter(api_keys=GROQ_API_KEYS, models=MODEL_POOL)

    # Step 1: deterministic 15-minute segmentation
    segments = _split_into_segments(transcript_text, segment_minutes=15)
    print(f"    Divided into {len(segments)} segments (~15 mins each) covering the full session")
    print(f"    Active API Keys: {len(GROQ_API_KEYS)} | Model pool: {' | '.join(m.split('/')[-1] for m in MODEL_POOL)}\n")

    segment_digests = []
    cleaned_segments = []
    for i, s in enumerate(segments, 1):
        print(f"    [Pass 1/2: Segment {i}/{len(segments)}] Auditing [{s['start']} – {s['end']}] ({len(s['text'])} chars)...")

        cleaned = _clean_segment_text(s['text'])
        cleaned_segments.append(cleaned)

        seg_start_sec = _ts_to_seconds(s['start'])
        seg_end_sec = _ts_to_seconds(s['end'])
        seg_dialogue = _dialogue_ranges(s['text'])

        prompt = (
            f"Transcript segment [{s['start']} – {s['end']}], cleaned for readability "
            f"(fillers removed, timestamps preserved):\n\n{cleaned}"
        )

        digest = None
        try:
            digest = router.call(
                messages=[
                    {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2200,
                temperature=0.0,
                allow_tpm_wait=True,
                is_synthesis=False
            )
        except Exception as e:
            print(f"        [WARN] Segment {i}: digest call failed ({str(e)[:60]})")
        chapters = _extract_chapters(digest, seg_start_sec, seg_end_sec) if digest else []

        if not chapters and digest:
            print(f"    [WARN] Segment {i}: no valid chapters parsed — retrying once...")
            time.sleep(CALL_COOLDOWN_SECS)
            retry_prompt = (
                "You produced no valid chapters. Output the chapters again, starting directly "
                "with a '### [MM:SS – MM:SS] — Title' header.\n\n"
                + prompt
            )
            try:
                digest = router.call(
                    messages=[
                        {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": retry_prompt}
                    ],
                    max_tokens=2200,
                    temperature=0.0,
                    allow_tpm_wait=True,
                    is_synthesis=False
                )
                chapters = _extract_chapters(digest, seg_start_sec, seg_end_sec)
            except Exception as e:
                print(f"        [WARN] Segment {i}: retry failed ({str(e)[:60]})")
                chapters = []

        # Coverage-based retry: an under-produced digest (early termination,
        # single chapter) silently drops real Q&A from the proctor log. First
        # re-digest each uncovered range IN ISOLATION (small focused prompts are
        # far more reliable), then fall back to a whole-segment re-ask.
        if chapters:
            gaps = _segment_gaps(chapters, seg_start_sec, seg_end_sec, seg_dialogue)
            coverage = _chapter_coverage(chapters, seg_start_sec, seg_end_sec, seg_dialogue)
            largest_gap = max((e - s for s, e in gaps), default=0)
            uncovered_dialogue = sum(e - s for s, e in gaps)
            # Under-production (model stops early) is the most common silent
            # failure: chapters legitimately cover what they do cover, but the
            # last chapter ends long before the segment's actual dialogue does.
            # Trigger gap-fill when >25% of real dialogue is uncovered, when any
            # single gap exceeds 3 min, or when the final chapter ends well before
            # the last dialogue turn. The tail threshold scales with session
            # length so short recordings (e.g. 7 min) don't lose their closing
            # remarks (a fixed 90s bound lets ~16% of a short session vanish).
            last_turn = seg_dialogue[-1][1] if seg_dialogue else seg_end_sec
            last_chapter_end = max((c['end'] or seg_end_sec) for c in chapters if c['start'] is not None) \
                if chapters else seg_start_sec
            tail_miss = last_turn - last_chapter_end
            seg_duration = max(1, seg_end_sec - seg_start_sec)
            tail_threshold = max(45, int(seg_duration * 0.12))
            if gaps and (coverage < 0.75 or largest_gap > 180 or uncovered_dialogue > 120 or tail_miss > tail_threshold):
                print(f"    [WARN] Segment {i}: chapters cover only {coverage:.0%} of dialogue "
                      f"(largest gap {largest_gap // 60}m{largest_gap % 60:02d}s, "
                      f"{len(gaps)} uncovered range(s)) — gap-filling...")
                time.sleep(CALL_COOLDOWN_SECS)
                try:
                    extra = []
                    for gs, ge in gaps:
                        gap_text = _extract_range_text(s['text'], gs, ge)
                        if not gap_text.strip():
                            continue
                        g_start = f"{gs // 60:02d}:{gs % 60:02d}"
                        g_end = f"{ge // 60:02d}:{ge % 60:02d}"
                        gap_prompt = (
                            f"The chapters below only covered part of the segment. Fill the gap "
                            f"[{g_start} – {g_end}] with chapters for every question/topic exchange "
                            f"in that range, using the REAL timestamps from this excerpt. "
                            f"Start directly with a '### [MM:SS – MM:SS] — Title' header.\n\n"
                            f"GAP TRANSCRIPT EXCERPT [{g_start} – {g_end}]:\n{gap_text}"
                        )
                        digest = router.call(
                            messages=[
                                {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                                {"role": "user", "content": gap_prompt}
                            ],
                            max_tokens=2200,
                            temperature=0.0,
                            allow_tpm_wait=True,
                            is_synthesis=False
                        )
                        extra.extend(_extract_chapters(digest, gs, ge))
                    if extra:
                        chapters = _merge_chapters(chapters, extra, seg_start_sec, seg_end_sec)
                        new_cov = _chapter_coverage(chapters, seg_start_sec, seg_end_sec)
                        print(f"        [OK] Segment {i}: {len(chapters)} chapters, "
                              f"coverage {new_cov:.0%} (gap-filled)")
                    else:
                        cover_prompt = (
                            "Your chapters left large gaps in the segment. You MUST cover the ENTIRE segment, "
                            "including these currently uncovered ranges: "
                            + _uncovered_ranges(chapters, seg_start_sec, seg_end_sec, seg_dialogue)
                            + ". Produce one or more chapters for every question/topic exchange in those ranges, "
                            "using their real timestamps from the transcript. Start directly with a "
                            "'### [MM:SS – MM:SS] — Title' header.\n\n"
                            + prompt
                        )
                        digest = router.call(
                            messages=[
                                {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                                {"role": "user", "content": cover_prompt}
                            ],
                            max_tokens=2200,
                            temperature=0.0,
                            allow_tpm_wait=True,
                            is_synthesis=False
                        )
                        retry_chapters = _extract_chapters(digest, seg_start_sec, seg_end_sec)
                        if _chapter_coverage(retry_chapters, seg_start_sec, seg_end_sec) > coverage:
                            chapters = retry_chapters
                            print(f"        [OK] Segment {i}: {len(chapters)} chapters (coverage improved)")
                except Exception as e:
                    print(f"        [WARN] Segment {i}: coverage retry failed ({str(e)[:60]}) — keeping current chapters")

        if chapters:
            segment_digests.append(_serialize_chapters(chapters))
            print(f"        [OK] Segment {i}: {len(chapters)} chapters in order")
        else:
            fallback = [{
                'start': seg_start_sec, 'end': seg_end_sec,
                'title': f"Segment {i} — [{s['start']} – {s['end']}] (digest unavailable)",
                'body': [
                    "- **Discussion & Events:** No chapter digest could be generated for this segment "
                    "(LLM calls failed).",
                    "- **Auditor Notes:** Refer to the raw transcript excerpt in the Executive Summary "
                    "grounding for this window.",
                ],
            }]
            segment_digests.append(_serialize_chapters(fallback))
            print(f"        [WARN] Segment {i}: still no valid chapters — added placeholder for window continuity")

        if i < len(segments):
            time.sleep(CALL_COOLDOWN_SECS)

    merged_chapters = "\n\n---\n\n".join(segment_digests)

    # Step 2: synthesize with token-aware budget sizing
    print(f"\n    [Pass 2/2: Synthesis] Generating Multi-Candidate Executive Summary, Scorecard & Action Items...")
    time.sleep(CALL_COOLDOWN_SECS)

    # The synthesis request is bounded by the free-tier TPM_CAP (a hard 413
    # past it). Size the GROUNDING EXCERPT and CHAPTER DIGESTS from an ESTIMATED
    # token budget so ANY transcript length fits.
    framework_tokens = _budget_tokens(REPORT_SYNTHESIS_PROMPT.format(grounding="", window_summaries=""))
    content_tokens = TPM_CAP - framework_tokens - SYNTH_MAX_TOKENS - SYNTH_TOKEN_SAFETY - 150
    if content_tokens < 1200:
        raise RuntimeError(
            f"Synthesis request cannot fit under the {TPM_CAP} TPM cap even with minimal inputs "
            f"(framework uses ~{framework_tokens} tokens).")

    grounding_tokens = int(content_tokens * 0.55)
    condensed_tokens = max(200, content_tokens - grounding_tokens)

    grounding = _build_grounding_context(cleaned_segments, max_tokens=grounding_tokens)
    synthesis_context = _condense_chapters_for_synthesis(merged_chapters, max_tokens=condensed_tokens)
    print(f"        Budget: framework ~{framework_tokens} tok | grounding {grounding_tokens} tok | condensed {condensed_tokens} tok")

    synthesis_prompt, synthesis_mt = _build_synthesis_call(
        SYNTHESIS_SYSTEM_PROMPT, REPORT_SYNTHESIS_PROMPT, grounding, synthesis_context)

    try:
        synthesis_result = router.call(
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=synthesis_mt,
            temperature=0.0,
            allow_tpm_wait=True,
            is_synthesis=True
        )
    except Exception as exc:
        print(f"    [WARN] Synthesis call failed ({str(exc)[:80]}) — retrying once with reduced grounding...")
        time.sleep(CALL_COOLDOWN_SECS)
        reduced_grounding = _build_grounding_context(cleaned_segments, max_tokens=max(300, grounding_tokens // 2))
        retry_prompt, retry_mt = _build_synthesis_call(
            SYNTHESIS_SYSTEM_PROMPT, REPORT_SYNTHESIS_PROMPT, reduced_grounding, synthesis_context)
        try:
            synthesis_result = router.call(
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": retry_prompt}
                ],
                max_tokens=retry_mt,
                temperature=0.0,
                allow_tpm_wait=True,
                is_synthesis=True
            )
        except Exception as exc2:
            print(f"    [WARN] Synthesis retry failed ({str(exc2)[:80]}) — building report from chapters only")
            synthesis_result = None

    if synthesis_result is None:
        missing = REQUIRED_SECTIONS
    else:
        missing = _has_required_sections(synthesis_result)
    if missing:
        print(f"    [WARN] Synthesis missing sections: {', '.join(missing)} — retrying once...")
        time.sleep(CALL_COOLDOWN_SECS)
        fix_template = (
            "Your previous output was incomplete. It is missing these sections: "
            + ", ".join(missing)
            + ". Output ONLY the complete report markdown with ALL sections present, "
            "starting with '## ⚡ Executive Summary (For TAs & Mentors)'. Reuse the grounded content above.\n\n"
            + REPORT_SYNTHESIS_PROMPT
        )
        fix_prompt, fix_mt = _build_synthesis_call(
            SYNTHESIS_SYSTEM_PROMPT, fix_template, grounding, synthesis_context)
        try:
            synthesis_result = router.call(
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": fix_prompt}
                ],
                max_tokens=fix_mt,
                temperature=0.0,
                allow_tpm_wait=True,
                is_synthesis=True
            )
        except Exception as exc:
            print(f"        [WARN] Synthesis retry failed ({exc}) — keeping first-pass output")
        if synthesis_result is None:
            missing = REQUIRED_SECTIONS
        else:
            missing = _has_required_sections(synthesis_result)
        if missing:
            print(f"    [WARN] Synthesis still missing: {', '.join(missing)} — generating missing section(s) directly...")
            time.sleep(CALL_COOLDOWN_SECS)
            missing_template = (
                "The following required report sections are MISSING from your previous output: "
                + ", ".join(missing)
                + ".\n\nGenerate ONLY those missing section(s), in the exact same Markdown style and with the "
                "exact '## ...' headers as the full report. Output ONLY the section(s), nothing else — no "
                "preamble, no duplicate of existing sections.\n\n"
                "Use the GROUNDING EXCERPT for exact timestamps/quotes and the CHAPTER DIGESTS for the "
                "full-session narrative.\n\n"
                "GROUNDING EXCERPT:\n{grounding}\n\n"
                "CHAPTER DIGESTS:\n{window_summaries}"
            )
            missing_prompt, missing_mt = _build_synthesis_call(
                SYNTHESIS_SYSTEM_PROMPT, missing_template, grounding, synthesis_context)
            try:
                sections = router.call(
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": missing_prompt}
                    ],
                    max_tokens=missing_mt,
                    temperature=0.0,
                    allow_tpm_wait=True,
                    is_synthesis=True,
                    raw=True,
                )
            except Exception as exc:
                print(f"        [WARN] Missing-section generation failed ({exc}) — skipping")
                sections = ""
            if synthesis_result is None:
                print(f"    [WARN] Synthesis unavailable — report will be chapters-only")
            elif not sections.strip():
                print(f"        [WARN] Missing sections still absent: {', '.join(missing)}")
                missing = _has_required_sections(synthesis_result)
            else:
                # Keep only blocks that correspond to genuinely missing sections.
                filtered = "\n\n".join(
                    _extract_section_block(sections, name) for name in REQUIRED_SECTIONS
                    if name.lower() in sections.lower() and name.lower() not in synthesis_result.lower())
                if filtered.strip():
                    synthesis_result = _splice_missing_sections(synthesis_result, filtered)
                missing = _has_required_sections(synthesis_result)
                if missing:
                    print(f"    [WARN] Synthesis still missing: {', '.join(missing)}")
                else:
                    print("        [OK] Missing sections recovered")

    # Step 3: assemble the complete document in Python
    if synthesis_result is None:
        full_report = f"""# 🎓 Viva Assessment & Meeting Intelligence Report

## ⚡ Executive Summary (For TAs & Mentors)
**Session Overview:** The synthesis pass could not complete because all configured
Groq API keys/models were unavailable (rate or daily-token limits). The chronological
proctor log below is complete and grounded in the transcript.

---

## 📑 5-Minute Chronological Proctor Log & Chapters

{merged_chapters}
"""
    else:
        full_report = f"""# 🎓 Viva Assessment & Meeting Intelligence Report

{synthesis_result}

---

## 📑 5-Minute Chronological Proctor Log & Chapters

{merged_chapters}
"""

    # Save Markdown report
    output_path = os.path.join(REPORTS_DIR, f"{file_id}_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\n[Report] Generated! Saved to: {output_path}")
    print(f"    Report length: {len(full_report)} chars")

    return full_report