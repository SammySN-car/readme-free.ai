"""
Shared parser for Google TimedText (json3) caption payloads.

Both transcript sources — the public timedtext endpoint and the
authenticated browser scraper — deliver the same event structure, so a
single parser keeps dialogue-turn formatting identical across tiers.

Turns are split on three signals so stretches where Google fails to flag
a speaker change still produce readable Meet-style dialogue blocks:
  1. Speaker-change markers (the authoritative signal, including ">>").
  2. Pauses >= TURN_PAUSE_MS between caption events (new speaker/turn).
  3. Sentence boundaries once a turn grows past TURN_MAX_CHARS, so a
     continuous monologue becomes capped, sentence-aligned blocks.
"""

import json

TURN_PAUSE_MS = 2500
TURN_MAX_CHARS = 350
TURN_MIN_CHARS = 250
TURN_HARD_MAX = 900


def _ends_sentence(text: str) -> bool:
    s = text.rstrip()
    return bool(s) and s[-1] in ".?!"


def parse(raw_json_str: str):
    """Parse a TimedText JSON string into '[MM:SS] dialogue' turns.

    Returns the formatted transcript (with a short provenance header),
    or None when the payload is not parsable."""
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

            cur.append((t_ms, text))
            if start_ms is None:
                start_ms = t_ms

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
            "=" * 60,
            "",
        ]
        return "\n".join(header) + "\n\n" + "\n\n".join(dialogue_turns)
    except Exception as e:
        print(f"    [JSON Parse Error] {e}")
        return None
