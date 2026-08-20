"""
Viva AI Analyzer — Main CLI Entry Point

Converts Google Drive / Meet viva recording links into graded Markdown + HTML reports.

Setup (one-time):
    1. Log in to Google so restricted Drive files can be read:
         python analyzer.py --login
    2. Set your API keys (Groq and/or OpenRouter). The pipeline auto-detects
       which provider each key belongs to:
         $env:GROQ_API_KEYS = "gsk_...,sk-or-..."   (comma-separated)

Usage:
    python analyzer.py "https://drive.google.com/file/d/FILE_ID/view"
    python analyzer.py "LINK_1" "LINK_2"
    python analyzer.py --file links.txt   (one link per line, '#' comments allowed)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import re
import time
from transcript_fetcher import fetch_transcript_from_gdrive
from report_generator import generate_report, LEAK_MARKERS
from html_generator import generate_html_report
from config import GROQ_API_KEYS, REPORTS_DIR, BROWSER_PROFILE_DIR


REQUIRED_HEADERS = [
    "Executive Summary",
    "Action Items",
    "Performance Scorecard",
    "Topic Competency Matrix",
    "Key Technical Questions",
    "Audit Notes",
    "Chronological",
]


def _auth_profile_ready() -> bool:
    """True if the user has run `python analyzer.py --login` (browser auth exists)."""
    return os.path.isdir(BROWSER_PROFILE_DIR) and os.path.exists(
        os.path.join(BROWSER_PROFILE_DIR, "Default")
    )


def check_prereqs():
    """Verify the user has completed the required one-time setup steps."""
    ok = True
    if not GROQ_API_KEYS:
        ok = False
        print("[ERROR] No API keys configured.")
        print("        Set GROQ_API_KEYS (comma-separated) to your Groq and/or")
        print("        OpenRouter keys, e.g.:")
        print("          PowerShell:  $env:GROQ_API_KEYS = \"gsk_...,sk-or-...\"")
        print("          Bash:        export GROQ_API_KEYS=\"gsk_...,sk-or-...\"")
        print("        Groq keys:      https://console.groq.com/keys")
        print("        OpenRouter:     https://openrouter.ai/keys")
    if not _auth_profile_ready():
        ok = False
        print("[ERROR] No authenticated Google login found. You must log in once")
        print("        before processing any recording:")
        print("          python analyzer.py --login")
        print("        A Chrome window will open; sign in to the Google account")
        print("        that can access your Drive recordings, then close it.")
    return ok


def validate_report(report: str) -> list:
    """Check a generated report for completeness and contamination.

    Returns a list of human-readable warnings (empty means healthy). This is
    advisory only — a warning never fails the pipeline.
    """
    warnings = []

    missing = [h for h in REQUIRED_HEADERS if h.lower() not in report.lower()]
    if missing:
        warnings.append(f"missing sections: {', '.join(missing)}")

    leaked = [m for m in LEAK_MARKERS if m.lower() in report.lower()]
    if leaked:
        warnings.append(f"prompt/instruction leakage detected: {', '.join(leaked)}")

    chapter_times = []
    for m in re.finditer(r'^###\s*[`\[]?(\d{1,2}):(\d{2})', report, flags=re.MULTILINE):
        chapter_times.append(int(m.group(1)) * 60 + int(m.group(2)))
    if len(chapter_times) > 1:
        out_of_order = sum(1 for a, b in zip(chapter_times, chapter_times[1:]) if b < a)
        if out_of_order:
            warnings.append(f"{out_of_order} chapter(s) out of chronological order")

    return warnings


def process_single_link(gdrive_url: str, index: int = 1, total: int = 1) -> str:
    """Process a single Google Drive viva recording link end-to-end."""

    print(f"\n{'='*60}")
    print(f"  Processing Link {index}/{total}")
    print(f"  URL: {gdrive_url[:80]}...")
    print(f"{'='*60}")

    start_time = time.time()

    # Step 1: Fetch transcript
    transcript_result = fetch_transcript_from_gdrive(gdrive_url)

    if not transcript_result["text"] or len(transcript_result["text"].strip()) < 50:
        print(f"[ERROR] Transcript is too short or empty. Skipping.")
        return None

    print(f"    Transcript method: {transcript_result['method']}")
    print(f"    Transcript length: {len(transcript_result['text'])} chars")
    print(f"    Segments: {len(transcript_result['segments'])}")

    # Step 2: Generate Markdown report
    report = generate_report(
        transcript_text=transcript_result["text"],
        file_id=transcript_result["file_id"]
    )

    # Validate report quality (advisory warnings, never fatal)
    report_warnings = validate_report(report)
    if report_warnings:
        print(f"\n[REPORT CHECK] {len(report_warnings)} warning(s):")
        for w in report_warnings:
            print(f"    - {w}")
    else:
        print("\n[REPORT CHECK] All sections present, no leakage, chapters in order")

    # Step 3: Generate HTML visual dashboard
    html_path = generate_html_report(
        markdown_text=report,
        file_id=transcript_result["file_id"],
        transcript_text=transcript_result["text"]
    )

    elapsed = time.time() - start_time
    print(f"\n[DONE] Link {index}/{total} processed in {elapsed:.1f} seconds")
    print(f"    Markdown: {REPORTS_DIR}\\{transcript_result['file_id']}_report.md")
    print(f"    HTML Dashboard: {html_path}")

    return report


def process_links(links: list[str]):
    """Process multiple Google Drive links sequentially."""

    print(f"\n{'='*60}")
    print(f"  Viva AI Analyzer")
    print(f"  Processing {len(links)} recording(s)")
    print(f"{'='*60}")

    results = []
    for i, link in enumerate(links, 1):
        link = link.strip()
        if not link or link.startswith("#"):
            continue

        try:
            report = process_single_link(link, index=i, total=len(links))
            if report:
                results.append({"link": link, "status": "success"})
            else:
                results.append({"link": link, "status": "skipped"})
        except Exception as e:
            print(f"\n[ERROR] Failed to process link {i}: {e}")
            results.append({"link": link, "status": "failed", "error": str(e)})

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Processing Summary")
    print(f"{'='*60}")
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"  Successful: {success}")
    print(f"  Failed:     {failed}")
    print(f"  Skipped:    {skipped}")
    print(f"\n  Reports saved in: {REPORTS_DIR}")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # --login flag: one-time authentication for Google Drive access
    if sys.argv[1] == "--login":
        from browser_scraper import login_to_google
        login_to_google()
        return

    # --file flag: read links from a text file
    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("Usage: python analyzer.py --file links.txt")
            sys.exit(1)
        filepath = sys.argv[2]
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            sys.exit(1)
        with open(filepath, "r") as f:
            links = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        # Links passed directly as arguments
        links = sys.argv[1:]

    if not check_prereqs():
        print("\nComplete the setup steps above, then re-run this command.")
        sys.exit(1)

    process_links(links)


if __name__ == "__main__":
    main()