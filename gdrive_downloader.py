"""
Google Drive Downloader
Downloads files from Google Drive shared links.
Supports:
  1. Standard public downloads via gdown.
  2. Domain-restricted downloads via an authenticated Playwright session
     (requires ``python analyzer.py --login`` to have been run once).
"""

import os
import re
import time
import gdown
from config import DOWNLOADS_DIR, BROWSER_PROFILE_DIR, google_profile_ready


def extract_file_id(url: str) -> str:
    """Extract file ID from Google Drive URL, YouTube URL, or raw ID."""
    url = url.strip()
    
    # YouTube URL check
    yt_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if 'youtube.com' in url or 'youtu.be' in url:
        if yt_match:
            return f"yt_{yt_match.group(1)}"
        return "yt_video"

    # Google Drive Pattern 1: /file/d/<ID>/
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    # Google Drive Pattern 2: ?id=<ID>
    match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    if re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url

    return "video_" + str(abs(hash(url)) % 1000000)


def download_from_gdrive(gdrive_url: str, output_filename: str = None) -> str:
    """
    Download a video from Google Drive, YouTube, or web link.
    """
    file_id = extract_file_id(gdrive_url)

    if output_filename:
        output_path = os.path.join(DOWNLOADS_DIR, output_filename)
    else:
        output_path = os.path.join(DOWNLOADS_DIR, f"{file_id}.mp4")

    # Reuse cached video if exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 1024:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Cache] Found existing downloaded video: {output_path} ({size_mb:.1f} MB)")
        return output_path

    # YouTube download via yt-dlp
    if "youtube.com" in gdrive_url or "youtu.be" in gdrive_url:
        print(f"[YouTube] Downloading audio/video stream with yt-dlp...")
        import yt_dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path.replace('.mp4', '.%(ext)s'),
            'quiet': False
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([gdrive_url])
        # Find downloaded file
        base = output_path.rsplit('.', 1)[0]
        for ext in ['.mp4', '.m4a', '.webm', '.mkv', '.mp3']:
            if os.path.exists(base + ext):
                return base + ext
        return output_path

    print(f"[Download] Downloading from GDrive...")
    print(f"    File ID: {file_id}")
    print(f"    Target:  {output_path}")

    # Attempt 1: Fast gdown download
    try:
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, output_path, quiet=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 1024:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[Download] Complete via gdown! File size: {size_mb:.1f} MB")
            return output_path
    except Exception as e:
        print(f"[Download] Public download notice: {e}")

    # Attempt 2: Authenticated Playwright Browser Download (requires --login)
    if not google_profile_ready():
        raise PermissionError(
            "The file is not publicly downloadable. Run `python analyzer.py --login` once to "
            "authenticate your Google account, then retry this link."
        )

    print(f"[Auth Download] Downloading domain-restricted file via authenticated session...")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=BROWSER_PROFILE_DIR,
            headless=True,
            channel="chrome",
            viewport={"width": 1400, "height": 900},
            accept_downloads=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        export_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        page.goto(export_url, wait_until="domcontentloaded")
        time.sleep(3)

        btn = page.locator('#uc-download-link, input[value*="Download anyway"], a:has-text("Download anyway"), button:has-text("Download anyway")').first

        if btn.is_visible():
            print("    Found 'Download anyway' button on virus scan warning page: downloading...")
            with page.expect_download(timeout=180000) as dl_info:
                btn.click()
            dl = dl_info.value
            dl.save_as(output_path)
        else:
            # Fallback: click download button in viewer
            page.goto(f"https://drive.google.com/file/d/{file_id}/view", wait_until="domcontentloaded")
            time.sleep(3)
            viewer_btn = page.locator('[aria-label*="Download"], button[data-tooltip*="Download"]').first
            if viewer_btn.is_visible():
                with page.expect_download(timeout=180000) as dl_info:
                    viewer_btn.click()
                dl = dl_info.value
                dl.save_as(output_path)

        ctx.close()

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        raise FileNotFoundError(f"Authenticated download failed for file: {file_id}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Auth Download] Complete! File size: {size_mb:.1f} MB")
    return output_path
