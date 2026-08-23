"""
Audio Extractor
Extracts audio from .mp4 video files and compresses to lightweight .mp3.
Groq Whisper has a 25MB file size limit, so we compress to 32kbps mono.
A 30-minute viva at 32kbps mono = ~7MB (well under the 25MB limit).
"""

import os
import subprocess
from config import AUDIO_DIR


def extract_audio(video_path: str, output_filename: str = None) -> str:
    """
    Extract audio from a video file and compress to a lightweight MP3.

    Uses ffmpeg to:
      - Extract audio stream
      - Convert to mono (1 channel — speech doesn't need stereo)
      - Resample to 16kHz (optimal for Whisper speech recognition)
      - Compress to 32kbps (tiny file size, perfect speech quality)

    Args:
        video_path: Path to the .mp4 video file.
        output_filename: Optional custom output name. Auto-generated if None.

    Returns:
        Absolute path to the compressed .mp3 audio file.
    """
    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}.mp3"
    elif not output_filename.endswith(".mp3"):
        output_filename = f"{output_filename}.mp3"

    output_path = os.path.join(AUDIO_DIR, output_filename)

    print(f"[Audio] Extracting audio from video...")
    print(f"    Input:  {video_path}")
    print(f"    Output: {output_path}")

    # ffmpeg command:
    #   -i          : input file
    #   -vn         : no video (audio only)
    #   -ac 1       : mono channel
    #   -ar 16000   : 16kHz sample rate (Whisper optimal)
    #   -b:a 32k    : 32kbps bitrate (tiny file, great for speech)
    #   -y          : overwrite output without asking
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "32k",
        "-y",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Audio] ffmpeg error:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Audio extraction failed. Output not found: {output_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Audio] Extraction complete! Audio size: {size_mb:.1f} MB")

    if size_mb > 25:
        print(f"[WARN] WARNING: File is {size_mb:.1f}MB, exceeds Groq's 25MB limit!")
        print(f"           The file may need to be split or further compressed.")

    return output_path


# Alias for backward compatibility
extract_audio_from_video = extract_audio
