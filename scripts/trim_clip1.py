#!/usr/bin/env python3
"""Trim clip 1 to start at 1:45, overwriting the original file."""
import os
import shutil
import subprocess
import sys
import tempfile

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIP1_PATH = os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3")
BACKUP_PATH = os.path.join(CLIP_DIR, "clip1_theater_conversation_ORIGINAL.mp3")

if not os.path.exists(CLIP1_PATH):
    print(f"ERROR: {CLIP1_PATH} not found")
    sys.exit(1)

# Backup original first
if not os.path.exists(BACKUP_PATH):
    print(f"Backing up original to {BACKUP_PATH}")
    shutil.copy2(CLIP1_PATH, BACKUP_PATH)
else:
    print(f"Backup already exists: {BACKUP_PATH}")

# Trim with ffmpeg: start at 1:45 (105s)
tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
tmp.close()
try:
    cmd = [
        "ffmpeg", "-y",
        "-i", CLIP1_PATH,
        "-ss", "00:01:45",
        "-c", "copy",
        tmp.name,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Get durations for verification
    def get_duration(path):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())

    orig_dur = get_duration(BACKUP_PATH)
    new_dur = get_duration(tmp.name)
    print(f"Original duration: {orig_dur:.1f}s")
    print(f"Trimmed duration:  {new_dur:.1f}s (removed first 105s)")

    # Overwrite original
    shutil.move(tmp.name, CLIP1_PATH)
    print(f"Overwrote {CLIP1_PATH}")
    print("Done.")
except Exception as e:
    print(f"ERROR: {e}")
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
    sys.exit(1)
