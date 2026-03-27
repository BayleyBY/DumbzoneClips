#!/usr/bin/env python3
"""
DumbzoneClips Pipeline
Automates: Drive download → WhisperX transcribe → phrase find → clip package → Drive upload
Stops at: HTML share folders ready for human review (speaker labeling + clip audition)

Usage:
  python pipeline.py               # process all new episodes
  python pipeline.py --month 26-03 # specific month only
  python pipeline.py --dry-run     # show what would run without doing it
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

# ── Google Drive helpers (via gog CLI) ───────────────────────────────────────

def gog_env(cfg: dict) -> dict:
    env = os.environ.copy()
    env["GOG_KEYRING_BACKEND"] = "file"
    env["GOG_KEYRING_PASSWORD"] = cfg["gog_keyring_password"]
    env["GOG_ACCOUNT"] = cfg["gog_account"]
    return env

def gog(cfg: dict, *args) -> str:
    result = subprocess.run(
        ["gog"] + list(args),
        env=gog_env(cfg),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"gog error: {result.stderr.strip()}")
    return result.stdout.strip()

def list_drive_children(cfg: dict, folder_id: str) -> list[dict]:
    """List files/folders directly inside a Drive folder."""
    output = gog(cfg, "drive", "search", f"'{folder_id}' in parents", "--max", "200")
    items = []
    for line in output.splitlines()[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 3:
            items.append({"id": parts[0], "name": parts[1], "type": parts[2]})
    return items

def download_file(cfg: dict, file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gog(cfg, "drive", "download", file_id, "--out", str(out_path))

def upload_file(cfg: dict, local_path: Path, parent_folder_id: str) -> str:
    """Upload a file to Drive, return the new file ID."""
    output = gog(cfg, "drive", "upload", str(local_path), "--parent", parent_folder_id)
    # parse file ID from output
    for line in output.splitlines():
        parts = line.split()
        if parts and re.match(r'^[A-Za-z0-9_-]{25,}$', parts[0]):
            return parts[0]
    return ""

def create_folder(cfg: dict, name: str, parent_id: str) -> str:
    """Create a Drive folder, return its ID."""
    output = gog(cfg, "drive", "mkdir", name, "--parent", parent_id)
    for line in output.splitlines():
        parts = line.split()
        if parts and re.match(r'^[A-Za-z0-9_-]{25,}$', parts[0]):
            return parts[0]
    return ""

def get_or_create_folder(cfg: dict, name: str, parent_id: str) -> str:
    """Get existing folder by name or create it."""
    children = list_drive_children(cfg, parent_id)
    for c in children:
        if c["name"] == name and c["type"] == "folder":
            return c["id"]
    return create_folder(cfg, name, parent_id)

# ── Episode discovery ─────────────────────────────────────────────────────────

def parse_month(filename: str) -> str | None:
    """
    Extract month string from episode filename.
    Handles both:
      YY-MM-DD   → YY-MM   (e.g. 25-11-03 → 25-11)
      YYYY-MM-DD → YYYY-MM  (e.g. 2026-01-25 → 2026-01)
    """
    stem = Path(filename).stem
    # Full year: 2026-01-25
    m = re.match(r'^(\d{4}-\d{2})-\d{2}', stem)
    if m:
        return m.group(1)
    # Short year: 25-11-03
    m = re.match(r'^(\d{2}-\d{2})-\d{2}', stem)
    if m:
        return m.group(1)
    return None

def get_done_stems(cfg: dict, month: str) -> set[str]:
    """
    Return stems already transcribed in Drive scripts/.
    Handles two layouts:
      scripts/YY-MM/YY-MM-DD/  (old style)
      scripts/YYYY-MM-DD/      (new style — episode folders at top level)
    """
    scripts_id = cfg["drive"]["scripts_folder_id"]
    children = list_drive_children(cfg, scripts_id)
    done = set()
    for c in children:
        if c["type"] != "folder":
            continue
        name = c["name"]
        # Old style: month subfolder (e.g. 25-12)
        if name == month:
            ep_folders = list_drive_children(cfg, c["id"])
            done.update(e["name"] for e in ep_folders if e["type"] == "folder")
        # New style: episode folder at top level (e.g. 2026-01-25)
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', name) or re.match(r'^\d{2}-\d{2}-\d{2}$', name):
            ep_month = parse_month(name)
            if ep_month == month:
                done.add(name)
    return done

def discover_new_episodes(cfg: dict, target_month: str | None) -> list[dict]:
    """
    List episodes in Drive sodes/ not yet transcribed.
    Returns list of {"id", "name", "stem", "month"} dicts.
    """
    sodes_id = cfg["drive"]["sodes_folder_id"]
    # sodes/ may have month subfolders or flat MP3s
    children = list_drive_children(cfg, sodes_id)

    episodes = []
    audio_exts = {".mp3", ".m4a", ".wav", ".mp4", ".aac", ".flac"}

    def check_file(item, month_hint=None):
        name = item["name"]
        ext = Path(name).suffix.lower()
        if ext not in audio_exts:
            return
        stem = Path(name).stem
        month = month_hint or parse_month(stem)
        if not month:
            print(f"  Skipping {name} — can't parse month from filename")
            return
        if target_month and month != target_month:
            return
        episodes.append({"id": item["id"], "name": name, "stem": stem, "month": month})

    for item in children:
        if item["type"] == "folder":
            # month subfolder
            month_name = item["name"]
            if re.match(r'^\d{2}-\d{2}$', month_name):
                sub = list_drive_children(cfg, item["id"])
                for sub_item in sub:
                    check_file(sub_item, month_name)
        else:
            check_file(item)

    # Filter out already-transcribed episodes
    months_seen = {e["month"] for e in episodes}
    done_by_month = {m: get_done_stems(cfg, m) for m in months_seen}
    new = [e for e in episodes if e["stem"] not in done_by_month.get(e["month"], set())]

    return sorted(new, key=lambda e: e["stem"])

# ── WhisperX ──────────────────────────────────────────────────────────────────

def transcribe_episode(cfg: dict, audio_path: Path, out_dir: Path) -> bool:
    """Run whisperx on one episode. Returns True on success."""
    wx = cfg["whisperx"]
    venv_python = Path(cfg["local"]["venv"]) / "bin" / "python"
    script = Path(cfg["local"]["scripts_dir"]) / "whisperxpy_v4.py"

    cmd = [
        str(venv_python), str(script),
        "--indir", str(audio_path.parent),
        "--outdir", str(out_dir),
        "--model", wx["model"],
        "--device", wx["device"],
        "--compute_type", wx["compute_type"],
        "--threads", str(wx["threads"]),
        "--progress",
        "--model_dir", cfg["local"]["whisperx_model_dir"],
        "--vad_method", wx["vad_method"],
        "--min_speakers", str(wx["min_speakers"]),
        "--max_speakers", str(wx["max_speakers"]),
        "--hf_token", cfg["hf_token"],
    ]
    print(f"  → Transcribing {audio_path.name} ...")
    result = subprocess.run(cmd, text=True)
    return result.returncode == 0

# ── Clip packager ─────────────────────────────────────────────────────────────

def package_clips(cfg: dict, scripts_dir: Path, audio_dir: Path, month: str, speaker_map_path: Path) -> bool:
    """Run multi_finder_packager for a month. Returns True on success."""
    pk = cfg["packager"]
    venv_python = Path(cfg["local"]["venv"]) / "bin" / "python"
    script = Path(cfg["local"]["scripts_dir"]) / "multi_finder_packager_v6.py"

    cmd = [
        str(venv_python), str(script),
        "--root", str(scripts_dir),
        "--audio_dir", str(audio_dir),
        "--month", month,
        "--pad_before", str(pk["pad_before"]),
        "--pad_after", str(pk["pad_after"]),
        "--pad_tp", pk["pad_tp"],
        "--pad_hab", pk["pad_hab"],
        "--pad_wife", pk["pad_wife"],
        "--pad_roseanne", pk["pad_roseanne"],
        "--pad_hillary", pk["pad_hillary"],
        "--speaker_map", str(speaker_map_path),
    ]
    print(f"  → Packaging clips for {month} ...")
    result = subprocess.run(cmd, text=True)
    return result.returncode == 0

# ── Drive upload ──────────────────────────────────────────────────────────────

def upload_share_folder(cfg: dict, local_share_dir: Path, month: str) -> str:
    """
    Upload a local share folder (e.g. 25-11_tp_portable/) to Drive share_readonly/YY-MM/.
    Returns a Drive folder URL for the month.
    """
    share_root_id = cfg["drive"]["share_folder_id"]
    month_folder_id = get_or_create_folder(cfg, month, share_root_id)

    # Upload each category folder
    for category_dir in sorted(local_share_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        print(f"  → Uploading {category_dir.name} ...")
        cat_folder_id = get_or_create_folder(cfg, category_dir.name, month_folder_id)

        # Upload index.html
        index = category_dir / "index.html"
        if index.exists():
            upload_file(cfg, index, cat_folder_id)

        # Upload clips/
        clips_dir = category_dir / "clips"
        if clips_dir.exists():
            clips_folder_id = get_or_create_folder(cfg, "clips", cat_folder_id)
            for clip in sorted(clips_dir.iterdir()):
                if clip.is_file():
                    upload_file(cfg, clip, clips_folder_id)

    return f"https://drive.google.com/drive/folders/{month_folder_id}"

# ── Notification ──────────────────────────────────────────────────────────────

def notify_discord(cfg: dict, month: str, episode_count: int, drive_url: str) -> None:
    """Send a completion message to the #github Discord channel via OpenClaw."""
    msg = (
        f"⚾ **DumbzoneClips — {month} ready for review**\n"
        f"{episode_count} episode(s) processed. Clips packaged by category.\n"
        f"📁 Drive: <{drive_url}>\n"
        f"Open each `index.html` to audition clips and label speakers, then let me know when you're ready for the final package."
    )
    # Write to a temp file for OpenClaw to pick up
    notify_path = Path(cfg["local"]["tmp_dir"]) / "discord_notify.txt"
    notify_path.write_text(msg)
    print(f"\n[NOTIFY] {msg}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DumbzoneClips automated pipeline")
    parser.add_argument("--month", default=None, help="Process specific month (YY-MM). Default: all new.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without doing it.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading results to Drive.")
    args = parser.parse_args()

    cfg = load_config()
    tmp = Path(cfg["local"]["tmp_dir"])
    tmp.mkdir(parents=True, exist_ok=True)
    Path(cfg["local"]["whisperx_model_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DumbzoneClips Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Discover new episodes
    print("\n[1/5] Checking Drive for new episodes...")
    new_episodes = discover_new_episodes(cfg, args.month)
    if not new_episodes:
        print("  No new episodes found. Nothing to do.")
        return
    print(f"  Found {len(new_episodes)} new episode(s):")
    for ep in new_episodes:
        print(f"    {ep['stem']} ({ep['month']})")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above. Exiting.")
        return

    # 2. Download speaker_map
    print("\n[2/5] Downloading speaker_map...")
    speaker_map_path = tmp / "speaker_map.json"
    download_file(cfg, cfg["drive"]["speaker_map_id"], speaker_map_path)
    print(f"  Downloaded speaker_map.json")

    # 3. Transcribe each episode
    months_processed = set()
    local_audio_by_month: dict[str, list[Path]] = {}

    print("\n[3/5] Transcribing episodes...")
    for ep in new_episodes:
        month = ep["month"]
        audio_tmp = tmp / "sodes" / month
        audio_tmp.mkdir(parents=True, exist_ok=True)
        audio_path = audio_tmp / ep["name"]

        # Download audio from Drive
        print(f"\n  Downloading {ep['name']} from Drive...")
        download_file(cfg, ep["id"], audio_path)

        # Local scripts output dir
        scripts_out = tmp / "scripts" / month / ep["stem"]
        scripts_out.mkdir(parents=True, exist_ok=True)

        # Transcribe
        ok = transcribe_episode(cfg, audio_path, tmp / "scripts" / month)
        if not ok:
            print(f"  ✗ Transcription failed for {ep['stem']}")
            continue

        months_processed.add(month)
        local_audio_by_month.setdefault(month, []).append(audio_path)
        print(f"  ✓ {ep['stem']} transcribed")

    if not months_processed:
        print("\nNo episodes transcribed successfully. Exiting.")
        sys.exit(1)

    # 4. Package clips per month
    print("\n[4/5] Packaging clips...")
    share_dirs_by_month: dict[str, Path] = {}

    for month in sorted(months_processed):
        scripts_dir = tmp / "scripts"
        audio_dir = tmp / "sodes"
        share_dir = tmp / "share" / month

        ok = package_clips(cfg, scripts_dir, audio_dir, month, speaker_map_path)
        if ok:
            share_dirs_by_month[month] = tmp / "scripts" / month
            print(f"  ✓ {month} clips packaged")
        else:
            print(f"  ✗ Packaging failed for {month}")

    # 5. Upload to Drive
    if not args.skip_upload:
        print("\n[5/5] Uploading to Drive...")
        for month, scripts_month_dir in share_dirs_by_month.items():
            drive_url = upload_share_folder(cfg, scripts_month_dir, month)
            ep_count = len([e for e in new_episodes if e["month"] == month])
            notify_discord(cfg, month, ep_count, drive_url)
            print(f"  ✓ {month} uploaded → {drive_url}")
    else:
        print("\n[5/5] Skipping upload (--skip-upload)")
        for month in share_dirs_by_month:
            local_path = share_dirs_by_month[month]
            print(f"  Local output: {local_path}")

    print("\n" + "=" * 60)
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
