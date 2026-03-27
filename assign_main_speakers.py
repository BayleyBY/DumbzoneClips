#!/usr/bin/env python3
"""
Assign main hosts (Dan/Jake/Blake) vs OTHER to diarized segments in WhisperX JSONs.

- Takes enrollment WAVs for the 3 hosts
- For each episode JSON:
  - Extracts each segment's audio from the original episode file (ffmpeg)
  - Computes speaker embeddings (speechbrain ECAPA-TDNN)
  - Assigns the segment to the closest host if similarity >= threshold, else OTHER
- Outputs either:
  (a) a speaker_map.json you can pass to multi_finder_packager --speaker_map
  or
  (b) in-place rewritten episode JSON (toggle with --rewrite)

Usage:
  python assign_main_speakers.py \
    --root "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/scripts/25-10" \
    --audio_dir "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/sodes_done/25-10" \
    --enroll Dan:/path/Dan_ref.wav Jake:/path/Jake_ref.wav Julie:/path/Julie_ref.wav \
    --threshold 0.55 \
    --outfile "/tmp/speaker_map_25-10.json"

Notes:
- Tune --threshold in ~[0.45, 0.65]. Lower = more aggressive assignment.
- Requires: ffmpeg, speechbrain, torch, torchaudio
"""

from __future__ import annotations
import argparse, json, subprocess, tempfile, os, sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier  # downloads spkrec-ecapa-voxceleb

SUPPORTED_EXTS = [".mp3", ".m4a", ".wav", ".mp4", ".aac", ".flac", ".mkv", ".mov"]

def which_or_die(cmd: str):
    from shutil import which
    if which(cmd) is None:
        print(f"Missing required command: {cmd}", file=sys.stderr)
        sys.exit(1)

def load_wav_16k_mono(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    # assume shape (1, D)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-9)
    return float((a * b).sum().item())

def find_audio(stem: str, audio_dir: Path) -> Path | None:
    direct = audio_dir / stem  # in case audio_dir already points at the month folder
    for ext in SUPPORTED_EXTS:
        p = audio_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for ext in SUPPORTED_EXTS:
        p = direct.with_suffix(ext)
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Month folder with episode subfolders (e.g., .../scripts/25-10)")
    ap.add_argument("--audio_dir", required=True, type=Path, help="Audio folder for the month (e.g., .../sodes_done/25-10)")
    ap.add_argument("--enroll", nargs="+", required=True,
                    help="List like Name:/path/to/ref.wav (mono, 16k preferred). Example: Dan:Dan_ref.wav")
    ap.add_argument("--threshold", type=float, default=0.55, help="Cosine sim threshold to assign host")
    ap.add_argument("--outfile", type=Path, default=None, help="Write speaker_map JSON here (default: print to stdout)")
    ap.add_argument("--rewrite", action="store_true", help="Rewrite episode JSON in-place with new names instead of map")
    ap.add_argument("--min_dur", type=float, default=0.8, help="Skip segments shorter than this many seconds")

    args = ap.parse_args()
    which_or_die("ffmpeg")

    # Load classifier (downloads on first run)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})  # or mps if desired

    # Enrollment embeddings
    enroll_vecs: Dict[str, torch.Tensor] = {}
    for spec in args.enroll:
        if ":" not in spec:
            print(f"Bad --enroll item (use Name:path): {spec}", file=sys.stderr); sys.exit(2)
        name, p = spec.split(":", 1)
        wav = load_wav_16k_mono(Path(p))
        with torch.no_grad():
            emb = classifier.encode_batch(wav).squeeze(0)  # shape(1,D) -> (D,) after squeeze? Keep batch dim
            if emb.dim() == 1: emb = emb.unsqueeze(0)
        enroll_vecs[name] = emb

    # Walk episodes
    ep_dirs = [p for p in sorted(args.root.iterdir()) if p.is_dir()]
    speaker_map: Dict[str, Dict[str, str]] = {}

    for ep_dir in ep_dirs:
        stem = ep_dir.name
        js = ep_dir / f"{stem}.json"
        if not js.exists():
            continue
        audio = find_audio(stem, args.audio_dir)
        if not audio:
            print(f"[WARN] Missing audio for {stem}", file=sys.stderr)
            continue

        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed JSON {js}: {e}", file=sys.stderr)
            continue

        segs = data.get("segments") or []
        spk_map: Dict[str, str] = {}
        tmpdir = Path(tempfile.mkdtemp(prefix=f"assign_spk_{stem}_"))

        for seg in segs:
            spk_id = seg.get("speaker") or "SPEAKER"
            start = float(seg.get("start", 0.0)); end = float(seg.get("end", 0.0))
            if end - start < args.min_dur:
                # too short, leave as OTHER
                spk_map.setdefault(spk_id, "OTHER")
                continue

            # extract snippet
            outwav = tmpdir / f"{start:.3f}_{end:.3f}.wav"
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(audio),
                   "-ac", "1", "-ar", "16000", str(outwav)]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                spk_map.setdefault(spk_id, "OTHER")
                continue

            # embed + compare
            wav = load_wav_16k_mono(outwav)
            with torch.no_grad():
                emb = classifier.encode_batch(wav).squeeze(0)
                if emb.dim() == 1: emb = emb.unsqueeze(0)

            best_name, best_sim = None, -1.0
            for name, ref in enroll_vecs.items():
                sim = cosine_sim(emb, ref)
                if sim > best_sim:
                    best_name, best_sim = name, sim

            assigned = best_name if (best_name and best_sim >= args.threshold) else "OTHER"
            # remember the *first* assignment per SPEAKER_XX; optional: majority vote instead
            if spk_id not in spk_map:
                spk_map[spk_id] = assigned

            # optionally rewrite per-segment immediately
            if args.rewrite:
                seg["speaker"] = assigned

        # merge map & write
        if args.rewrite:
            js.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if spk_map:
            speaker_map[stem] = spk_map

    # Output mapping
    out_json = json.dumps(speaker_map, ensure_ascii=False, indent=2)
    if args.outfile:
        args.outfile.write_text(out_json, encoding="utf-8")
        print(f"Wrote: {args.outfile}")
    else:
        print(out_json)

if __name__ == "__main__":
    main()
