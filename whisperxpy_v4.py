#!/usr/bin/env python3
"""
Batch-transcribe & diarize audio with WhisperX (CPU + int8 by default),
using correct CLI flags (chunk_size + silero VAD) and QoL options
(model caching, threads, progress, speaker bounds, warning suppression).

Usage examples:
  python whisperx_batch.py --indir "/path/to/podcasts" --outdir "$HOME/transcripts_x"
  python whisperx_batch.py --indir "/path" --outdir "$HOME/transcripts_x" --model medium.en
  HF_TOKEN=hf_xxx python whisperx_batch.py --indir "/path" --outdir "$HOME/transcripts_x"
  python whisperx_batch.py --indir "/path" --outdir "$HOME/transcripts_x" --workers 2 --progress
  python whisperx_batch.py --indir "/path" --outdir "$HOME/transcripts_x" --min_speakers 2 --max_speakers 3
"""

from __future__ import annotations
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

SUPPORTED_EXTS = {".mp3", ".m4a", ".wav", ".mp4", ".aac", ".flac", ".mkv", ".mov"}

### Timer to see how long the process took
def fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:   return f"{h:d}h {m:02d}m {s:02d}s"
    if m:   return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def which_or_die(cmd: str, hint: str = "") -> None:
    if shutil.which(cmd) is None:
        msg = f"Required command '{cmd}' not found on PATH."
        if hint:
            msg += f" {hint}"
        print(msg, file=sys.stderr)
        sys.exit(1)


def build_whisperx_cmd(
    infile: Path,
    outdir: Path,
    model: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    chunk_size: int,
    use_vad: bool,
    vad_method: str,
    vad_onset: float,
    vad_offset: float,
    hf_token: Optional[str],
    model_dir: Optional[Path],
    threads: int,
    print_progress: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> List[str]:
    cmd = [
        "whisperx",
        str(infile),
        "--model", model,
        "--diarize",
        "--device", device,
        "--compute_type", compute_type,
        "--output_dir", str(outdir),
        "--chunk_size", str(chunk_size),
    ]
    if use_vad:
        cmd += ["--vad_method", vad_method]
        if vad_method == "silero":
            cmd += ["--vad_onset", str(vad_onset), "--vad_offset", str(vad_offset)]
    if language:
        cmd += ["--language", language]
    if hf_token:
        cmd += ["--hf_token", hf_token]
    if model_dir:
        cmd += ["--model_dir", str(model_dir)]
    if threads and threads > 0:
        cmd += ["--threads", str(threads)]
    if print_progress:
        cmd += ["--print_progress", "True"]
    if min_speakers is not None:
        cmd += ["--min_speakers", str(min_speakers)]
    if max_speakers is not None:
        cmd += ["--max_speakers", str(max_speakers)]
    return cmd



def discover_audio_files(indir: Path, patterns: List[str]) -> List[Path]:
    exts = {e.lower().strip() for e in patterns if e.strip()}
    if not exts:
        exts = SUPPORTED_EXTS
    files: List[Path] = []
    for p in indir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def stem_for_output(infile: Path) -> str:
    return infile.stem


def already_done(outdir: Path, stem: str) -> bool:
    """
    Consider a file done if <outdir>/<stem>/<stem>.txt exists.
    """
    subdir = outdir / stem
    return (subdir / f"{stem}.txt").exists()


def run_one(
    infile: Path,
    outdir: Path,
    args: argparse.Namespace,
    log_to_file: bool,
) -> Tuple[Path, int]:
    stem = stem_for_output(infile)
    subdir = outdir / stem
    subdir.mkdir(parents=True, exist_ok=True)

    if already_done(outdir, stem):
        print(f"Skip (exists): {stem}")
        return infile, 0

    cmd = build_whisperx_cmd(
        infile=infile,
        outdir=subdir,                    # per-episode subfolder
        model=args.model,
        language=args.language if args.language else None,
        device=args.device,
        compute_type=args.compute_type,
        chunk_size=args.chunk_size,
        use_vad=not args.no_vad,
        vad_method=args.vad_method,
        vad_onset=args.vad_onset,
        vad_offset=args.vad_offset,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        model_dir=args.model_dir,
        threads=args.threads,
        print_progress=args.progress,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print(f"Transcribing + diarizing: {infile.name}")

    env = os.environ.copy()
    if args.suppress_warnings:
        env["PYTHONWARNINGS"] = "ignore"

    if log_to_file:
        log_path = subdir / f"{stem}.log"
        with open(log_path, "w", encoding="utf-8") as logf:
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            logf.write("# CMD: " + cmd_str + "\n")
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            return infile, proc.returncode
    else:
        proc = subprocess.run(cmd, env=env)
        return infile, proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch WhisperX (CPU + int8 + diarization) with QoL options.")
    parser.add_argument("--indir", required=True, type=Path, help="Folder with audio/video files")
    parser.add_argument("--outdir", required=True, type=Path, help="Folder for transcripts")
    parser.add_argument("--model", default="small.en", help="ASR model (e.g., small.en, medium.en, large-v3)")
    parser.add_argument("--language", default="en", help="Language code (e.g., en). Leave empty to auto-detect.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="Compute device")
    parser.add_argument("--compute_type", default="int8", help="Compute type for faster-whisper (int8|float32, etc.). int8 is fast on CPU.")
    parser.add_argument("--chunk_size", type=int, default=30, help="ASR chunk size in seconds (30 is a good starting point)")
    parser.add_argument("--no_vad", action="store_true", help="Disable VAD (silero) trimming")
    parser.add_argument("--vad_method", choices=["silero", "pyannote"], default="silero", help="VAD backend for diarization (pyannote requires HF token)")
    parser.add_argument("--vad_onset", type=float, default=0.5, help="VAD onset threshold (silero)")
    parser.add_argument("--vad_offset", type=float, default=0.5, help="VAD offset threshold (silero)")
    parser.add_argument("--patterns", default=".mp3,.m4a,.wav,.mp4,.aac,.flac,.mkv,.mov", help="Comma list of file extensions to include")
    parser.add_argument("--hf_token", default=None, help="HF token (or set env HF_TOKEN)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel files to process")

    # QoL additions
    parser.add_argument("--model_dir", type=Path, default=Path.home() / "whisperx_models", help="Cache dir for models (prevents re-downloads)")
    parser.add_argument("--threads", type=int, default=0, help="Threads to pass to WhisperX (0 = do not set)")
    parser.add_argument("--progress", action="store_true", help="Show incremental progress timestamps from WhisperX")
    parser.add_argument("--min_speakers", type=int, default=None, help="Lower bound for speaker count (diarization)")
    parser.add_argument("--max_speakers", type=int, default=None, help="Upper bound for speaker count (diarization)")
    parser.add_argument("--suppress_warnings", action="store_true", help="Suppress Python warnings in child processes")

    args = parser.parse_args()

    ### Timer Start
    t0 = time.perf_counter()

    which_or_die("whisperx", "Activate your venv or `pip install whisperx`.")
    which_or_die("ffmpeg", "Install via Homebrew: `brew install ffmpeg`.")

    if not args.indir.exists():
        print(f"--indir not found: {args.indir}", file=sys.stderr)
        sys.exit(2)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.model_dir:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    files = discover_audio_files(args.indir, patterns)
    if not files:
        print("No input files found with given patterns.", file=sys.stderr)
        sys.exit(0)

    log_to_file = args.workers > 1

    t0 = time.perf_counter()

    errs = 0
    if args.workers <= 1:
        for f in files:
            _, rc = run_one(f, args.outdir, args, log_to_file=False)
            if rc != 0:
                print(f"[ERROR] Non-zero exit for {f.name} (code {rc})", file=sys.stderr)
                errs += 1
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(run_one, f, args.outdir, args, True) for f in files]
            for fut in as_completed(futs):
                f, rc = fut.result()
                if rc != 0:
                    print(f"[ERROR] Non-zero exit for {f.name} (code {rc})", file=sys.stderr)
                    errs += 1

    t1 = time.perf_counter()
    elapsed_sec = time.perf_counter() - t0
    elapsed_td = timedelta(seconds=int(elapsed_sec))
    print(f"Total processing time: {elapsed_td} ({elapsed_sec:.2f} seconds)")

    if errs:
        print(f"Completed with {errs} error(s). Check logs in: {args.outdir}", file=sys.stderr)
        print(f"Total processing time: {elapsed_td} ({elapsed_sec:.2f} seconds)")
        sys.exit(3)
    else:
        print("All done ✅")
        print(f"Total processing time: {elapsed_td} ({elapsed_sec:.2f} seconds)")


if __name__ == "__main__":
    main()