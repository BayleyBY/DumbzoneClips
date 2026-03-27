#!/usr/bin/env python3
"""
build_speaker_map_ecapa.py

Use SpeechBrain ECAPA to:
  1) Build anchor embeddings from one or more labeled payload JSONs
     (payload_updated.json from your clip pages).
  2) For a given month, compute embeddings for each diarized SPEAKER_XX
     and assign the closest anchor name.
  3) Emit a speaker_map.json suitable for multi_finder_packager_v6.py.

Assumed layout (same as your current project):
  TRANSCRIPTS ROOT (e.g. /Volumes/BagEnd/.../scripts):
    YY-MM/
      YY-MM-DD/
        YY-MM-DD.json   (WhisperX segments + "speaker")

  AUDIO ROOT (e.g. /Volumes/BagEnd/.../sodes_done):
    YY-MM/YY-MM-DD.<ext>
    or
    YY-MM-DD.<ext>

Training data:
  payload_updated.json (or payload.json) with entries such as:
  {
    "episode": "25-09-15",
    "start": 6635.17,
    "end": 6635.79,
    "speaker_id": "SPEAKER_06",
    "speaker": "Dan",
    ...
  }
We treat "speaker" as the canonical name and only use entries where
speaker is NOT of the form "SPEAKER_XX".

Example:

  python build_speaker_map_ecapa.py \
    --root "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/scripts" \
    --audio_dir "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/sodes_done" \
    --train_payload "/path/to/September_25_tp_payload_updated.json" \
    --month 25-10 \
    --out "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/speaker_map_auto_25-10.json" \
    --threshold 0.65 \
    --device cpu

Then plug that file into multi_finder_packager_v6.py:

  python multi_finder_packager_v6.py ... \
    --speaker_map "/Volumes/BagEnd/.../speaker_map_auto_25-10.json"

Requirements:
  pip install speechbrain torch torchaudio
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

SUPPORTED_AUDIO_EXTS = [".mp3", ".m4a", ".wav", ".flac", ".aac", ".mp4", ".mkv", ".mov"]

# ------------------------------ IO helpers ------------------------------


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_month_dirs(root: Path) -> List[Path]:
    return [
        p for p in sorted(root.iterdir())
        if p.is_dir() and re.match(r"^\d{2}-\d{2}$", p.name)
    ]


def group_episodes_by_month(root: Path) -> Dict[str, Dict[str, Path]]:
    months: Dict[str, Dict[str, Path]] = {}
    for mdir in find_month_dirs(root):
        ep_map: Dict[str, Path] = {}
        for ep in sorted(p for p in mdir.iterdir() if p.is_dir()):
            stem = ep.name
            if (ep / f"{stem}.json").exists():
                ep_map[stem] = ep
        if ep_map:
            months[mdir.name] = ep_map
    return months


def locate_audio(stem: str, audio_dir: Path) -> Optional[Path]:
    """
    Try:
      AUDIO_DIR/<stem>.<ext>
      AUDIO_DIR/<YY-MM>/<stem>.<ext>
    """
    # direct
    for ext in SUPPORTED_AUDIO_EXTS:
        p = audio_dir / f"{stem}{ext}"
        if p.exists():
            return p

    # with month subdir
    if len(stem) >= 5 and stem[2] == "-":
        month = stem[:5]
        for ext in SUPPORTED_AUDIO_EXTS:
            p = audio_dir / month / f"{stem}{ext}"
            if p.exists():
                return p

    return None


def load_segments(ep_folder: Path, stem: str) -> List[dict]:
    js = ep_folder / f"{stem}.json"
    if not js.exists():
        eprint(f"[WARN] Missing transcript: {js}")
        return []
    try:
        data = json.loads(js.read_text(encoding="utf-8"))
    except Exception as e:
        eprint(f"[WARN] Failed to read JSON {js}: {e}")
        return []
    segs = data.get("segments")
    if not isinstance(segs, list):
        eprint(f"[WARN] JSON missing 'segments' list: {js}")
        return []
    out: List[dict] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        start = s.get("start")
        end = s.get("end")
        text = s.get("text", "")
        spk = s.get("speaker", "?")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        rec = {
            "start": float(start),
            "end": float(end),
            "text": str(text),
            "speaker": spk if isinstance(spk, str) else "?",
        }
        out.append(rec)
    return out


# ------------------------------ Embedding helpers ------------------------------


def load_audio_cached(
    cache: Dict[Path, Tuple[torch.Tensor, int]],
    path: Path,
) -> Tuple[torch.Tensor, int]:
    """Load audio with torchaudio, caching per file."""
    if path in cache:
        return cache[path]
    wav, sr = torchaudio.load(str(path))
    cache[path] = (wav, sr)
    return wav, sr


def slice_waveform(
    wav: torch.Tensor,
    sr: int,
    start: float,
    end: float,
    min_dur: float = 0.5,
) -> Optional[torch.Tensor]:
    """Return mono slice [time] in seconds, or None if too short."""
    s = max(0.0, float(start))
    e = max(s, float(end))
    if e - s < min_dur:
        return None
    s_idx = int(s * sr)
    e_idx = int(e * sr)
    if s_idx >= wav.shape[1]:
        return None
    e_idx = min(e_idx, wav.shape[1])
    if e_idx - s_idx < int(min_dur * sr):
        return None
    seg = wav[:, s_idx:e_idx]
    # downmix to mono if needed
    if seg.shape[0] > 1:
        seg = seg.mean(dim=0, keepdim=True)
    return seg.squeeze(0)  # [time]


def embed_clip(
    classifier: EncoderClassifier,
    wav_1d: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Compute ECAPA embedding for a 1D waveform tensor."""
    wav_1d = wav_1d.to(device)
    if wav_1d.dim() == 1:
        wav_1d = wav_1d.unsqueeze(0)  # [1, time]
    with torch.no_grad():
        emb = classifier.encode_batch(wav_1d)  # [1, 1, feat] or [1, feat]
    emb = emb.squeeze().detach().cpu().numpy().astype("float32")
    # L2-normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ------------------------------ Phase 1: build anchors ------------------------------


def build_anchor_embeddings(
    classifier: EncoderClassifier,
    device: torch.device,
    audio_dir: Path,
    payload_paths: List[Path],
    max_clips_per_name: int = 40,
) -> Dict[str, np.ndarray]:
    """
    From one or more payload JSON files, build an average embedding per
    canonical speaker name (Dan, Jake, Blake, etc).

    Only uses entries where "speaker" does NOT look like "SPEAKER_07".
    """
    audio_cache: Dict[Path, Tuple[torch.Tensor, int]] = {}
    per_name_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)

    def is_generic_speaker(s: str) -> bool:
        return bool(re.match(r"^SPEAKER_\d+$", s.strip()))

    for path in payload_paths:
        eprint(f"[anchors] Loading payload: {path}")
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            eprint(f"[anchors]   ERROR reading {path}: {e}")
            continue
        if not isinstance(items, list):
            eprint(f"[anchors]   Skipping {path}: not a list")
            continue

        for idx, item in enumerate(items):
            name = str(item.get("speaker", "")).strip()
            if not name or is_generic_speaker(name):
                continue  # not labeled / not helpful

            ep = str(item.get("episode", "")).strip()
            start = item.get("start")
            end = item.get("end")
            if not ep or start is None or end is None:
                continue

            audio_path = locate_audio(ep, audio_dir)
            if not audio_path:
                eprint(f"[anchors]   WARNING: audio not found for episode {ep}")
                continue

            # Limit number of training clips per name to keep things balanced
            if len(per_name_vecs[name]) >= max_clips_per_name:
                continue

            try:
                wav, sr = load_audio_cached(audio_cache, audio_path)
                seg = slice_waveform(wav, sr, float(start), float(end), min_dur=0.8)
                if seg is None:
                    continue
                emb = embed_clip(classifier, seg, device)
                per_name_vecs[name].append(emb)
            except Exception as e:
                eprint(f"[anchors]   ERROR embedding {ep} idx={idx}: {e}")

    anchors: Dict[str, np.ndarray] = {}
    for name, vecs in per_name_vecs.items():
        if not vecs:
            continue
        mat = np.stack(vecs, axis=0)
        mean = mat.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        anchors[name] = mean.astype("float32")
        eprint(f"[anchors] Name '{name}': {len(vecs)} clips -> anchor built.")

    if not anchors:
        eprint("[anchors] No anchors built! Check your payloads and labels.")
    else:
        eprint(f"[anchors] Built {len(anchors)} anchor(s): {', '.join(sorted(anchors.keys()))}")
    return anchors


# ------------------------------ Phase 2: assign speakers ------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def assign_speakers_for_month(
    classifier: EncoderClassifier,
    device: torch.device,
    root: Path,
    audio_dir: Path,
    month: str,
    anchors: Dict[str, np.ndarray],
    max_segs_per_speaker: int = 10,
    min_dur_seg: float = 0.8,
    threshold: float = 0.6,
) -> Dict[str, Dict[str, str]]:
    """
    For each episode and per diarized speaker (SPEAKER_XX), build an
    embedding from that speaker's segments and assign the closest anchor
    name if above threshold.

    Returns mapping:
      { "YY-MM-DD": { "SPEAKER_06": "Dan", "SPEAKER_07": "Jake", ... }, ... }
    """
    months = group_episodes_by_month(root)
    if month not in months:
        eprint(f"[assign] Month not found under root: {month}")
        return {}

    ep_map = months[month]
    audio_cache: Dict[Path, Tuple[torch.Tensor, int]] = {}
    anchor_names = sorted(anchors.keys())
    if not anchor_names:
        eprint("[assign] No anchors provided; aborting.")
        return {}

    # Stack anchor vectors for vectorized similarity
    anchor_mat = np.stack([anchors[n] for n in anchor_names], axis=0)  # [K, D]

    def embed_speaker(
        episode: str,
        segs: List[dict],
        audio_path: Path,
    ) -> Optional[np.ndarray]:
        # Group segs by speaker_id
        by_spk: Dict[str, List[dict]] = defaultdict(list)
        for s in segs:
            sp = s.get("speaker", "?")
            by_spk[sp].append(s)

        wav, sr = load_audio_cached(audio_cache, audio_path)
        mapping_for_ep: Dict[str, np.ndarray] = {}

        for spk_id, s_list in by_spk.items():
            # pick up to max_segs_per_speaker segments, preferring longer ones
            s_list = sorted(s_list, key=lambda s: s["end"] - s["start"], reverse=True)
            s_list = s_list[:max_segs_per_speaker]
            vecs: List[np.ndarray] = []
            for s in s_list:
                st = float(s["start"])
                en = float(s["end"])
                seg_wav = slice_waveform(wav, sr, st, en, min_dur=min_dur_seg)
                if seg_wav is None:
                    continue
                try:
                    emb = embed_clip(classifier, seg_wav, device)
                    vecs.append(emb)
                except Exception as e:
                    eprint(f"[assign]   ERROR embedding seg {episode} {spk_id}: {e}")
            if not vecs:
                continue
            mat = np.stack(vecs, axis=0)
            mean = mat.mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                mean = mean / norm
            mapping_for_ep[spk_id] = mean.astype("float32")
        return mapping_for_ep

    result: Dict[str, Dict[str, str]] = {}

    for ep, ep_folder in ep_map.items():
        eprint(f"[assign] Episode {ep} ...")
        segs = load_segments(ep_folder, ep)
        if not segs:
            eprint(f"[assign]   No segments; skipping.")
            continue

        audio_path = locate_audio(ep, audio_dir)
        if not audio_path:
            eprint(f"[assign]   WARNING: audio not found; skipping.")
            continue

        spk_embs = embed_speaker(ep, segs, audio_path)
        if not spk_embs:
            eprint(f"[assign]   No usable speaker embeddings; skipping.")
            continue

        ep_map_out: Dict[str, str] = {}
        for spk_id, emb in spk_embs.items():
            # compute cosine sim vs anchors
            sims = anchor_mat @ emb  # [K]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_name = anchor_names[best_idx]
            eprint(f"[assign]   {ep} {spk_id} -> {best_name} (sim={best_sim:.3f})")
            if best_sim >= threshold:
                ep_map_out[spk_id] = best_name
        if ep_map_out:
            result[ep] = ep_map_out

    return result


# ------------------------------ Main ------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Build a speaker_map.json using SpeechBrain ECAPA anchors from labeled payloads."
    )
    ap.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Transcripts root (YY-MM/YY-MM-DD/YY-MM-DD.json).",
    )
    ap.add_argument(
        "--audio_dir",
        required=True,
        type=Path,
        help="Audio root (contains YY-MM/YY-MM-DD.<ext> or YY-MM-DD.<ext>).",
    )
    ap.add_argument(
        "--train_payload",
        type=Path,
        action="append",
        default=[],
        help="Path to payload.json or payload_updated.json with labeled speakers. "
             "Can be passed multiple times.",
    )
    ap.add_argument(
        "--month",
        required=True,
        help="Target month (YY-MM) to build mapping for.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output speaker_map.json path.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold to accept an anchor match (default 0.6).",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="torch device for ECAPA (cpu, cuda, or mps). Default: cpu.",
    )
    ap.add_argument(
        "--max_train_per_name",
        type=int,
        default=40,
        help="Max training clips per name when building anchors.",
    )
    ap.add_argument(
        "--max_segs_per_speaker",
        type=int,
        default=10,
        help="Max diarized segments per SPEAKER_XX to use for its embedding.",
    )
    ap.add_argument(
        "--min_seg_dur",
        type=float,
        default=0.8,
        help="Minimum seconds per diarized segment to be considered.",
    )

    args = ap.parse_args()

    if not args.root.exists():
        eprint(f"--root not found: {args.root}")
        sys.exit(2)
    if not args.audio_dir.exists():
        eprint(f"--audio_dir not found: {args.audio_dir}")
        sys.exit(2)
    if not args.train_payload:
        eprint("You must provide at least one --train_payload payload JSON.")
        sys.exit(2)

    device = torch.device(args.device)
    eprint(f"[model] Loading ECAPA on {device} ...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
        savedir=str(Path("pretrained_models") / "spkrec-ecapa-voxceleb"),
    )

    # Phase 1: anchors
    anchors = build_anchor_embeddings(
        classifier,
        device,
        args.audio_dir,
        args.train_payload,
        max_clips_per_name=args.max_train_per_name,
    )
    if not anchors:
        eprint("No anchors built; exiting.")
        sys.exit(1)

    # Phase 2: assignment
    mapping = assign_speakers_for_month(
        classifier,
        device,
        args.root,
        args.audio_dir,
        args.month,
        anchors,
        max_segs_per_speaker=args.max_segs_per_speaker,
        min_dur_seg=args.min_seg_dur,
        threshold=args.threshold,
    )

    if not mapping:
        eprint("No episode-speaker mappings produced; nothing to write.")
        sys.exit(1)

    # Write out in the format multi_finder_packager expects:
    # { "YY-MM-DD": { "SPEAKER_06": "Dan", ... }, ... }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    eprint(f"[done] Wrote speaker map to {args.out}")


if __name__ == "__main__":
    main()
