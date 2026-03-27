#!/usr/bin/env python3
"""
Multi Finder Packager — Unified

Build portable monthly clip pages (HTML + local audio clips) for multiple
categories:
  • The Point Is            ("the point is")
  • Have/Had a Buddy        ("have a buddy" / "had a buddy")
  • Roseanne                (mentions)
  • Hillary                 (mentions)
  • Fight With Wife         (topic heuristic: fight-words + spouse-words)

Key features
  • Works with your layout:
      ROOT/
        YY-MM/
          YY-MM-DD/
            YY-MM-DD.json   (WhisperX segments + optional words[] + speaker)
    Audio lookup tries both:
      AUDIO_DIR/<YY-MM-DD>.<ext>
      AUDIO_DIR/<YY-MM>/<YY-MM-DD>.<ext>
  • Smart sentence-like clipping using word-level timestamps
    (expands window to pause boundaries; min ~3s, max ~12s)
  • Per-month packaging; one folder per category with index.html + clips/
  • Inline speaker relabel in the HTML (per episode-speaker group or per-clip),
    with an exportable speaker_map.json you can reuse next run.

Example
  python multi_finder_packager.py \
    --root "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/scripts" \
    --audio_dir "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/sodes_done" \
    --month 25-10 \
    --mono --loudnorm --format mp3 --bitrate 96k \
    --pad_before 0.4 --pad_after 0.6

Outputs per month (under ROOT/YY-MM):
  YY-MM_tp_portable/
  YY-MM_hab_portable/
  YY-MM_roseanne_portable/
  YY-MM_hillary_portable/
  YY-MM_wife_fight_portable/
Each contains index.html + clips/.
"""
from __future__ import annotations
import argparse
import calendar
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_AUDIO_EXTS = [".mp3", ".m4a", ".wav", ".mp4", ".aac", ".flac", ".mkv", ".mov"]

# --------------------- Utils ---------------------

def which_or_die(cmd: str, hint: str = "") -> None:
    if shutil.which(cmd) is None:
        msg = f"Required command '{cmd}' not found."
        if hint:
            msg += f" {hint}"
        print(msg, file=sys.stderr)
        sys.exit(1)

def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "clip"

def parse_pad_pair(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Pad must be 'before,after' (e.g. 0.4,0.6)")
    return (float(parts[0]), float(parts[1]))

# --------------------- Discovery ---------------------

def find_month_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and re.match(r"^\d{2}-\d{2}$", p.name)]

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

def pretty_month_title(month_str: str) -> str:
    try:
        yy, mm = month_str.split("-")
        year = int(yy)
        # assume 2000–2099 for two-digit years
        if 0 <= year <= 99:
            year += 2000
        return f"{calendar.month_name[int(mm)]} {year}"
    except Exception:
        return month_str

def pretty_episode_label(ep_str: str) -> str:
    try:
        yy, mm, dd = ep_str.split("-")
        return f"{calendar.month_name[int(mm)]} {int(dd)}"
    except Exception:
        return ep_str

# Audio can be in AUDIO_DIR/<stem>.<ext> or AUDIO_DIR/<month>/<stem>.<ext>

def locate_audio(stem: str, audio_dir: Optional[Path], month: Optional[str] = None) -> Optional[Path]:
    if not audio_dir:
        return None
    for ext in SUPPORTED_AUDIO_EXTS:
        p = audio_dir / f"{stem}{ext}"
        if p.exists():
            return p
    if month:
        for ext in SUPPORTED_AUDIO_EXTS:
            p = audio_dir / month / f"{stem}{ext}"
            if p.exists():
                return p
    if audio_dir.exists():
        for sub in audio_dir.iterdir():
            if not sub.is_dir():
                continue
            for ext in SUPPORTED_AUDIO_EXTS:
                p = sub / f"{stem}{ext}"
                if p.exists():
                    return p
    return None

# --------------------- Load ---------------------

def load_segments(ep_folder: Path, stem: str) -> List[dict]:
    js = ep_folder / f"{stem}.json"
    if not js.exists():
        print(f"ERROR: Missing {js}", file=sys.stderr)
        return []
    try:
        data = json.loads(js.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to read JSON: {e}", file=sys.stderr)
        return []
    segs = data.get("segments")
    if not isinstance(segs, list):
        print("ERROR: JSON missing 'segments' list", file=sys.stderr)
        return []
    out: List[dict] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        start = s.get("start"); end = s.get("end"); text = s.get("text", ""); words = s.get("words")
        spk = s.get("speaker", "?")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and isinstance(text, str):
            rec = {"start": float(start), "end": float(end), "text": text.strip(), "speaker": spk if isinstance(spk, str) else "?"}
            if isinstance(words, list):
                clean_words = []
                for w in words:
                    try:
                        ws = float(w.get("start", 0.0)); we = float(w.get("end", 0.0)); wt = str(w.get("word", w.get("text", "")).strip())
                        if wt:
                            clean_words.append({"start": ws, "end": we, "word": wt})
                    except Exception:
                        pass
                if clean_words:
                    rec["words"] = clean_words
            out.append(rec)
    return out

# --------------------- Finders ---------------------

PHRASES = {
    'tp': {
        'label': 'The Point Is',
        'targets_any': [
            ['point','is'],           # “Point Is”
            ['point','was'],          # “Point was”
        ],
    },
    'hab':      { 'label': 'Have A Buddy',   'targets_any': [ ['have','a','buddy'], ['had','a','buddy'] ] },
    'roseanne': { 'label': 'Roseanne',       'single': ['roseanne'] },
    'hillary':  { 'label': 'Hillary',        'single': ['hillary'] },
    'tmm':      { 'label': 'Too Much Money', 'targets': ['too','much','money'] },
}

FIGHT_WORDS  = { 'fight','fighting','argue','argued','arguing','argument','arguments','blowup','blow-up','disagreement','bicker','bickering' }
SPOUSE_WORDS = { 'wife','spouse','girlfriend','partner' }
CO_WINDOW    = 12

def _match_token_sequence(words: List[dict], targets: List[str], case_sensitive: bool) -> List[Tuple[float,float]]:
    if not words:
        return []
    toks = [w['word'] if case_sensitive else w['word'].lower() for w in words]
    k = len(targets)
    hits: List[Tuple[float,float]] = []
    i = 0
    while i <= len(toks) - k:
        if toks[i:i+k] == targets:
            st = words[i]['start']; en = words[i+k-1]['end']
            hits.append((st,en))
            i += k
        else:
            i += 1
    return hits

def find_hits_for_phrase(segments: List[dict], key: str, case_sensitive: bool) -> List[Tuple[float,float,str,str]]:
    meta = PHRASES[key]
    hits: List[Tuple[float,float,str,str]] = []

    # Special robust matcher for Too Much Money
    tmm_re = re.compile(r"\btoo\s+much\s+money\b", re.IGNORECASE) if key == "tmm" else None

    for seg in segments:
        sp = seg.get('speaker', '?')
        seg_text = seg.get('text', '') or ''

        # Remember how many hits we had before this segment
        before = len(hits)

        if 'words' in seg:
            w = seg['words']
            if 'targets' in meta:
                ts = meta['targets'] if case_sensitive else [t.lower() for t in meta['targets']]
                for (st, en) in _match_token_sequence(w, ts, case_sensitive):
                    hits.append((st, en, seg_text, sp))
            if 'targets_any' in meta:
                for alt in meta['targets_any']:
                    ts = alt if case_sensitive else [t.lower() for t in alt]
                    for (st, en) in _match_token_sequence(w, ts, case_sensitive):
                        hits.append((st, en, seg_text, sp))
            if 'single' in meta:
                singles = set(meta['single'] if case_sensitive else [s.lower() for s in meta['single']])
                toks = [x['word'] if case_sensitive else x['word'].lower() for x in w]
                for i, tok in enumerate(toks):
                    if tok in singles:
                        hits.append((w[i]['start'], w[i]['end'], seg_text, sp))
        else:
            t = seg_text if case_sensitive else seg_text.lower()
            if 'targets' in meta:
                pattern = ' '.join(meta['targets'] if case_sensitive else [t.lower() for t in meta['targets']])
                if pattern in t:
                    hits.append((seg['start'], seg['end'], seg_text, sp))
            if 'targets_any' in meta:
                for alt in meta['targets_any']:
                    pattern = ' '.join(alt if case_sensitive else [x.lower() for x in alt])
                    if pattern in t:
                        hits.append((seg['start'], seg['end'], seg_text, sp))
            if 'single' in meta:
                singles = meta['single'] if case_sensitive else [x.lower() for x in meta['single']]
                for sng in singles:
                    if sng in t:
                        hits.append((seg['start'], seg['end'], seg_text, sp))

        # Fallback for TMM: if token logic found nothing for this segment,
        # but the raw text contains "too much money" (any case), add one hit
        if key == "tmm" and tmm_re is not None and len(hits) == before:
            if tmm_re.search(seg_text):
                start = float(seg.get('start', 0.0))
                end = float(seg.get('end', 0.0))
                hits.append((start, end, seg_text, sp))

    hits.sort(key=lambda x: x[0])
    return hits

def find_hits_topic_wife_fight(segments: List[dict]) -> List[Tuple[float,float,str,str]]:
    hits: List[Tuple[float,float,str,str]] = []
    for seg in segments:
        sp = seg.get('speaker','?')
        if 'words' in seg:
            toks = [w['word'].lower() for w in seg['words']]
            idx_f = [i for i,t in enumerate(toks) if t in FIGHT_WORDS]
            idx_s = [i for i,t in enumerate(toks) if t in SPOUSE_WORDS]
            if idx_f and idx_s:
                for i in idx_f:
                    for j in idx_s:
                        if abs(i-j) <= CO_WINDOW:
                            st = seg['words'][min(i,j)]['start']
                            en = seg['words'][max(i,j)]['end']
                            hits.append((st,en,seg['text'],sp))
                            break
        else:
            t = seg['text'].lower()
            if any(w in t for w in FIGHT_WORDS) and any(w in t for w in SPOUSE_WORDS):
                hits.append((seg['start'], seg['end'], seg['text'], sp))
    hits.sort(key=lambda x: x[0])
    return hits

def find_hits_topic_is_back(segments: List[dict], case_sensitive: bool) -> List[Tuple[float,float,str,str]]:
    """
    Heuristic for 'X is back (in style/trend)' / 'making a comeback'.
    Returns (start, end, text, speaker).
    """
    # helpers / lexicons
    AUX_BE = {"is","are","was","were","be","been","being","'s","'re"}
    BAD_FOLLOWS = {"to","from","on","at","up","down","home","out","off","there","here","again"}
    # direct positive multi-token sequences
    POS_SEQS = [
        ["back","in","style"], ["back","in","fashion"], ["back","in","vogue"],
        ["back","on","trend"],
        ["making","a","comeback"], ["make","a","comeback"], ["made","a","comeback"],
        ["is","making","a","comeback"], ["are","making","a","comeback"],
    ]
    # simple stoplist for subjects we don't want to count as “trend objects”
    BORING_SUBJ = {"i","you","we","they","he","she","it","this","that","there","here"}

    def toks_of_words(words, case_sensitive):
        return [ (w["word"], float(w["start"]), float(w["end"])) for w in words
                 if isinstance(w, dict) and "word" in w and "start" in w and "end" in w ]

    def matches_seq(toks, i, seq):
        if i + len(seq) > len(toks): return False
        for k, tok in enumerate(seq):
            if toks[i+k][0] != tok: return False
        return True

    hits: List[Tuple[float,float,str,str]] = []

    for seg in segments:
        sp = seg.get("speaker","?")
        txt = seg.get("text","")
        # -------- Word-timestamp path
        if "words" in seg and isinstance(seg["words"], list) and seg["words"]:
            toks = toks_of_words(seg["words"], case_sensitive)
            if not case_sensitive:
                toks = [(w.lower(), s, e) for (w,s,e) in toks]

            # 1) Direct positive sequences
            for i in range(len(toks)):
                for seq in POS_SEQS:
                    if matches_seq(toks, i, seq):
                        st = toks[i][1]; en = toks[i+len(seq)-1][2]
                        hits.append((st, en, txt, sp))

            # 2) Generic “is/are … back” (avoid movement senses)
            for i in range(1, len(toks)):  # need a previous token
                w, s_back, e_back = toks[i]
                if w != "back": 
                    continue
                wprev = toks[i-1][0]
                if wprev not in AUX_BE:
                    continue
                # exclude bad follow-ups like “back to …”
                if i+1 < len(toks) and toks[i+1][0] in BAD_FOLLOWS:
                    continue
                # look left a bit for a plausible subject (non-boring, ≥3 chars)
                left_ok = False
                for j in range(max(0, i-5), i-1):
                    cand = toks[j][0]
                    if cand not in AUX_BE and cand not in BORING_SUBJ and len(cand) >= 3:
                        left_ok = True
                        break
                if not left_ok:
                    continue
                st = toks[i-1][1]  # start from the be-verb
                en = e_back
                hits.append((st, en, txt, sp))

        # -------- Fallback: regex over segment text when no words[]
        else:
            hay = txt if case_sensitive else txt.lower()
            # Positive patterns
            POS_PATTERNS = [
                r"\bback in (style|fashion|vogue)\b",
                r"\bback on trend\b",
                r"\b(making|made|make)(?: a)? comeback\b",
                r"\b(?:is|are|was|were|'s|'re)\s+back\b(?!\s+(to|from|on|at|up|down|home|out|off|there|here|again))",
            ]
            if any(re.search(p, hay) for p in POS_PATTERNS):
                # use seg bounds as a coarse proxy
                hits.append((float(seg["start"]), float(seg["end"]), txt, sp))

    hits.sort(key=lambda x: x[0])
    return hits


# --------------------- Smart sentence window ---------------------

def _find_nearest_word_idx(words: List[dict], t: float, side: str = 'left') -> Optional[int]:
    if not words:
        return None
    lo, hi = 0, len(words)-1
    best = 0
    while lo <= hi:
        mid = (lo+hi)//2
        ws = float(words[mid]['start']); we = float(words[mid]['end'])
        if ws <= t <= we:
            return mid
        if we < t:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return min(best, len(words)-1) if side == 'left' else max(best+1, 0)

def expand_to_sentence_window(words: List[dict], hit_start_t: float, hit_end_t: float,
                              pause_sec: float = 0.45, min_len: float = 3.0, max_len: float = 12.0) -> Tuple[float,float]:
    if not words:
        return hit_start_t, hit_end_t
    iL = _find_nearest_word_idx(words, hit_start_t, side='left') or 0
    iR = _find_nearest_word_idx(words, hit_end_t, side='right') or (len(words)-1)
    iL = max(0, iL); iR = min(len(words)-1, iR)
    def gap_left(idx: int) -> float:
        if idx <= 0: return 1e9
        return float(words[idx]['start']) - float(words[idx-1]['end'])
    def gap_right(idx: int) -> float:
        if idx >= len(words)-1: return 1e9
        return float(words[idx+1]['start']) - float(words[idx]['end'])
    L, R = iL, iR
    while L > 0:
        newL = L - 1
        new_start = float(words[newL]['start']); new_end = float(words[R]['end'])
        if (float(words[L]['start']) - float(words[newL]['end'])) >= pause_sec: break
        if (new_end - new_start) > max_len: break
        L = newL
    while R < len(words)-1:
        newR = R + 1
        new_start = float(words[L]['start']); new_end = float(words[newR]['end'])
        if (float(words[newR]['start']) - float(words[R]['end'])) >= pause_sec: break
        if (new_end - new_start) > max_len: break
        R = newR
    start_t = float(words[L]['start']); end_t = float(words[R]['end'])
    cur_len = end_t - start_t
    if cur_len < min_len:
        while cur_len < min_len and R < len(words)-1 and gap_right(R) < pause_sec:
            R += 1; end_t = float(words[R]['end']); cur_len = end_t - start_t
        while cur_len < min_len and L > 0 and gap_left(L) < pause_sec:
            L -= 1; start_t = float(words[L]['start']); cur_len = end_t - start_t
    if (end_t - start_t) > max_len:
        end_t = start_t + max_len
    return start_t, end_t

def find_segment_for_time(segments: List[dict], t: float) -> Optional[dict]:
    for s in segments:
        try:
            if float(s.get('start', 0.0)) <= t <= float(s.get('end', 0.0)):
                return s
        except Exception:
            pass
    return None

def build_context_text(segs: List[dict],
                       hit_start: float,
                       window_before: int = 1,
                       window_after: int = 1) -> str:
    """
    Build a text snippet using the segment containing `hit_start`
    plus `window_before` segments before and `window_after` segments after.

    Includes *all* speakers in that range, concatenated in order.
    Falls back to just the center segment if we can't find neighbours.
    """
    if not segs:
        return ""

    # Find the center segment by time
    center = find_segment_for_time(segs, hit_start)
    if not center:
        # fallback: just find the first segment whose start is after hit_start
        center_idx = None
        for i, s in enumerate(segs):
            if float(s.get("start", 0.0)) >= hit_start:
                center_idx = i
                break
        if center_idx is None:
            center_idx = len(segs) - 1
    else:
        # find its index
        center_idx = None
        for i, s in enumerate(segs):
            if s is center:
                center_idx = i
                break
        if center_idx is None:
            # match by start/end as a backup
            cs = float(center.get("start", 0.0))
            ce = float(center.get("end", 0.0))
            for i, s in enumerate(segs):
                if float(s.get("start", 0.0)) == cs and float(s.get("end", 0.0)) == ce:
                    center_idx = i
                    break
            if center_idx is None:
                center_idx = 0

    start_idx = max(0, center_idx - window_before)
    end_idx = min(len(segs), center_idx + window_after + 1)

    parts: List[str] = []
    for j in range(start_idx, end_idx):
        txt = (segs[j].get("text") or "").strip()
        if txt:
            parts.append(txt)

    return " ".join(parts).strip()

def window_for_hit(segs: List[dict],
                   hit_start: float,
                   hit_end: float,
                   clip_mode: str,
                   pad_before: float,
                   pad_after: float,
                   cap_len: float) -> Tuple[float,float,float,float]:
    """
    Decide final [start,end] and [pad_before,pad_after] for a hit.
    - segment: use the diarized segment (speaker line) boundaries.
    - smart:   use expand_to_sentence_window() if words[] present, else fallback to segment, else hit window.
    - cap_len: if >0 and the selected window is longer, trim to <= cap_len (prefer centering on the phrase).
    Returns (start, end, pad_before, pad_after).
    """
    seg = find_segment_for_time(segs, hit_start)
    # If exact start isn’t inside a segment (rare), try midpoint
    if not seg:
        mid = (hit_start + hit_end) / 2.0
        seg = find_segment_for_time(segs, mid)

    # 1) Choose base window
    if clip_mode == 'segment' and seg:
        start, end = float(seg['start']), float(seg['end'])
        pb, pa = 0.0, 0.0
    elif clip_mode == 'smart' and seg and 'words' in seg:
        start, end = expand_to_sentence_window(seg['words'], hit_start, hit_end,
                                               pause_sec=0.45, min_len=3.0, max_len=12.0)
        pb, pa = 0.0, 0.0
    elif seg:
        # fallback to segment
        start, end = float(seg['start']), float(seg['end'])
        pb, pa = 0.0, 0.0
    else:
        # last resort: use raw hit with legacy pads
        start, end = hit_start, hit_end
        pb, pa = pad_before, pad_after

    # 2) Optionally cap very long segments
    if cap_len and (end - start) > cap_len:
        # Try to center around the phrase midpoint, but stay within segment window
        mid = (hit_start + hit_end) / 2.0
        half = cap_len / 2.0
        new_start = max(start, mid - half)
        new_end = new_start + cap_len
        if new_end > end:
            new_end = end
            new_start = max(start, end - cap_len)
        start, end = new_start, new_end
        # keep pb/pa = 0 for segment/smart windows

    return start, end, pb, pa

def build_context_window(
    segments: List[dict],
    hit_start: float,
    hit_end: float,
    window_before: int,
    window_after: int,
) -> Tuple[Optional[str], float, float]:
    """
    Build a text + time window around a hit using *segment-level* context.

    - segments: list of WhisperX segments (each has start, end, text, speaker, maybe words)
    - hit_start / hit_end: phrase time range
    - window_before / window_after: how many segments before/after to pull in

    Returns (context_text, ctx_start, ctx_end). If something goes wrong,
    returns (None, hit_start, hit_end).
    """
    if not segments:
        return None, hit_start, hit_end

    # Find index of the segment that contains hit_start (or the closest after it)
    idx = None
    for i, s in enumerate(segments):
        try:
            s_start = float(s.get("start", 0.0))
            s_end   = float(s.get("end", 0.0))
        except Exception:
            continue

        if s_start <= hit_start <= s_end:
            idx = i
            break
        if s_start > hit_start and idx is None:
            idx = i
            break

    if idx is None:
        # Fallback: just use the hit itself
        return None, hit_start, hit_end

    start_idx = max(0, idx - window_before)
    end_idx   = min(len(segments) - 1, idx + window_after)

    ctx_segments = segments[start_idx:end_idx + 1]
    # Build combined text
    texts = [s.get("text", "").strip() for s in ctx_segments if s.get("text")]
    ctx_text = " ".join(texts).strip() or None

    ctx_start = float(ctx_segments[0].get("start", hit_start))
    ctx_end   = float(ctx_segments[-1].get("end", hit_end))

    return ctx_text, ctx_start, ctx_end

# --------------------- Speaker mapping ---------------------

def resolve_speaker(ep: str, spk: str, mapping: dict) -> str:
    try:
        if mapping:
            if ep in mapping and spk in mapping[ep]:
                return mapping[ep][spk]
            if "*" in mapping and spk in mapping["*"]:
                return mapping["*"][spk]
    except Exception:
        pass
    return spk

# --------------------- Clip export ---------------------

def export_clip(infile: Path, outpath: Path, start: float, end: float,
                pad_before: float, pad_after: float, fmt: str, bitrate: str,
                mono: bool, loudnorm: bool) -> None:
    ss = max(0.0, start - pad_before)
    dur = max(0.1, (end + pad_after) - ss)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", str(infile)]
    if mono:
        cmd += ["-ac", "1"]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-16:LRA=11:TP=-1.5"]
    if fmt == "mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", bitrate]
    elif fmt == "wav":
        cmd += ["-codec:a", "pcm_s16le"]
    else:
        raise ValueError("Unsupported format: " + fmt)
    cmd += [str(outpath)]
    subprocess.run(cmd, check=True)

# --------------------- HTML ---------------------

MONTH_HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{page_label} — {title}</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; line-height: 1.45; margin: 24px; }}
    header {{ display:flex; align-items:center; gap:16px; margin-bottom:16px; flex-wrap: wrap; }}
    .row {{ display:flex; align-items:center; gap:8px; margin:6px 0; flex-wrap: wrap; }}
    .card {{ border:1px solid #ddd; border-radius:12px; padding:12px 14px; margin:12px 0; }}
    .small {{ color:#777; font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    audio {{ width: 320px; }}
    select {{ padding:6px 8px; border-radius:8px; border:1px solid #ccc; }}
    input[type=text] {{ padding:6px 8px; border-radius:8px; border:1px solid #ccc; min-width: 160px; }}
    button {{ padding:6px 10px; border-radius:8px; border:1px solid #ccc; cursor:pointer; }}
  </style>
</head>
<body>
  <header>
    <h1>{page_label} — <span class=\"mono\">{title}</span></h1>
    <div class=\"row\">
      <label for=\"episodeFilter\">Filter by episode:</label>
      <select id=\"episodeFilter\"><option value=\"\">All</option></select>
    </div>
    <div class=\"row\">
      <button id=\"downloadMap\">Download speaker_map.json</button>
      <button id=\"downloadPayload\">Download updated payload.json</button>
    </div>
    <div class=\"small\">Portable page with local clips. Type speaker names to relabel per clip; export mapping to reuse next time.</div>
  </header>
  <div id=\"hits\"></div>
<script id=\"payload\" type=\"application/json\">{payload_json}</script>
<script>
let data = [];
try {{
  data = JSON.parse(document.getElementById('payload').textContent);
  console.log('[multi-finder] loaded items:', Array.isArray(data) ? data.length : 'n/a');
}} catch (e) {{
  const pre = document.createElement('pre');
  pre.textContent = 'JSON parse error: ' + e.message;
  pre.style.color = 'crimson';
  document.body.insertBefore(pre, document.body.firstChild);
}}
function buildUI() {{
  const container = document.getElementById('hits');
  const select = document.getElementById('episodeFilter');
  const epNames = {{}};
  data.forEach(d => {{ epNames[d.episode] = d.episode_pretty || d.episode; }});
  const episodes = Object.keys(epNames).sort();
  episodes.forEach(ep => {{ const opt = document.createElement('option'); opt.value = ep; opt.textContent = epNames[ep]; select.appendChild(opt); }});

  function render(filterEp) {{
    container.innerHTML = '';
    const items = filterEp ? data.filter(d => d.episode === filterEp) : data;
    if (!items.length) {{ const d = document.createElement('div'); d.className = 'small'; d.textContent = 'No matches.'; container.appendChild(d); return; }}
    items.forEach((item, idx) => {{
      const card = document.createElement('div'); card.className = 'card';
      const title = document.createElement('div'); title.className = 'row';
      const left = document.createElement('div');
      const strongEl = document.createElement('strong'); strongEl.textContent = '#' + (idx + 1);
      const smallEl = document.createElement('span'); smallEl.className = 'small';
      smallEl.textContent = '(episode: ' + (item.episode_pretty || item.episode) + ', speaker: ' + item.speaker + ')';
      left.appendChild(strongEl); left.appendChild(document.createTextNode(' ')); left.appendChild(smallEl);
      const right = document.createElement('div'); right.className = 'small mono'; right.textContent = '@ ' + item.start.toFixed(2) + 's – ' + item.end.toFixed(2) + 's';
      title.appendChild(left); title.appendChild(right);

      const row = document.createElement('div'); row.className = 'row';
      const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = item.clip;
      const text = document.createElement('span');
      text.textContent = '  ' + (item.context || item.text);
      row.appendChild(audio);
      row.appendChild(text);
      const inp = document.createElement('input');
      inp.type = 'text'; inp.placeholder = 'Name'; inp.value = item.speaker; inp.style.marginLeft = '8px';
      function applyRenameClipOnly() {{
        const name = inp.value.trim() || item.speaker_id || item.speaker;
        const targetIdx = data.indexOf(item);
        if (targetIdx >= 0) data[targetIdx].speaker = name;
        render(select.value || '');
      }}
      inp.addEventListener('change', applyRenameClipOnly);
      inp.addEventListener('blur', applyRenameClipOnly);

      row.appendChild(audio); row.appendChild(text); row.appendChild(inp);
      card.appendChild(title); card.appendChild(row); container.appendChild(card);
    }});
  }}

  select.onchange = () => render(select.value || '');
  render('');
}}
// Export buttons
const dlMap = document.getElementById('downloadMap');
const dlPayload = document.getElementById('downloadPayload');

dlMap.onclick = function() {{
  // Build {{ episode: {{ SPEAKER_XX: name }} }} only when uniform across clips
  const groups = new Map(); // key: ep+'||'+id -> array of names
  data.forEach(d => {{
    const ep = d.episode, id = d.speaker_id || d.speaker;
    if (!id) return;
    const key = ep + '||' + id;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(d.speaker);
  }});
  const mapping = {{}};
  groups.forEach((names, key) => {{
    const [ep, id] = key.split('||');
    const first = names[0];
    const uniform = names.every(n => n === first);
    const changed = uniform && first && first != id;
    if (changed) {{ (mapping[ep] ||= {{}})[id] = first; }}
  }});
  const blob = new Blob([JSON.stringify(mapping, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'speaker_map.json'; a.click();
}};

dlPayload.onclick = function() {{
  const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'payload_updated.json'; a.click();
}};

buildUI();
</script>
</body>
</html>
"""

# --------------------- Driver ---------------------

def process_one_month(
    root: Path,
    audio_dir: Path,
    month: str,
    pad_before: float,
    pad_after: float,
    fmt: str,
    bitrate: str,
    mono: bool,
    loudnorm: bool,
    max_hits: int,
    case_sensitive: bool,
    speaker_map: dict,
    clip_mode: str,
    cap_len: float,
    ctx_before: int,
    ctx_after: int,
    pad_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:

    pad_overrides = pad_overrides or {}

    months = group_episodes_by_month(root)
    if month not in months:
        print(f"Month not found under root: {month}", file=sys.stderr); return
    ep_map = months[month]

    categories = [
        ('tp',        'The Point Is',
            lambda segs: find_hits_for_phrase(segs, 'tp', False),        '_tp_'),

        ('hab',       'Have A Buddy',
            lambda segs: find_hits_for_phrase(segs, 'hab', False),       '_hab_'),

        ('roseanne',  'Roseanne',
            lambda segs: find_hits_for_phrase(segs, 'roseanne', False),  '_roseanne_'),

        ('hillary',   'Hillary',
            lambda segs: find_hits_for_phrase(segs, 'hillary', False),   '_hillary_'),

        ('wife_fight','Fight With Wife',
            lambda segs: find_hits_topic_wife_fight(segs),               '_wife_'),

        ('tmm',       'Too Much Money',
            lambda segs: find_hits_for_phrase(segs, 'tmm', False),       '_tmm_'),
    ]

    for key, label, finder, suf in categories:
        # Effective pads for this category
        cat_pb, cat_pa = pad_before, pad_after
        if key in pad_overrides and pad_overrides[key]:
            try:
                cat_pb, cat_pa = pad_overrides[key]
            except Exception:
                pass

        # -----------------------------------------------------------

        share = root / month / f"{month}_{key}_portable"
        clips = share / "clips"
        clips.mkdir(parents=True, exist_ok=True)
        payload: List[dict] = []

        for ep, ep_folder in ep_map.items():
            segs = load_segments(ep_folder, ep)
            audio = locate_audio(ep, audio_dir, month)
            if not audio:
                print(f"WARNING: missing audio for {ep} in {audio_dir} (also checked {audio_dir/month})")
                continue
            hits = finder(segs)
            if max_hits and len(hits) > max_hits:
                hits = hits[:max_hits]
            for i, (st, en, tx, sp) in enumerate(hits, start=1):
                # 1) Build context window: text + combined time span across segments
                ctx_text, ctx_start, ctx_end = build_context_window(
                    segs,
                    hit_start=st,
                    hit_end=en,
                    window_before=ctx_before,   # e.g. 1–2 segments before
                    window_after=ctx_after,     # e.g. 1–2 segments after
                )

                # Fallback: if context fails, just use the hit span
                if ctx_text is None:
                    ctx_text = tx
                    ctx_start, ctx_end = st, en

                # 2) Use the *context* span directly for the clip;
                # let export_clip handle padding.
                s2 = ctx_start
                e2 = ctx_end
                pb = pad_before
                pa = pad_after

                # 3) Export clip
                clip_name = safe_name(f"{ep}{suf}{i:03d}.{fmt}")
                outpath = clips / clip_name
                export_clip(audio, outpath, s2, e2, pb, pa, fmt, bitrate, mono, loudnorm)

                # 4) Add to payload
                resolved = resolve_speaker(ep, sp, speaker_map)
                payload.append({
                    "episode": ep,
                    "episode_pretty": pretty_episode_label(ep),
                    "start": s2,
                    "end": e2,
                    "text": tx,                  # the hit line
                    "context": ctx_text or tx,   # full context text
                    "speaker_id": sp,
                    "speaker": resolved,
                    "clip": f"clips/{clip_name}",
                })

        payload.sort(key=lambda d: (d.get('episode',''), d['start']))
        html = MONTH_HTML.format(
            page_label=label,
            title=pretty_month_title(month),
            payload_json=json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
        (share / "index.html").write_text(html, encoding="utf-8")
        print(f"Wrote: {share}/index.html  (clips: {len(payload)})")

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description='Multi finder monthly clip packager (portable HTML + clips).')
    ap.add_argument('--clip_mode', choices=['segment','smart'], default='segment',
                help="How to choose clip range: 'segment' = whole diarized line (recommended), 'smart' = pause-based expansion using word gaps.")
    ap.add_argument('--cap_len', type=float, default=0.0, help="If > 0, cap overly long segment clips to this many seconds (centered on the phrase if possible).")
    ap.add_argument('--root', required=True, type=Path, help='Transcripts root with month/episode folders')
    ap.add_argument('--audio_dir', required=True, type=Path, help='Directory containing original audio files (may include month subfolders)')
    ap.add_argument('--month', default=None, help='Specific month (YY-MM) for aggregate pages')
    ap.add_argument('--all_months', action='store_true', help='Build aggregate pages for all months under --root')
    ap.add_argument('--pad_before', type=float, default=0.4, help='Seconds of context before phrase (fallback when no words)')
    ap.add_argument('--pad_after', type=float, default=0.6, help='Seconds of context after phrase (fallback when no words)')
    ap.add_argument('--max_hits', type=int, default=0, help='Limit number of matches per episode (0 = no limit)')
    ap.add_argument('--format', choices=['mp3','wav'], default='mp3', help='Audio format for clips')
    ap.add_argument('--bitrate', default='96k', help='Bitrate for mp3')
    ap.add_argument('--mono', action='store_true', help='Downmix to mono')
    ap.add_argument('--loudnorm', action='store_true', help='Apply loudness normalization')
    ap.add_argument('--zip', action='store_true', help='Zip each resulting share folder')
    ap.add_argument('--speaker_map', type=Path, default=None, help='Optional JSON mapping of {"YY-MM-DD": {"SPEAKER_07": "Alice"}, "*": {...}}')
    ap.add_argument('--pad_tp', type=str, default=None, help="Override pads for The Point Is as 'before,after' (e.g. 0.3,0.7)")
    ap.add_argument('--pad_hab', type=str, default=None, help="Override pads for Have A Buddy")
    ap.add_argument('--pad_roseanne', type=str, default=None, help="Override pads for Roseanne")
    ap.add_argument('--pad_hillary', type=str, default=None, help="Override pads for Hillary")
    ap.add_argument('--pad_wife', type=str, default=None, help="Override pads for Fight With Wife")
    ap.add_argument('--ctx_before', type=int, default=1, help='How many segments before the hit to include in context text')
    ap.add_argument('--ctx_after', type=int, default=1, help='How many segments after the hit to include in context text')
    ap.add_argument('--case_sensitive', action='store_true', help='Match case-sensitively (default: case-insensitive)'
)

    args = ap.parse_args()

    pad_overrides = {
    "tp":        parse_pad_pair(args.pad_tp),
    "hab":       parse_pad_pair(args.pad_hab),
    "roseanne":  parse_pad_pair(args.pad_roseanne),
    "hillary":   parse_pad_pair(args.pad_hillary),
    "wife_fight":parse_pad_pair(args.pad_wife),
}
    if not args.root.exists():
        print(f"--root not found: {args.root}", file=sys.stderr); sys.exit(2)
    if not args.audio_dir.exists():
        print(f"--audio_dir not found: {args.audio_dir}", file=sys.stderr); sys.exit(2)

    which_or_die("ffmpeg", "Install via Homebrew: brew install ffmpeg")

    months = group_episodes_by_month(args.root)

    speaker_map = {}
    if args.speaker_map and args.speaker_map.exists():
        try:
            speaker_map = json.loads(args.speaker_map.read_text(encoding='utf-8'))
            print(f"Loaded speaker_map from {args.speaker_map}")
        except Exception as e:
            print(f"WARNING: failed to read speaker_map: {e}")

    targets = [args.month] if args.month else (sorted(months.keys()) if args.all_months else None)
    if not targets:
        print("Specify --month YY-MM or --all_months", file=sys.stderr); sys.exit(2)

    for m in targets:
        process_one_month(
            args.root,
            args.audio_dir,
            m,
            args.pad_before,
            args.pad_after,
            args.format,
            args.bitrate,
            args.mono,
            args.loudnorm,
            args.max_hits,
            args.case_sensitive,
            speaker_map,
            args.clip_mode,
            args.cap_len,
            args.ctx_before,
            args.ctx_after,
            pad_overrides=pad_overrides,
        )

        if args.zip:
            for key in ["tp","hab","roseanne","hillary","wife_fight"]:
                share = args.root / m / f"{m}_{key}_portable"
                if share.exists():
                    z = shutil.make_archive(str(share), 'zip', root_dir=share)
                    print(f"Zipped: {z}")

if __name__ == '__main__':
    main()