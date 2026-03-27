#!/usr/bin/env python3
"""
Create an interactive HTML page to *audition* each diarized speaker for an episode:
- Plays short audio snippets from the original file at the right timestamps 
- Shows transcript snippets for context
- Lets you type a name for each SPEAKER_XX
- One-click to download a mapping JSON, or copy a --map string for speaker_labeler.py

Works with the per-episode layout produced by whisperx_batch.py:
  transcripts_x/
    Episode_001/
      Episode_001.json  (WhisperX segments incl. "speaker", "start", "end", "text")
      Episode_001.srt / .vtt / .txt

Usage examples
  python speaker_auditioner.py --root "$HOME/transcripts_x" --audio_dir "/path/to/podcasts" --episode 25-09-15
  python speaker_auditioner.py --root "$HOME/transcripts_x" --audio_dir "/path/to/podcasts" --all

This writes <episode>/<stem>_audition.html which you can open in a browser.

Notes
- The HTML references the original audio file in-place (no copy). Ensure your browser can access that path.
- If your audio is stored elsewhere, point --audio_dir at the folder that contains <stem>.<ext>.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

SUPPORTED_AUDIO_EXTS = [".mp3", ".m4a", ".wav", ".mp4", ".aac", ".flac", ".mkv", ".mov"]


def find_episode_folders(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        stem = sub.name
        if (sub / f"{stem}.json").exists():
            out[stem] = sub
        else:
            # fallback: try to find a single plausible JSON that has segments
            jsons = list(sub.glob("*.json"))
            for js in jsons:
                try:
                    data = json.loads(js.read_text(encoding="utf-8"))
                    if isinstance(data.get("segments"), list):
                        out[js.stem] = sub
                        break
                except Exception:
                    pass
    return out

def locate_audio(stem: str, audio_dir: Optional[Path]) -> Optional[Path]:
    if not audio_dir:
        return None
    for ext in SUPPORTED_AUDIO_EXTS:
        p = audio_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_segments(ep_folder: Path, stem: str) -> List[dict]:
    js = ep_folder / f"{stem}.json"
    if not js.exists():
        print(f"ERROR: Missing {js} (need WhisperX JSON with segments)", file=sys.stderr)
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
        sp = s.get("speaker"); start = s.get("start"); end = s.get("end"); text = s.get("text")
        if isinstance(sp, str) and isinstance(start, (int, float)) and isinstance(end, (int, float)) and isinstance(text, str):
            out.append({"speaker": sp, "start": float(start), "end": float(end), "text": text.strip()})
    return out


def pick_samples(
    segments: List[dict],
    max_per_speaker: int = 5,
    min_chars: int = 12,
    min_seconds: float = 3.0,
    max_seconds: float = 5.0,
) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = {}
    for s in segments:
        text = (s.get("text") or "").strip()
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        dur = max(0.0, end - start)

        # keep only 3–5s (or whatever you pass)
        if dur < min_seconds or dur > max_seconds:
            continue
        if len(text) < min_chars:
            continue

        sp = s.get("speaker")
        if isinstance(sp, str):
            buckets.setdefault(sp, []).append(
                {"speaker": sp, "start": start, "end": end, "text": text}
            )

    out: Dict[str, List[dict]] = {}
    for sp, segs in buckets.items():
        segs.sort(key=lambda x: x["start"])
        if len(segs) <= max_per_speaker:
            out[sp] = segs
        else:
            step = max(1, len(segs) // max_per_speaker)
            out[sp] = [segs[i] for i in range(0, len(segs), step)][:max_per_speaker]
    return out

def seconds_to_mmss(s: float) -> str:
    m = int(s // 60)
    ss = int(round(s - m * 60))
    return f"{m:02d}:{ss:02d}"


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Speaker Auditioner: {stem}</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; line-height: 1.45; margin: 24px; }}
    header {{ display:flex; align-items:center; gap:16px; margin-bottom:16px; }}
    .spk {{ border:1px solid #ddd; border-radius:12px; padding:12px 14px; margin:12px 0; }}
    .spk h3 {{ margin: 0 0 8px 0; }}
    .row {{ display:flex; align-items:center; gap:8px; margin:6px 0; }}
    button {{ padding:6px 10px; border-radius:8px; border:1px solid #ccc; cursor:pointer; }}
    input[type=text] {{ padding:6px 8px; border-radius:8px; border:1px solid #ccc; min-width: 220px; }}
    .samples {{ margin-left: 6px; }}
    .small {{ color:#777; font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .footer {{ margin-top: 24px; display:flex; gap:12px; align-items:center; flex-wrap: wrap; }}
    .codebox {{ background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:8px 10px; }}
  </style>
</head>
<body>
  <header>
    <h1>Speaker Auditioner — <span class="mono">{stem}</span></h1>
  </header>
  <div class="small">Audio file: <span class="mono">{audio_path}</span></div>
  <audio id="player" controls preload="auto" src="{audio_src}"></audio>

  <div id="speakers"></div>

  <div class="footer">
    <button id="downloadJson">Download mapping.json</button>
    <button id="copyMap">Copy --map string</button>
    <span class="small">Use with <span class="mono">speaker_labeler.py</span></span>
  </div>

<script>
const samples = {samples_json};
const player = document.getElementById('player');

let stopGuard = null; // holds the current timeupdate handler so we can remove it

function playSnippet(start, end) {{
  if (!player.src) return;

  // clear any previous guard
  if (stopGuard) {{
    player.removeEventListener('timeupdate', stopGuard);
    stopGuard = null;
  }}

  // small epsilon so we don't miss the boundary
  const target = Math.max(0, end - 0.05);

  // seek then play
  try {{ player.currentTime = Math.max(0, start); }} catch (e) {{}}
  player.play().catch(() => {{ /* user gesture might be required */ }});

  // pause precisely when we cross 'end'
  stopGuard = function onTimeUpdate() {{
    if (player.currentTime >= target) {{
      player.pause();
      player.removeEventListener('timeupdate', onTimeUpdate);
      stopGuard = null;
    }}
  }};
  player.addEventListener('timeupdate', stopGuard);
}}

function buildUI() {{
  const container = document.getElementById('speakers');
  Object.keys(samples).sort().forEach(function(spk) {{
    const card = document.createElement('div');
    card.className = 'spk';
    const title = document.createElement('h3');
    title.textContent = spk;
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('label');
    label.textContent = 'Name:';
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Name for ' + spk;
    input.id = 'name_' + spk;
    row.appendChild(label);
    row.appendChild(input);

    const samplesDiv = document.createElement('div');
    samplesDiv.className = 'samples';

    (samples[spk] || []).forEach(function(seg, idx) {{
      const row2 = document.createElement('div');
      row2.className = 'row';
      const btn = document.createElement('button');
      btn.textContent = 'Play ' + (idx+1) + ' (' + (seg.end - seg.start).toFixed(1) + 's @ ' + seg.start.toFixed(1) + 's)';
      btn.onclick = function() {{ playSnippet(seg.start, seg.end); }};
      const text = document.createElement('div');
      text.textContent = seg.text;
      text.style.flex = '1';
      row2.appendChild(btn);
      row2.appendChild(text);
      samplesDiv.appendChild(row2);
    }});

    card.appendChild(title);
    card.appendChild(row);
    card.appendChild(samplesDiv);
    container.appendChild(card);
  }});
}}

function gatherMapping() {{
  const out = {{}};
  Object.keys(samples).forEach(function(spk) {{
    const inp = document.getElementById('name_' + spk);
    if (inp && inp.value.trim()) out[spk] = inp.value.trim();
  }});
  return out;
}}

document.getElementById('downloadJson').onclick = function() {{
  const mapping = gatherMapping();
  const blob = new Blob([JSON.stringify(mapping, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '{stem}_mapping.json';
  a.click();
}};

document.getElementById('copyMap').onclick = async function() {{
  const mapping = gatherMapping();
  const parts = Object.keys(mapping).sort().map(function(k) {{ return k + '=' + mapping[k]; }});
  const s = parts.join(',');
  try {{ await navigator.clipboard.writeText(s); alert('Copied: ' + s); }}
  catch (e) {{ alert(s); }}
}};

buildUI();
</script>
</body>
</html>
"""

def generate_html(ep_folder: Path, stem: str, audio_path: Path, samples: Dict[str, List[dict]]) -> Path:
    audio_src = audio_path.resolve().as_uri()
    html = HTML_TEMPLATE.format(
        stem=stem,
        audio_path=str(audio_path),
        audio_src=audio_src,
        samples_json=json.dumps(samples, ensure_ascii=False),
    )
    out = ep_folder / f"{stem}_audition.html"
    out.write_text(html, encoding='utf-8')
    return out


def main():
    ap = argparse.ArgumentParser(description='Create an interactive per-episode speaker audition HTML.')
    ap.add_argument('--root', required=True, type=Path, help='Transcripts root containing per-episode folders')
    ap.add_argument('--audio_dir', required=True, type=Path, help='Directory containing original audio files')
    ap.add_argument('--episode', default=None, help='Specific episode stem (folder name/stem).')
    ap.add_argument('--all', action='store_true', help='Generate audition pages for all episodes in --root')
    ap.add_argument('--samples_per_speaker', type=int, default=5, help='Max samples per speaker (default 5)')
    ap.add_argument('--min_chars', type=int, default=12, help='Minimum transcript length per sample (default 12)')
    ap.add_argument('--min_seconds', type=float, default=3.0, help='Minimum segment duration to audition')
    ap.add_argument('--max_seconds', type=float, default=5.0, help='Maximum segment duration to audition')

    args = ap.parse_args()

    if not args.root.exists():
        print(f"--root not found: {args.root}", file=sys.stderr)
        sys.exit(2)
    if not args.audio_dir.exists():
        print(f"--audio_dir not found: {args.audio_dir}", file=sys.stderr)
        sys.exit(2)

    episodes = find_episode_folders(args.root)
    targets: List[str] = []
    if args.episode:
        if args.episode in episodes:
            targets = [args.episode]
        else:
            print(f"Episode not found in root: {args.episode}", file=sys.stderr)
            sys.exit(2)
    elif args.all:
        targets = list(episodes.keys())
    else:
        print("Provide --episode <stem> or --all", file=sys.stderr)
        sys.exit(2)

    for stem in targets:
        ep_folder = episodes[stem]
        segs = load_segments(ep_folder, stem)
        if not segs:
            continue
        samples = pick_samples(
            segs,
            max_per_speaker=args.samples_per_speaker,
            min_chars=args.min_chars,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
        )
        if not samples:
            print(f"NOTE: No samples selected for {stem} (try adjusting --min_chars/--min_seconds/--max_seconds)")
            continue

        audio = locate_audio(stem, args.audio_dir)
        if not audio:
            print(f"WARNING: Could not locate audio for {stem} in {args.audio_dir}")
            continue

        out = generate_html(ep_folder, stem, audio, samples)
        print(f"Wrote: {out}")


if __name__ == '__main__':
    main()