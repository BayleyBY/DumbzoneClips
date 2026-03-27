#!/usr/bin/env python3
"""
Read-only packager for ALL phrases from saved payloads.

- Scans a glob pattern (default looks for **/payload_updated.json)
- For each payload, emits a clean, read-only index.html next to (or under) an output root
- Leaves audio clips in their existing relative paths (e.g., clips/*.mp3)
- Optional zip per page

Usage example (BagEnd paths):
  python readonly_pack_all.py \
    --payload_glob "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/**/payload_updated.json" \
    --outdir_root "/Volumes/BagEnd/Projects/WhisperAI/dumbzone/share_readonly" \
    --zip

Assumptions:
- Each payload is an array of items like:
  { "episode":"25-10-02", "speaker":"Jake", "start":..., "end":..., "text":"...", "clip":"clips/25-10-02_tp_001.mp3" }
- The clip paths are valid relative to the payload’s directory.
"""

from __future__ import annotations
import argparse, json, zipfile, io, re, calendar
from pathlib import Path
from typing import List, Dict

# ---- Helpers ----

PHRASE_SYNONYMS = [
    # Shortcodes in folder/file names
    (re.compile(r'(?:^|[/_])tp(?:[_/]|$)', re.I),            "The Point Is",      "tp"),
    (re.compile(r'(?:^|[/_])hab(?:[_/]|$)', re.I),           "Have A Buddy",      "hab"),
    (re.compile(r'(?:^|[/_])tmm(?:[_/]|$)', re.I),           "Too Much Money",    "tmm"),

    # Plain-English fallbacks (when shortcodes aren’t present)
    (re.compile(r"\bthe\s+point\s+is\b", re.I),              "The Point Is",      "tp"),
    (re.compile(r"\bhave(?:\s+|/|_)a\b|\bhad(?:\s+|/|_)a\b", re.I), "Have A Buddy", "hab"),
    (re.compile(r"\btoo\s+much\s+money\b", re.I),            "Too Much Money",    "tmm"),
    (re.compile(r"\broseanne\b", re.I),                      "Roseanne",          "roseanne"),
    (re.compile(r"\bhillary\b", re.I),                       "Hillary",           "hillary"),
    (re.compile(r"\bwife.*fight|fight.*wife\b", re.I),       "Fight With Wife",   "wife_fight"),
]

def is_default_speaker(item: dict) -> bool:
    spk = (item.get("speaker") or "").strip()
    sid = (item.get("speaker_id") or "").strip()
    if not spk:
        return True
    # Looks like an untouched default tag (SPEAKER_XX / SPEAKER 03 etc.)
    up = spk.upper()
    if re.match(r"^SPEAKER[\s_:-]*\d{1,3}$", up):
        return True
    # If speaker text equals the original ID (not renamed)
    if sid and spk == sid:
        return True
    return False

LABEL_BY_KEY = {
    "tp": "The Point Is",
    "hab": "Have A Buddy",
    "roseanne": "Roseanne",
    "hillary": "Hillary",
    "wife": "Fight With Wife",
    "wife_fight": "Fight With Wife",
}

def pretty_month_title(yy_mm: str) -> str:
    try:
        yy, mm = yy_mm.split("-")
        year = int(yy)
        if 0 <= year <= 99:
            year += 2000
        month_name = calendar.month_name[int(mm)]
        return f"{month_name} {year}"
    except Exception:
        return yy_mm

def infer_month_from_path(p: Path) -> str | None:
    """
    Walk parents looking for a folder that looks like YY-MM (e.g., 25-10).
    """
    for ancestor in [p] + list(p.parents):
        m = re.search(r"(\d{2}-\d{2})", ancestor.name)
        if m:
            return m.group(1)
    return None

def resolve_speaker(ep: str, spk: str, mapping: dict) -> str:
    """Return mapped speaker name if provided, else original tag."""
    try:
        if mapping:
            if ep in mapping and spk in mapping[ep]:
                return mapping[ep][spk]
            if "*" in mapping and spk in mapping["*"]:
                return mapping["*"][spk]
    except Exception:
        pass
    return spk

def infer_phrase_from_path(p: Path) -> str:
    """
    Guess phrase label from path parts using folder keys (e.g., _tp_portable)
    or, failing that, PHRASE_SYNONYMS. Falls back to parent folder name.
    """
    hay = "/".join(x.lower() for x in p.parts)

    # Prefer explicit monthly share folder keys like 25-11_tp_portable/
    m = re.search(r"_(tp|hab|roseanne|hillary|wife(?:_fight)?)_portable", hay)
    if m:
        key = m.group(1)
        key = "wife_fight" if key == "wife" else key
        if key in LABEL_BY_KEY:
            return LABEL_BY_KEY[key]

    # Otherwise, try textual synonyms (e.g., when folders are “The Point Is - …”)
    for rx, label, _key in PHRASE_SYNONYMS:
        if rx.search(hay):
            return label

    # Fallback: parent folder name (last resort)
    return p.parent.name

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def make_zip(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(src_dir))

# ---- HTML template (read-only) ----

MONTH_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; line-height: 1.45; margin: 24px; }}
    header {{ display:flex; align-items:center; gap:16px; margin-bottom:16px; flex-wrap: wrap; }}
    .row {{ display:flex; align-items:center; gap:8px; margin:6px 0; flex-wrap: wrap; }}
    .card {{ border:1px solid #ddd; border-radius:12px; padding:12px 14px; margin:12px 0; }}
    .small {{ color:#777; font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    audio {{ width: 320px; }}
    select {{ padding:6px 8px; border-radius:8px; border:1px solid #ccc; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="row">
      <label for="episodeFilter">Filter by episode:</label>
      <select id="episodeFilter"><option value="">All</option></select>
    </div>
    <div class="small">Portable read-only page. Share this folder as-is (keep <span class="mono">clips/</span> with it).</div>
  </header>
  <div id="hits"></div>

<script id="payload" type="application/json">{payload_json}</script>
<script>
let data = [];
try {{
  data = JSON.parse(document.getElementById('payload').textContent);
  if (!Array.isArray(data)) data = [];
}} catch (e) {{
  const pre = document.createElement('pre');
  pre.textContent = 'JSON parse error: ' + e.message;
  pre.style.color = 'crimson';
  document.body.insertBefore(pre, document.body.firstChild);
}}

function render() {{
  const container = document.getElementById('hits');
  const select = document.getElementById('episodeFilter');
  if (!select.dataset.init) {{
    const episodes = Array.from(new Set(data.map(d => d.episode))).sort();
    episodes.forEach(ep => {{ const opt = document.createElement('option'); opt.value = ep; opt.textContent = ep; select.appendChild(opt); }});
    select.dataset.init = "1";
    select.onchange = () => render();
  }}
  container.innerHTML = '';
  const filterEp = select.value || '';
  const items = filterEp ? data.filter(d => d.episode === filterEp) : data;
  if (!items.length) {{
    const d = document.createElement('div'); d.className = 'small'; d.textContent = 'No matches.'; container.appendChild(d); return;
  }}
  items.forEach((item, idx) => {{
    const card = document.createElement('div'); card.className = 'card';
    const title = document.createElement('div'); title.className = 'row';
    const left = document.createElement('div');
    left.innerHTML = '<strong>#' + (idx+1) + '</strong> ' +
                     '<span class="small">(episode: ' + (item.episode || '?') +
                     ', speaker: ' + (item.speaker || item.speaker_id || '?') + ')</span>';
    const right = document.createElement('div'); right.className = 'small mono';
    const s = Number(item.start || 0).toFixed(2), e = Number(item.end || 0).toFixed(2);
    right.textContent = '@ ' + s + 's – ' + e + 's';
    title.appendChild(left); title.appendChild(right);

    const row = document.createElement('div'); row.className = 'row';
    const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata';
    audio.src = item.clip;
    const text = document.createElement('span');
    // Prefer extended context if present; fall back to the hit line
    const display = (item.context && item.context.trim())
      ? item.context
      : (item.context_text && item.context_text.trim()) // optional legacy key
        ? item.context_text
        : (item.text || '');
    text.textContent = '  ' + display;
    row.appendChild(audio); row.appendChild(text);

    card.appendChild(title); card.appendChild(row); container.appendChild(card);
  }});
}}
render();
</script>
</body>
</html>
"""

def build_readonly_page(payload_path: Path, out_dir: Path, title: str, speaker_map: dict | None = None) -> Path:
    ensure_dir(out_dir)

    # Load payload
    raw = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Payload is not a list: {payload_path}")

    speaker_map = speaker_map or {}

    filtered = []
    for item in raw:
        ep = str(item.get("episode", ""))
        # prefer stable id when present; fall back to current 'speaker'
        sp_id = str(item.get("speaker_id") or item.get("speaker") or "").strip()
        # current display name in payload
        cur_name = str(item.get("speaker") or "").strip()

        # 1) If payload already has a hard "KILL", drop it immediately
        if cur_name.upper() == "KILL":
            continue

        # 2) Apply mapping (if any) based on speaker_id (or fallback to the shown speaker)
        resolved = resolve_speaker(ep, sp_id, speaker_map) if sp_id else cur_name or "?"

        # 3) If the resolved name is KILL, omit this clip
        if isinstance(resolved, str) and resolved.upper() == "KILL":
            continue

        # 4) Keep the clip; write back the resolved name (if we actually changed it)
        if resolved and resolved != cur_name:
            item["speaker"] = resolved

        filtered.append(item)

    # Escape </script>
    payload_s = json.dumps(filtered, ensure_ascii=False).replace("</", "<\\/")
    html = MONTH_HTML.format(title=title, payload_json=payload_s)
    out_html = out_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")
    return out_html

# ---- Main ----

def main():
    ap = argparse.ArgumentParser(description="Build read-only pages from all payloads.")
    ap.add_argument("--payload_glob", required=True, help="Glob to find payload JSONs (e.g., '/.../**/payload_updated.json')")
    ap.add_argument("--outdir_root", required=True, type=Path, help="Root folder where read-only pages will be written")
    ap.add_argument("--zip", action="store_true", help="Zip each page folder after writing")
    ap.add_argument("--title_prefix", default="", help="Optional prefix for page title")
    ap.add_argument("--copy_clips", action="store_true", help="Copy the payload's sibling clips/ folder into each output page folder.")
    ap.add_argument("--index_root", type=Path, default=None, help="If set, write a root index.html that links to all built pages under this folder.")
    ap.add_argument("--speaker_map",type=Path, default=None, help=('Optional JSON mapping like ' '{"YY-MM-DD": {"SPEAKER_07": "Alice"}, "*": {...}}. ' 'If the resolved name is "KILL", the snippet is removed.'),)

    args = ap.parse_args()

    speaker_map = {}
    if args.speaker_map and args.speaker_map.exists():
        try:
            speaker_map = json.loads(args.speaker_map.read_text(encoding="utf-8"))
            print(f"Loaded speaker_map from {args.speaker_map}")
        except Exception as e:
            print(f"WARNING: failed to read speaker_map: {e}")

    payloads = [Path(p) for p in sorted(Path("/").glob(args.payload_glob.lstrip("/")))]
    if not payloads:
        print(f"No payloads matched: {args.payload_glob}")
        return

    built: List[Path] = []
    for payload in payloads:
        # Infer phrase and month for title + destination
        phrase = infer_phrase_from_path(payload)
        yy_mm = infer_month_from_path(payload) or "unknown"
        pretty_month = pretty_month_title(yy_mm)
        title = f"{args.title_prefix}{phrase} — {pretty_month}".strip()

        # Destination folder:
        # <outdir_root>/<YY-MM>/<safe-phrase>/ (index.html here)
        safe_phrase = re.sub(r"[^A-Za-z0-9._ -]+", "_", phrase).strip().replace("  ", " ")
        out_dir = args.outdir_root / yy_mm / f"{safe_phrase} - {pretty_month}"
        html_path = build_readonly_page(payload, out_dir, title, speaker_map)
        # Optionally copy clips/ folder next to the payload
        if args.copy_clips:
            src_clips = payload.parent / "clips"
            dst_clips = out_dir / "clips"
            if src_clips.exists() and src_clips.is_dir():
                # Safe copy (overwrites files if present)
                if not dst_clips.exists():
                    dst_clips.mkdir(parents=True, exist_ok=True)
                for p in src_clips.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(src_clips)
                        (dst_clips / rel).parent.mkdir(parents=True, exist_ok=True)
                        (dst_clips / rel).write_bytes(p.read_bytes())

        print(f"Wrote: {html_path}")

        # Zip (optional)
        if args.zip:
            zip_path = out_dir.with_suffix(".zip")
            make_zip(out_dir, zip_path)
            print(f"Zipped: {zip_path}")

        built.append(html_path)

    if args.index_root:
        args.index_root.mkdir(parents=True, exist_ok=True)

        # Build relative links from index_root to each page
        items = []
        for html_path in built:
            # infer month and phrase again for labeling
            phrase = infer_phrase_from_path(html_path)
            yy_mm = infer_month_from_path(html_path) or "unknown"
            title = f"{phrase} — {pretty_month_title(yy_mm)}"

            rel = html_path.parent.relative_to(args.index_root) if html_path.is_relative_to(args.index_root) \
                  else os.path.relpath(html_path.parent, args.index_root)
            items.append((yy_mm, title, f"{rel}/"))

        items.sort(key=lambda x: (x[0], x[1]))

        INDEX_HTML = """<!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Read-only Clip Pages — Index</title>
      <style>
        body { font-family: -apple-system, system-ui, sans-serif; margin:24px; }
        h1 { margin:0 0 10px 0; }
        .muted { color:#666; font-size:12px; margin-bottom:12px; }
        .list { display:flex; flex-direction:column; gap:8px; }
        a { text-decoration:none; color:#111; border:1px solid #e5e7eb; border-radius:10px; padding:8px 10px; background:#fff; }
        .chip { font-size:12px; color:#fff; background:#2563eb; padding:2px 8px; border-radius:999px; margin-right:8px; }
      </style>
    </head>
    <body>
      <h1>Read-only Clip Pages</h1>
      <div class="muted">Auto-generated index linking to each page folder.</div>
      <div class="list">
        {rows}
      </div>
    </body>
    </html>"""

        rows = []
        for mm, title, href in items:
            rows.append(f'<a href="{href}"><span class="chip">{mm}</span>{title}</a>')
        (args.index_root / "index.html").write_text(
            INDEX_HTML.format(rows="\n    ".join(rows)), encoding="utf-8")
        print(f"Wrote root index: {args.index_root}/index.html")


    print(f"\nDone. Wrote {len(built)} pages.")

if __name__ == "__main__":
    main()
