# DumbzoneClips — Workflow

A 4-step pipeline to transcribe podcast episodes, identify speakers, find recurring phrases, and build shareable audio clip pages.

---

## Environment Setup

```bash
conda activate dumbzone-env
# or: pip install -r requirements.txt
```

Dependencies: `whisperx`, `speechbrain`, `ffmpeg` (system), HuggingFace token for pyannote diarization.

---

## Step 1 — Transcribe & Diarize

Uses WhisperX to produce timestamped transcripts with speaker labels (`SPEAKER_00`, `SPEAKER_01`, etc.)

```bash
python whisperxpy_v4.py \
  --indir "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/sodes" \
  --outdir "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/scripts" \
  --model large-v3 \
  --workers 1 \
  --threads 16 \
  --progress \
  --model_dir "/Volumes/Flame_Archive/AI_Projects/WhisperAI/whisperx_models" \
  --vad_method pyannote \
  --min_speakers 5 --max_speakers 16 \
  --hf_token "$HF_TOKEN"
```

**Output:** `scripts/YY-MM/YY-MM-DD/YY-MM-DD.json` — WhisperX JSON with segments + word-level timestamps

---

## Step 2 — Build Speaker Map (ECAPA voice fingerprinting)

Maps anonymous speaker IDs to real names (Dan, Jake, Blake) using voice sample enrollment.

```bash
python build_speaker_map_ecapa.py \
  --root "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/scripts/25-11" \
  --audio_dir "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/sodes_done/25-11" \
  --enroll \
    Dan:/path/to/voice_samples/dan_sample.wav \
    Jake:/path/to/voice_samples/jake_sample.wav \
    Blake:/path/to/voice_samples/blake_sample.wav \
  --threshold 0.55 \
  --outfile "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/scripts/25-11/speaker_map.json"
```

> ⚠️ Note: `assign_main_speakers.py` (older approach) is **not currently working reliably**. Use `build_speaker_map_ecapa.py` instead.

**Output:** `speaker_map.json` — maps `{ "YY-MM-DD": { "SPEAKER_07": "Dan" } }`

---

## Step 3 — Find Phrases & Package Clips

Searches all transcripts for key phrases and builds a self-contained HTML audition page with extracted audio clips.

```bash
python multi_finder_packager_v6.py \
  --root "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/scripts" \
  --audio_dir "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/sodes_done" \
  --month 25-11 \
  --pad_before 0.4 --pad_after 0.6 \
  --pad_tp 1.2,2.8 \
  --pad_hab 2.6,5.6 \
  --pad_wife 2.0,8.0 \
  --pad_roseanne 2.0,6.0 \
  --pad_hillary 1.0,3.0
```

**Phrases detected:**
- "The Point Is" / "The Point Was"
- "Have/Had a Buddy"
- Roseanne mentions
- Hillary mentions
- Fight with Wife (heuristic: fight-words + spouse-words in proximity)
- Too Much Money

**Output:** One portable folder per category: `scripts/25-11/25-11_tp_portable/`, `25-11_hab_portable/`, etc. Each contains `index.html` + `clips/`.

### Step 3b — Label Speakers in HTML

Open each `index.html` in a browser, type speaker names in the input fields next to each clip, then click **Download updated payload.json**. Save the file alongside the script for future runs.

---

## Step 4 — Build Read-Only Share Package

Bundles all approved clip pages into a single shareable package.

```bash
python readonly_pack_all_v3.py \
  --payload_glob "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/**/payload*.json" \
  --outdir_root "/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/share_readonly" \
  --title_prefix "" \
  --copy_clips
```

**Output:** `share_readonly/` — static HTML + audio, ready to upload to GitHub Pages (`releases` branch).

---

## Directory Structure (local)

```
/Volumes/Flame_Archive/AI_Projects/WhisperAI/dumbzone/
├── sodes/              ← raw podcast MP3s (input)
├── sodes_done/         ← processed/organized MP3s
│   └── 25-11/
├── scripts/            ← WhisperX JSON transcripts
│   └── 25-11/
│       └── 25-11-03/
│           └── 25-11-03.json
├── share_readonly/     ← final shareable output → publish to `releases` branch
├── voice_samples/      ← short WAV clips for speaker enrollment
│   ├── dan_sample.wav
│   ├── jake_sample.wav
│   └── blake_sample.wav
└── whisperx_models/    ← cached model weights (large-v3, etc.)
```

---

## Publishing to GitHub Pages

The `releases` branch hosts the public-facing clip pages at:
`https://bayleyby.github.io/DumbzoneClips/`

To update after a new month is processed:
1. Run Step 4 to generate `share_readonly/`
2. Check out the `releases` branch
3. Copy the new month folder into the repo root
4. Update `index.html` with a link to the new month
5. Commit and push
