# DumbzoneClips

AI-powered pipeline for [The Dumbzone Podcast](https://www.thedumbzone.com) — transcribes episodes, identifies speakers, finds recurring phrases, and builds shareable audio clip pages.

## What It Does

1. **Transcribe & diarize** — WhisperX produces timestamped transcripts with speaker labels
2. **Identify speakers** — SpeechBrain ECAPA maps anonymous speaker IDs to real names (Dan, Jake, Blake)
3. **Find phrases** — searches transcripts for recurring bits and exports audio clips
4. **Package for sharing** — builds self-contained HTML pages with playable clips

### Phrases Tracked

- *"The Point Is"* / *"The Point Was"*
- *"Have/Had a Buddy"*
- Roseanne mentions
- Hillary mentions
- Fight with Wife (proximity heuristic)
- Too Much Money

## Scripts

| Script | Purpose |
|--------|---------|
| `whisperxpy_v4.py` | Batch transcribe + diarize audio with WhisperX |
| `build_speaker_map_ecapa.py` | Build speaker name map via ECAPA voice fingerprinting |
| `assign_main_speakers.py` | (Legacy) assign speaker names — use `build_speaker_map_ecapa.py` instead |
| `speaker_auditioner_v3.py` | Audition individual speaker segments |
| `multi_finder_packager_v6.py` | Find phrases, extract clips, generate HTML audition pages |
| `readonly_pack_all_v3.py` | Bundle approved clips into a shareable read-only package |

## Setup

```bash
conda create -n dumbzone-env python=3.10
conda activate dumbzone-env
pip install -r requirements.txt
brew install ffmpeg
```

You'll also need a [HuggingFace token](https://huggingface.co/settings/tokens) for pyannote diarization.

## Usage

See [docs/workflow.md](docs/workflow.md) for the full step-by-step pipeline.

## Public Clip Pages

Monthly clip pages are published on the [`releases`](https://github.com/BayleyBY/DumbzoneClips/tree/releases) branch and hosted via GitHub Pages at:

**[https://bayleyby.github.io/DumbzoneClips/](https://bayleyby.github.io/DumbzoneClips/)**
