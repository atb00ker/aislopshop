# Align SRT

Re-aligns SRT subtitle timestamps against source audio using
[stable-ts](https://github.com/jianfch/stable-ts) forced alignment.
Works on any SRT file and any audio/video format that ffmpeg supports.

## Dependencies

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/download.html) — must be on your `PATH`
- stable-ts Python package

> **Important:** The PyPI package name is `stable-ts` (hyphen).
> Do **not** run `pip install stable_whisper` — that package does not exist on PyPI.

```bash
pip install -r requirements.txt
```

## Usage

### Full sync (most common)

Re-aligns every cue from the start of the audio. Output defaults to
`<input>-aligned.srt` in the same directory.

```bash
python align-srt.py --srt talk.srt --audio talk.m4a
```

### Specify output path

```bash
python align-srt.py --srt talk.srt --audio talk.m4a --output talk-fixed.srt
```

### Better accuracy

Use a larger Whisper model for dense or technical vocabulary.
`small` is a good balance of speed and accuracy.

```bash
python align-srt.py --srt talk.srt --audio talk.m4a --model small
```

Available models (slowest/most accurate last): `tiny`, `base`, `small`, `medium`, `large`

### Partial sync

Preserve cues whose numbers are ≤ N verbatim, and re-align everything after
that from a given audio timestamp.

```bash
# Preserve cues 1–10 (which end at 00:01:30,000 = 90 s), align the rest
python align-srt.py \
    --srt talk.srt --audio talk.m4a \
    --keep-through 10 --anchor 90.0
```

> `--anchor` must be set to the **end time in seconds** of the last preserved
> cue whenever `--keep-through` is used.

### Duplicate cues

If consecutive cues share identical text (e.g. a repeated line), they are
excluded from the main alignment pass and re-evaluated afterwards. A duplicate
is kept only when the gap before it is at least `--dup-gap-min` seconds (default: 10)
in both the original SRT and the aligned output, suggesting the line was
genuinely spoken twice.

```bash
# Lower the threshold to 5 s
python align-srt.py --srt talk.srt --audio talk.m4a --dup-gap-min 5
```

## All options

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--srt` | *(required)* | Input SRT file |
| `--audio` | *(required)* | Source audio/video file |
| `--output` | `<srt>-aligned.srt` | Output SRT path |
| `--model` | `base` | Whisper model size |
| `--language` | `en` | Audio language code |
| `--anchor` | `0` | Audio start time for alignment (seconds) |
| `--keep-through` | `0` | Preserve cues with number ≤ N |
| `--dup-gap-min` | `10` | Min gap (s) to keep a consecutive duplicate cue |

## How it works

1. Parses the SRT into cue blocks.
2. Builds one alignment segment per cue (proportional windows for full sync,
   existing SRT times as hints for partial sync).
3. Converts the audio to 16 kHz mono WAV (via ffmpeg).
4. Runs `model.align_words()` — forced alignment per cue segment.
5. Falls back to `model.align()` with `original_split=True` if needed.
6. Writes the corrected SRT; reports any cues that could not be matched.

## Troubleshooting

**`Expected a callable, got Tensor`** — usually a version mismatch or a bug in
the old single-string `align()` path. This script now uses per-cue
`align_words()` by default. Reinstall dependencies:

```bash
pip install -U -r requirements.txt
```

**`pip install stable_whisper` fails** — use `stable-ts` instead (see above).

**Poor alignment quality** — try `--model small` for better accuracy on
technical vocabulary.
