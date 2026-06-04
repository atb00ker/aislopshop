#!/usr/bin/env python3
"""
align-srt.py

Re-aligns SRT timestamps against source audio using stable-ts forced alignment.
By default the entire file is synced from 00:00:00.  Use --keep-through and
--anchor to preserve a verified prefix and only re-align the remainder.

Requirements
------------
    pip install stable-ts       # PyPI package name is "stable-ts" (with a hyphen)
    # ffmpeg must also be installed and on your PATH

Usage
-----
    # Sync the entire SRT (default; output is <input>-aligned.srt)
    python align-srt.py --srt talk.srt --audio talk.m4a

    # Specify output path explicitly
    python align-srt.py --srt talk.srt --audio talk.m4a --output talk-fixed.srt

    # Better accuracy for dense/technical vocabulary
    python align-srt.py --srt talk.srt --audio talk.m4a --model small

    # Partial sync — preserve cues 1-10, re-align from 00:01:30,000 onward
    python align-srt.py \\
        --srt talk.srt --audio talk.m4a \\
        --keep-through 10 --anchor 90.0

Notes
-----
  * --output defaults to <srt-stem>-aligned.srt in the same directory.
  * --anchor is the audio time (seconds) where alignment begins.
    Default: 0 (start of audio).  For partial sync, set to the end timestamp
    of the last preserved cue (e.g. 90.0 = 00:01:30,000).
  * --keep-through N preserves every cue whose printed number is ≤ N verbatim.
    Default: 0 (align everything).  This handles non-sequential numbering
    (e.g. gaps in cue numbers).
  * --dup-gap-min sets the minimum gap (seconds) required before a consecutive
    duplicate cue is kept in the output.  Default: 10.
  * Consecutive duplicate cues are excluded from the main alignment pass
    and re-evaluated afterwards.  A duplicate is retained only when both the
    original gap and the aligned gap are at least --dup-gap-min seconds.
  * Alignment uses align_words (one segment per cue).  If that fails, the
    script falls back to align() with original_split=True.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Cue:
    position: int   # 0-based index in the parsed list (stable for ordering)
    cue_num:  int   # the number printed in the .srt file
    start_ms: int
    end_ms:   int
    text:     str   # possibly multi-line (joined with '\n')


# ---------------------------------------------------------------------------
# SRT I/O
# ---------------------------------------------------------------------------

_TS_RE = re.compile(
    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})'
)


def parse_srt(path: Path) -> List[Cue]:
    raw = path.read_text(encoding='utf-8-sig')
    cues: List[Cue] = []
    for i, block in enumerate(re.split(r'\n{2,}', raw.strip())):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            cue_num = int(lines[0].strip())
        except ValueError:
            continue
        m = _TS_RE.match(lines[1].strip())
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start_ms = (h1 * 3600 + m1 * 60 + s1) * 1000 + ms1
        end_ms   = (h2 * 3600 + m2 * 60 + s2) * 1000 + ms2
        cues.append(Cue(i, cue_num, start_ms, end_ms, '\n'.join(lines[2:])))
    return cues


def ms_to_ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def write_srt(cues: List[Cue], path: Path) -> None:
    blocks = [
        f'{c.cue_num}\n{ms_to_ts(c.start_ms)} --> {ms_to_ts(c.end_ms)}\n{c.text}'
        for c in cues
    ]
    path.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word_tokens(text: str) -> List[str]:
    """Alphanumeric + apostrophe tokens — used for word counting and dedup."""
    return re.findall(r"[A-Za-z0-9']+", text)


def normalize(text: str) -> str:
    """Lowercase word sequence for duplicate detection."""
    return ' '.join(word_tokens(text.lower()))


def audio_duration_s(path: str) -> float:
    """Return audio duration in seconds via ffprobe."""
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.exit(f'ffprobe failed:\n{r.stderr}')
    return float(r.stdout.strip())


def trim_audio_to_wav(src: str, start_s: float, dst: str) -> None:
    """Extract audio from start_s seconds onward; write as 16 kHz mono WAV."""
    r = subprocess.run(
        ['ffmpeg', '-y', '-ss', str(start_s), '-i', src,
         '-ac', '1', '-ar', '16000', '-f', 'wav', dst],
        capture_output=True,
    )
    if r.returncode != 0:
        sys.exit(f'ffmpeg failed:\n{r.stderr.decode(errors="replace")}')


def cue_text(cue: Cue) -> str:
    return cue.text.replace('\n', ' ').strip()


def build_segments_proportional(cues: List[Cue], avail_s: float) -> List[dict]:
    """Estimate segment windows by word count when SRT times are unreliable."""
    weights = [max(1, len(word_tokens(cue_text(c)))) for c in cues]
    total = sum(weights)
    t = 0.0
    segments: List[dict] = []
    for cue, weight in zip(cues, weights):
        dur = avail_s * (weight / total)
        end = min(avail_s, t + max(dur, 0.3))
        segments.append({'start': t, 'end': end, 'text': cue_text(cue)})
        t = end
    if segments:
        segments[-1]['end'] = avail_s
    return segments


def build_segments_from_srt(cues: List[Cue], anchor_ms: int, avail_s: float) -> List[dict]:
    """Use existing SRT times as alignment hints (relative to anchor)."""
    segments: List[dict] = []
    for cue in cues:
        start_s = max(0.0, (cue.start_ms - anchor_ms) / 1000)
        end_s = max(start_s + 0.1, (cue.end_ms - anchor_ms) / 1000)
        end_s = min(end_s, avail_s)
        segments.append({'start': start_s, 'end': end_s, 'text': cue_text(cue)})
    return segments


def _seg_time(seg, attr: str) -> Optional[float]:
    val = getattr(seg, attr, None)
    if val is None and attr == 'start':
        val = getattr(seg, 'start_time', None)
    if val is None and attr == 'end':
        val = getattr(seg, 'end_time', None)
    return val


def extract_segment_times(result, offset_ms: int) -> List[Tuple[int, int]]:
    """Pull (start_ms, end_ms) for each aligned segment."""
    out: List[Tuple[int, int]] = []
    for seg in getattr(result, 'segments', []):
        s = _seg_time(seg, 'start')
        e = _seg_time(seg, 'end')
        if s is not None and e is not None:
            out.append((int(s * 1000) + offset_ms, int(e * 1000) + offset_ms))
    return out


def run_align_words(model, wav_path: str, segments: List[dict], language: str):
    """Align one segment per cue — stable-ts recommended path for subtitles."""
    return model.align_words(
        wav_path,
        segments,
        language=language,
        stream=False,
        verbose=False,
    )


def run_align_split(model, wav_path: str, cues: List[Cue], language: str):
    """Fallback: align with one line per cue via original_split."""
    text = '\n'.join(cue_text(c) for c in cues)
    return model.align(
        wav_path,
        text,
        language=language,
        original_split=True,
        stream=False,
        verbose=False,
        fast_mode=True,
    )


def align_cues(model, wav_path: str, cues: List[Cue], language: str,
               anchor_s: float, audio_dur_s: float, use_srt_times: bool):
    """
    Align cues against audio.  Tries align_words first (per-cue segments),
    falls back to align() with original_split if needed.
    """
    offset_ms = int(anchor_s * 1000)
    avail_s = max(0.1, audio_dur_s - anchor_s)

    if use_srt_times:
        segments = build_segments_from_srt(cues, offset_ms, avail_s)
    else:
        segments = build_segments_proportional(cues, avail_s)

    print(f'Aligning {len(cues)} cues with align_words …')
    try:
        result = run_align_words(model, wav_path, segments, language)
        times = extract_segment_times(result, offset_ms)
        if len(times) == len(cues):
            return times, 'align_words'
        print(f'WARNING: align_words returned {len(times)} segments '
              f'for {len(cues)} cues — trying fallback.')
    except Exception as exc:
        print(f'WARNING: align_words failed ({exc}) — trying fallback.')

    print('Trying align() with original_split …')
    result = run_align_split(model, wav_path, cues, language)
    times = extract_segment_times(result, offset_ms)
    if len(times) != len(cues):
        print(f'WARNING: align returned {len(times)} segments for {len(cues)} cues.')
    return times, 'align'


def apply_segment_times(cues: List[Cue], times: List[Tuple[int, int]],
                        anchor_ms: int) -> Tuple[List[Cue], List[int]]:
    """Build updated cues from aligned segment times."""
    prev_ms = anchor_ms
    unaligned: List[int] = []
    updated: List[Cue] = []

    for i, orig in enumerate(cues):
        if i < len(times):
            s, e = times[i]
            s = max(s, prev_ms)
            e = max(e, s + 100)
        else:
            dur = max(orig.end_ms - orig.start_ms, 500)
            s = prev_ms + 200
            e = s + dur
            unaligned.append(orig.cue_num)

        updated.append(Cue(orig.position, orig.cue_num, s, e, orig.text))
        prev_ms = e

    return updated, unaligned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--srt',          required=True,
                    help='Input SRT file')
    ap.add_argument('--audio',        required=True,
                    help='Source audio/video file (any format ffmpeg supports)')
    ap.add_argument('--output',       default=None,
                    help='Output SRT file.  Default: <srt-stem>-aligned.srt '
                         'in the same directory as --srt.')
    ap.add_argument('--anchor',       type=float, default=0.0,
                    help='Audio time (seconds) at which alignment starts.  '
                         'Default: 0 (full file).  For partial sync, set to '
                         'the end time of the last preserved cue.')
    ap.add_argument('--keep-through', type=int, dest='keep_through', default=0,
                    help='Preserve every cue whose cue-number is ≤ N verbatim.  '
                         'Default: 0 (align entire file).')
    ap.add_argument('--dup-gap-min',  type=float, dest='dup_gap_min', default=10.0,
                    help='Minimum gap in seconds before a consecutive duplicate '
                         'cue is kept in the output.  Default: 10.')
    ap.add_argument('--model',        default='base',
                    help='Whisper model: tiny | base | small | medium | large.  '
                         'Default: base.  Use "small" for better accuracy.')
    ap.add_argument('--language',     default='en')
    args = ap.parse_args()

    srt_path   = Path(args.srt)
    audio_path = Path(args.audio)
    out_path   = (Path(args.output) if args.output
                  else srt_path.with_stem(srt_path.stem + '-aligned'))

    for p in (srt_path, audio_path):
        if not p.exists():
            sys.exit(f'File not found: {p}')

    if out_path == srt_path:
        sys.exit('ERROR: --output must differ from --srt to avoid overwriting the original.')

    try:
        import stable_whisper
    except ImportError:
        sys.exit(
            'stable-ts is not installed.\n'
            'Run:  pip install -r requirements.txt\n'
            '  (PyPI package name is "stable-ts", not "stable_whisper".)'
        )

    # ------------------------------------------------------------------
    # 1. Parse — split into preserved vs. to-align
    # ------------------------------------------------------------------
    all_cues    = parse_srt(srt_path)
    preserved   = [c for c in all_cues if c.cue_num <= args.keep_through]
    to_align    = [c for c in all_cues if c.cue_num >  args.keep_through]
    audio_dur_s = audio_duration_s(str(audio_path))

    if args.keep_through:
        print(f'Parsed {len(all_cues)} cues — '
              f'preserving {len(preserved)} (cue_num ≤ {args.keep_through}), '
              f'aligning {len(to_align)} from {args.anchor}s.')
    else:
        print(f'Parsed {len(all_cues)} cues — aligning entire file '
              f'({audio_dur_s:.1f}s audio).')

    if not to_align:
        sys.exit('Nothing to align — all cues are within --keep-through range.')

    if args.keep_through > 0 and args.anchor == 0.0:
        sys.exit(
            'ERROR: --keep-through is set but --anchor is 0.\n'
            'Set --anchor to the end time (in seconds) of the last preserved cue\n'
            'so aligned cues do not overlap with preserved ones.\n'
            'Example: --keep-through 10 --anchor 90.0'
        )

    # ------------------------------------------------------------------
    # 2. Detect consecutive duplicates
    #    A cue is a "duplicate" when it immediately follows a cue with
    #    identical normalised text.  These are excluded from the main
    #    alignment pass (sequential word-counting breaks for duplicates)
    #    and re-evaluated afterwards.
    # ------------------------------------------------------------------
    dup_gap_min_ms = int(args.dup_gap_min * 1000)
    main_cues: List[Cue] = []
    dup_cues:  List[Tuple[Cue, Cue]] = []   # (duplicate_cue, its predecessor)

    for i, cue in enumerate(to_align):
        if i > 0 and normalize(cue.text) == normalize(to_align[i - 1].text):
            pred = to_align[i - 1]
            orig_gap_s = (cue.start_ms - pred.end_ms) / 1000
            print(f'  Duplicate: cue {cue.cue_num} == cue {pred.cue_num} '
                  f'(original gap {orig_gap_s:.1f}s) — excluded from alignment pass.')
            dup_cues.append((cue, pred))
        else:
            main_cues.append(cue)

    # ------------------------------------------------------------------
    # 3. Convert/trim audio, then run per-cue forced alignment
    # ------------------------------------------------------------------
    fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    method = 'unknown'
    try:
        if args.anchor > 0:
            print(f'\nTrimming audio from {args.anchor}s → temporary WAV …')
        else:
            print('\nConverting audio → temporary WAV …')
        trim_audio_to_wav(str(audio_path), args.anchor, tmp_wav)

        print(f'Loading stable-ts model "{args.model}" …')
        model = stable_whisper.load_model(args.model)

        # Full sync: SRT times may be wrong → proportional segment windows.
        # Partial sync: preserved prefix is trusted → use SRT times as hints.
        use_srt_times = args.keep_through > 0
        print(f'Alignment input: {len(main_cues)} cues '
              f'({"SRT times" if use_srt_times else "proportional windows"}).')

        print('Running forced alignment (may take 1–3 minutes) …')
        times, method = align_cues(
            model, tmp_wav, main_cues, args.language,
            args.anchor, audio_dur_s, use_srt_times,
        )
        print(f'Aligned {len(times)} segments via {method}.')

    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    # ------------------------------------------------------------------
    # 4. Apply aligned segment times to cues
    # ------------------------------------------------------------------
    updated_cues, unaligned_ns = apply_segment_times(
        main_cues, times, int(args.anchor * 1000),
    )

    # ------------------------------------------------------------------
    # 5. Re-evaluate consecutive duplicates
    #    Keep a duplicate only if both the original gap and the aligned gap
    #    are ≥ --dup-gap-min seconds (line likely spoken twice).
    # ------------------------------------------------------------------
    by_num = {c.cue_num: c for c in updated_cues}
    extra:   List[Tuple[int, Cue]] = []

    for dup, pred_orig in dup_cues:
        pred_new = by_num.get(pred_orig.cue_num)
        if pred_new is None:
            print(f'\nDup cue {dup.cue_num}: predecessor {pred_orig.cue_num} '
                  f'not in output → dropped.')
            continue

        dup_idx  = next(
            (i for i, c in enumerate(to_align) if c.cue_num == dup.cue_num), None
        )
        nxt_orig = (to_align[dup_idx + 1]
                    if dup_idx is not None and dup_idx + 1 < len(to_align)
                    else None)
        nxt_new  = by_num.get(nxt_orig.cue_num) if nxt_orig else None

        audio_end_ms = int(audio_dur_s * 1000)
        pred_end_ms  = pred_new.end_ms
        nxt_start_ms = nxt_new.start_ms if nxt_new else audio_end_ms
        avail_gap_ms = nxt_start_ms - pred_end_ms
        orig_gap_ms  = dup.start_ms - pred_orig.end_ms
        dur_ms       = max(dup.end_ms - dup.start_ms, 500)

        print(f'\nDup cue {dup.cue_num}:  '
              f'original gap = {orig_gap_ms/1000:.1f}s | '
              f'aligned gap = {avail_gap_ms/1000:.1f}s | '
              f'line duration ≈ {dur_ms/1000:.1f}s')

        if orig_gap_ms >= dup_gap_min_ms and avail_gap_ms >= dur_ms + 500:
            dup_start = pred_end_ms + (avail_gap_ms - dur_ms) // 2
            dup_end   = min(dup_start + dur_ms, nxt_start_ms - 100)
            dup_start = dup_end - dur_ms
            print(f'  → Kept at {ms_to_ts(dup_start)} → {ms_to_ts(dup_end)}')
            extra.append((dup.position,
                          Cue(dup.position, dup.cue_num, dup_start, dup_end, dup.text)))
        else:
            print(f'  → Dropped (gap too small for a genuine second utterance).')

    # ------------------------------------------------------------------
    # 6. Assemble, validate, write
    # ------------------------------------------------------------------
    all_pairs = [(c.position, c) for c in updated_cues] + extra
    all_pairs.sort(key=lambda x: x[0])
    final_cues = preserved + [c for _, c in all_pairs]

    overlap_count = 0
    for i in range(1, len(final_cues)):
        p, c = final_cues[i - 1], final_cues[i]
        if c.start_ms < p.end_ms:
            print(f'OVERLAP: cue {p.cue_num} ends {ms_to_ts(p.end_ms)} | '
                  f'cue {c.cue_num} starts {ms_to_ts(c.start_ms)}')
            overlap_count += 1

    write_srt(final_cues, out_path)

    print(f'\n{"="*60}')
    print(f'Output: {out_path}  ({len(final_cues)} cues)')
    if overlap_count:
        print(f'WARNING: {overlap_count} timestamp overlap(s) — review manually.')
    if unaligned_ns:
        print(f'Cues with no alignment match (original duration kept): {unaligned_ns}')
    if not overlap_count and not unaligned_ns:
        print('All cues aligned cleanly.')
    print(f'\nVerify {out_path} against the audio before replacing the original.')


if __name__ == '__main__':
    main()
