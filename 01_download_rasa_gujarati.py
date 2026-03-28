#!/usr/bin/env python3
"""
Download AI4Bharat/Rasa Gujarati dataset, analyze it, and convert to LJSpeech format.

Dataset: https://huggingface.co/datasets/ai4bharat/Rasa (subset: Gujarati)
Columns: filename, text, language, gender, style, duration, wav_path, audio
~27.9K rows total (train + test splits)

Output:
  ~/rasa_analysis/         ← gender/style/duration stats
  ~/rasa_ljspeech/         ← LJSpeech format (wavs/ + metadata.csv) for training
  ~/rasa_ljspeech_eval/    ← LJSpeech format for evaluation (test split)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

# Check dependencies
try:
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "datasets", "soundfile", "numpy", "--break-system-packages", "-q"
    ])
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np


def download_and_analyze(cache_dir):
    """Download Rasa Gujarati and return both splits."""
    print("=" * 60)
    print("Downloading AI4Bharat/Rasa — Gujarati subset...")
    print("=" * 60)
    print(f"Cache dir: {cache_dir}")
    print("This may take a while on first run...\n")

    # Load both splits
    train_ds = load_dataset(
        "ai4bharat/Rasa",
        name="Gujarati",
        split="train",
        cache_dir=cache_dir
    )
    print(f"Train split loaded: {len(train_ds)} rows")

    test_ds = load_dataset(
        "ai4bharat/Rasa",
        name="Gujarati",
        split="test",
        cache_dir=cache_dir
    )
    print(f"Test split loaded:  {len(test_ds)} rows")

    return train_ds, test_ds


def analyze_dataset(ds, split_name, output_dir):
    """Analyze duration by gender, style, etc. WITHOUT decoding audio."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Access raw Arrow table — completely bypasses torchcodec
    table = ds.data

    genders = table.column("gender").to_pylist()
    styles = table.column("style").to_pylist()
    durations_raw = table.column("duration").to_pylist()  # strings!
    filenames = table.column("filename").to_pylist()

    # Check sample rate from audio struct
    sample_rates = set()
    try:
        audio_col = table.column("audio")
        # Check first non-null entry for path pattern to guess SR
        first_audio = audio_col[0].as_py()
        if first_audio and "path" in first_audio and first_audio["path"]:
            print(f"Audio path sample: {first_audio['path']}")
        has_bytes = first_audio and first_audio.get("bytes") is not None
        print(f"Audio has inline bytes: {has_bytes}")
    except Exception as e:
        print(f"Could not inspect audio column: {e}")

    gender_duration = defaultdict(float)
    gender_count = defaultdict(int)
    style_duration = defaultdict(float)
    style_count = defaultdict(int)
    gender_style_duration = defaultdict(float)
    gender_style_count = defaultdict(int)

    durations = []

    for i in range(len(genders)):
        gender = genders[i] or "unknown"
        style = styles[i] or "unknown"
        try:
            duration = float(durations_raw[i])
        except (ValueError, TypeError):
            duration = 0.0

        gender_duration[gender] += duration
        gender_count[gender] += 1
        style_duration[style] += duration
        style_count[style] += 1
        gender_style_duration[f"{gender}|{style}"] += duration
        gender_style_count[f"{gender}|{style}"] += 1
        durations.append(duration)

    total_duration = sum(durations)
    durations_arr = np.array(durations)

    # Print report
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS — {split_name} ({len(ds)} rows)")
    print(f"{'=' * 60}")

    print(f"\nTotal duration: {total_duration/3600:.2f} hours ({total_duration:.0f}s)")
    print(f"Sample rates found: {sample_rates if sample_rates else 'checking...'}")
    print(f"Duration stats: min={durations_arr.min():.2f}s, max={durations_arr.max():.2f}s, "
          f"mean={durations_arr.mean():.2f}s, median={np.median(durations_arr):.2f}s")

    print(f"\n--- Duration by Gender ---")
    print(f"{'Gender':<12} {'Count':>8} {'Hours':>10} {'Avg (s)':>10}")
    print("-" * 42)
    for g in sorted(gender_duration.keys()):
        hrs = gender_duration[g] / 3600
        avg = gender_duration[g] / gender_count[g] if gender_count[g] > 0 else 0
        print(f"{g:<12} {gender_count[g]:>8} {hrs:>10.2f} {avg:>10.2f}")

    print(f"\n--- Duration by Style ---")
    print(f"{'Style':<20} {'Count':>8} {'Hours':>10} {'Avg (s)':>10}")
    print("-" * 50)
    for s in sorted(style_duration.keys(), key=lambda x: style_duration[x], reverse=True):
        hrs = style_duration[s] / 3600
        avg = style_duration[s] / style_count[s] if style_count[s] > 0 else 0
        print(f"{s:<20} {style_count[s]:>8} {hrs:>10.2f} {avg:>10.2f}")

    print(f"\n--- Duration by Gender × Style ---")
    print(f"{'Gender|Style':<30} {'Count':>8} {'Hours':>10}")
    print("-" * 50)
    for gs in sorted(gender_style_duration.keys(), key=lambda x: gender_style_duration[x], reverse=True):
        hrs = gender_style_duration[gs] / 3600
        print(f"{gs:<30} {gender_style_count[gs]:>8} {hrs:>10.2f}")

    # Save analysis
    analysis = {
        "split": split_name,
        "total_rows": len(ds),
        "total_duration_hours": round(total_duration / 3600, 2),
        "sample_rates": list(sample_rates),
        "duration_stats": {
            "min": round(float(durations_arr.min()), 3),
            "max": round(float(durations_arr.max()), 3),
            "mean": round(float(durations_arr.mean()), 3),
            "median": round(float(np.median(durations_arr)), 3),
            "p95": round(float(np.percentile(durations_arr, 95)), 3),
            "p99": round(float(np.percentile(durations_arr, 99)), 3),
        },
        "gender": {
            g: {"count": gender_count[g], "hours": round(gender_duration[g] / 3600, 2)}
            for g in sorted(gender_duration.keys())
        },
        "style": {
            s: {"count": style_count[s], "hours": round(style_duration[s] / 3600, 2)}
            for s in sorted(style_duration.keys())
        },
    }

    analysis_path = output_dir / f"analysis_{split_name.lower()}.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis saved: {analysis_path}")

    return analysis


def convert_to_ljspeech(ds, output_dir, target_sr=22050, gender_filter=None):
    """Convert HF dataset to LJSpeech format — bypasses torchcodec entirely."""
    import io
    import librosa

    output_dir = Path(output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_lines = []
    skipped = 0
    processed = 0
    errors = 0

    # Access raw Arrow table — NO torchcodec
    table = ds.data
    filenames = table.column("filename").to_pylist()
    texts = table.column("text").to_pylist()
    genders = table.column("gender").to_pylist()
    audio_col = table.column("audio")

    total = len(filenames)
    print(f"\nConverting {total} rows to LJSpeech format...")
    if gender_filter:
        print(f"Gender filter: {gender_filter}")
    print(f"Target sample rate: {target_sr} Hz")
    print(f"Output: {output_dir}\n")

    for i in range(total):
        # Apply gender filter
        if gender_filter and (genders[i] or "").lower() != gender_filter.lower():
            skipped += 1
            continue

        text = (texts[i] or "").strip()
        filename = filenames[i] or f"utt_{i:06d}"

        if not text:
            skipped += 1
            continue

        # Clean text
        text = text.replace("|", " ")
        text = text.replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split())

        if not text:
            skipped += 1
            continue

        try:
            # Get raw audio from Arrow struct — no torchcodec!
            audio_struct = audio_col[i].as_py()

            if audio_struct is None:
                skipped += 1
                continue

            # Decode audio bytes with soundfile
            audio_bytes = audio_struct.get("bytes")
            if audio_bytes is not None:
                audio_array, orig_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            else:
                skipped += 1
                continue

            # Resample if needed
            if orig_sr != target_sr:
                audio_array = librosa.resample(
                    audio_array, orig_sr=orig_sr, target_sr=target_sr
                )

            # Save WAV
            wav_path = wavs_dir / f"{filename}.wav"
            sf.write(str(wav_path), audio_array, target_sr, subtype="PCM_16")

            # LJSpeech format: file_id|text|text
            metadata_lines.append(f"{filename}|{text}|{text}")
            processed += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Error on {filename}: {e}")

        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i+1}/{total} (processed={processed}, skipped={skipped})")

    # Sort and write metadata
    metadata_lines.sort()
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for line in metadata_lines:
            f.write(line + "\n")

    print(f"\n{'=' * 60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Metadata:  {csv_path}")
    print(f"WAVs dir:  {wavs_dir}")

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Download AI4Bharat/Rasa Gujarati, analyze, and convert to LJSpeech"
    )
    parser.add_argument("--cache-dir", type=str,
                        default=os.path.expanduser("~/hf_cache"),
                        help="HuggingFace cache directory")
    parser.add_argument("--analysis-dir", type=str,
                        default=os.path.expanduser("~/rasa_analysis"),
                        help="Output directory for analysis reports")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.expanduser("~/rasa_ljspeech"),
                        help="Output directory for LJSpeech training data")
    parser.add_argument("--eval-output-dir", type=str,
                        default=os.path.expanduser("~/rasa_ljspeech_eval"),
                        help="Output directory for LJSpeech eval data")
    parser.add_argument("--target-sr", type=int, default=22050,
                        help="Target sample rate (default: 22050 for PiperTTS)")
    parser.add_argument("--gender", type=str, default=None,
                        choices=["Male", "Female", None],
                        help="Filter by gender (default: analyze first, then choose)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only download and analyze, don't convert")
    args = parser.parse_args()

    # Install librosa for resampling
    try:
        import librosa
    except ImportError:
        print("Installing librosa for resampling...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "librosa",
            "--break-system-packages", "-q"
        ])

    # Step 1: Download
    train_ds, test_ds = download_and_analyze(args.cache_dir)

    # Step 2: Analyze
    print("\n")
    train_analysis = analyze_dataset(train_ds, "Train", args.analysis_dir)
    test_analysis = analyze_dataset(test_ds, "Test", args.analysis_dir)

    # Combined summary
    print(f"\n{'=' * 60}")
    print(f"COMBINED SUMMARY (Train + Test)")
    print(f"{'=' * 60}")
    total_hrs = train_analysis["total_duration_hours"] + test_analysis["total_duration_hours"]
    total_rows = train_analysis["total_rows"] + test_analysis["total_rows"]
    print(f"Total: {total_rows} rows, {total_hrs:.2f} hours")

    # Merge gender stats
    for g in set(list(train_analysis["gender"].keys()) + list(test_analysis["gender"].keys())):
        t_hrs = train_analysis["gender"].get(g, {}).get("hours", 0)
        e_hrs = test_analysis["gender"].get(g, {}).get("hours", 0)
        t_cnt = train_analysis["gender"].get(g, {}).get("count", 0)
        e_cnt = test_analysis["gender"].get(g, {}).get("count", 0)
        print(f"  {g}: {t_cnt + e_cnt} clips, {t_hrs + e_hrs:.2f} hours")

    if args.analyze_only:
        print(f"\n--analyze-only flag set. Stopping here.")
        print(f"Re-run with --gender Male or --gender Female to convert.")
        return

    if args.gender is None:
        print(f"\nNo --gender specified. Run with --analyze-only first to see stats,")
        print(f"then re-run with --gender Male or --gender Female.")
        print(f"\nExample:")
        print(f"  python {sys.argv[0]} --gender Female --output-dir ~/rasa_ljspeech")
        return

    # Step 3: Convert
    print(f"\n{'=' * 60}")
    print(f"Converting to LJSpeech — Gender: {args.gender}")
    print(f"{'=' * 60}")

    print(f"\n--- Train split ---")
    convert_to_ljspeech(
        train_ds, args.output_dir,
        target_sr=args.target_sr,
        gender_filter=args.gender
    )

    print(f"\n--- Test split ---")
    convert_to_ljspeech(
        test_ds, args.eval_output_dir,
        target_sr=args.target_sr,
        gender_filter=args.gender
    )

    print(f"\n{'=' * 60}")
    print(f"ALL DONE!")
    print(f"{'=' * 60}")
    print(f"Train: {args.output_dir}")
    print(f"Eval:  {args.eval_output_dir}")
    print(f"\nNext step — train PiperTTS:")
    print(f"  cd ~/piper1-gpl")
    print(f"  python3 -m piper.train fit \\")
    print(f"      --data.voice_name gu_IN-rasa-medium \\")
    print(f"      --data.csv_path {args.output_dir}/metadata.csv \\")
    print(f"      --data.audio_dir {args.output_dir}/wavs/ \\")
    print(f"      --model.sample_rate 22050 \\")
    print(f"      --data.espeak_voice gu \\")
    print(f"      --data.cache_dir ~/piper_cache/ \\")
    print(f"      --data.config_path ~/piper_output/config.json")


if __name__ == "__main__":
    main()
