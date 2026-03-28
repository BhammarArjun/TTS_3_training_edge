# PiperTTS Gujarati Male — Complete Training Journey

**Project:** Training a Gujarati Male TTS model using PiperTTS (VITS architecture)
**Dataset:** AI4Bharat/Rasa — Gujarati subset, Male speaker
**Machine:** Lightning.ai L4 GPU Studio (24 GB VRAM, 31 GB RAM, Python 3.12)
**Date:** March 22–24, 2026
**Repo:** `OHF-Voice/piper1-gpl` (new actively maintained Piper repo)
**Trained Model:** `Arjun4707/piper-gujarati-male` (HuggingFace, private)

---

## Table of Contents

1. [Dataset Selection](#1-dataset-selection)
2. [Environment Setup](#2-environment-setup)
3. [Data Download & Preparation](#3-data-download--preparation)
4. [PiperTTS Repository Setup](#4-pipertts-repository-setup)
5. [Patches Applied](#5-patches-applied)
6. [Training](#6-training)
7. [Testing & Export](#7-testing--export)
8. [Upload to HuggingFace](#8-upload-to-huggingface)
9. [Inference on Local Machine](#9-inference-on-local-machine)
10. [Resume Training](#10-resume-training)
11. [Key Lessons](#11-key-lessons)
12. [Architecture Reference](#12-architecture-reference)

---

## 1. Dataset Selection

### Datasets Evaluated

| Dataset | Source | Hours | Issues |
|---------|--------|-------|--------|
| IISc SPIRE Gujarati Female | IISc Bengaluru | 33.5h | tar.gz only extracted 1,788 of 8,242 WAVs — incomplete download |
| AI4Bharat/Rasa Gujarati | HuggingFace | 51.4h total | Gated dataset, requires HF agreement. Works well. |

### Rasa Gujarati Analysis

```
Total: 27,946 rows, 51.39 hours
  Female: 12,149 clips, 25.38 hours
  Male:   15,797 clips, 26.01 hours
```

**Chose Male voice** — 15,797 clips, ~26 hours.

Columns: `filename`, `text`, `language`, `gender`, `style`, `duration`, `wav_path`, `audio`

Styles include: WIKI, BOOK, CONVERSATION, PROPER NOUN, NAMES, and Ekman emotions (ANGER, FEAR, SURPRISE, HAPPY, SAD, DISGUST).

### Important Notes

- **Rasa is a gated dataset.** Must accept terms at https://huggingface.co/datasets/ai4bharat/Rasa and use a token with "Access to public gated repositories" enabled.
- **Audio is embedded as bytes in Parquet.** NOT file paths. Decoded via `soundfile` from the Arrow table.
- **Duration column is a string**, not float. Must cast with `float()`.

---

## 2. Environment Setup

### Lightning.ai Studio Constraints

- **No venv creation allowed.** Lightning.ai studios have a single managed conda environment. Install everything with `--break-system-packages`.
- **Python 3.12** comes preinstalled. PiperTTS piper1-gpl works with 3.12.
- **GPU must be manually enabled** in the studio settings. Without it, `torch.cuda.is_available()` returns `False`.

### System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build espeak-ng ffmpeg
```

### Verify Gujarati Phoneme Support

```bash
# espeak-ng has Gujarati (gu) built-in
espeak-ng --voices | grep gu
# Should show: 5  gu  --/M  Gujarati  inc/gu

# Test phonemization
espeak-ng -v gu -q --ipa "નમસ્તે"
# Should output IPA phonemes
```

### Python Dependencies

```bash
pip install datasets soundfile librosa scikit-build cmake ninja --break-system-packages
```

---

## 3. Data Download & Preparation

### Script: `01_download_rasa_gujarati.py`

This script handles everything: download → analyze → filter by gender → resample → convert to LJSpeech format.

#### Key Technical Details

**Problem: `torchcodec` ImportError**

HuggingFace `datasets` library (2024+) uses `torchcodec` for audio decoding, which requires FFmpeg shared libraries not present on Lightning.ai. ANY access to dataset rows (iterating, indexing, column slicing via `ds['column']`) triggers audio decoding and crashes.

**Solution: Bypass HuggingFace Dataset wrapper entirely — use the raw Arrow table.**

```python
# WRONG — triggers torchcodec
for row in ds:           # crashes
ds[0]                    # crashes
ds['gender']             # crashes

# CORRECT — raw Arrow table, no audio decoding
table = ds.data
genders = table.column("gender").to_pylist()    # plain Python list
durations = table.column("duration").to_pylist() # strings, need float()
audio_col = table.column("audio")               # raw struct, decode manually
```

**Audio decoding without torchcodec:**

```python
import io
import soundfile as sf

audio_struct = audio_col[i].as_py()  # dict with 'bytes' and 'path'
audio_array, orig_sr = sf.read(io.BytesIO(audio_struct["bytes"]), dtype="float32")
```

#### Running the Script

```bash
# Step 1: Login to HuggingFace (required — gated dataset)
huggingface-cli login

# Step 2: Analyze (no conversion, fast)
python ~/01_download_rasa_gujarati.py --analyze-only

# Step 3: Convert Male clips to LJSpeech format
python ~/01_download_rasa_gujarati.py --gender Male
```

#### Output

```
~/rasa_ljspeech/
├── wavs/
│   ├── GUJ_M_WIKI_01679.wav    (22050Hz, 16-bit, mono)
│   ├── GUJ_M_SURPRISE_00382.wav
│   └── ...  (~13,800 training clips)
└── metadata.csv
    # GUJ_M_WIKI_01679|ગુજરાતી ટેક્સ્ટ...|ગુજરાતી ટેક્સ્ટ...

~/rasa_ljspeech_eval/
├── wavs/  (~2,000 test clips)
└── metadata.csv
```

LJSpeech format: `filename|text|text` (pipe-delimited, no header).

---

## 4. PiperTTS Repository Setup

### Clone the New Repo

**IMPORTANT:** The old `rhasspy/piper` repo is archived (Oct 2025). Use the new `OHF-Voice/piper1-gpl` repo.

The old repo requires `piper-phonemize` which has no Python 3.12 Linux wheels. The new repo embeds espeak-ng directly — no `piper-phonemize` needed.

```bash
cd ~
git clone https://github.com/OHF-Voice/piper1-gpl.git
cd piper1-gpl
pip install -e '.[train]' --break-system-packages
pip install scikit-build cmake ninja --break-system-packages
```

### Build Monotonic Alignment (Cython C extension)

The VITS model requires a `monotonic_align` C extension for alignment search during training.

```bash
cd ~/piper1-gpl/src/piper/train/vits/monotonic_align

# Build with Cython + GCC directly (setup.py has path issues)
pip install cython --break-system-packages
cython core.pyx
gcc -shared -O2 -fPIC $(python3-config --includes) -o core.cpython-312-x86_64-linux-gnu.so core.c

# Verify
ls *.so
# Should show: core.cpython-312-x86_64-linux-gnu.so
```

### Fix Import Path

The `__init__.py` has a wrong nested import path:

```bash
# Fix: .monotonic_align.core → .core
sed -i 's/from .monotonic_align.core import maximum_path_c/from .core import maximum_path_c/' \
    ~/piper1-gpl/src/piper/train/vits/monotonic_align/__init__.py
```

---

## 5. Patches Applied

### Patch 1: PyTorch `weights_only=True` (PosixPath)

**File:** `src/piper/train/__main__.py`
**Problem:** PyTorch 2.6+ changed `torch.load` default to `weights_only=True`. Checkpoints with `pathlib.PosixPath` objects get blocked.
**Fix:** Add after `import torch`:

```python
import pathlib
torch.serialization.add_safe_globals([pathlib.PosixPath])
```

```bash
sed -i '2i import pathlib\ntorch.serialization.add_safe_globals([pathlib.PosixPath])' \
    ~/piper1-gpl/src/piper/train/__main__.py
```

### Patch 2: espeak-ng Data Path (Gujarati voice not found)

**File:** `src/piper/phonemize_espeak.py`
**Problem:** Piper's bundled espeak-ng data doesn't properly support Gujarati. `set_voice('gu')` fails with `RuntimeError: Failed to set voice: gu`.
**Root cause:** The bundled `espeakbridge` C extension can't initialize Gujarati despite having the `gu_dict` and `lang/inc/gu` files.
**Fix:** Point to system espeak-ng data instead:

```bash
sed -i 's|ESPEAK_DATA_DIR = .*|ESPEAK_DATA_DIR = Path("/usr/lib/x86_64-linux-gnu/espeak-ng-data")|' \
    ~/piper1-gpl/src/piper/phonemize_espeak.py
```

**Verification:**

```python
from piper import espeakbridge
espeakbridge.initialize('/usr/lib/x86_64-linux-gnu/espeak-ng-data')
espeakbridge.set_voice('gu')
result = espeakbridge.get_phonemes('નમસ્તે')
print(result)  # [('nəmˈʌsteː', '', True)]
```

Note: `get_phonemes()` takes only 1 argument (text). Voice is set separately via `set_voice()`.

### Patch 3: Monotonic Alignment Import Path

**File:** `src/piper/train/vits/monotonic_align/__init__.py`
**Problem:** Import path `from .monotonic_align.core` expects a nested subdirectory that doesn't exist.
**Fix:**

```bash
sed -i 's/from .monotonic_align.core import maximum_path_c/from .core import maximum_path_c/' \
    ~/piper1-gpl/src/piper/train/vits/monotonic_align/__init__.py
```

### Summary of All Patches

| File | Change | Reason |
|------|--------|--------|
| `src/piper/train/__main__.py` | Add `pathlib.PosixPath` to safe globals | PyTorch 2.6 weights_only default |
| `src/piper/phonemize_espeak.py` | Point `ESPEAK_DATA_DIR` to system path | Bundled espeak-ng fails for Gujarati |
| `src/piper/train/vits/monotonic_align/__init__.py` | Fix import `.monotonic_align.core` → `.core` | Wrong nested import path |
| `src/piper/train/vits/monotonic_align/` | Build `core.so` manually with cython+gcc | `setup.py build_ext` has output path issues |

---

## 6. Training

### Base Checkpoint — NOT Compatible

The old Piper checkpoints (from `rhasspy/piper-checkpoints`) are incompatible with `piper1-gpl`. They contain `model.sample_bytes` which the new repo doesn't recognize. Even `--weights_only false` doesn't help.

**Solution:** Train from scratch. With 26 hours of data, this is totally viable.

### Training Command

```bash
cd ~/piper1-gpl

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m piper.train fit \
    --data.voice_name "gu_IN-rasa-medium" \
    --data.csv_path ~/rasa_ljspeech/metadata.csv \
    --data.audio_dir ~/rasa_ljspeech/wavs/ \
    --model.sample_rate 22050 \
    --data.espeak_voice "gu" \
    --data.cache_dir ~/piper_cache/ \
    --data.config_path ~/piper_output/config.json \
    --trainer.max_epochs 300 \
    --trainer.devices 1 \
    --trainer.accelerator gpu \
    --trainer.precision 32 \
    --data.batch_size 8 \
    --data.num_workers 4 \
    --trainer.check_val_every_n_epoch 10
```

### Hyperparameter Notes

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | **8** | batch_size=16 OOMs on longer clips. batch_size=32 OOMs at step ~119. batch_size=8 is stable and actually faster (2 it/s vs 1.1 it/s). |
| precision | **32** | fp16-mixed (`16-mixed`) is SLOWER for VITS (0.48 it/s vs 2 it/s) because VITS uses complex STFT operations incompatible with half precision. |
| num_workers | **4** | Workers ARE running (confirmed via `ps aux`). Speed is GPU-bound, not data-bound. |
| PYTORCH_CUDA_ALLOC_CONF | `expandable_segments:True` | Prevents CUDA memory fragmentation. Without it, reserved-but-unallocated memory causes OOM. |
| check_val_every_n_epoch | **10** | Checkpoints saved every 10 epochs. Lightning overwrites previous checkpoint (keeps only latest). |

### OOM Debugging History

| batch_size | VRAM | Result |
|------------|------|--------|
| 32 | ~22 GB | OOM at step ~0 (flow model) |
| 16 | ~20 GB | OOM at step ~119 (long audio spike) |
| 8 | ~12-14 GB | Stable, 2.02 it/s |

### Training Speed

```
14,217 utterances processed
1,599 steps per epoch at batch_size=8
~2.0 it/s on L4 GPU
~13 minutes per epoch
```

| Epoch | Time | Expected Quality |
|-------|------|-----------------|
| 10 | ~2 hours | Noise/babble |
| 50 | ~11 hours | Recognizable Gujarati phonemes |
| 100 | ~22 hours | Rough but intelligible |
| 144 | ~30 hours | Decent quality (where we stopped) |
| 200 | ~43 hours | Good quality |
| 500 | ~4.5 days | High quality |

### Training Metrics at Epoch 57

```
Val Loss Trend:
  epoch~  9: val_loss=35.3776
  epoch~ 19: val_loss=33.0964
  epoch~ 29: val_loss=33.3922
  epoch~ 39: val_loss=37.0717  (spike — normal for GAN)
  epoch~ 49: val_loss=34.0209

Generator loss:  32-37 (oscillating, healthy)
Discriminator loss: 1.6-3.0 (balanced, healthy)
```

**Healthy training indicators:**
- Val loss fluctuating but not consistently rising → no overfitting
- Generator and discriminator losses balanced (neither dominating)
- Discriminator NOT dropping to 0 (would overpower generator)
- Discriminator NOT above 10+ (generator winning too easily)

### Reading TensorBoard

```bash
tensorboard --logdir ~/piper1-gpl/lightning_logs/ --port 6006

# Or read programmatically:
python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('lightning_logs/version_6/')
ea.Reload()
for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    last = events[-1]
    print(f'{tag}: step={last.step}, value={last.value:.4f}')
"
```

---

## 7. Testing & Export

### Export Checkpoint to ONNX

Piper inference requires ONNX format, not raw `.ckpt`.

```bash
cd ~/piper1-gpl

CKPT=$(ls -t lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
EPOCH=$(echo "$CKPT" | grep -oP 'epoch=\K[0-9]+')
echo "Exporting epoch $EPOCH"

python3 -m piper.train.export_onnx \
    --checkpoint "$CKPT" \
    --output-file ~/piper_test/gu_epoch${EPOCH}.onnx

cp ~/piper_output/config.json ~/piper_test/gu_epoch${EPOCH}.onnx.json
```

### Test Inference

```bash
echo "ગુજરાતી ભાષામાં આ પ્રથમ પરીક્ષણ છે" | \
    python3 -m piper \
        --model ~/piper_test/gu_epoch${EPOCH}.onnx \
        --config ~/piper_test/gu_epoch${EPOCH}.onnx.json \
        --output_file ~/piper_test/test1.wav

echo "નમસ્તે, તમે કેમ છો" | \
    python3 -m piper \
        --model ~/piper_test/gu_epoch${EPOCH}.onnx \
        --config ~/piper_test/gu_epoch${EPOCH}.onnx.json \
        --output_file ~/piper_test/test2.wav

du -h ~/piper_test/*.wav
```

### Model Size

- Raw checkpoint: ~281 MB (generator 23.7M + discriminator 46.7M params)
- **Exported ONNX: ~63 MB** (generator only — discriminator discarded at export)
- At inference, only the text encoder + flow + duration predictor + HiFi-GAN decoder are kept

---

## 8. Upload to HuggingFace

```python
from huggingface_hub import HfApi
import os

api = HfApi()
home = os.path.expanduser('~')
epoch = '144'  # adjust to your epoch

# Create private repo (first time only)
try:
    api.create_repo('Arjun4707/piper-gujarati-male', repo_type='model', private=True)
except:
    pass

# Upload ONNX model + config
api.upload_file(
    path_or_fileobj=os.path.join(home, 'piper_test', f'gu_epoch{epoch}.onnx'),
    path_in_repo=f'gu_epoch{epoch}.onnx',
    repo_id='Arjun4707/piper-gujarati-male'
)
api.upload_file(
    path_or_fileobj=os.path.join(home, 'piper_test', f'gu_epoch{epoch}.onnx.json'),
    path_in_repo=f'gu_epoch{epoch}.onnx.json',
    repo_id='Arjun4707/piper-gujarati-male'
)

# Upload raw checkpoint for resume training later
CKPT = os.popen("ls -t ~/piper1-gpl/lightning_logs/version_*/checkpoints/*.ckpt | head -1").read().strip()
api.upload_file(
    path_or_fileobj=CKPT,
    path_in_repo=f'checkpoints/epoch{epoch}.ckpt',
    repo_id='Arjun4707/piper-gujarati-male'
)
```

---

## 9. Inference on Local Machine

### Mac / Linux (CPU only — no GPU needed)

```bash
pip install piper-tts

# Download from HuggingFace
huggingface-cli download Arjun4707/piper-gujarati-male --local-dir ./piper-gu

# Generate speech
echo "ગુજરાતી ભાષામાં આ પ્રથમ પરીક્ષણ છે" | \
    piper --model piper-gu/gu_epoch144.onnx \
          --config piper-gu/gu_epoch144.onnx.json \
          --output_file test.wav

# Play (Mac)
afplay test.wav
```

---

## 10. Resume Training

When you get GPU access again, download the checkpoint from HuggingFace and resume:

```bash
# Download checkpoint
huggingface-cli download Arjun4707/piper-gujarati-male \
    checkpoints/epoch144.ckpt \
    --local-dir ~/piper_resume

# Clone and setup repo (if new studio)
cd ~
git clone https://github.com/OHF-Voice/piper1-gpl.git
cd piper1-gpl
pip install -e '.[train]' --break-system-packages
pip install scikit-build cmake ninja cython --break-system-packages

# Apply all patches (Section 5)
sed -i '2i import pathlib\ntorch.serialization.add_safe_globals([pathlib.PosixPath])' src/piper/train/__main__.py

sed -i 's|ESPEAK_DATA_DIR = .*|ESPEAK_DATA_DIR = Path("/usr/lib/x86_64-linux-gnu/espeak-ng-data")|' src/piper/phonemize_espeak.py

# Build monotonic align
cd src/piper/train/vits/monotonic_align
cython core.pyx
gcc -shared -O2 -fPIC $(python3-config --includes) -o core.cpython-312-x86_64-linux-gnu.so core.c
sed -i 's/from .monotonic_align.core import maximum_path_c/from .core import maximum_path_c/' __init__.py
cd ~/piper1-gpl

# Build espeakbridge
python3 setup.py build_ext --inplace

# Re-download and convert dataset (if not cached)
python ~/01_download_rasa_gujarati.py --gender Male

# Resume training from checkpoint
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m piper.train fit \
    --data.voice_name "gu_IN-rasa-medium" \
    --data.csv_path ~/rasa_ljspeech/metadata.csv \
    --data.audio_dir ~/rasa_ljspeech/wavs/ \
    --model.sample_rate 22050 \
    --data.espeak_voice "gu" \
    --data.cache_dir ~/piper_cache/ \
    --data.config_path ~/piper_output/config.json \
    --trainer.max_epochs 500 \
    --trainer.devices 1 \
    --trainer.accelerator gpu \
    --trainer.precision 32 \
    --data.batch_size 8 \
    --data.num_workers 4 \
    --trainer.check_val_every_n_epoch 10 \
    --ckpt_path ~/piper_resume/checkpoints/epoch144.ckpt
```

---

## 11. Key Lessons

### Lightning.ai Specifics

1. **No venv** — always use `--break-system-packages`
2. **GPU must be enabled manually** in studio settings
3. **`/dev/shm` is limited** — doesn't affect training but good to know
4. **Studios can stop unexpectedly** — always upload checkpoints to HuggingFace
5. **Home directory persists** across studio restarts (same studio)

### PiperTTS / VITS Specifics

1. **Use `piper1-gpl`**, not the old `rhasspy/piper` — old repo is archived, `piper-phonemize` has no Python 3.12 wheels
2. **VITS is slow to train** — it's training 2 networks adversarially (generator + discriminator). 2 it/s on L4 is normal.
3. **fp16-mixed is SLOWER** for VITS — complex STFT operations don't play well with half precision. Stick with fp32.
4. **batch_size=8 is the sweet spot for L4** — bigger batches OOM on longer audio clips due to variable-length sequences
5. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** prevents fragmentation OOM
6. **Lightning overwrites checkpoints** — only the latest is kept. Export ONNX at milestones.
7. **Old Piper checkpoints are incompatible** with piper1-gpl (different hyperparameter schema)
8. **Gujarati espeak-ng works** but needs the system data path, not the bundled one

### Data Preparation

1. **HuggingFace `datasets` + `torchcodec`** — always use `ds.data` (Arrow table) to bypass audio decoding
2. **Duration column in Rasa is a string** — cast with `float()`
3. **Rasa is gated** — requires HF login + terms acceptance + token with gated repo permission
4. **SPIRE dataset** (IISc) — human-curated, skip diarization and CPS filtering. But our download was incomplete (1,788 of 8,242 WAVs).

### Training Quality

1. **Val loss for VITS is noisy** — GAN training inherently fluctuates. Use your ears, not just the numbers.
2. **Balanced adversarial losses = healthy** — disc_loss 1.5-3.0, gen_loss 32-37 at our scale
3. **Overfitting unlikely before epoch 300** with 26 hours of varied data (multiple styles + emotions)
4. **Recognizable Gujarati at epoch 50**, decent quality at epoch 144

---

## 12. Architecture Reference

### VITS Model (what Piper uses)

```
Total params: 70.4M
├── Generator (model_g): 23.7M params
│   ├── Text Encoder — phonemes → hidden representations
│   ├── Duration Predictor — phoneme → frame alignment
│   ├── Posterior Encoder — spectrogram → latent z (training only)
│   ├── Normalizing Flow — z ↔ z_p (reversible transform)
│   ├── Monotonic Alignment Search — hard text↔audio alignment (training only)
│   └── HiFi-GAN Decoder — latent z → raw waveform
│
└── Discriminator (model_d): 46.7M params
    └── Multi-Period Discriminator — 5 sub-discriminators (periods 2,3,5,7,11)
```

### Training Step (each iteration)

1. Load batch (16 phoneme+spectrogram pairs)
2. Generator forward pass (text encoder → posterior encoder → flow → MAS → HiFi-GAN)
3. Discriminator judges real vs generated waveforms
4. Compute 4 losses: mel reconstruction, KL divergence, generator adversarial, discriminator adversarial
5. Update 2 optimizers: one for generator, one for discriminator

### At Inference (ONNX export)

Posterior encoder and discriminator are **discarded**. Only kept:
- Text encoder
- Duration predictor
- Normalizing flow
- HiFi-GAN decoder

This is why ONNX is ~63MB vs ~281MB training model.

---

## File Inventory

| File | Purpose |
|------|---------|
| `01_download_rasa_gujarati.py` | Download, analyze, convert Rasa dataset to LJSpeech format |
| `02_train_rasa_gujarati.sh` | Training launcher script |
| `03_test_and_export.sh` | Test inference + ONNX export |
| `~/rasa_ljspeech/` | Converted training data (22050Hz WAVs + metadata.csv) |
| `~/rasa_ljspeech_eval/` | Evaluation data (test split) |
| `~/piper_cache/` | Phonemized utterance cache (reusable across runs) |
| `~/piper_output/config.json` | Voice config (phoneme map, sample rate, etc.) |
| `~/piper1-gpl/lightning_logs/version_*/` | Training logs + checkpoints |
| `~/piper_test/gu_epoch*.onnx` | Exported ONNX models at various epochs |

---

## Citation

```
@misc{ai4bharat2024rasa,
    author={Praveen Srinivasa Varadhan and Ashwin Sankar and Giri Raju and Mitesh M. Khapra},
    title={Rasa: Building Expressive Speech Synthesis Systems for Indian Languages in Low-resource Settings},
    year={2024},
    booktitle={Proc. Interspeech}
}
```
