# Piper TTS — Gujarati Male (Edge Deployment)

Lightweight Gujarati TTS model trained from scratch using [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) (VITS architecture), exported to ONNX (~63 MB) for edge deployment.

## Model

Trained model: [Arjun4707/piper-gujarati-male](https://huggingface.co/Arjun4707/piper-gujarati-male)

## Dataset

**AI4Bharat/Rasa Gujarati Male** — 15,797 clips, ~26 hours (CC-BY-4.0, openly licensed).

No YouTube-scraped data was used for this model.

## Quick start

```bash
pip install piper-tts
echo "નમસ્તે, કેમ છો?" | piper --model epoch_144.onnx --output_file output.wav
```

## Training details

See [PiperTTS_Gujarati_Male_Training_Journey.md](PiperTTS_Gujarati_Male_Training_Journey.md) for the complete training journal covering:
- Three required patches for piper1-gpl on Lightning.ai
- espeak-ng Gujarati phoneme setup
- Batch size tuning (fp32 faster than fp16 for VITS)
- ONNX export at milestone epochs

## Project structure

```
├── README.md
├── PiperTTS_Gujarati_Male_Training_Journey.md  ← Complete training journal
├── 01_download_rasa_gujarati.py                ← Dataset download + LJSpeech conversion
├── example_samples/                            ← Generated audio samples
└── ...
```

## License

- **Training scripts**: Apache-2.0
- **Trained model weights**: Apache-2.0 (training data is CC-BY-4.0, no restrictions)
- Free for commercial and non-commercial use

## Author

**Arjun Bhammar** — [HuggingFace](https://huggingface.co/Arjun4707) | [GitHub](https://github.com/BhammarArjun)
