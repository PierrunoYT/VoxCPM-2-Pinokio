# VoxCPM 2 — Pinokio Launcher

A Pinokio 1-click launcher for **VoxCPM 2** — a tokenizer-free Text-to-Speech system with context-aware speech generation, voice cloning, and voice design. 2B parameters, 48kHz output, 30 languages.

## Features

- **Voice Design** — Describe the voice you want with a text instruction (no reference audio needed)
- **Controllable Cloning** — Clone a voice with an optional style/control instruction
- **Ultimate Cloning** — Auto-transcribes reference audio via SenseVoice ASR for maximum vocal fidelity
- **Streaming Mode** — Chunked generation for faster time-to-first-audio
- **Reference audio enhancement** — Optional ZipEnhancer denoising before cloning
- **Text normalization** — Normalize numbers, dates, and abbreviations

## Requirements

- **GPU**: NVIDIA GPU recommended (~8GB VRAM); Apple Silicon and CPU fallback supported
- **Storage**: ~10GB for models

## Installation

Click **Install** in Pinokio. This will:

1. Install Python dependencies via `uv`
2. Install PyTorch for your platform
3. Download `openbmb/VoxCPM2` from Hugging Face
4. Download `iic/speech_zipenhancer_ans_multiloss_16k_base` and `iic/SenseVoiceSmall` from ModelScope

Then click **Start**. Once the terminal prints the Gradio URL, the sidebar shows **Open Web UI**.

## Usage

### Voice Design

Leave Reference Audio empty. Enter a Control Instruction describing the voice, then enter the target text.

```
Control Instruction: A warm young woman, slow and melancholic
Target Text: I never asked you to stay… But why does it still hurt?
```

### Controllable Cloning

Upload a reference audio clip. Optionally add a Control Instruction to steer style. The model clones the uploaded voice with the requested style applied.

### Ultimate Cloning

Upload a reference audio clip and toggle **Ultimate Cloning Mode**. The SenseVoice ASR model auto-transcribes the reference audio. The model continues from the reference for maximum fidelity. Note: a brief artifact may appear at the start of the output due to the continuation approach.

### Advanced Settings

| Parameter | Default | Description |
|---|---|---|
| CFG (guidance scale) | 2.0 | Higher → closer to prompt/reference; lower → more creative |
| LocDiT flow-matching steps | 10 | More steps → better quality, slower generation |
| Text normalization | off | Normalize numbers, dates, abbreviations |
| Reference audio enhancement | off | Apply ZipEnhancer denoising before cloning |

## API

When the app is running, the URL is shown in the Pinokio **Open Web UI** button. Replace `<BASE_URL>` below with that URL (e.g. `http://127.0.0.1:7860`).

Two endpoints are exposed:

- `POST /run/generate` — standard generation
- `POST /run/generate_streaming` — streaming generation

### Python (Gradio client)

```python
from gradio_client import Client

client = Client("<BASE_URL>")

result = client.predict(
    text="VoxCPM2 brings multilingual support and controllable voice cloning.",
    control_instruction="A warm young woman, calm and expressive",
    reference_audio=None,
    prompt_text="",
    use_ultimate_cloning=False,
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=False,
    denoise=False,
    api_name="/generate",
)
audio_path, status = result
print(status)  # ✅ Generated (voice design)! Sample rate: 48000Hz
```

### JavaScript (fetch)

```javascript
const BASE_URL = "<BASE_URL>";

const response = await fetch(`${BASE_URL}/run/generate`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    data: [
      "VoxCPM2 brings multilingual support and controllable voice cloning.",
      "A warm young woman, calm and expressive",
      null,
      "",
      false,
      2.0,
      10,
      false,
      false,
    ],
  }),
});
const result = await response.json();
console.log(result.data[1]); // status message
```

### curl

```bash
curl -s -X POST "<BASE_URL>/run/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      "VoxCPM2 brings multilingual support and controllable voice cloning.",
      "A warm young woman, calm and expressive",
      null,
      "",
      false,
      2.0,
      10,
      false,
      false
    ]
  }'
```

The response `data[0]` contains the generated audio file path and `data[1]` the status string.

## Technical Details

- **Model**: VoxCPM 2 (`openbmb/VoxCPM2`)
- **Sample Rate**: 48kHz
- **Parameters**: ~2B
- **Architecture**: Tokenizer-free TTS on MiniCPM-4 backbone
- **Languages**: 30
- **ASR** (Ultimate Cloning): `iic/SenseVoiceSmall` via FunASR
- **Enhancement** (optional): `iic/speech_zipenhancer_ans_multiloss_16k_base`
- **License**: Apache-2.0

## Troubleshooting

**Model download fails during install**
Re-run **Install** — `hf download` and `snapshot_download` are idempotent.

**GPU memory errors**
Reduce inference timesteps or process shorter text segments.

**Audio sounds robotic at the start (Ultimate Cloning)**
This is expected — switch to Controllable Cloning for clean output without artifacts.

**Sidebar doesn't show "Open Web UI" after Start**
Check the terminal (click **Terminal** in the sidebar) to confirm Gradio printed a URL.

## Resources

- [VoxCPM GitHub](https://github.com/OpenBMB/VoxCPM)
- [Model on Hugging Face](https://huggingface.co/openbmb/VoxCPM2)
- [Gradio Documentation](https://gradio.app/docs/)
