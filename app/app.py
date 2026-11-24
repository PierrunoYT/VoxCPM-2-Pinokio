import os
import re

# Disable torch.compile completely to avoid Dynamo errors on Windows
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
# Disable dynamo before any model imports.
# NOTE: torch._dynamo.disable() with no argument only returns a decorator, it does
# not disable anything globally -- the config flag is what actually turns it off.
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import gradio as gr
import soundfile as sf
import numpy as np
from voxcpm import VoxCPM
from funasr import AutoModel
import tempfile
import time

# Generated audio is written here so the files are grouped and easy to clean up,
# rather than being scattered through the system temp root.
_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "voxcpm2_outputs")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# Nothing else ever deletes these, so drop outputs left over from earlier
# sessions. Gradio copies results into its own cache before serving them, and
# the age cutoff leaves files from a concurrently running instance alone.
_OUTPUT_MAX_AGE_S = 24 * 60 * 60


def _purge_old_outputs():
    cutoff = time.time() - _OUTPUT_MAX_AGE_S
    for name in os.listdir(_OUTPUT_DIR):
        path = os.path.join(_OUTPUT_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


_purge_old_outputs()

# Global models
model = None
asr_model = None

def _pick_device():
    """CUDA, then Apple Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models():
    global model, asr_model
    print("Loading VoxCPM2...")
    # torch.compile is disabled globally (see top of file), so optimize=True
    # would only add a pointless warm-up generation at startup.
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", optimize=False)
    print(f"VoxCPM2 loaded! Sample rate: {model.tts_model.sample_rate}Hz")

    # ASR only auto-fills the Ultimate Cloning transcript, so a failure here
    # must not take the whole app down -- users can still type it manually.
    print("Loading SenseVoice ASR model...")
    try:
        asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            device=_pick_device(),
        )
        print("SenseVoice loaded!")
    except Exception as e:
        asr_model = None
        print(f"SenseVoice failed to load, auto-transcription disabled: {e}")

def _to_numpy(wav):
    """Model outputs may be torch tensors; soundfile needs numpy."""
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav)


def _write_wav(wav):
    """Write audio to a uniquely named temp file.

    tempfile.mktemp() is deprecated and racy -- mkstemp() creates the file
    atomically. We close the descriptor immediately so soundfile can reopen it
    by path, which is safe because the name is already reserved by us.
    """
    fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="voxcpm2_", dir=_OUTPUT_DIR)
    os.close(fd)
    sf.write(output_path, _to_numpy(wav), model.tts_model.sample_rate)
    return output_path


# SenseVoice prefixes its output with tags such as <|en|><|NEUTRAL|><|Speech|>.
_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")


def _transcribe(audio_path):
    """Return SenseVoice's transcript of audio_path, or "" if none is available."""
    if asr_model is None:
        return ""
    try:
        print("Running ASR on reference audio...")
        res = asr_model.generate(input=audio_path, language="auto", use_itn=True)
    except Exception as e:
        print(f"ASR error: {e}")
        return ""
    text = _SENSEVOICE_TAG.sub("", res[0].get("text") or "").strip() if res else ""
    if not text:
        print("ASR returned no text.")
        return ""
    print(f"ASR result: {text[:60]}...")
    return text


def transcribe_audio(checked, audio_path):
    """Auto-transcribe reference audio using SenseVoice (only when ultimate cloning is ON)"""
    if not checked:
        return gr.update()
    # Reference audio was cleared -- drop the stale transcript instead of
    # leaving the previous clip's text behind.
    if audio_path is None:
        return gr.update(value="")
    if asr_model is None:
        # Don't put the warning in the textbox: its contents are sent to the
        # model as the reference transcript.
        gr.Warning("ASR model not loaded -- type the transcript manually.")
        return gr.update(value="")
    return gr.update(value=_transcribe(audio_path))


def _build_generation_kwargs(
    text,
    control_instruction,
    reference_audio,
    prompt_text,
    use_ultimate_cloning,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
):
    """Assemble the kwargs shared by both the standard and streaming paths.

    Returns (kwargs, mode). Raises ValueError with a user-facing message when
    Ultimate Cloning can't run, rather than silently falling back to a
    different mode.
    """
    text = text.strip()
    control = (control_instruction or "").strip()

    kwargs = dict(
        cfg_value=float(cfg_value),
        inference_timesteps=int(inference_timesteps),
        normalize=normalize,
        denoise=denoise,
    )

    if use_ultimate_cloning:
        if reference_audio is None:
            raise ValueError("Ultimate Cloning needs reference audio.")
        transcript = (prompt_text or "").strip()
        if not transcript:
            # The auto-transcription may not have finished when this request
            # was submitted, or it failed -- transcribe now.
            transcript = _transcribe(reference_audio)
        if not transcript:
            raise ValueError(
                "Ultimate Cloning needs a transcript of the reference audio, "
                "and ASR produced none -- type it manually."
            )
        # The control instruction is ignored in this mode; the reference is
        # passed as both the continuation prompt and the cloning reference.
        kwargs.update(
            text=text,
            reference_wav_path=reference_audio,
            prompt_wav_path=reference_audio,
            prompt_text=transcript,
        )
        return kwargs, "ultimate cloning"

    # Prepend control instruction in parentheses (for voice design / controllable cloning)
    kwargs["text"] = f"({control}){text}" if control else text
    if reference_audio is not None:
        kwargs["reference_wav_path"] = reference_audio
        return kwargs, "controllable cloning"
    return kwargs, "voice design" if control else "basic"


def _validate_request(text):
    """Return an error status string, or None when the request is usable."""
    if model is None:
        return "⚠️ Model not loaded yet."
    if not text or text.strip() == "":
        return "⚠️ Please enter some text to synthesize."
    return None


def generate_speech(
    text,
    control_instruction,
    reference_audio,
    prompt_text,
    use_ultimate_cloning,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
):
    """Generate speech using VoxCPM2"""
    error = _validate_request(text)
    if error:
        return None, error

    try:
        kwargs, mode = _build_generation_kwargs(
            text,
            control_instruction,
            reference_audio,
            prompt_text,
            use_ultimate_cloning,
            cfg_value,
            inference_timesteps,
            normalize,
            denoise,
        )
    except ValueError as e:
        return None, f"⚠️ {e}"

    try:
        wav = model.generate(**kwargs)
        output_path = _write_wav(wav)
        return output_path, f"✅ Generated ({mode})! Sample rate: {model.tts_model.sample_rate}Hz"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return None, error_msg


def generate_streaming_speech(
    text,
    control_instruction,
    reference_audio,
    prompt_text,
    use_ultimate_cloning,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
):
    """Generate speech using VoxCPM2 in streaming mode.

    A generator so the status box updates as chunks arrive, instead of the UI
    sitting silent until the whole clip is done.
    """
    error = _validate_request(text)
    if error:
        yield None, error
        return

    try:
        kwargs, mode = _build_generation_kwargs(
            text,
            control_instruction,
            reference_audio,
            prompt_text,
            use_ultimate_cloning,
            cfg_value,
            inference_timesteps,
            normalize,
            denoise,
        )
    except ValueError as e:
        yield None, f"⚠️ {e}"
        return

    try:
        sample_rate = model.tts_model.sample_rate
        chunks = []
        samples = 0
        for chunk in model.generate_streaming(**kwargs):
            chunk = _to_numpy(chunk)
            chunks.append(chunk)
            samples += chunk.shape[-1]
            yield gr.update(), f"⏳ Streaming… {samples / sample_rate:.1f}s of audio generated"
        if not chunks:
            yield None, "⚠️ Streaming generation produced no audio."
            return
        wav = np.concatenate(chunks)

        output_path = _write_wav(wav)

        yield output_path, f"✅ Generated ({mode}, streaming)! Sample rate: {sample_rate}Hz"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        yield None, error_msg


def on_ultimate_toggle(checked):
    """Toggle UI when ultimate cloning mode changes"""
    if checked:
        # Only lock the control instruction; _build_generation_kwargs already
        # ignores it in this mode, and keeping the text means toggling back
        # off doesn't lose what the user typed.
        return (
            gr.update(visible=True),
            gr.update(interactive=False),
        )
    return (
        gr.update(visible=False),
        gr.update(interactive=True),
    )

# Create Gradio interface
_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

_CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
}
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

# Gradio 6 takes theme/css in launch(); passing them to Blocks is deprecated.
with gr.Blocks(title="VoxCPM 2 - Text-to-Speech") as demo:
    gr.Markdown("""
    # 🎙️ VoxCPM 2 - Tokenizer-Free TTS
    
    **2B params · 48kHz · 30 languages · Voice Design · Voice Cloning**
    
    **Three modes of speech generation:**
    - 🎨 **Voice Design** — Describe the voice you want (no reference audio needed)
    - 🎛️ **Controllable Cloning** — Clone a voice with optional style guidance
    - 🎙️ **Ultimate Cloning** — Auto-transcribe reference audio for maximum fidelity
    """)

    with gr.Row():
        with gr.Column():
            # Reference audio
            reference_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="🎤 Reference Audio (optional — upload for cloning)",
            )

            # Ultimate cloning toggle
            use_ultimate_cloning = gr.Checkbox(
                value=False,
                label="🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
                info="Auto-transcribes reference audio for maximum vocal fidelity. Disables Control Instruction.",
                elem_classes=["switch-toggle"],
            )
            gr.Markdown(
                "⚠️ **Note:** Ultimate Cloning works by continuing from the reference audio. "
                "This may produce a brief artifact at the start of the output. "
                "For clean results, use **Controllable Cloning** instead (reference audio + Control Instruction, without Ultimate Cloning).",
            )

            # Prompt text (auto-transcribed, shown only in ultimate mode)
            prompt_text = gr.Textbox(
                value="",
                label="Transcript of Reference Audio (auto-filled via ASR, editable)",
                placeholder="The transcript will appear here after toggling Ultimate Cloning...",
                lines=2,
                visible=False,
            )

            # Control instruction
            control_instruction = gr.Textbox(
                value="",
                label="🎛️ Control Instruction (optional — voice design / style control)",
                placeholder="e.g. A warm young woman / Excited and fast-paced / Male, deep voice, calm",
                lines=2,
            )

            # Target text
            text_input = gr.Textbox(
                value="VoxCPM2 brings multilingual support, creative voice design, and controllable voice cloning.",
                label="✍️ Target Text — the content to speak",
                lines=4,
                max_lines=10,
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                denoise_checkbox = gr.Checkbox(
                    value=False,
                    label="Reference audio enhancement",
                    info="Apply ZipEnhancer denoising to the reference audio before cloning",
                    elem_classes=["switch-toggle"],
                )
                normalize_checkbox = gr.Checkbox(
                    value=False,
                    label="Text normalization",
                    info="Normalize numbers, dates, and abbreviations",
                    elem_classes=["switch-toggle"],
                )
                cfg_value_slider = gr.Slider(
                    minimum=1.0, maximum=3.0, value=2.0, step=0.1,
                    label="CFG (guidance scale)",
                    info="Higher → closer to the prompt/reference; lower → more creative",
                )
                inference_steps_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="LocDiT flow-matching steps",
                    info="More steps → better quality, but slower",
                )

            with gr.Row():
                generate_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg")
                generate_streaming_btn = gr.Button("⚡ Streaming", variant="secondary", size="lg")

        with gr.Column():
            output_audio = gr.Audio(label="Generated Audio")
            status_output = gr.Textbox(label="Status", lines=2)

    # Info section
    gr.Markdown("""
    ---
    ## 💡 Voice Description Examples
    
    | Control Instruction | Target Text |
    |---|---|
    | *A young girl with a soft, sweet voice. Speaks slowly with a melancholic tone.* | *I never asked you to stay… But why does it still hurt?* |
    | *Relaxed young male voice, slightly nasal, lazy drawl, very casual.* | *Dude, did you see that set? The waves are totally gnarly today.* |
    | *Male, deep voice, calm, slow pace* | *The universe doesn't owe you an explanation.* |
    | *Child, cheerful, energetic* | *Let's go to the park! I want ice cream!* |
    
    **Model**: VoxCPM 2 (2B params, 48kHz, ~8GB VRAM) | **License**: Apache-2.0
    """)

    # Wire up ultimate cloning toggle
    use_ultimate_cloning.change(
        fn=on_ultimate_toggle,
        inputs=[use_ultimate_cloning],
        outputs=[prompt_text, control_instruction],
    ).then(
        fn=transcribe_audio,
        inputs=[use_ultimate_cloning, reference_audio],
        outputs=[prompt_text],
    )

    # Re-transcribe whenever the reference audio changes while Ultimate Cloning is on
    reference_audio.change(
        fn=transcribe_audio,
        inputs=[use_ultimate_cloning, reference_audio],
        outputs=[prompt_text],
    )

    # Generate
    generate_btn.click(
        fn=generate_speech,
        inputs=[
            text_input,
            control_instruction,
            reference_audio,
            prompt_text,
            use_ultimate_cloning,
            cfg_value_slider,
            inference_steps_slider,
            normalize_checkbox,
            denoise_checkbox,
        ],
        outputs=[output_audio, status_output],
        show_progress=True,
        api_name="generate",
    )

    # Streaming
    generate_streaming_btn.click(
        fn=generate_streaming_speech,
        inputs=[
            text_input,
            control_instruction,
            reference_audio,
            prompt_text,
            use_ultimate_cloning,
            cfg_value_slider,
            inference_steps_slider,
            normalize_checkbox,
            denoise_checkbox,
        ],
        outputs=[output_audio, status_output],
        show_progress=True,
        api_name="generate_streaming",
    )

if __name__ == "__main__":
    load_models()

    # The launcher's icon.png sits in the repo root, one level above app/.
    _favicon = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icon.png")
    if not os.path.isfile(_favicon):
        _favicon = None
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        favicon_path=_favicon,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )
