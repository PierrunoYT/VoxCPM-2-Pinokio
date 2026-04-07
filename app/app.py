import os
import sys

# Disable torch.compile completely to avoid Dynamo errors on Windows
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
# Disable dynamo before any model imports
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

import gradio as gr
import soundfile as sf
import numpy as np
from voxcpm import VoxCPM
from funasr import AutoModel
import tempfile

# Global models
model = None
asr_model = None

def load_models():
    global model, asr_model
    print("Loading VoxCPM2...")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", optimize=True)
    print(f"VoxCPM2 loaded! Sample rate: {model.tts_model.sample_rate}Hz")

    print("Loading SenseVoice ASR model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr_model = AutoModel(
        model="iic/SenseVoiceSmall",
        disable_update=True,
        device=device,
    )
    print("SenseVoice loaded!")

def transcribe_audio(checked, audio_path):
    """Auto-transcribe reference audio using SenseVoice (only when ultimate cloning is ON)"""
    if not checked or audio_path is None:
        return gr.update()
    if asr_model is None:
        return gr.update(value="⚠️ ASR model not loaded.")
    try:
        print("Running ASR on reference audio...")
        res = asr_model.generate(input=audio_path, language="auto", use_itn=True)
        text = res[0]["text"].split("|>")[-1]
        print(f"ASR result: {text[:60]}...")
        return gr.update(value=text)
    except Exception as e:
        print(f"ASR error: {e}")
        return gr.update(value="")

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
    if model is None:
        return None, "⚠️ Model not loaded yet."
    if not text or text.strip() == "":
        return None, "⚠️ Please enter some text to synthesize."

    try:
        text = text.strip()
        control = (control_instruction or "").strip()
        reference_wav_path = reference_audio if reference_audio is not None else None
        prompt_text_clean = prompt_text.strip() if prompt_text and prompt_text.strip() else None

        # In ultimate cloning mode, control instruction is disabled
        if use_ultimate_cloning:
            control = ""

        # Prepend control instruction in parentheses (for voice design / controllable cloning)
        final_text = f"({control}){text}" if control else text

        kwargs = dict(
            text=final_text,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            normalize=normalize,
            denoise=denoise,
        )

        if reference_wav_path:
            kwargs["reference_wav_path"] = reference_wav_path

        # Ultimate cloning: pass reference as both prompt and reference
        if use_ultimate_cloning and reference_wav_path and prompt_text_clean:
            kwargs["prompt_wav_path"] = reference_wav_path
            kwargs["prompt_text"] = prompt_text_clean

        wav = model.generate(**kwargs)

        output_path = tempfile.mktemp(suffix=".wav")
        sf.write(output_path, wav, model.tts_model.sample_rate)

        mode = "ultimate cloning" if (use_ultimate_cloning and prompt_text_clean) else \
               "controllable cloning" if reference_wav_path else \
               "voice design" if control else "basic"
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
    """Generate speech using VoxCPM2 in streaming mode"""
    if model is None:
        return None, "⚠️ Model not loaded yet."
    if not text or text.strip() == "":
        return None, "⚠️ Please enter some text to synthesize."

    try:
        text = text.strip()
        control = (control_instruction or "").strip()
        reference_wav_path = reference_audio if reference_audio is not None else None
        prompt_text_clean = prompt_text.strip() if prompt_text and prompt_text.strip() else None

        if use_ultimate_cloning:
            control = ""

        final_text = f"({control}){text}" if control else text

        kwargs = dict(
            text=final_text,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            normalize=normalize,
            denoise=denoise,
        )

        if reference_wav_path:
            kwargs["reference_wav_path"] = reference_wav_path

        if use_ultimate_cloning and reference_wav_path and prompt_text_clean:
            kwargs["prompt_wav_path"] = reference_wav_path
            kwargs["prompt_text"] = prompt_text_clean

        chunks = []
        for chunk in model.generate_streaming(**kwargs):
            chunks.append(chunk)
        wav = np.concatenate(chunks)

        output_path = tempfile.mktemp(suffix=".wav")
        sf.write(output_path, wav, model.tts_model.sample_rate)

        return output_path, f"✅ Generated (streaming)! Sample rate: {model.tts_model.sample_rate}Hz"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return None, error_msg

def on_ultimate_toggle(checked):
    """Toggle UI when ultimate cloning mode changes"""
    if checked:
        return (
            gr.update(visible=True),
            gr.update(interactive=False, value=""),
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

with gr.Blocks(
    title="VoxCPM 2 - Text-to-Speech",
    theme=_APP_THEME,
    css=_CUSTOM_CSS,
) as demo:
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

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _favicon = None
    for _candidate in (
        os.path.join(_here, "icon.png"),
        os.path.join(_here, "icon.jpeg"),
        os.path.join(_here, "icon.jpg"),
        os.path.join(_root, "icon.jpeg"),
        os.path.join(_root, "icon.jpg"),
        os.path.join(_root, "icon.png"),
    ):
        if os.path.isfile(_candidate):
            _favicon = _candidate
            break
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        favicon_path=_favicon,
    )
