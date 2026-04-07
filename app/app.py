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
import tempfile

# Global model
model = None

def load_model():
    global model
    print("Loading VoxCPM2...")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    print(f"VoxCPM2 loaded! Sample rate: {model.tts_model.sample_rate}Hz")

def generate_speech(
    text,
    prompt_audio,
    prompt_text,
    reference_audio,
    use_prompt_enhancement,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
    retry_badcase,
    retry_badcase_max_times,
    retry_badcase_ratio_threshold
):
    """Generate speech using VoxCPM2"""
    if model is None:
        return None, "⚠️ Model not loaded yet."
    if not text or text.strip() == "":
        return None, "⚠️ Please enter some text to synthesize."

    try:
        prompt_wav_path = prompt_audio if prompt_audio is not None else None
        prompt_text_input = prompt_text.strip() if prompt_text and prompt_text.strip() else None
        reference_wav_path = reference_audio if reference_audio is not None else None

        kwargs = dict(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise if use_prompt_enhancement else False,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        )
        if reference_wav_path:
            kwargs["reference_wav_path"] = reference_wav_path
        if prompt_wav_path:
            kwargs["prompt_wav_path"] = prompt_wav_path
        if prompt_text_input:
            kwargs["prompt_text"] = prompt_text_input

        wav = model.generate(**kwargs)

        output_path = tempfile.mktemp(suffix=".wav")
        sf.write(output_path, wav, model.tts_model.sample_rate)

        return output_path, f"✅ Speech generated! Sample rate: {model.tts_model.sample_rate}Hz"

    except Exception as e:
        error_msg = f"❌ Error generating speech: {str(e)}"
        print(error_msg)
        return None, error_msg

def generate_streaming_speech(
    text,
    prompt_audio,
    prompt_text,
    reference_audio,
    use_prompt_enhancement,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
    retry_badcase,
    retry_badcase_max_times,
    retry_badcase_ratio_threshold
):
    """Generate speech using VoxCPM2 in streaming mode"""
    if model is None:
        return None, "⚠️ Model not loaded yet."
    if not text or text.strip() == "":
        return None, "⚠️ Please enter some text to synthesize."

    try:
        prompt_wav_path = prompt_audio if prompt_audio is not None else None
        prompt_text_input = prompt_text.strip() if prompt_text and prompt_text.strip() else None
        reference_wav_path = reference_audio if reference_audio is not None else None

        kwargs = dict(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise if use_prompt_enhancement else False,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        )
        if reference_wav_path:
            kwargs["reference_wav_path"] = reference_wav_path
        if prompt_wav_path:
            kwargs["prompt_wav_path"] = prompt_wav_path
        if prompt_text_input:
            kwargs["prompt_text"] = prompt_text_input

        chunks = []
        for chunk in model.generate_streaming(**kwargs):
            chunks.append(chunk)
        wav = np.concatenate(chunks)

        output_path = tempfile.mktemp(suffix=".wav")
        sf.write(output_path, wav, model.tts_model.sample_rate)

        return output_path, f"✅ Speech generated (streaming)! Sample rate: {model.tts_model.sample_rate}Hz"

    except Exception as e:
        error_msg = f"❌ Error generating speech: {str(e)}"
        print(error_msg)
        return None, error_msg

# Create Gradio interface
with gr.Blocks(
    title="VoxCPM 2 - Text-to-Speech",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav button {
            font-size: 16px !important;
        }
    """
) as demo:
    gr.Markdown("""
    # 🎙️ VoxCPM 2 - Tokenizer-Free TTS
    
    **2B params · 48kHz · 30 languages · Voice Design · Voice Cloning**
    
    - 🎭 Context-aware, expressive speech generation
    - 🎨 Voice Design — describe the voice you want in natural language
    - 🎤 Zero-shot voice cloning with controllable style
    - 🌍 30 languages supported (no language tag needed)
    - ⚡ Real-time streaming synthesis
    """)

    with gr.Tabs():
        # Tab 1: Basic Generation
        with gr.Tab("🎵 Basic Generation"):
            gr.Markdown("### Generate speech from text. The model infers prosody from content automatically.")
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter your text here...",
                        lines=5,
                        value="VoxCPM2 brings multilingual support, creative voice design, and controllable voice cloning.",
                        max_lines=10,
                        show_copy_button=True
                    )
                    normalize_checkbox = gr.Checkbox(
                        label="Text Normalization",
                        value=False,
                        info="Enable for regular text (numbers, abbreviations). Disable for phoneme input."
                    )
                    generate_btn = gr.Button("🎙️ Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    output_audio = gr.Audio(label="Generated Speech", type="filepath")
                    status_output = gr.Textbox(label="Status", lines=2)

        # Tab 2: Voice Design
        with gr.Tab("🎨 Voice Design"):
            gr.Markdown("""
            ### 🎨 Design a Voice from Description
            Put a voice description in parentheses at the start of your text. The model generates a matching voice — no reference audio needed.
            
            **Example**: `(A young woman, gentle and sweet voice)Hello, welcome!`
            """)
            with gr.Row():
                with gr.Column():
                    text_input_design = gr.Textbox(
                        label="Text with Voice Description",
                        placeholder="(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
                        lines=5,
                        value="(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
                        max_lines=10,
                        show_copy_button=True
                    )
                    gr.Markdown("""
                    **Voice description tips**: gender, age, accent, tone, emotion, pace, pitch...  
                    e.g. `(Male, deep voice, calm, slow pace)`, `(Child, cheerful, energetic)`
                    """)
                    normalize_design = gr.Checkbox(label="Text Normalization", value=False)
                    generate_design_btn = gr.Button("🎨 Generate with Voice Design", variant="primary", size="lg")
                with gr.Column():
                    output_audio_design = gr.Audio(label="Generated Speech", type="filepath")
                    status_output_design = gr.Textbox(label="Status", lines=2)

        # Tab 3: Voice Cloning
        with gr.Tab("🎤 Voice Cloning"):
            gr.Markdown("""
            ### Clone a voice from reference audio
            - **Controllable Cloning**: Provide reference audio, optionally add style control in parentheses
            - **Ultimate Cloning**: Also provide the transcript of the reference audio for maximum fidelity
            """)
            with gr.Row():
                with gr.Column():
                    text_input_clone = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter text... Optionally prepend style in parentheses: (slightly faster, cheerful tone)Your text here",
                        lines=5,
                        value="This is a demonstration of voice cloning with VoxCPM2.",
                        max_lines=10,
                        show_copy_button=True
                    )
                    reference_audio_input = gr.Audio(
                        label="Reference Audio (for cloning)",
                        type="filepath"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Reference Transcript (optional, for ultimate cloning)",
                        placeholder="Exact transcript of the reference audio for maximum fidelity...",
                        lines=3,
                        max_lines=8,
                        show_copy_button=True
                    )
                    use_enhancement = gr.Checkbox(
                        label="Prompt Speech Enhancement",
                        value=False,
                        info="Enable for clean voice (16kHz). Disable for high-quality cloning."
                    )
                    normalize_checkbox_clone = gr.Checkbox(label="Text Normalization", value=False)
                    generate_clone_btn = gr.Button("🎤 Clone Voice", variant="primary", size="lg")
                with gr.Column():
                    output_audio_clone = gr.Audio(label="Generated Speech", type="filepath")
                    status_output_clone = gr.Textbox(label="Status", lines=2)

        # Tab 4: Advanced Settings
        with gr.Tab("⚙️ Advanced Settings"):
            gr.Markdown("### Full control over all generation parameters.")
            with gr.Row():
                with gr.Column():
                    text_input_advanced = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter your text here...",
                        lines=5,
                        value="Advanced settings allow you to fine-tune the speech generation process.",
                        max_lines=10,
                        show_copy_button=True
                    )
                    reference_audio_advanced = gr.Audio(
                        label="Reference Audio (for cloning)",
                        type="filepath"
                    )
                    prompt_audio_advanced = gr.Audio(
                        label="Prompt Audio (for ultimate cloning — same as reference for max fidelity)",
                        type="filepath"
                    )
                    prompt_text_advanced = gr.Textbox(
                        label="Prompt Text / Transcript",
                        placeholder="Transcript of the prompt audio...",
                        lines=2,
                        max_lines=8,
                        show_copy_button=True
                    )
                    with gr.Accordion("🎛️ Generation Parameters", open=True):
                        cfg_value_slider = gr.Slider(
                            minimum=0.5, maximum=5.0, value=2.0, step=0.1,
                            label="CFG Value",
                            info="Higher = better adherence to prompt. Lower = more relaxed."
                        )
                        inference_steps_slider = gr.Slider(
                            minimum=5, maximum=50, value=10, step=1,
                            label="Inference Timesteps",
                            info="Higher = better quality, slower."
                        )
                        normalize_advanced = gr.Checkbox(label="Text Normalization", value=False)
                        denoise_checkbox = gr.Checkbox(
                            label="Denoise Output", value=False,
                            info="Enable external denoising (may cause distortion, limits to 16kHz)"
                        )
                        use_enhancement_advanced = gr.Checkbox(label="Prompt Speech Enhancement", value=False)
                    with gr.Accordion("🔄 Retry Settings", open=False):
                        retry_badcase = gr.Checkbox(label="Enable Retry for Bad Cases", value=True)
                        retry_max_times = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Maximum Retry Times")
                        retry_ratio_threshold = gr.Slider(
                            minimum=3.0, maximum=10.0, value=6.0, step=0.5,
                            label="Retry Ratio Threshold"
                        )
                    with gr.Row():
                        generate_advanced_btn = gr.Button("🎙️ Generate", variant="primary")
                        generate_streaming_btn = gr.Button("⚡ Generate (Streaming)", variant="secondary")
                with gr.Column():
                    output_audio_advanced = gr.Audio(label="Generated Speech", type="filepath")
                    status_output_advanced = gr.Textbox(label="Status", lines=3)

    # Info section
    gr.Markdown("""
    ---
    ## 👩‍🍳 Quick Tips
    
    - **Voice Design**: Put description in parentheses at start of text: `(gentle female voice)Hello!`
    - **Controllable Cloning**: Reference audio + style description in text
    - **Ultimate Cloning**: Reference audio + its transcript for maximum fidelity
    - **Short Sentences**: Increase CFG value for better clarity
    - **Long Texts**: Lower CFG value for better stability
    - **Fast Draft**: Use lower inference timesteps (5-10)
    - **High Quality**: Use higher inference timesteps (20-50)
    
    ## ⚠️ Important Notes
    
    - Voice Design and style control results may vary — try generating 1-3 times
    - Output quality depends on prompt speech quality for cloning
    - 30 languages supported — no language tag needed, just input text directly
    - Please use responsibly and mark AI-generated content appropriately
    
    **Model**: VoxCPM 2 (2B params, 48kHz, ~8GB VRAM) | **License**: Apache-2.0
    """)

    # Connect buttons — Basic generation
    generate_btn.click(
        fn=generate_speech,
        inputs=[
            text_input,
            gr.State(None), gr.State(None), gr.State(None),
            gr.State(False), gr.State(2.0), gr.State(10),
            normalize_checkbox,
            gr.State(False), gr.State(True), gr.State(3), gr.State(6.0),
        ],
        outputs=[output_audio, status_output]
    )

    # Voice Design
    generate_design_btn.click(
        fn=generate_speech,
        inputs=[
            text_input_design,
            gr.State(None), gr.State(None), gr.State(None),
            gr.State(False), gr.State(2.0), gr.State(10),
            normalize_design,
            gr.State(False), gr.State(True), gr.State(3), gr.State(6.0),
        ],
        outputs=[output_audio_design, status_output_design]
    )

    # Voice Cloning — reference_audio used for both reference_wav_path and prompt_wav_path (ultimate cloning)
    generate_clone_btn.click(
        fn=generate_speech,
        inputs=[
            text_input_clone,
            reference_audio_input,  # prompt_audio (for ultimate cloning)
            prompt_text_input,
            reference_audio_input,  # reference_audio
            use_enhancement,
            gr.State(2.0), gr.State(10),
            normalize_checkbox_clone,
            use_enhancement,
            gr.State(True), gr.State(3), gr.State(6.0),
        ],
        outputs=[output_audio_clone, status_output_clone]
    )

    # Advanced generation
    generate_advanced_btn.click(
        fn=generate_speech,
        inputs=[
            text_input_advanced,
            prompt_audio_advanced,
            prompt_text_advanced,
            reference_audio_advanced,
            use_enhancement_advanced,
            cfg_value_slider, inference_steps_slider,
            normalize_advanced, denoise_checkbox,
            retry_badcase, retry_max_times, retry_ratio_threshold,
        ],
        outputs=[output_audio_advanced, status_output_advanced]
    )

    # Advanced streaming
    generate_streaming_btn.click(
        fn=generate_streaming_speech,
        inputs=[
            text_input_advanced,
            prompt_audio_advanced,
            prompt_text_advanced,
            reference_audio_advanced,
            use_enhancement_advanced,
            cfg_value_slider, inference_steps_slider,
            normalize_advanced, denoise_checkbox,
            retry_badcase, retry_max_times, retry_ratio_threshold,
        ],
        outputs=[output_audio_advanced, status_output_advanced]
    )

if __name__ == "__main__":
    load_model()

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _favicon = None
    for _candidate in (
        os.path.join(_here, "icon.png"),
        os.path.join(_here, "icon.jpg"),
        os.path.join(_root, "icon.jpg"),
        os.path.join(_root, "icon.png"),
    ):
        if os.path.isfile(_candidate):
            _favicon = _candidate
            break
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        show_api=True,
        favicon_path=_favicon,
    )
