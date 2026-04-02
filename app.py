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

# Initialize the model
print("Loading VoxCPM model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
print("Model loaded successfully!")

def generate_speech(
    text,
    prompt_audio,
    prompt_text,
    use_prompt_enhancement,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
    retry_badcase,
    retry_badcase_max_times,
    retry_badcase_ratio_threshold
):
    """Generate speech using VoxCPM"""
    
    if not text or text.strip() == "":
        return None, "⚠️ Please enter some text to synthesize."
    
    try:
        # Prepare prompt audio path and text (must both be provided or both be None)
        prompt_wav_path = prompt_audio if prompt_audio is not None else None
        prompt_text_input = prompt_text.strip() if prompt_text and prompt_text.strip() else None
        
        # If only one is provided, drop both and warn
        if (prompt_wav_path is None) != (prompt_text_input is None):
            if prompt_wav_path and not prompt_text_input:
                return None, "⚠️ Please provide the reference text (transcript of your reference audio) for voice cloning."
            else:
                prompt_wav_path = None
                prompt_text_input = None
        
        status_msg = "🎙️ Generating speech..."
        
        # Generate speech
        wav = model.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text_input,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise if use_prompt_enhancement else False,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        )
        
        # Save to temporary file
        output_path = tempfile.mktemp(suffix=".wav")
        sf.write(output_path, wav, model.tts_model.sample_rate)
        
        status_msg = f"✅ Speech generated successfully! Sample rate: {model.tts_model.sample_rate}Hz"
        
        return output_path, status_msg
        
    except Exception as e:
        error_msg = f"❌ Error generating speech: {str(e)}"
        print(error_msg)
        return None, error_msg

def generate_streaming_speech(
    text,
    prompt_audio,
    prompt_text,
    use_prompt_enhancement,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
    retry_badcase,
    retry_badcase_max_times,
    retry_badcase_ratio_threshold
):
    """Generate speech using VoxCPM in streaming mode"""
    
    if not text or text.strip() == "":
        return None, "⚠️ Please enter some text to synthesize."
    
    try:
        # Prepare prompt audio path and text (must both be provided or both be None)
        prompt_wav_path = prompt_audio if prompt_audio is not None else None
        prompt_text_input = prompt_text.strip() if prompt_text and prompt_text.strip() else None
        
        # If only one is provided, drop both and warn
        if (prompt_wav_path is None) != (prompt_text_input is None):
            if prompt_wav_path and not prompt_text_input:
                return None, "⚠️ Please provide the reference text (transcript of your reference audio) for voice cloning."
            else:
                prompt_wav_path = None
                prompt_text_input = None
        
        status_msg = "🎙️ Generating speech (streaming mode)..."
        
        # Generate speech in streaming mode
        chunks = []
        for chunk in model.generate_streaming(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text_input,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise if use_prompt_enhancement else False,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        ):
            chunks.append(chunk)
        
        wav = np.concatenate(chunks)
        
        # Save to temporary file
        output_path = tempfile.mktemp(suffix=".wav")
        sf.write(output_path, wav, model.tts_model.sample_rate)
        
        status_msg = f"✅ Speech generated successfully (streaming)! Sample rate: {model.tts_model.sample_rate}Hz"
        
        return output_path, status_msg
        
    except Exception as e:
        error_msg = f"❌ Error generating speech: {str(e)}"
        print(error_msg)
        return None, error_msg

# Create Gradio interface with improved styling
with gr.Blocks(
    title="VoxCPM - Text-to-Speech",
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
    # 🎙️ VoxCPM - Tokenizer-Free TTS
    
    **Context-Aware Speech Generation and True-to-Life Voice Cloning**
    
    VoxCPM is a novel tokenizer-free Text-to-Speech system that enables:
    - 🎭 Context-aware, expressive speech generation
    - 🎤 True-to-life zero-shot voice cloning
    - ⚡ High-efficiency synthesis (RTF 0.17 on RTX 4090)
    """)
    
    with gr.Tabs():
        # Tab 1: Basic Generation
        with gr.Tab("🎵 Basic Generation"):
            gr.Markdown("""
            ### 🥚 Step 1: Prepare Your Content
            Enter the text you want to synthesize. You can use regular text or phoneme input.
            """)
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter your text here...",
                        lines=5,
                        value="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.",
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
        
        # Tab 2: Voice Cloning
        with gr.Tab("🎤 Voice Cloning"):
            gr.Markdown("""
            ### 🍳 Step 2: Choose Your Voice Style
            Upload a reference audio to clone a specific voice, or leave empty for the model to improvise.
            """)
            
            with gr.Row():
                with gr.Column():
                    text_input_clone = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter your text here...",
                        lines=5,
                        value="This is a demonstration of voice cloning with VoxCPM.",
                        max_lines=10,
                        show_copy_button=True
                    )
                    
                    prompt_audio_input = gr.Audio(
                        label="Reference Audio (Prompt Speech)",
                        type="filepath"
                    )
                    
                    prompt_text_input = gr.Textbox(
                        label="Reference Text (Required for Voice Cloning)",
                        placeholder="Transcript of the reference audio...",
                        lines=3,
                        info="You must provide the transcript of the reference audio for voice cloning to work",
                        max_lines=8,
                        show_copy_button=True
                    )
                    
                    use_enhancement = gr.Checkbox(
                        label="Prompt Speech Enhancement",
                        value=False,
                        info="Enable for clean voice (16kHz). Disable for high-quality cloning (up to 44.1kHz)"
                    )
                    
                    normalize_checkbox_clone = gr.Checkbox(
                        label="Text Normalization",
                        value=False
                    )
                    
                    generate_clone_btn = gr.Button("🎤 Clone Voice", variant="primary", size="lg")
                    
                with gr.Column():
                    output_audio_clone = gr.Audio(label="Generated Speech", type="filepath")
                    status_output_clone = gr.Textbox(label="Status", lines=2)
        
        # Tab 3: Advanced Settings
        with gr.Tab("⚙️ Advanced Settings"):
            gr.Markdown("""
            ### 🧂 Step 3: Fine-Tune Your Results
            Adjust these parameters to control the quality and characteristics of the generated speech.
            """)
            
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
                    
                    prompt_audio_advanced = gr.Audio(
                        label="Reference Audio (Optional)",
                        type="filepath"
                    )
                    
                    prompt_text_advanced = gr.Textbox(
                        label="Reference Text (Optional)",
                        placeholder="Transcript of the reference audio...",
                        lines=2,
                        max_lines=8,
                        show_copy_button=True
                    )
                    
                    with gr.Accordion("🎛️ Generation Parameters", open=True):
                        cfg_value_slider = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            value=2.0,
                            step=0.1,
                            label="CFG Value",
                            info="Higher = better adherence to prompt, but may sound strained. Lower = more relaxed."
                        )
                        
                        inference_steps_slider = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Inference Timesteps",
                            info="Higher = better quality, slower. Lower = faster, draft quality."
                        )
                        
                        normalize_advanced = gr.Checkbox(
                            label="Text Normalization",
                            value=False
                        )
                        
                        denoise_checkbox = gr.Checkbox(
                            label="Denoise Output",
                            value=False,
                            info="Enable external denoising (may cause distortion, limits to 16kHz)"
                        )
                        
                        use_enhancement_advanced = gr.Checkbox(
                            label="Prompt Speech Enhancement",
                            value=False
                        )
                    
                    with gr.Accordion("🔄 Retry Settings", open=False):
                        retry_badcase = gr.Checkbox(
                            label="Enable Retry for Bad Cases",
                            value=True,
                            info="Automatically retry if generation is unstable"
                        )
                        
                        retry_max_times = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Maximum Retry Times"
                        )
                        
                        retry_ratio_threshold = gr.Slider(
                            minimum=3.0,
                            maximum=10.0,
                            value=6.0,
                            step=0.5,
                            label="Retry Ratio Threshold",
                            info="Maximum length restriction for bad case detection"
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
    
    - **Regular Text**: Keep "Text Normalization" ON for natural text with numbers and punctuation
    - **Phoneme Input**: Turn "Text Normalization" OFF for precise pronunciation control
    - **Clean Voice Cloning**: Enable "Prompt Speech Enhancement" (16kHz limit)
    - **High-Quality Cloning**: Disable enhancement for up to 44.1kHz sampling
    - **Short Sentences**: Increase CFG value for better clarity
    - **Long Texts**: Lower CFG value for better stability
    - **Fast Draft**: Use lower inference timesteps (5-10)
    - **High Quality**: Use higher inference timesteps (20-50)
    
    ## ⚠️ Important Notes
    
    - Output quality depends on prompt speech quality
    - Model is trained on Chinese and English data
    - Please use responsibly and mark AI-generated content appropriately
    - For research and development purposes only
    
    **License**: Apache-2.0 | **Model**: VoxCPM1.5 (44.1kHz, 6.25Hz token rate)
    """)
    
    # Connect buttons to functions
    # Basic generation
    generate_btn.click(
        fn=generate_speech,
        inputs=[
            text_input,
            gr.State(None),  # No prompt audio
            gr.State(None),  # No prompt text
            gr.State(False),  # No enhancement
            gr.State(2.0),  # Default CFG
            gr.State(10),  # Default timesteps
            normalize_checkbox,
            gr.State(False),  # No denoise
            gr.State(True),  # Retry enabled
            gr.State(3),  # Max retry times
            gr.State(6.0),  # Retry threshold
        ],
        outputs=[output_audio, status_output]
    )
    
    # Voice cloning
    generate_clone_btn.click(
        fn=generate_speech,
        inputs=[
            text_input_clone,
            prompt_audio_input,
            prompt_text_input,
            use_enhancement,
            gr.State(2.0),
            gr.State(10),
            normalize_checkbox_clone,
            use_enhancement,  # Use enhancement as denoise
            gr.State(True),
            gr.State(3),
            gr.State(6.0),
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
            use_enhancement_advanced,
            cfg_value_slider,
            inference_steps_slider,
            normalize_advanced,
            denoise_checkbox,
            retry_badcase,
            retry_max_times,
            retry_ratio_threshold,
        ],
        outputs=[output_audio_advanced, status_output_advanced]
    )
    
    # Advanced streaming generation
    generate_streaming_btn.click(
        fn=generate_streaming_speech,
        inputs=[
            text_input_advanced,
            prompt_audio_advanced,
            prompt_text_advanced,
            use_enhancement_advanced,
            cfg_value_slider,
            inference_steps_slider,
            normalize_advanced,
            denoise_checkbox,
            retry_badcase,
            retry_max_times,
            retry_ratio_threshold,
        ],
        outputs=[output_audio_advanced, status_output_advanced]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        show_api=True,
        favicon_path="icon.png" if os.path.exists("icon.png") else None
    )