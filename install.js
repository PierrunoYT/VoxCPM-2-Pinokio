module.exports = {
  run: [
    // Install VoxCPM dependencies from requirements.txt
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Install PyTorch with CUDA support
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          // xformers: true   // VoxCPM doesn't require xformers
        }
      }
    },
    // Pre-download VoxCPM1.5 model
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "huggingface-cli download openbmb/VoxCPM1.5 --local-dir-use-symlinks False && dir"
        ],
      }
    },
    // Optional: Pre-download enhancement models (ZipEnhancer and SenseVoice)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python -c \"from modelscope import snapshot_download; snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base'); snapshot_download('iic/SenseVoiceSmall')\" && dir"
        ],
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch VoxCPM."
      }
    }
  ]
}
