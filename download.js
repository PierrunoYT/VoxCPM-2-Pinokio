// Model downloads shared by install.js and update.js.
// Both `hf download` and modelscope's `snapshot_download` are idempotent,
// so re-running this is safe and resumes partial downloads.
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "hf download openbmb/VoxCPM2"
        ],
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python -c \"from modelscope import snapshot_download; snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base'); snapshot_download('iic/SenseVoiceSmall')\""
        ],
      }
    }
  ]
}
