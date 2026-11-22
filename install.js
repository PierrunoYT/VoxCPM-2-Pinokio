module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r app/requirements.txt"
        ],
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env"
        }
      }
    },
    {
      method: "script.start",
      params: {
        uri: "download.js"
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch VoxCPM 2."
      }
    }
  ]
}
