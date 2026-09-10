module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // Keep TorchDynamo off (mirrors app/app.py) so a compile error can
          // never take the daemon down, and let unimplemented MPS ops fall
          // back to CPU on Apple Silicon instead of raising.
          TORCH_COMPILE_DISABLE: "1",
          TORCHDYNAMO_DISABLE: "1",
          PYTORCH_ENABLE_MPS_FALLBACK: "1"
        },
        message: [
          "python app/app.py"
        ],
        on: [{
          "event": "/(http:\/\/\\S+)/",
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
