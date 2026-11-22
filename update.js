module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      // Upgrade the app's own packages but leave torch alone: a blanket -U
      // would pull the latest PyPI torch over the build torch.js installs.
      message: [
        "uv pip install -r app/requirements.txt --upgrade-package voxcpm --upgrade-package gradio --upgrade-package funasr --upgrade-package modelscope --upgrade-package huggingface_hub"
      ],
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env"
      }
    }
  }, {
    method: "script.start",
    params: {
      uri: "download.js"
    }
  }, {
    method: "notify",
    params: {
      html: "Update complete! Click 'Start' to launch VoxCPM 2."
    }
  }]
}
