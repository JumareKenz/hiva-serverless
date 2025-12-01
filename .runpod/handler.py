import runpod
import os

# vLLM is already running on port 8000 inside the container
# (started by your Dockerfile CMD or entrypoint)
# We just proxy every request straight to it

def handler(job):
    # job["input"] contains everything the user sent to /v1/chat/completions, /v1/completions, etc.
    # RunPod automatically forwards it to your container port 8000
    # → nothing else to do
    return job  # not used because we let RunPod proxy directly

# This tells RunPod to simply forward all HTTP traffic to port 8000
runpod.serverless.start({
    "handler": handler,
    "forward": {"port": 8000}
})
