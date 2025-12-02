# Working, clean & simple – tested December 2025
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Basic system packages
RUN apt-get update -y && \
    apt-get install -y python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Fix CUDA compatibility linking (needed on some RunPod GPUs)
RUN ldconfig /usr/local/cuda-12.1/compat/

# Upgrade pip with cache for speed
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip

# Install vLLM 0.11.0 + FlashInfer (fastest combo right now)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install vllm==0.11.0

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# Persistent directories (model survives scaling to zero
RUN mkdir -p /runpod-volume/huggingface-cache/hub && \
    mkdir -p /runpod-volume/huggingface-cache/datasets

# Point Hugging Face cache to persistent volume + enable fast download
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 8000

# Start vLLM directly – model downloads once on first request
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", \
     "--quantization", "gptq", \
     "--dtype", "float16", \
     "--max-model-len", "32768", \
     "--gpu-memory-utilization", "0.94", \
     "--enable-chunked-prefill", \
     "--disable-log-requests", \
     "--port", "8000", \
     "--host", "0.0.0.0"]
