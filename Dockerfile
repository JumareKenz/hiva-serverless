FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Install vLLM (switching back to pip installs since issues that required building fork are fixed and space optimization is not as important since caching) and FlashInfer 
RUN python3 -m pip install vllm==0.11.0 && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# Pass HF token from RunPod UI (for gated Qwen model)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Persistent volume + cache directories (model survives scaling to zero)
RUN mkdir -p /runpod-volume/model && \
    mkdir -p /runpod-volume/huggingface-cache/hub && \
    mkdir -p /runpod-volume/huggingface-cache/datasets

# HuggingFace cache points to persistent volume
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Expose OpenAI-compatible port
EXPOSE 8000

# Start vLLM directly (model downloads on first request – no RUN timeout, no handler.py needed)
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", \
     "--quantization", "gptq", \
     "--dtype", "float16", \
     "--max-model-len", "32768", \
     "--gpu-memory-utilization", "0.94", \
     "--tensor-parallel-size", "1", \
     "--enable-chunked-prefill", \
     "--disable-log-requests", \
     "--port", "8000", \
     "--host", "0.0.0.0"]
