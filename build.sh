#!/bin/bash

set -e

echo "Building PaddleOCR Local Docker images..."

if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "Warning: Docker GPU support was not detected. PaddleOCR-VL may not run correctly."
    read -r -p "Continue? (y/N) " reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [ ! -f "env.txt" ]; then
    echo "env.txt does not exist. Creating RTX 50 / Blackwell defaults..."
    cat > env.txt << EOF
API_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
VLM_BACKEND=vllm
VLM_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
PANDOCR_GPU_DEVICE_ID=0
PADDLEOCR_VL_MODEL_NAME=PaddleOCR-VL-1.6-0.9B
PPOCR_V6_MODEL_NAME=PP-OCRv6_medium
PANDOCR_MODEL_CONTROL=docker
PANDOCR_ACTIVE_MODEL_ON_START=paddleocr-vl-1.6
PANDOCR_MODEL_SWITCH_TIMEOUT=1200
PADDLE_REQUEST_TIMEOUT=3600
PANDOCR_CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
PANDOCR_MAX_UPLOAD_MB=512
PANDOCR_API_TOKEN=
PANDOCR_ENABLE_API_DOCS=0
EOF
fi

echo "Pulling PaddleOCR-VL images..."
docker compose --env-file env.txt pull paddleocr-vlm-server paddleocr-vl-api

echo "Building local images..."
docker compose --env-file env.txt build paddleocr-ocr-api pandocr-web

echo "Build complete."
echo ""
echo "Next:"
echo "  docker compose --env-file env.txt up -d --no-start"
echo "  docker compose --env-file env.txt start pandocr-web"
echo "  docker compose --env-file env.txt logs -f"
echo "  docker compose --env-file env.txt down"
