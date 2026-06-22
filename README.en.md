# PaddleOCR Local - PaddleOCR-VL & PP-OCRv6 & MinerU WebUI

**Language / 语言**: [简体中文](README.md) | English

PaddleOCR Local is a lightweight Web frontend for PaddleOCR-VL, PP-OCRv6, and MinerU. The frontend handles file upload, queueing, preview, model switching, and download, while the FastAPI backend serves static files, converts Office files to PDF, and proxies requests. OCR inference runs in separate PaddleOCR services. The NVIDIA path uses official Docker services, and the macOS Apple Silicon path uses local PaddleX/MLX services.

**Four supported models:**

| Model | Use case | Notes |
|-------|----------|-------|
| PaddleOCR-VL 1.6 | Document parsing | Layout analysis, tables, formulas, seal recognition |
| PP-OCRv6 | Text recognition | Lightweight text-only OCR |
| MinerU | Document parsing | MinerU2.5-Pro-2605-1.2B, hybrid engine |
| GLM-OCR (Ollama) | Text recognition | Zhipu GLM-OCR model, requires Ollama |

On single-GPU deployments, the WebUI model selector automatically manages container start/stop, so only one model occupies VRAM at a time.

<img width="1920" height="945" alt="image" src="https://github.com/user-attachments/assets/85a247a0-c796-4a20-b596-1cc4148df964" />

## One-Click Deployment

The default goal of this project is open-source self-hosting: new users should usually run one command, let the script check the environment, install dependencies, start services, and open the WebUI.

macOS Apple Silicon:

```bash
./macos-one-click.command
```

Windows + NVIDIA:

```powershell
.\windows-one-click.bat
```

Before deployment, or after a failed run, use the doctor:

```bash
make doctor
```

macOS uses local PaddlePaddle + PaddleX + optional MLX-VLM. Windows/NVIDIA uses Docker Compose. By default, services bind to localhost for a zero-config local setup.

## Current Architecture

```text
Browser
  -> pandocr-web:8000
       - FastAPI
       - static WebUI
       - Office to PDF conversion
       - PaddleOCR-VL request proxy
       - PP-OCRv6 OCR request proxy
       - MinerU request proxy
       - GLM-OCR (Ollama) request proxy
       - Translation proxy (OpenAI-compatible API)
  -> PaddleOCR services
       - NVIDIA: paddleocr-vl-api + paddleocr-ocr-api + paddleocr-vlm-server in docker compose
       - macOS: local paddlex --serve, optionally with mlx_vlm.server
  -> MinerU service
       - NVIDIA: mineru-api in docker compose (profile: mineru)
  -> GLM-OCR (Ollama)
       - Local or Docker Ollama service
```

The NVIDIA Compose stack keeps five core services + two optional services:

- `pandocr-web`
- `paddleocr-vl-api`
- `paddleocr-ocr-api`
- `paddleocr-vlm-server`
- `mineru-api` (Docker profile `mineru`, must be explicitly enabled)
- `ollama` (Docker profile `glm-ocr`, must be explicitly enabled; or use an existing Ollama instance)

For single-GPU machines, the Docker deployment keeps only one OCR model hot-loaded by default. `pandocr-web` stays online and controls the model containers through the Docker socket: selecting `PaddleOCR-VL 1.6` starts `paddleocr-vlm-server` + `paddleocr-vl-api` and stops other model containers; selecting `PP-OCRv6` or `MinerU` does the reverse; selecting `GLM-OCR` does not require Docker container management — it connects directly to the Ollama service. The top bar UI polls this runtime state in real time, showing whether the model is ready, starting, stopped, or failed.

## Features

- Supports image, PDF, PPT/PPTX, and DOC/DOCX uploads.
- Supports model switching between `PaddleOCR-VL 1.6` document parsing, `PP-OCRv6` text OCR, `MinerU` document parsing, and `GLM-OCR` text recognition; Docker single-GPU deployments start and stop models on demand to avoid multiple models consuming VRAM simultaneously.
- The left "Parsing Settings" panel automatically switches based on the currently selected model: PaddleOCR models show options for layout detection, chart recognition, document rectification, seal recognition, etc.; MinerU shows options for formula parsing, table parsing, image analysis, parsing method, etc.
- The WebUI supports one-click Chinese/English switching, remembers the user's choice, and keeps translations centralized in `static/i18n.js` for future languages.
- **OCR result translation**: Translate OCR Markdown results into 20 languages (Simplified Chinese, Traditional Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Arabic, Italian, Dutch, Polish, Turkish, Vietnamese, Thai, Indonesian, Malay, Hindi). Requires `PANDOCR_TRANSLATE_API_URL` and `PANDOCR_TRANSLATE_API_KEY` (OpenAI API compatible).
- Sends PDFs to PaddleOCR-VL page by page, making it easier to compare with the official online parsing result and reliably keep the raw JSON for each page.
- Renders PP-OCRv6 results with an official-style visual OCR layer: source/result pages stay aligned, scrolling and zooming are synchronized, recognized text can be copied or corrected, and raw JSON remains available.
- Persists parsing tasks locally under `data/tasks/`, so history remains available after refreshing the page. Deleting a task also removes the local record.
- Markdown preview supports horizontally scrollable tables, KaTeX math rendering, and correction for literal `\n` line breaks in OCR output.
- Supports parsing options including layout detection, chart recognition, document rectification, orientation recognition, seal recognition, formula numbering, and Markdown tag ignoring.
- Downloads package both Markdown output and OCR-extracted images.
- **Large file support**: Files over 1GB are processed in streaming batches to avoid OOM; supports resuming parsing from the last checkpoint after interruption.

## Deployment

This project supports two deployment paths. Do not mix them:

- **NVIDIA Docker version**: for Linux/Windows Docker environments with an NVIDIA GPU, using the official PaddleOCR-VL Docker services.
- **macOS Apple Silicon version**: for Apple M1/M2/M3/M4 chips, following the official Apple Silicon flow with local PaddlePaddle + PaddleX serving, optionally accelerated by MLX-VLM.

### Option 1: NVIDIA Docker

For Windows NVIDIA users, the recommended path is the one-click script:

```powershell
.\windows-one-click.bat
```

It checks Docker, detects the NVIDIA GPU, selects `env.txt` or `env.docker`, pulls the official PaddleOCR-VL images, builds `pandocr-web`, clears old containers, creates all model containers without starting both models, starts the WebUI, and waits for the active model health check.

Useful one-click options:

```powershell
.\windows-one-click.bat -DryRun
.\windows-one-click.bat -GpuId 1
.\windows-one-click.bat -EnvFile env.docker
```

Manual deployment is still available:

Choose the environment file based on your GPU model:

| GPU | Recommended env file | Image tag |
| --- | --- | --- |
| RTX 30 series | `env.docker` | `latest-nvidia-gpu-offline` |
| RTX 40 series | `env.docker` | `latest-nvidia-gpu-offline` |
| RTX 50 series / Blackwell | `env.txt` | `latest-nvidia-gpu-sm120-offline` |

The commands below use `env.txt` for RTX 50 series as an example. For RTX 30/40 series, replace `env.txt` with `env.docker`.

**Step 1: Pull official PaddleOCR images and build local services**

```bash
docker compose --env-file env.txt pull paddleocr-vlm-server paddleocr-vl-api
docker compose --env-file env.txt build paddleocr-ocr-api pandocr-web
```

**Step 2: Build the MinerU image (optional, only if you need MinerU)**

```bash
# Use China mirror for faster downloads
docker build -t mineru:latest https://github.com/opendatalab/MinerU.git#master:docker/china

# Or use the international source
# docker build -t mineru:latest https://github.com/opendatalab/MinerU.git#master:docker/global
```

> The MinerU image is based on vllm/vllm-openai, approximately 13GB in size, and takes 15-30 minutes to build (including model weight downloads).

**Step 3: Create and start containers**

```bash
# Create all containers (without starting)
docker compose --env-file env.txt up -d --no-start

# If you built the MinerU image, also create the mineru-api container
docker compose --env-file env.txt --profile mineru up -d --no-start mineru-api

# Start only the WebUI (it will start/stop model containers on demand)
docker compose --env-file env.txt start pandocr-web
```

Keep this `up -d --no-start` then `start pandocr-web` order for single-GPU deployments. Starting the whole compose stack with a plain `docker compose up -d` can hot-load multiple models at the same time and waste VRAM. After the WebUI is online, use the top-right model selector to switch models; the UI calls `/api/model-runtime/switch`, starts only the selected model containers, stops the inactive model containers, and keeps the top bar runtime badge synchronized with the real container state.

Open:

- WebUI: http://localhost:18000 (default port, configurable via `PANDOCR_PORT` in `env.txt`)
- PaddleOCR-VL API health: http://localhost:8081/health, available when `PaddleOCR-VL 1.6` is the active model.
- PP-OCRv6 API health: http://localhost:8082/health, available when `PP-OCRv6` is the active model.
- MinerU API health: http://localhost:8083/health, available when `MinerU` is the active model.

**LAN access:**

By default, the port binds to `0.0.0.0`, so other devices on the LAN can access the WebUI via `http://<your-IP>:18000`. `PANDOCR_CORS_ORIGINS` must include the LAN access address.

`pandocr-web` mounts `/var/run/docker.sock` so it can start and stop only the model containers defined in this compose file; treat this as Docker host management access and do not expose the WebUI to untrusted networks without additional controls.

Check status:

```powershell
docker compose --env-file env.txt ps
curl http://localhost:18000/api/model-runtime
curl http://localhost:18000/api/models
```

Common environment variables:

`env.txt` is the current recommended configuration for RTX 50 / Blackwell:

```text
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
PANDOCR_CORS_ORIGINS=http://localhost:18000,http://127.0.0.1:18000
PANDOCR_PORT=18000
PANDOCR_MAX_UPLOAD_MB=512
PANDOCR_MAX_CONCURRENT_OCR=1
PANDOCR_ENFORCE_ORIGIN_CHECK=1
PANDOCR_API_TOKEN=
PANDOCR_ENABLE_API_DOCS=0
MINERU_SERVICE_URL=http://mineru-api:8000
MINERU_MODEL_NAME=MinerU2.5-Pro-2605-1.2B
```

RTX 30/40 series and other non-Blackwell NVIDIA GPUs should use `env.docker`, where both image tags are `latest-nvidia-gpu-offline`.

Leave `PANDOCR_API_TOKEN` empty for local single-user use. If you expose the WebUI through a reverse proxy, LAN, or shared environment, set a long random token. `PANDOCR_ENFORCE_ORIGIN_CHECK=1` rejects cross-origin API mutations from origins outside the allowlist, but it is not a replacement for token authentication. The frontend prompts for the token after an API 401 and stores it in browser local storage. `/docs` and `/redoc` are only enabled when `PANDOCR_ENABLE_API_DOCS=1`; the WebUI OpenAPI JSON is always available at `/api/openapi.json`.

Useful commands:

```powershell
docker compose --env-file env.txt ps
docker compose --env-file env.txt logs -f pandocr-web
docker compose --env-file env.txt restart pandocr-web
docker compose --env-file env.txt down
```

### Option 2: macOS Apple Silicon

macOS Apple Silicon follows the official PaddleOCR-VL documentation for local deployment and does not use the NVIDIA Docker Compose images. References:

- PaddleOCR-VL Apple Silicon Usage Tutorial: https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html
- PaddleOCR-VL Usage Tutorial: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
- PaddleX Serving Guide: https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_deploy/serving.html

Supported chips:

- Apple M1 / M2 / M3 / M4 series chips (arm64)
- The project scripts check for `Darwin + arm64`, so M1-M4 use the same local Mac deployment path.
- The official PaddleOCR-VL Apple Silicon documentation currently states that accuracy validation has been completed on Apple M4. M1/M2/M3 can use the same path, but actual speed and stability depend on the chip model, memory, system version, and model cache state.

The default official PaddleX pipeline name on Mac is:

```text
PaddleOCR-VL-1.6
```

Do not use the bare `PaddleOCR-VL` name as the Mac default pipeline. In current PaddleX 3.6.x, the bare name maps to the older v1 configuration. `PaddleOCR-VL-1.6` uses `PP-DocLayoutV3`, `PaddleOCR-VL-1.6-0.9B`, and the native PaddlePaddle backend.

One-click deployment (recommended):

```bash
./macos-one-click.command
```

Equivalent Make command:

```bash
make mac-one-click
```

This one-click command checks the Apple Silicon environment, installs macOS dependencies, enables MLX-VLM acceleration by default, starts `mlx_vlm.server` / PaddleX API / PaddleOCR Local WebUI, runs health checks, and opens http://127.0.0.1:8000 automatically.

The first startup downloads `PP-DocLayoutV3`, `PaddleOCR-VL-1.6-0.9B`, and MLX model weights. The time required depends on network and disk speed. After the model cache is ready, subsequent runs of the same command reuse the installed environment and running services.

Advanced manual startup:

```bash
make mac-setup
make mac-up
```

Then open:

- WebUI: http://127.0.0.1:8000
- PaddleOCR-VL API health: http://127.0.0.1:8081/health

Test, stop, and view logs:

```bash
make mac-test
make mac-down
make mac-logs
```

If native mode is too slow, install and enable the MLX-VLM path from the official Apple Silicon documentation:

```bash
make mac-setup-mlx
make mac-down
make mac-up-mlx
make mac-test-mlx
```

MLX mode starts three local services:

- `mlx_vlm.server`: `127.0.0.1:8111`
- PaddleX full parsing API: `127.0.0.1:8081`
- PaddleOCR Local WebUI: `127.0.0.1:8000`

If Hugging Face downloads are slow, set `HF_TOKEN` to improve Hugging Face rate limits. Startup is much faster after models are cached. To change the MLX port, set `MLX_PORT`; the startup scripts generate the PaddleX configuration from the template.

Common environment variables:

```bash
PANDOCR_HOST=127.0.0.1
PANDOCR_PORT=8000
PADDLEX_HOST=127.0.0.1
PADDLEX_PORT=8081
PADDLEX_PIPELINE=PaddleOCR-VL-1.6
PANDOCR_MACOS_BACKEND=mlx
MLX_HOST=127.0.0.1
MLX_PORT=8111
MLX_MODEL=PaddlePaddle/PaddleOCR-VL-1.6
PADDLEPADDLE_VERSION=3.3.0
STARTUP_TIMEOUT_SECONDS=900
```

If the port is occupied, for example to move the WebUI to `18000`:

```bash
PANDOCR_PORT=18000 make mac-up
```

## MinerU Integration

This project integrates the [MinerU](https://github.com/opendatalab/MinerU) document parsing service on top of the original PaddleOCR version. The integration covers four areas:

1. **Backend** (`server.py`): Registers the `mineru` model in `MODEL_RUNTIME_CONFIG` and `model_catalog()`, adds the `/api/mineru` proxy route that forwards WebUI requests as MinerU's `multipart/form-data` format to the `mineru-api` `/file_parse` endpoint, and converts MinerU responses to the paddleocr-local unified format.

2. **Frontend** (`index.html` + `app.js`): The left "Parsing Settings" panel dynamically switches based on the currently selected model -- PaddleOCR models show options for layout detection, chart recognition, document rectification, seal recognition, etc.; MinerU shows options for formula parsing, table parsing, image analysis, parsing method, etc.

3. **Docker** (`docker-compose.yml`): Adds the `mineru-api` service using `profiles: ["mineru"]` so it does not start with a default `docker compose up`. You must build the MinerU image first, then explicitly enable the profile to create the container.

4. **Model switching**: Consistent with other models. When the WebUI model selector switches to MinerU, it automatically stops the current model container and starts `mineru-api`, waiting for the health check to pass before marking it as ready.

## Main APIs

- `GET /`: WebUI home page.
- `GET /api/models`: Returns available models and their proxy endpoints.
- `GET /api/model-runtime`: Returns active model, readiness, container state, and current switch operation.
- `POST /api/model-runtime/switch`: Starts the selected model containers and stops the inactive model containers when Docker model control is enabled.
- `GET /api/tasks`: Reads the local persistent task summary list without returning large source files or OCR results.
- `GET /api/tasks/{task_id}`: Reads the full details of one task.
- `PUT /api/tasks/{task_id}`: Saves one task to `data/tasks/`; `task.json` stores lightweight metadata, while Markdown, OCR JSON, images, and batch Markdown are split into `result.json`, with summaries in `summary.json`.
- `DELETE /api/tasks/{task_id}`: Deletes one local task.
- `DELETE /api/tasks`: Clears local task history by deleting only valid task id directories, avoiding files outside the task store.
- `POST /api/convert/to-pdf`: Converts PPT/PPTX/DOC/DOCX to PDF.
- `POST /api/paddleocr-vl-1.6`: Proxies OCR requests to the PaddleOCR-VL layout-parsing service.
- `POST /api/pp-ocrv6`: Proxies OCR requests to the PP-OCRv6 service and returns page images, recognized text lines, boxes, scores, and raw JSON.
- `POST /api/mineru`: Proxies OCR requests to the MinerU `/file_parse` service and returns Markdown and extracted images.
- `POST /api/glm-ocr`: Proxies OCR requests to the GLM-OCR (Ollama) service, using PP-OCRv6 layout detection + GLM-OCR text recognition.
- `POST /api/tasks/{task_id}/translate`: Translates the OCR Markdown result of a task into the target language, streaming translation progress.
- `GET /api/translate/config`: Returns whether translation is configured and the model in use.
- `GET /api/openapi.json`: OpenAPI JSON for this WebUI backend. `paddle-layout-openapi.json` in the repo documents the upstream Paddle layout-parsing service.

## Project Structure

```text
.
├── server.py                  # FastAPI backend (including MinerU / GLM-OCR / translation proxy)
├── layout_detect_server.py    # PP-OCRv6 layout detection helper service
├── requirements.txt
├── requirements-macos.txt
├── requirements-macos-mlx.txt
├── macos-one-click.command
├── windows-one-click.bat
├── Dockerfile                 # pandocr-web image
├── Dockerfile.ocr             # paddleocr-ocr-api image
├── docker-compose.yml         # includes mineru-api / ollama optional services
├── data/                      # Local task data directory, not committed by default
├── model_cache_mineru/        # MinerU model cache, not committed by default
├── env.txt                    # RTX 50 / Blackwell environment variables
├── env.docker                 # RTX 30 / 40 environment variables
├── pipeline_config_ocr_v6.yaml
├── pipeline_config_vllm.yaml
├── pipeline_config_macos_mlx.template.yaml
├── scripts/                   # Deployment helper scripts
│   ├── windows-one-click.ps1
├── static/
│   ├── index.html             # includes MinerU / GLM-OCR settings panel
│   ├── app.js                 # includes MinerU / GLM-OCR model switching and request logic
│   ├── style.css
│   ├── i18n.js                # Chinese/English translations
│   ├── latex_unicode.json     # LaTeX Unicode mapping
│   └── vendor/
│       ├── katex/             # KaTeX math formula rendering
│       └── pdfjs/             # PDF.js preview (with CJK CMap support)
├── QUICKSTART.md
├── webui-openapi.json
├── paddle-layout-openapi.json
├── DOCKER_DEPLOY.md
└── PROJECT_SUMMARY.md
```

## Local Development

When running `server.py` locally outside Docker, set `PANDOCR_MODEL_CONTROL=none` and start the model services yourself. You need an existing PaddleOCR-VL service listening at `http://localhost:8081/layout-parsing`. To use PP-OCRv6 locally, also start a PaddleX OCR service at `http://localhost:8082/ocr` or set `PADDLE_OCR_SERVICE_URL`. To use MinerU locally, start a MinerU API service at `http://localhost:8083` or set `MINERU_SERVICE_URL`. To use GLM-OCR, start Ollama and pull the `glm-ocr` model, or set `PANDOCR_OLLAMA_BASE_URL` and `PANDOCR_OLLAMA_MODEL`.

```powershell
pip install -r requirements.txt
python server.py
```

Then open http://localhost:18000.

Run the local quality gate:

```bash
make check
```

## Differences from Original

This project is forked from [CHEN010325/paddleocr-local](https://github.com/CHEN010325/paddleocr-local) and adds the following features on top of the original:

| Feature | Original | This Fork |
|---------|----------|-----------|
| OCR models | PaddleOCR-VL 1.6, PP-OCRv6 | + MinerU document parsing, GLM-OCR (Ollama) text recognition |
| Translation | None | 20-language OCR result translation (requires OpenAI-compatible API) |
| Large files | May OOM | >1GB files processed in streaming batches, supports resume after interruption |
| PDF preview | Basic preview | Full CJK CMap support — Chinese/Japanese/Korean PDFs no longer garbled |
| Model cache | Single directory | Separate cache directories (`model_cache_mineru/`, etc.) to avoid model conflicts |
| Parsing settings | Fixed panel | Dynamically switches based on selected model |
| LAN access | Binds to 127.0.0.1 only | Binds to 0.0.0.0 by default, supports LAN access |
| Port | Fixed 8000 | Configurable via `PANDOCR_PORT` environment variable |

## Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — Baidu PaddlePaddle OCR open-source project, providing PaddleOCR-VL and PP-OCRv6 models
- [CHEN010325/paddleocr-local](https://github.com/CHEN010325/paddleocr-local) — Upstream repository this project is forked from, providing the WebUI and Docker deployment framework
- [MinerU](https://github.com/opendatalab/MinerU) — OpenDataLab open-source document parsing project, MinerU2.5-Pro model
- [Ollama](https://github.com/ollama/ollama) — Local LLM inference framework, used to run GLM-OCR
- [Zhipu AI](https://www.zhipuai.cn/) — GLM-OCR model provider
- [KaTeX](https://github.com/KaTeX/KaTeX) — Math formula rendering
- [PDF.js](https://github.com/nicedoc/pdf.js) — PDF preview rendering

## License

This project is licensed under the [Apache License 2.0](LICENSE).

This project integrates multiple upstream OCR/model services. Please verify their licenses before commercial use:

| Project | License | Commercial use |
|---------|---------|---------------|
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 | ✅ Allowed |
| [MinerU](https://github.com/opendatalab/MinerU) | AGPL-3.0 (server-side) | ⚠️ Note: network services require source disclosure |
| [Ollama](https://github.com/ollama/ollama) | MIT | ✅ Allowed |
| [GLM-OCR / Zhipu AI](https://www.zhipuai.cn/) | Apache-2.0 (model) | ✅ Allowed, subject to model license |

> **Note**: MinerU uses AGPL-3.0. If you provide MinerU functionality to third-party users via a network service, you must open-source your server-side code. Internal use is not subject to this restriction. See [MinerU License](https://github.com/opendatalab/MinerU/blob/master/LICENSE).
