import os
import asyncio
import base64
import httpx
import subprocess
import tempfile
import shutil
import io
import json
import re
import logging
import time
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import List, Optional, Union
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("pandocr")
logging.basicConfig(level=os.getenv("PANDOCR_LOG_LEVEL", "INFO"))
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_csv_env(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def parse_positive_int_env(name: str, default: str) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return max(1, int(default))


PADDLE_SERVICE_URL = os.getenv("PADDLE_SERVICE_URL", "http://localhost:8081/layout-parsing")
PADDLEOCR_VL_MODEL_NAME = os.getenv("PADDLEOCR_VL_MODEL_NAME", "PaddleOCR-VL-1.6-0.9B")
PADDLE_OCR_SERVICE_URL = os.getenv("PADDLE_OCR_SERVICE_URL", "http://localhost:8082/ocr")
PPOCR_V6_MODEL_NAME = os.getenv("PPOCR_V6_MODEL_NAME", "PP-OCRv6_medium")
MINERU_SERVICE_URL = os.getenv("MINERU_SERVICE_URL", "http://localhost:8083")
MINERU_MODEL_NAME = os.getenv("MINERU_MODEL_NAME", "MinerU2.5-Pro-2605-1.2B")
_ollama_default_url = "http://ollama:11434" if os.getenv("PANDOCR_MODEL_CONTROL", "docker").strip().lower() == "docker" else "http://localhost:11434"
OLLAMA_BASE_URL = os.getenv("PANDOCR_OLLAMA_BASE_URL", _ollama_default_url).strip()
OLLAMA_MODEL = os.getenv("PANDOCR_OLLAMA_MODEL", "glm-ocr").strip()
OLLAMA_NUM_CTX = int(os.getenv("PANDOCR_OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("PANDOCR_OLLAMA_NUM_PREDICT", "4096"))
PADDLE_REQUEST_TIMEOUT = float(os.getenv("PADDLE_REQUEST_TIMEOUT", "3600"))
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DATA_DIR = Path(os.getenv("PANDOCR_TASK_DATA_DIR", "data/tasks")).resolve()
MAX_REQUEST_BYTES = int(float(os.getenv("PANDOCR_MAX_UPLOAD_MB", "512")) * 1024 * 1024)
MAX_TOTAL_UPLOAD_BYTES = int(float(os.getenv("PANDOCR_MAX_TOTAL_UPLOAD_MB", "4096")) * 1024 * 1024)
DEFAULT_CHUNK_SIZE = int(float(os.getenv("PANDOCR_DEFAULT_CHUNK_SIZE_MB", "10")) * 1024 * 1024)
CHUNKED_UPLOAD_THRESHOLD = int(float(os.getenv("PANDOCR_CHUNKED_UPLOAD_THRESHOLD_MB", "100")) * 1024 * 1024)
MAX_BATCH_BYTES = int(float(os.getenv("PANDOCR_MAX_BATCH_MB", "200")) * 1024 * 1024)
UPLOAD_SESSION_DIR = Path(os.getenv("PANDOCR_UPLOAD_DIR", "data/uploads")).resolve()
UPLOAD_SESSION_TTL_HOURS = float(os.getenv("PANDOCR_UPLOAD_TTL_HOURS", "24"))
PANDOCR_HOST = os.getenv("PANDOCR_HOST", "0.0.0.0")
PANDOCR_PORT = int(os.getenv("PANDOCR_PORT", "8000"))
MODEL_CONTROL_MODE = os.getenv("PANDOCR_MODEL_CONTROL", "docker").strip().lower()
MODEL_RUNTIME_STARTUP = os.getenv("PANDOCR_ACTIVE_MODEL_ON_START", "paddleocr-vl-1.6").strip()
DOCKER_SOCKET_PATH = os.getenv("PANDOCR_DOCKER_SOCKET", "/var/run/docker.sock")
MODEL_SWITCH_TIMEOUT = float(os.getenv("PANDOCR_MODEL_SWITCH_TIMEOUT", "1200"))
API_TOKEN = os.getenv("PANDOCR_API_TOKEN", "").strip()
ENABLE_API_DOCS = parse_bool_env("PANDOCR_ENABLE_API_DOCS", "0")
ENFORCE_ORIGIN_CHECK = parse_bool_env("PANDOCR_ENFORCE_ORIGIN_CHECK", "1")
MAX_CONCURRENT_OCR = parse_positive_int_env("PANDOCR_MAX_CONCURRENT_OCR", "1")
TRANSLATE_API_URL = os.getenv("PANDOCR_TRANSLATE_API_URL", "").strip()
TRANSLATE_API_KEY = os.getenv("PANDOCR_TRANSLATE_API_KEY", "").strip()
TRANSLATE_MODEL = os.getenv("PANDOCR_TRANSLATE_MODEL", "gpt-4o-mini").strip()
TASK_STORE_MARKER = ".pandocr-task-store"
TASK_RESULT_FILE = "result.json"
TASK_MARKDOWN_FILE = "markdown.md"
TASK_SUMMARY_FILE = "summary.json"
FOLDER_STORE_FILE = "folders.json"
UPLOAD_CHUNK_SIZE = 1024 * 1024
CORS_ORIGINS = parse_csv_env(
    "PANDOCR_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
)

MODEL_RUNTIME_CONFIG = {
    "paddleocr-vl-1.6": {
        "containers": ["paddleocr-vlm-server", "paddleocr-vl-api"],
        "start_order": ["paddleocr-vlm-server", "paddleocr-vl-api"],
        "stop_order": ["paddleocr-vl-api", "paddleocr-vlm-server"],
        "health_url": PADDLE_SERVICE_URL.rsplit("/", 1)[0] + "/health",
    },
    "pp-ocrv6": {
        "containers": ["paddleocr-ocr-api"],
        "start_order": ["paddleocr-ocr-api"],
        "stop_order": ["paddleocr-ocr-api"],
        "health_url": PADDLE_OCR_SERVICE_URL.rsplit("/", 1)[0] + "/health",
    },
    "mineru": {
        "containers": ["mineru-api"],
        "start_order": ["mineru-api"],
        "stop_order": ["mineru-api"],
        "health_url": f"{MINERU_SERVICE_URL}/health",
    },
    "glm-ocr": {
        "containers": ["paddleocr-ocr-api"],  # PP-OCRv6 stays running for layout detection
        "start_order": ["paddleocr-ocr-api"],
        "stop_order": [],  # Don't stop PP-OCRv6 when switching away from glm-ocr
        "health_url": f"{OLLAMA_BASE_URL}/api/tags",
    },
}
DEFAULT_RUNTIME_MODEL_ID = MODEL_RUNTIME_STARTUP if MODEL_RUNTIME_STARTUP in MODEL_RUNTIME_CONFIG else "paddleocr-vl-1.6"

model_runtime_lock = asyncio.Lock()
ocr_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)
model_runtime_operation = {
    "targetModelId": DEFAULT_RUNTIME_MODEL_ID,
    "state": "idle",
    "message": "",
    "startedAt": None,
    "updatedAt": None,
}
model_runtime_task: asyncio.Task | None = None
ocr_active_count = 0


class ModelSwitchRequest(BaseModel):
    modelId: str


class CreateUploadRequest(BaseModel):
    filename: str
    totalSize: int
    chunkSize: int = DEFAULT_CHUNK_SIZE
    taskId: str | None = None

    @field_validator("filename")
    @classmethod
    def sanitize_filename(cls, v: str) -> str:
        # Strip directory components and reject suspicious names
        safe_name = Path(v).name
        if not safe_name or safe_name.startswith("."):
            raise ValueError("Invalid filename")
        return safe_name


class ProcessRequest(BaseModel):
    modelId: str
    ocrOptions: dict = Field(default_factory=dict)


def model_catalog() -> list[dict]:
    return [
        {
            "id": "paddleocr-vl-1.6",
            "name": PADDLEOCR_VL_MODEL_NAME,
            "label": "PaddleOCR-VL 1.6",
            "kind": "document_parsing",
            "endpoint": "/api/paddleocr-vl-1.6",
        },
        {
            "id": "pp-ocrv6",
            "name": PPOCR_V6_MODEL_NAME,
            "label": "PP-OCRv6",
            "kind": "text_ocr",
            "endpoint": "/api/pp-ocrv6",
        },
        {
            "id": "mineru",
            "name": MINERU_MODEL_NAME,
            "label": "MinerU",
            "kind": "document_parsing",
            "endpoint": "/api/mineru",
        },
        {
            "id": "glm-ocr",
            "name": OLLAMA_MODEL,
            "label": "GLM-OCR (Ollama)",
            "kind": "document_parsing",
            "endpoint": "/api/glm-ocr",
        },
    ]


def model_control_available() -> bool:
    return MODEL_CONTROL_MODE == "docker" and Path(DOCKER_SOCKET_PATH).exists()


async def docker_api_request(method: str, path: str, *, timeout: float = 30) -> httpx.Response:
    transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET_PATH)
    async with httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=timeout) as client:
        return await client.request(method, path)


async def inspect_container(name: str) -> dict:
    if not model_control_available():
        return {
            "name": name,
            "exists": False,
            "running": False,
            "state": "unknown",
            "health": "unknown",
        }

    response = await docker_api_request("GET", f"/containers/{name}/json")
    if response.status_code == 404:
        return {
            "name": name,
            "exists": False,
            "running": False,
            "state": "missing",
            "health": "missing",
        }
    response.raise_for_status()
    payload = response.json()
    state = payload.get("State") or {}
    health = state.get("Health") or {}
    return {
        "name": name,
        "exists": True,
        "running": bool(state.get("Running")),
        "state": state.get("Status") or "unknown",
        "health": health.get("Status") or "none",
    }


async def docker_container_action(name: str, action: str) -> None:
    if not model_control_available():
        raise RuntimeError("Docker model control is not available")
    if action == "stop":
        response = await docker_api_request("POST", f"/containers/{name}/stop?t=20", timeout=45)
        if response.status_code in {204, 304, 404}:
            return
    elif action == "start":
        response = await docker_api_request("POST", f"/containers/{name}/start", timeout=45)
        if response.status_code in {204, 304}:
            return
    else:
        raise ValueError(f"Unsupported container action: {action}")
    if response.status_code >= 400:
        raise RuntimeError(f"Docker {action} failed for {name}: {response.text}")


async def check_http_health(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url)
        return 200 <= response.status_code < 300
    except Exception:
        return False


async def model_runtime_status(model_id: str) -> dict:
    config = MODEL_RUNTIME_CONFIG[model_id]

    # GLM-OCR (Ollama) has no Docker containers; use HTTP health check directly
    if model_id == "glm-ocr":
        containers = []
        health_ok = await check_http_health(config["health_url"])
        # For Ollama, also verify the model is loaded
        if health_ok:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        has_model = any(OLLAMA_MODEL in m for m in models)
                        if has_model:
                            state = "ready"
                        else:
                            state = "model_missing"
                        return {
                            "id": model_id,
                            "containers": containers,
                            "running": True,
                            "ready": has_model,
                            "state": state,
                            "healthUrl": config["health_url"],
                            "ollamaModel": OLLAMA_MODEL,
                        }
            except Exception:
                pass
        return {
            "id": model_id,
            "containers": containers,
            "running": False,
            "ready": False,
            "state": "offline",
            "healthUrl": config["health_url"],
            "ollamaModel": OLLAMA_MODEL,
        }

    containers = [await inspect_container(name) for name in config["containers"]]
    if not model_control_available():
        health_ok = await check_http_health(config["health_url"])
        return {
            "id": model_id,
            "containers": containers,
            "running": health_ok,
            "ready": health_ok,
            "state": "ready" if health_ok else "unknown",
            "healthUrl": config["health_url"],
        }

    any_running = any(container["running"] for container in containers)
    all_running = all(container["running"] for container in containers)
    any_missing = any(not container["exists"] for container in containers)
    health_ok = await check_http_health(config["health_url"]) if all_running else False

    if any_missing:
        state = "missing"
    elif health_ok:
        state = "ready"
    elif any_running:
        state = "starting" if all_running else "partial"
    else:
        state = "stopped"

    return {
        "id": model_id,
        "containers": containers,
        "running": any_running,
        "ready": health_ok,
        "state": state,
        "healthUrl": config["health_url"],
    }


async def build_model_runtime_payload() -> dict:
    models = {
        model_id: await model_runtime_status(model_id)
        for model_id in MODEL_RUNTIME_CONFIG
    }
    active_model = model_runtime_operation.get("targetModelId", DEFAULT_RUNTIME_MODEL_ID)
    
    # If the target model failed, and another model is ready, we could fallback,
    # but it's better to stay on the target model and show the error state.
    
    return {
        "controlMode": MODEL_CONTROL_MODE,
        "controlAvailable": model_control_available(),
        "activeModelId": active_model,
        "defaultModelId": DEFAULT_RUNTIME_MODEL_ID,
        "operation": dict(model_runtime_operation),
        "ocrActiveCount": ocr_active_count,
        "maxConcurrentOcr": MAX_CONCURRENT_OCR,
        "models": models,
    }


def set_model_runtime_operation(state: str, message: str = "", target_model_id: str | None = None) -> None:
    now = time.time()
    if target_model_id:
        model_runtime_operation["targetModelId"] = target_model_id
    model_runtime_operation["state"] = state
    model_runtime_operation["message"] = message
    model_runtime_operation["updatedAt"] = now
    if state == "switching":
        model_runtime_operation["startedAt"] = now


async def wait_model_ready(model_id: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await model_runtime_status(model_id)
        if status["ready"]:
            return
        await asyncio.sleep(3)
    raise TimeoutError(f"Timed out waiting for {model_id} to become ready")


async def wait_container_runtime_ready(container_name: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await inspect_container(container_name)
        if not status["exists"]:
            raise RuntimeError(f"Docker container {container_name} is missing. Run docker compose up --no-start first.")
        if status["running"] and status["health"] in {"healthy", "none"}:
            return
        await asyncio.sleep(3)
    raise TimeoutError(f"Timed out waiting for Docker container {container_name} to become healthy")


async def activate_model_runtime(model_id: str) -> None:
    if model_id not in MODEL_RUNTIME_CONFIG:
        raise ValueError(f"Unknown model id: {model_id}")
    # glm-ocr doesn't need Docker — Ollama runs externally
    if model_id != "glm-ocr" and not model_control_available():
        raise RuntimeError("Docker model control is not available")

    async with model_runtime_lock:
        set_model_runtime_operation("switching", f"Switching to {model_id}", model_id)
        switch_started_at = time.monotonic()
        try:
            # Stop other models' containers (skip if Docker is unavailable)
            if model_control_available():
                for other_model_id, config in MODEL_RUNTIME_CONFIG.items():
                    if other_model_id == model_id:
                        continue
                    for container_name in config["stop_order"]:
                        await docker_container_action(container_name, "stop")

                for container_name in MODEL_RUNTIME_CONFIG[model_id]["start_order"]:
                    remaining_timeout = max(3, MODEL_SWITCH_TIMEOUT - (time.monotonic() - switch_started_at))
                    await docker_container_action(container_name, "start")
                    await wait_container_runtime_ready(container_name, remaining_timeout)

            remaining_timeout = max(3, MODEL_SWITCH_TIMEOUT - (time.monotonic() - switch_started_at))
            await wait_model_ready(model_id, remaining_timeout)
            set_model_runtime_operation("ready", f"{model_id} is ready", model_id)
        except Exception as err:
            logger.exception("Model runtime switch failed")
            set_model_runtime_operation("error", str(err), model_id)


async def schedule_model_runtime_activation(model_id: str) -> None:
    global model_runtime_task
    if model_id not in MODEL_RUNTIME_CONFIG:
        raise HTTPException(status_code=400, detail="Unknown model id")
    # glm-ocr doesn't need Docker — Ollama runs externally
    if model_id != "glm-ocr" and not model_control_available():
        raise HTTPException(status_code=503, detail="Docker model control is not available")
    async with model_runtime_lock:
        if ocr_active_count > 0:
            raise HTTPException(status_code=409, detail="OCR is running. Wait for the active task before switching models.")
        if model_runtime_task and not model_runtime_task.done():
            model_runtime_task.cancel()
        set_model_runtime_operation("switching", f"Switching to {model_id}", model_id)
        model_runtime_task = asyncio.create_task(activate_model_runtime(model_id))


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_task_data_dir()
    ensure_upload_dir()
    # Clean up stale chunked upload sessions
    await run_in_threadpool(cleanup_stale_uploads)
    # Crash recovery: reset tasks stuck in "processing" state
    await run_in_threadpool(reset_stuck_processing_tasks)
    if model_control_available():
        await schedule_model_runtime_activation(DEFAULT_RUNTIME_MODEL_ID)
    yield


app = FastAPI(
    title="PaddleOCR Local WebUI",
    version="0.2.0",
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


SAFE_API_METHODS = {"GET", "HEAD", "OPTIONS"}


def normalize_origin(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def configured_origins_for_request(request: Request) -> set[str]:
    origins = {normalize_origin(origin) for origin in CORS_ORIGINS if origin != "*"}
    request_origin = f"{request.url.scheme}://{request.url.netloc}".lower()
    origins.add(request_origin)
    return {origin for origin in origins if origin}


def request_origin_is_allowed(request: Request) -> bool:
    if not ENFORCE_ORIGIN_CHECK or not request.url.path.startswith("/api/"):
        return True
    if request.method in SAFE_API_METHODS:
        return True
    origin = request.headers.get("origin")
    if not origin:
        return True
    if "*" in CORS_ORIGINS:
        return True
    return normalize_origin(origin) in configured_origins_for_request(request)


@app.middleware("http")
async def enforce_request_security(request: Request, call_next):
    if not request_origin_is_allowed(request):
        return JSONResponse(status_code=403, content={"detail": "Cross-origin API request is not allowed"})

    if API_TOKEN and request.url.path.startswith("/api/") and not request_is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid API token"})

    if request.method in {"POST", "PUT", "PATCH"} and MAX_REQUEST_BYTES > 0:
        # Chunk upload endpoints are exempt: each chunk is small (≤ chunkSize)
        is_chunk_upload = request.url.path.startswith("/api/uploads/") and "/chunks/" in request.url.path
        if not is_chunk_upload:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > MAX_REQUEST_BYTES:
                        max_mb = MAX_REQUEST_BYTES / 1024 / 1024
                        return JSONResponse(
                            status_code=413,
                            content={"detail": f"Request body is too large. Max upload size is {max_mb:.0f} MB."},
                        )
                except ValueError:
                    pass

    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/") and not API_TOKEN:
        response.headers.setdefault("X-Pandocr-Auth-Warning", "PANDOCR_API_TOKEN is not set")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "worker-src 'self' blob:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'",
    )
    return response


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.get("/api/models")
async def get_models():
    """Return OCR models available through this proxy."""
    return {
        "default": DEFAULT_RUNTIME_MODEL_ID,
        "data": model_catalog(),
        "maxUploadBytes": MAX_REQUEST_BYTES,
        "maxTotalUploadBytes": MAX_TOTAL_UPLOAD_BYTES,
        "chunkedUploadThreshold": CHUNKED_UPLOAD_THRESHOLD,
        "defaultChunkSize": DEFAULT_CHUNK_SIZE,
        "maxBatchBytes": MAX_BATCH_BYTES,
        "authRequired": bool(API_TOKEN),
        "originProtection": ENFORCE_ORIGIN_CHECK,
        "maxConcurrentOcr": MAX_CONCURRENT_OCR,
    }


@app.get("/api/model-runtime")
async def get_model_runtime():
    return await build_model_runtime_payload()


@app.post("/api/model-runtime/switch")
async def switch_model_runtime(request: ModelSwitchRequest):
    await schedule_model_runtime_activation(request.modelId)
    return await build_model_runtime_payload()


def request_is_authenticated(request: Request) -> bool:
    if not API_TOKEN:
        return True
    header = request.headers.get("authorization", "")
    token = ""
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()
    token = token or request.headers.get("x-pandocr-token", "").strip()
    return bool(token) and secrets.compare_digest(token, API_TOKEN)


def validate_task_data_dir() -> None:
    task_dir = TASK_DATA_DIR.resolve()
    forbidden = {
        Path(task_dir.anchor).resolve(),
        PROJECT_ROOT.resolve(),
        PROJECT_ROOT.parent.resolve(),
        Path.home().resolve(),
    }
    if task_dir in forbidden:
        raise RuntimeError(f"Unsafe PANDOCR_TASK_DATA_DIR: {task_dir}")


def ensure_task_data_dir() -> None:
    validate_task_data_dir()
    TASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    marker = TASK_DATA_DIR / TASK_STORE_MARKER
    if not marker.exists():
        marker.write_text("PaddleOCR Local task store\n", encoding="utf-8")


def safe_task_id(task_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,80}", task_id or ""):
        raise HTTPException(status_code=400, detail="Invalid task id")
    return task_id


def task_file_path(task_id: str) -> Path:
    return TASK_DATA_DIR / safe_task_id(task_id) / "task.json"


def task_summary_path(task_id: str) -> Path:
    return task_dir_path(task_id) / TASK_SUMMARY_FILE


def task_result_path(task_id: str) -> Path:
    return task_dir_path(task_id) / TASK_RESULT_FILE


def task_dir_path(task_id: str) -> Path:
    return TASK_DATA_DIR / safe_task_id(task_id)


def task_source_path(task_id: str) -> Path:
    return task_dir_path(task_id) / "source.bin"


def task_source_url(task_id: str) -> str:
    return f"/api/tasks/{safe_task_id(task_id)}/source"


def split_task_for_storage(task: dict) -> tuple[dict, dict | None]:
    """Keep task.json as metadata and move heavy OCR results into result.json."""
    task_id = task.get("id")
    source_url = task.get("sourceUrl")
    has_external_source = bool(source_url) or (isinstance(task_id, str) and task_source_path(task_id).exists())

    stored = dict(task)
    stored.pop("detailLoaded", None)
    preserve_result = bool(stored.pop("_preserveResult", False))

    result_payload = {}
    for key in ("markdown", "images", "ocrResults", "translation", "translationLang"):
        if key in stored:
            result_payload[key] = stored.pop(key)

    if has_external_source:
        stored["sourceUrl"] = source_url or task_source_url(task_id)
        stored.pop("sourceDataUrl", None)

    batches = stored.get("batches") if isinstance(stored.get("batches"), list) else []
    compact_batches = []
    batch_markdown = {}
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        compact = dict(batch)
        compact.pop("payloadDataUrl", None)
        compact.pop("payloadBlob", None)
        if "markdown" in compact:
            batch_id = compact.get("id")
            if batch_id:
                batch_markdown[str(batch_id)] = compact.pop("markdown")
            else:
                compact.pop("markdown", None)
        compact_batches.append(compact)
    if batch_markdown:
        result_payload["batchMarkdown"] = batch_markdown

    has_result_payload = any(
        bool(result_payload.get(key))
        for key in ("markdown", "images", "ocrResults", "batchMarkdown")
    )
    if preserve_result and not has_result_payload and isinstance(task_id, str):
        previous_state = {}
        previous_path = task_file_path(task_id)
        if previous_path.exists():
            try:
                previous = read_task_file(previous_path)
                previous_state = previous.get("_resultState") if isinstance(previous.get("_resultState"), dict) else {}
            except (OSError, ValueError, json.JSONDecodeError):
                previous_state = {}
        stored["batches"] = compact_batches
        stored["_storage"] = {
            "version": 2,
            "resultPath": TASK_RESULT_FILE if task_result_path(task_id).exists() else None,
        }
        stored["_resultState"] = previous_state
        return stored, None

    stored["batches"] = compact_batches
    stored["_storage"] = {
        "version": 2,
        "resultPath": TASK_RESULT_FILE if has_result_payload else None,
    }
    stored["_resultState"] = {
        "hasMarkdown": bool(result_payload.get("markdown") or result_payload.get("batchMarkdown")),
        "hasImages": bool(result_payload.get("images")),
        "hasOcrResults": bool(result_payload.get("ocrResults")),
    }
    return stored, result_payload


def task_summary(task: dict) -> dict:
    batches = task.get("batches") if isinstance(task.get("batches"), list) else []
    result_state = task.get("_resultState") if isinstance(task.get("_resultState"), dict) else {}
    completed_pages = sum(
        int(batch.get("pageCount") or 0)
        for batch in batches
        if isinstance(batch, dict) and batch.get("status") == "completed"
    )
    return {
        "id": task.get("id"),
        "name": task.get("name"),
        "originalName": task.get("originalName"),
        "sourceKind": task.get("sourceKind"),
        "mimeType": task.get("mimeType"),
        "size": task.get("size"),
        "createdAt": task.get("createdAt"),
        "updatedAt": task.get("updatedAt"),
        "status": task.get("status"),
        "pageCount": task.get("pageCount"),
        "pdfBatchSize": task.get("pdfBatchSize"),
        "sourceUrl": task.get("sourceUrl"),
        "modelId": task.get("modelId"),
        "modelName": task.get("modelName"),
        "error": task.get("error"),
        "completedPages": completed_pages,
        "batchCount": len(batches),
        "hasMarkdown": bool(result_state.get("hasMarkdown") or task.get("markdown")),
        "hasOcrResults": bool(result_state.get("hasOcrResults") or task.get("ocrResults")),
        "folderId": task.get("folderId"),
        "folderName": task.get("folderName"),
        "detailLoaded": False,
    }


def read_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def read_task_file(path: Path) -> dict:
    return read_json_file(path)


def write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    temp_path.replace(path)


def write_task_bundle(task_id: str, task: dict) -> dict:
    ensure_task_data_dir()
    stored_task, result_payload = split_task_for_storage(task)
    task_dir = task_dir_path(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    result_path = task_result_path(task_id)

    # Smart merge with existing result.json:
    # When the task object was hydrated in lite mode (missing images/ocrResults),
    # we must preserve the existing heavy data and only update what changed.
    if result_payload is not None and result_path.exists():
        try:
            existing = read_json_file(result_path)
            if isinstance(existing, dict):
                # Merge images: new images override existing, but keep existing ones not in new
                if isinstance(result_payload.get("images"), dict) and isinstance(existing.get("images"), dict):
                    merged_images = dict(existing["images"])
                    merged_images.update(result_payload["images"])
                    result_payload["images"] = merged_images
                elif not result_payload.get("images") and existing.get("images"):
                    result_payload["images"] = existing["images"]

                # Merge ocrResults: if new has fewer, keep existing
                if isinstance(result_payload.get("ocrResults"), list) and isinstance(existing.get("ocrResults"), list):
                    if len(result_payload["ocrResults"]) < len(existing["ocrResults"]):
                        result_payload["ocrResults"] = existing["ocrResults"]

                # Keep existing translation if not in new
                if not result_payload.get("translation") and existing.get("translation"):
                    result_payload["translation"] = existing["translation"]
        except (OSError, ValueError, json.JSONDecodeError) as err:
            logger.warning("Failed to merge existing result.json for task %s: %s", task_id, err)

    if result_payload is None:
        pass
    elif stored_task.get("_storage", {}).get("resultPath"):
        write_json_file(result_path, result_payload)
    elif result_path.exists():
        # SAFETY: Never delete an existing result.json even if the current
        # payload is empty. The task may have accumulated results from
        # previous processing runs that aren't in the current task object
        # (e.g., after crash recovery without hydration).
        # Only overwrite if we actually have data to write.
        if any(bool(result_payload.get(key)) for key in ("markdown", "images", "ocrResults", "batchMarkdown", "translation")):
            write_json_file(result_path, result_payload)
        # else: keep existing result.json intact

    # Also save markdown to a standalone .md file for fast lite hydration.
    # This allows the lite endpoint to load markdown without parsing the
    # entire result.json (which can be hundreds of MB for large documents).
    md_path = task_dir / TASK_MARKDOWN_FILE
    markdown_text = task.get("markdown") if isinstance(task, dict) else None
    if markdown_text:
        try:
            md_path.write_text(markdown_text, encoding="utf-8")
        except OSError as err:
            logger.warning("Failed to write markdown file for task %s: %s", task_id, err)
    elif md_path.exists() and result_payload is None:
        # Task was reset — remove stale markdown file
        try:
            md_path.unlink()
        except OSError:
            pass

    write_json_file(task_file_path(task_id), stored_task)
    summary = task_summary(stored_task)
    write_json_file(task_summary_path(task_id), summary)
    return stored_task


def hydrate_task_detail_lite(task_id: str, task: dict) -> dict:
    """Load lightweight task details — markdown and metadata only.

    Omits `images` and `ocrResults` to avoid loading hundreds of MB
    of base64 data into memory.  The frontend fetches these on demand
    via the /api/tasks/{id}/result endpoint.

    Markdown is loaded from a separate .md file when available,
    avoiding the need to parse the entire result.json.
    """
    task_dir = task_dir_path(task_id)
    md_path = task_dir / TASK_MARKDOWN_FILE

    # Fast path: load markdown from standalone file (no JSON parsing needed)
    if md_path.exists():
        try:
            task["markdown"] = md_path.read_text(encoding="utf-8")
        except OSError as err:
            logger.warning("Failed to read markdown file %s: %s", md_path, err)

    # Determine _resultState without loading result.json
    # Check file existence and first bytes to infer what data exists
    storage = task.get("_storage") if isinstance(task.get("_storage"), dict) else {}
    result_name = storage.get("resultPath") or TASK_RESULT_FILE
    result_path = task_dir / result_name

    result_state = task.get("_resultState") if isinstance(task.get("_resultState"), dict) else {}
    if result_path.exists() and not result_state:
        # Infer _resultState from file existence and size — avoid loading the JSON
        try:
            # Peek at the first few bytes to quickly determine if key fields exist
            with result_path.open("r", encoding="utf-8") as f:
                preview = f.read(4096)
            task["_resultState"] = {
                "hasMarkdown": bool(task.get("markdown")) or '"markdown"' in preview,
                "hasImages": '"images"' in preview,
                "hasOcrResults": '"ocrResults"' in preview or '"ocr_lines"' in preview.lower(),
            }
        except OSError:
            task["_resultState"] = {"hasMarkdown": bool(task.get("markdown")), "hasImages": False, "hasOcrResults": False}

    # Load translation data — it's typically small
    if result_path.exists():
        try:
            # Only parse result.json if we need translation (usually small)
            # or if markdown wasn't loaded from .md file
            if not task.get("markdown") or not task.get("translation"):
                result_payload = read_json_file(result_path)
                if not task.get("markdown") and "markdown" in result_payload:
                    task["markdown"] = result_payload["markdown"]
                if "translation" in result_payload:
                    task["translation"] = result_payload["translation"]
                if "translationLang" in result_payload:
                    task["translationLang"] = result_payload["translationLang"]
                batch_markdown = result_payload.get("batchMarkdown")
                if isinstance(batch_markdown, dict) and isinstance(task.get("batches"), list):
                    for batch in task["batches"]:
                        if isinstance(batch, dict) and batch.get("id") in batch_markdown:
                            batch["markdown"] = batch_markdown[batch["id"]]
        except (OSError, ValueError, json.JSONDecodeError) as err:
            logger.warning("Failed to hydrate lite task result %s: %s", result_path, err)

    task.setdefault("markdown", "")
    task.setdefault("images", {})
    task.setdefault("ocrResults", [])

    return task


def hydrate_task_detail(task_id: str, task: dict) -> dict:
    storage = task.get("_storage") if isinstance(task.get("_storage"), dict) else {}
    result_name = storage.get("resultPath") or TASK_RESULT_FILE
    result_path = task_dir_path(task_id) / result_name
    if result_path.exists():
        try:
            result_payload = read_json_file(result_path)
            for key in ("markdown", "images", "ocrResults", "translation", "translationLang"):
                if key in result_payload:
                    task[key] = result_payload[key]
            batch_markdown = result_payload.get("batchMarkdown")
            if isinstance(batch_markdown, dict) and isinstance(task.get("batches"), list):
                for batch in task["batches"]:
                    if isinstance(batch, dict) and batch.get("id") in batch_markdown:
                        batch["markdown"] = batch_markdown[batch["id"]]
        except (OSError, ValueError, json.JSONDecodeError) as err:
            logger.warning("Failed to hydrate task result %s: %s", result_path, err)

    task.setdefault("markdown", "")
    task.setdefault("images", {})
    task.setdefault("ocrResults", [])

    # Migrate legacy nested format: old compact_ocr_json_result wrapped
    # the page result inside a "result" key. The frontend expects fields
    # like parser, ocrLines, pageImage at the top level.
    ocr_results = task.get("ocrResults")
    if isinstance(ocr_results, list):
        for i, item in enumerate(ocr_results):
            if not isinstance(item, dict):
                continue
            inner = item.get("result")
            if isinstance(inner, dict) and ("parser" in inner or "ocrLines" in inner):
                # Merge inner fields to top level, preserving metadata
                merged = dict(inner)
                for meta_key in ("batchId", "pageIndex", "label", "sourcePage"):
                    if meta_key in item and meta_key not in merged:
                        merged[meta_key] = item[meta_key]
                ocr_results[i] = merged

    return task


def task_needs_compaction(task: dict) -> bool:
    if any(key in task for key in ("markdown", "images", "ocrResults", "detailLoaded")):
        return True
    batches = task.get("batches") if isinstance(task.get("batches"), list) else []
    return any(
        isinstance(batch, dict) and any(key in batch for key in ("markdown", "payloadDataUrl", "payloadBlob"))
        for batch in batches
    )


def task_sort_timestamp(task: dict) -> float:
    value = task.get("updatedAt") or task.get("createdAt")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return float(text)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0
    return 0


def list_task_summaries() -> list[dict]:
    ensure_task_data_dir()
    tasks = []
    for path in TASK_DATA_DIR.glob("*/task.json"):
        try:
            summary_path = path.parent / TASK_SUMMARY_FILE
            if summary_path.exists():
                tasks.append(read_json_file(summary_path))
                continue

            task = read_task_file(path)
            if task.get("id") == path.parent.name and task_needs_compaction(task):
                task = write_task_bundle(path.parent.name, task)
            summary = task_summary(task)
            write_json_file(summary_path, summary)
            tasks.append(summary)
        except (OSError, ValueError, json.JSONDecodeError) as err:
            logger.warning("Skipping invalid task file %s: %s", path, err)
    tasks.sort(key=task_sort_timestamp, reverse=True)
    return tasks


# ---------------------------------------------------------------------------
# Folder management
# ---------------------------------------------------------------------------

def folder_store_path() -> Path:
    return TASK_DATA_DIR / FOLDER_STORE_FILE


def read_folder_store() -> dict:
    ensure_task_data_dir()
    path = folder_store_path()
    if not path.exists():
        return {"folders": []}
    try:
        return read_json_file(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"folders": []}


def write_folder_store(store: dict) -> None:
    ensure_task_data_dir()
    write_json_file(folder_store_path(), store)


def sanitize_folder_name(name: str) -> str:
    """Sanitize folder name: strip, collapse whitespace, reject empty or dots-only."""
    clean = re.sub(r"\s+", " ", name.strip())
    if not clean or clean.strip(".") == "":
        raise HTTPException(status_code=400, detail="Invalid folder name")
    if len(clean) > 100:
        raise HTTPException(status_code=400, detail="Folder name too long (max 100 characters)")
    return clean


@app.get("/api/folders")
async def list_folders():
    """Return all folders with their task counts."""
    store = await run_in_threadpool(read_folder_store)
    folder_task_counts = {}
    for folder in store.get("folders", []):
        folder_task_counts[folder["id"]] = 0
    # Count tasks per folder
    tasks = await run_in_threadpool(list_task_summaries)
    for task in tasks:
        fid = task.get("folderId")
        if fid and fid in folder_task_counts:
            folder_task_counts[fid] += 1
    result = []
    for folder in store.get("folders", []):
        result.append({**folder, "taskCount": folder_task_counts.get(folder["id"], 0)})
    return {"folders": result}


class CreateFolderRequest(BaseModel):
    name: str


@app.post("/api/folders")
async def create_folder(req: CreateFolderRequest):
    """Create a new folder."""
    name = await run_in_threadpool(sanitize_folder_name, req.name)
    store = await run_in_threadpool(read_folder_store)
    folder_id = secrets.token_urlsafe(8)
    now = time.time()
    folder = {
        "id": folder_id,
        "name": name,
        "createdAt": now,
        "updatedAt": now,
    }
    store.setdefault("folders", []).append(folder)
    await run_in_threadpool(write_folder_store, store)
    return {**folder, "taskCount": 0}


class RenameFolderRequest(BaseModel):
    name: str


@app.put("/api/folders/{folder_id}")
async def rename_folder(folder_id: str, req: RenameFolderRequest):
    """Rename a folder."""
    name = await run_in_threadpool(sanitize_folder_name, req.name)
    store = await run_in_threadpool(read_folder_store)
    found = False
    for folder in store.get("folders", []):
        if folder["id"] == folder_id:
            folder["name"] = name
            folder["updatedAt"] = time.time()
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Folder not found")
    await run_in_threadpool(write_folder_store, store)
    return {"ok": True}


@app.delete("/api/folders/{folder_id}")
async def delete_folder(folder_id: str):
    """Delete a folder. Tasks in the folder are moved to root (folderId cleared)."""
    store = await run_in_threadpool(read_folder_store)
    original_len = len(store.get("folders", []))
    store["folders"] = [f for f in store.get("folders", []) if f["id"] != folder_id]
    if len(store["folders"]) == original_len:
        raise HTTPException(status_code=404, detail="Folder not found")
    await run_in_threadpool(write_folder_store, store)
    # Clear folderId from all tasks in this folder
    await run_in_threadpool(clear_folder_from_tasks, folder_id)
    return {"ok": True}


def clear_folder_from_tasks(folder_id: str) -> None:
    """Remove folderId from all tasks in the given folder."""
    for path in TASK_DATA_DIR.glob("*/task.json"):
        try:
            task = read_task_file(path)
            if task.get("folderId") == folder_id:
                task.pop("folderId", None)
                task.pop("folderName", None)
                write_task_bundle(path.parent.name, task)
        except (OSError, ValueError, json.JSONDecodeError):
            pass


class MoveTaskToFolderRequest(BaseModel):
    folderId: str | None = None  # None = move to root


@app.put("/api/tasks/{task_id}/folder")
async def move_task_to_folder(task_id: str, req: MoveTaskToFolderRequest):
    """Move a task to a folder (or root if folderId is null)."""
    path = task_file_path(task_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    task = await run_in_threadpool(read_task_file, path)
    # Hydrate to preserve result.json
    task = hydrate_task_detail(task_id, task)

    if req.folderId is None:
        task.pop("folderId", None)
        task.pop("folderName", None)
    else:
        store = await run_in_threadpool(read_folder_store)
        folder = next((f for f in store.get("folders", []) if f["id"] == req.folderId), None)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        task["folderId"] = req.folderId
        task["folderName"] = folder["name"]

    stored = await run_in_threadpool(write_task_bundle, task_id, task)
    return {"ok": True, "task": task_summary(stored)}


def remove_task_dir(task_id: str) -> None:
    ensure_task_data_dir()
    path = task_dir_path(task_id).resolve()
    if path.parent != TASK_DATA_DIR:
        raise HTTPException(status_code=400, detail="Invalid task path")
    if path.exists():
        shutil.rmtree(path)


def clear_task_dirs() -> None:
    ensure_task_data_dir()
    for path in TASK_DATA_DIR.iterdir():
        if path.is_dir() and re.fullmatch(r"[A-Za-z0-9_-]{6,80}", path.name):
            shutil.rmtree(path)


def reset_stuck_processing_tasks() -> None:
    """Reset tasks stuck in 'processing' state from a previous server crash.

    IMPORTANT: Must hydrate the task before writing to preserve result.json.
    Without hydration, split_task_for_storage sees no results and deletes result.json,
    permanently losing all OCR data.
    """
    ensure_task_data_dir()
    reset_count = 0
    for path in TASK_DATA_DIR.iterdir():
        if not path.is_dir():
            continue
        task_path = path / "task.json"
        if not task_path.exists():
            continue
        try:
            task = read_task_file(task_path)
            if task.get("status") == "processing":
                # Hydrate to preserve result.json — without this, write_task_bundle
                # would see an empty result_payload and DELETE result.json
                task = hydrate_task_detail(path.name, task)
                # Reset processing batches to pending so they can be resumed
                for batch in task.get("batches", []):
                    if isinstance(batch, dict) and batch.get("status") == "processing":
                        batch["status"] = "pending"
                task["status"] = "pending"
                task["error"] = "Server restarted during processing; task reset to pending."
                write_task_bundle(path.name, task)
                reset_count += 1
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if reset_count:
        logger.info("Reset %d stuck processing task(s) on startup", reset_count)


# ---------------------------------------------------------------------------
# Chunked upload session helpers
# ---------------------------------------------------------------------------

# Per-upload locks to prevent TOCTOU races on meta.json during concurrent chunk uploads
_upload_locks: dict[str, asyncio.Lock] = {}


def _upload_session_dir(upload_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,80}", upload_id):
        raise HTTPException(status_code=400, detail="Invalid upload id")
    return UPLOAD_SESSION_DIR / upload_id


def ensure_upload_dir() -> None:
    """Ensure the upload session directory exists and is writable."""
    UPLOAD_SESSION_DIR.mkdir(parents=True, exist_ok=True)


def _upload_meta_path(upload_id: str) -> Path:
    return _upload_session_dir(upload_id) / "meta.json"


def _read_upload_meta(upload_id: str) -> dict:
    path = _upload_meta_path(upload_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise HTTPException(status_code=500, detail=f"Failed to read upload session: {err}") from err


def _write_upload_meta(upload_id: str, meta: dict) -> None:
    path = _upload_meta_path(upload_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def cleanup_stale_uploads() -> None:
    """Remove upload sessions older than UPLOAD_SESSION_TTL_HOURS."""
    if not UPLOAD_SESSION_DIR.exists():
        return
    cutoff = time.time() - UPLOAD_SESSION_TTL_HOURS * 3600
    for entry in UPLOAD_SESSION_DIR.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            shutil.rmtree(entry, ignore_errors=True)
            _upload_locks.pop(entry.name, None)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("createdAt", 0) < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                _upload_locks.pop(entry.name, None)
                logger.info("Cleaned up stale upload session: %s", entry.name)
        except (OSError, json.JSONDecodeError):
            shutil.rmtree(entry, ignore_errors=True)
            _upload_locks.pop(entry.name, None)


def reassemble_chunks(upload_id: str, target_path: Path) -> int:
    """Reassemble all chunks into a single file at *target_path*. Returns total bytes."""
    meta = _read_upload_meta(upload_id)
    session_dir = _upload_session_dir(upload_id)
    total_chunks = meta["totalChunks"]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with target_path.open("wb") as out:
        for index in range(total_chunks):
            chunk_path = session_dir / "chunks" / str(index)
            if not chunk_path.exists():
                raise HTTPException(status_code=400, detail=f"Missing chunk {index}")
            # Stream-copy to avoid loading each chunk fully into memory
            with chunk_path.open("rb") as inp:
                while True:
                    block = inp.read(UPLOAD_CHUNK_SIZE)
                    if not block:
                        break
                    out.write(block)
                    total += len(block)
    return total


async def read_upload_bytes(file: UploadFile, max_bytes: int | None = None) -> bytes:
    chunks = []
    total = 0
    limit = max_bytes if max_bytes and max_bytes > 0 else None
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if limit and total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file is too large. Max upload size is {limit / 1024 / 1024:.0f} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def write_upload_to_path(file: UploadFile, path: Path, max_bytes: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    limit = max_bytes if max_bytes and max_bytes > 0 else None
    try:
        with path.open("wb") as buffer:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if limit and total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file is too large. Max upload size is {limit / 1024 / 1024:.0f} MB.",
                    )
                buffer.write(chunk)
    except Exception:
        if path.exists():
            path.unlink()
        raise
    return total


def get_pdf_page_count(source_path: Path) -> int:
    """Read only the PDF cross-reference table to get page count. O(1) for most PDFs."""
    import fitz

    doc = fitz.open(str(source_path))
    try:
        return doc.page_count
    finally:
        doc.close()


def extract_pdf_pages(source_path: Path, start_page: int, end_page: int, output_path: Path | None = None) -> Path:
    """Extract page range from source PDF using PyMuPDF (fitz).

    PyMuPDF streams pages from disk without loading the entire PDF into memory,
    making it suitable for files >1 GB.  Falls back to pypdf if fitz is unavailable.
    """
    try:
        import fitz

        doc = fitz.open(str(source_path))
        total_pages = doc.page_count
        if total_pages <= 0:
            doc.close()
            raise ValueError("Source PDF has no pages")
        if start_page < 1 or end_page < start_page or start_page > total_pages:
            doc.close()
            raise ValueError(f"Invalid page range {start_page}-{end_page} for {total_pages} pages")

        end_page = min(end_page, total_pages)

        # Create a new PDF with only the selected pages
        out_doc = fitz.open()
        for page_index in range(start_page - 1, end_page):
            out_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
        doc.close()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_doc.save(str(output_path), deflate=True, garbage=3)
            out_doc.close()
            return output_path

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                out_doc.save(tmp_file.name, deflate=True, garbage=3)
                out_doc.close()
                return Path(tmp_file.name)
        except Exception:
            raise
    except ImportError:
        pass

    # Fallback: pypdf (loads entire PDF into memory — not ideal for large files)
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(source_path))
    total_pages = len(reader.pages)
    if total_pages <= 0:
        raise ValueError("Source PDF has no pages")
    if start_page < 1 or end_page < start_page or start_page > total_pages:
        raise ValueError(f"Invalid page range {start_page}-{end_page} for {total_pages} pages")

    end_page = min(end_page, total_pages)
    writer = PdfWriter()
    for page_index in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_index])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            writer.write(f)
        return output_path

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            writer.write(tmp_file)
            return Path(tmp_file.name)
    except Exception:
        raise


@app.post("/api/tasks/{task_id}/source")
async def upload_task_source(task_id: str, file: UploadFile = File(...)):
    """Persist the original uploaded source outside task.json."""
    source_path = task_source_path(task_id)
    temp_path = source_path.with_suffix(".tmp")
    size = await write_upload_to_path(file, temp_path, MAX_REQUEST_BYTES)
    temp_path.replace(source_path)
    return {
        "ok": True,
        "url": task_source_url(task_id),
        "size": size,
        "filename": Path(file.filename or "source").name,
        "contentType": file.content_type or "application/octet-stream",
    }


@app.get("/api/tasks/{task_id}/source")
async def get_task_source(task_id: str, request: Request):
    """Return the original uploaded source file for previewing or resumable parsing.

    Supports HTTP Range requests so that PDF.js can load individual pages
    on demand without downloading the entire file — critical for files >1 GB.
    """
    source_path = task_source_path(task_id)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Task source not found")

    media_type = "application/octet-stream"
    filename = "source"
    task_path = task_file_path(task_id)
    if task_path.exists():
        try:
            task = await run_in_threadpool(read_task_file, task_path)
            media_type = task.get("mimeType") or media_type
            filename = task.get("originalName") or task.get("name") or filename
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    file_size = source_path.stat().st_size
    range_header = request.headers.get("range")

    # Handle Range request (for PDF.js lazy page loading)
    if range_header:
        import re as _re
        match = _re.match(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else file_size - 1
            if start >= file_size:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
            end = min(end, file_size - 1)
            content_length = end - start + 1

            async def _range_stream():
                with source_path.open("rb") as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = f.read(min(UPLOAD_CHUNK_SIZE, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                _range_stream(),
                status_code=206,
                media_type=media_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Content-Length": str(content_length),
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": f'inline; filename="{filename}"',
                },
            )

    # Full file response
    return FileResponse(source_path, media_type=media_type, filename=filename)


@app.get("/api/tasks/{task_id}/source/pages")
async def get_task_source_pages(
    task_id: str,
    start_page: int = Query(..., ge=1),
    end_page: int = Query(..., ge=1),
):
    """Return a compact PDF containing only a page range from the source PDF."""
    source_path = task_source_path(task_id)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Task source not found")
    if end_page < start_page:
        raise HTTPException(status_code=400, detail="end_page must be greater than or equal to start_page")

    try:
        batch_dir = task_dir_path(task_id) / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        output_path = batch_dir / f"pages_{start_page}_{end_page}.pdf"
        result_path = await run_in_threadpool(extract_pdf_pages, source_path, start_page, end_page, output_path)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("Failed to extract PDF pages")
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF pages: {err}") from err

    return FileResponse(result_path, media_type="application/pdf", filename=f"pages_{start_page}_{end_page}.pdf")


@app.get("/api/tasks")
async def list_tasks():
    """List locally persisted document parsing task summaries."""
    tasks = await run_in_threadpool(list_task_summaries)
    return {"tasks": tasks}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str, lite: bool = Query(True)):
    """Return one locally persisted task.

    By default returns a lightweight response that omits heavy fields
    (images, ocrResults) to avoid OOM on large results.  Set ?lite=false
    to load the full result payload.
    """
    path = task_file_path(task_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        task = await run_in_threadpool(read_task_file, path)
    except (OSError, ValueError, json.JSONDecodeError) as err:
        logger.warning("Failed to read task file %s: %s", path, err)
        raise HTTPException(status_code=500, detail="Failed to read task")
    if task_source_path(task_id).exists() and not task.get("sourceUrl"):
        task["sourceUrl"] = task_source_url(task_id)

    if lite:
        task = hydrate_task_detail_lite(task_id, task)
    else:
        task = hydrate_task_detail(task_id, task)

    task["detailLoaded"] = True
    return task


@app.get("/api/tasks/{task_id}/result")
async def get_task_result(
    task_id: str,
    fields: str = Query("", description="Comma-separated fields to include: images,ocrResults,markdown,translation"),
    image_offset: int = Query(0, ge=0),
    image_limit: int = Query(200, ge=1, le=1000),
    ocr_offset: int = Query(0, ge=0),
    ocr_limit: int = Query(100, ge=1, le=1000),
):
    """Return heavy result data for a task, with pagination support.

    This endpoint allows the frontend to lazily load images and ocrResults
    on demand, avoiding OOM when a task has hundreds of pages of results.
    """
    result_path = task_result_path(task_id)
    if not result_path.exists():
        return {"images": {}, "ocrResults": []}

    try:
        result_payload = await run_in_threadpool(read_json_file, result_path)
    except (OSError, ValueError, json.JSONDecodeError) as err:
        logger.warning("Failed to read result file for task %s: %s", task_id, err)
        raise HTTPException(status_code=500, detail="Failed to read task result")

    requested = set(f.strip() for f in fields.split(",") if f.strip()) if fields else set()
    # If no fields specified, return all
    include_all = len(requested) == 0
    response = {}

    if include_all or "images" in requested:
        all_images = result_payload.get("images", {})
        if isinstance(all_images, dict):
            image_keys = list(all_images.keys())
            paginated_keys = image_keys[image_offset:image_offset + image_limit]
            response["images"] = {k: all_images[k] for k in paginated_keys if k in all_images}
            response["imageTotal"] = len(image_keys)
        else:
            response["images"] = {}
            response["imageTotal"] = 0

    if include_all or "ocrResults" in requested:
        all_ocr = result_payload.get("ocrResults", [])
        if isinstance(all_ocr, list):
            response["ocrResults"] = all_ocr[ocr_offset:ocr_offset + ocr_limit]
            response["ocrTotal"] = len(all_ocr)
        else:
            response["ocrResults"] = []
            response["ocrTotal"] = 0

    if include_all or "markdown" in requested:
        response["markdown"] = result_payload.get("markdown", "")
        bm = result_payload.get("batchMarkdown")
        if isinstance(bm, dict):
            response["batchMarkdown"] = bm

    if include_all or "translation" in requested:
        response["translation"] = result_payload.get("translation")
        response["translationLang"] = result_payload.get("translationLang")

    return response


@app.put("/api/tasks/{task_id}")
async def save_task(task_id: str, request: Request):
    """Persist one task to the local project data directory."""
    task = await request.json()
    if not isinstance(task, dict):
        raise HTTPException(status_code=400, detail="Task payload must be a JSON object")
    if task.get("id") != task_id:
        raise HTTPException(status_code=400, detail="Task id mismatch")

    stored_task = await run_in_threadpool(write_task_bundle, task_id, task)
    return {"ok": True, "task": task_summary(stored_task)}


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete one locally persisted task."""
    await run_in_threadpool(remove_task_dir, task_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Translation endpoint
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    targetLang: str = "zh-CN"  # Target language code
    sourceLang: str | None = None  # Auto-detect if not specified
    chunkIndex: int | None = None  # Translate specific batch chunk (0-based), None = all


@app.post("/api/tasks/{task_id}/translate")
async def translate_task(task_id: str, req: TranslateRequest):
    """Translate the OCR markdown result while preserving formatting.

    Streams the translation in SSE so the frontend can show progress.
    Uses an OpenAI-compatible API (e.g., GPT-4o-mini, DeepSeek, Qwen).
    """
    safe_task_id(task_id)
    if not TRANSLATE_API_URL:
        raise HTTPException(status_code=501, detail="Translation API not configured. Set PANDOCR_TRANSLATE_API_URL and PANDOCR_TRANSLATE_API_KEY.")
    if not TRANSLATE_API_KEY:
        raise HTTPException(status_code=501, detail="Translation API key not configured. Set PANDOCR_TRANSLATE_API_KEY.")

    task = read_task_file(task_file_path(task_id))
    task = hydrate_task_detail(task_id, task)
    markdown = task.get("markdown", "")

    if not markdown or not markdown.strip():
        raise HTTPException(status_code=400, detail="No markdown content to translate.")

    # Split into chunks of ~2000 chars at paragraph boundaries
    chunks = split_markdown_for_translation(markdown)
    if req.chunkIndex is not None:
        if req.chunkIndex < 0 or req.chunkIndex >= len(chunks):
            raise HTTPException(status_code=400, detail=f"chunkIndex must be in [0, {len(chunks)})")
        chunks = [chunks[req.chunkIndex]]

    async def translate_stream():
        translated_parts = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            progress = {"currentChunk": i, "totalChunks": total, "percent": int(i / total * 100)}
            yield f"data: {json.dumps({'type': 'progress', **progress})}\n\n"

            try:
                translated = await translate_chunk(chunk, req.targetLang, req.sourceLang)
                translated_parts.append(translated)
                yield f"data: {json.dumps({'type': 'chunk', 'index': i, 'text': translated})}\n\n"
            except Exception as err:
                logger.error("Translation chunk %s failed: %s", i, err)
                yield f"data: {json.dumps({'type': 'error', 'index': i, 'error': str(err)})}\n\n"
                translated_parts.append(chunk)  # Keep original on error

        # Merge and save
        full_translation = "\n\n".join(translated_parts)
        task["translation"] = full_translation
        task["translationLang"] = req.targetLang
        write_task_bundle(task_id, task)

        yield f"data: {json.dumps({'type': 'done', 'percent': 100, 'lang': req.targetLang})}\n\n"

    return StreamingResponse(translate_stream(), media_type="text/event-stream")


def split_markdown_for_translation(markdown: str, max_chars: int = 3000) -> list[str]:
    """Split markdown into chunks for translation.

    Strategy:
    1. Split at paragraph boundaries (double newline).
    2. If a single paragraph exceeds max_chars, split at sentence boundaries.
    3. Never break mid-sentence — always end at a sentence-ending punctuation
       (. ! ? 。！？) followed by whitespace or end-of-string.
    4. Each chunk includes enough context (the previous chunk's last sentence)
       to maintain translation coherence across chunk boundaries.
    """
    if not markdown or not markdown.strip():
        return [markdown] if markdown else []

    paragraphs = markdown.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if not para.strip():
            # Preserve empty paragraphs for structure
            if current:
                current += "\n\n"
            continue

        # If adding this paragraph stays within limit, just append
        if len(current) + len(para) + 2 <= max_chars:
            current = current + "\n\n" + para if current else para
            continue

        # Current chunk is full — flush it
        if current:
            chunks.append(current)

        # If the paragraph itself fits, start new chunk with it
        if len(para) <= max_chars:
            current = para
            continue

        # Single paragraph too long — split at sentence boundaries
        sentences = _split_sentences(para)
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 > max_chars and current:
                chunks.append(current)
                current = sent
            else:
                current = current + " " + sent if current else sent

    if current:
        chunks.append(current)

    return chunks if chunks else [markdown]


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries, preserving the delimiter.

    Handles both CJK (。！？) and Latin (. ! ?) sentence endings.
    Keeps the punctuation attached to the preceding sentence.
    """
    import re
    # Split after sentence-ending punctuation followed by whitespace or end
    parts = re.split(r'(?<=[。！？.!?])\s+', text)
    # Filter empty parts but preserve structure
    return [p for p in parts if p.strip()] if parts else [text]


async def translate_chunk(text: str, target_lang: str, source_lang: str | None = None) -> str:
    """Call an OpenAI-compatible API to translate text while preserving Markdown/HTML formatting."""
    lang_name = LANG_CODE_TO_NAME.get(target_lang, target_lang)
    lang_native = LANG_CODE_TO_NATIVE.get(target_lang, lang_name)
    source_hint = (
        f"The source language is {LANG_CODE_TO_NAME.get(source_lang, source_lang)}."
        if source_lang else "Auto-detect the source language."
    )

    system_prompt = (
        f"You are an expert academic and technical translator. Translate the following text into {lang_name}（{lang_native}）. "
        f"{source_hint}\n\n"
        "ABSOLUTE RULES — ANY VIOLATION IS UNACCEPTABLE:\n\n"
        "1. OUTPUT: Return ONLY the translated text. No explanations, notes, commentary, alternatives, or hedging words. Nothing except the translation.\n\n"
        "2. TRANSLATE EVERYTHING — including:\n"
        "   - ALL headings and titles (## 三、 电吉他 → ## 3. Electric Guitar, # 绪论 → # Introduction)\n"
        "   - Chinese numerals and ordinals in headings (一、→ 1., 三、→ 3., 第十二章 → Chapter 12)\n"
        "   - Table headers, figure captions, footnotes, labels, annotations\n"
        "   - Partial sentences at chunk boundaries — translate them fully even if the start/end seems cut off\n"
        "   - Do NOT skip any line, heading, label, or annotation — everything readable must be translated\n\n"
        "3. FIDELITY — translate accurately, do NOT add, omit, reinterpret, or embellish:\n"
        "   - Every sentence in the source MUST appear in the translation — no skipping, no summarizing, no expanding\n"
        "   - Do NOT add transitional phrases, explanatory asides, or 'helpful' context not present in the original\n"
        "   - Do NOT soften, rephrase, or simplify the author's tone — preserve the original voice and register\n"
        "   - If the source is ambiguous, preserve the ambiguity faithfully rather than resolving it\n\n"
        "4. TECHNICAL & SCIENTIFIC PRECISION:\n"
        "   - Use established terminology for the relevant field (mathematics, physics, chemistry, biology, engineering, medicine, law, etc.)\n"
        "   - When a term has a standard translation in the target language, use that standard — do not invent alternatives\n"
        "   - Preserve all numeric values exactly: 3.14 stays 3.14, 10^6 stays 10^6, 1/2 stays 1/2\n"
        "   - Preserve unit symbols unchanged: kg, m/s, MHz, kJ/mol, etc.\n"
        "   - Preserve chemical formulas and equations: H2O, CH3COOH, E=mc2, F=ma\n"
        "   - Preserve variable names and symbols used in formulas: x, theta, alpha, etc.\n\n"
        "5. MATH & CODE — copy verbatim, zero alteration:\n"
        "   - LaTeX: $...$, $$...$$, \\[...\\], \\(...\\) — do NOT translate content inside math delimiters\n"
        "   - Inline code (`...`) and code blocks (```...```) — copy verbatim\n"
        "   - Equations, theorems, proofs, algorithms — preserve formatting exactly\n\n"
        "6. FORMATTING PRESERVATION:\n"
        "   - Markdown: # headers, **bold**, *italic*, [links](url), - lists, > blockquotes, | tables |, --- rules\n"
        "   - HTML tags: keep all tags and attributes; only translate text content inside tags\n"
        "   - Do NOT translate: URLs, file paths, image filenames, CSS class names, data: URIs\n"
        "   - Keep the same paragraph count, heading hierarchy, list item count, table dimensions\n\n"
        "7. QUALITY:\n"
        "   - Natural, fluent target language — not stiff or machine-sounding\n"
        "   - Same academic/professional register as the original\n"
        "   - Proper nouns: use the established convention in the target language (e.g. Einstein → 爱因斯坦, Fourier → 傅里叶)"
    )

    headers = {
        "Authorization": f"Bearer {TRANSLATE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": TRANSLATE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
    }

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            TRANSLATE_API_URL,
            headers=headers,
            json=payload,
        )
        if resp.status_code != 200:
            logger.error("Translation API error %s: %s", resp.status_code, resp.text[:500])
            raise HTTPException(status_code=resp.status_code, detail=f"Translation API error: {resp.text[:200]}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]


# Language code to name mapping
LANG_CODE_TO_NAME = {
    "zh-CN": "Simplified Chinese", "zh-TW": "Traditional Chinese",
    "en": "English", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "it": "Italian", "nl": "Dutch", "pl": "Polish",
    "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
    "id": "Indonesian", "ms": "Malay", "hi": "Hindi",
}

LANG_CODE_TO_NATIVE = {
    "zh-CN": "简体中文", "zh-TW": "繁體中文",
    "en": "English", "ja": "日本語", "ko": "한국어",
    "fr": "Français", "de": "Deutsch", "es": "Español",
    "pt": "Português", "ru": "Русский", "ar": "العربية",
    "it": "Italiano", "nl": "Nederlands", "pl": "Polski",
    "tr": "Türkçe", "vi": "Tiếng Việt", "th": "ไทย",
    "id": "Bahasa Indonesia", "ms": "Bahasa Melayu", "hi": "हिन्दी",
}


@app.get("/api/translate/config")
async def get_translate_config():
    """Check if translation is configured."""
    return {
        "available": bool(TRANSLATE_API_URL and TRANSLATE_API_KEY),
        "model": TRANSLATE_MODEL if TRANSLATE_API_URL else None,
    }


@app.delete("/api/tasks")
async def clear_tasks():
    """Delete all locally persisted tasks."""
    await run_in_threadpool(clear_task_dirs)
    return {"ok": True}


@app.get("/api/tasks/{task_id}/source/info")
async def get_task_source_info(task_id: str):
    """Return PDF metadata (page count, size, mimeType) without loading full content."""
    source_path = task_source_path(task_id)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Task source not found")

    size = source_path.stat().st_size
    mime_type = "application/octet-stream"
    source_kind = "unknown"
    page_count = 0

    task_path = task_file_path(task_id)
    if task_path.exists():
        try:
            task = await run_in_threadpool(read_task_file, task_path)
            mime_type = task.get("mimeType") or mime_type
            source_kind = task.get("sourceKind") or source_kind
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    # Detect PDF and get page count
    try:
        with source_path.open("rb") as f:
            header = f.read(5)
        if header == b"%PDF-":
            mime_type = "application/pdf"
            source_kind = "pdf"
            page_count = await run_in_threadpool(get_pdf_page_count, source_path)
    except Exception as err:
        logger.warning("Failed to read source file header: %s", err)

    return {
        "pageCount": page_count,
        "size": size,
        "mimeType": mime_type,
        "sourceKind": source_kind,
    }


# ---------------------------------------------------------------------------
# Chunked upload endpoints
# ---------------------------------------------------------------------------

@app.post("/api/uploads")
async def create_upload_session(request: CreateUploadRequest):
    """Create a chunked upload session. Returns upload_id and chunk plan."""
    if request.totalSize <= 0:
        raise HTTPException(status_code=400, detail="totalSize must be positive")
    if request.totalSize > MAX_TOTAL_UPLOAD_BYTES:
        max_mb = MAX_TOTAL_UPLOAD_BYTES / 1024 / 1024
        raise HTTPException(status_code=413, detail=f"File too large. Max total upload size is {max_mb:.0f} MB.")
    if request.chunkSize < 1024 * 1024:
        raise HTTPException(status_code=400, detail="chunkSize must be at least 1 MB")
    if request.chunkSize > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="chunkSize must be at most 100 MB")

    upload_id = secrets.token_urlsafe(12)
    total_chunks = (request.totalSize + request.chunkSize - 1) // request.chunkSize

    meta = {
        "uploadId": upload_id,
        "filename": request.filename,
        "totalSize": request.totalSize,
        "chunkSize": request.chunkSize,
        "totalChunks": total_chunks,
        "receivedChunks": [],
        "taskId": request.taskId,
        "createdAt": time.time(),
        "completed": False,
    }

    session_dir = _upload_session_dir(upload_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "chunks").mkdir(parents=True, exist_ok=True)
    _write_upload_meta(upload_id, meta)

    return {
        "uploadId": upload_id,
        "chunkSize": request.chunkSize,
        "totalChunks": total_chunks,
        "receivedChunks": [],
    }


@app.put("/api/uploads/{upload_id}/chunks/{chunk_index}")
async def upload_chunk(upload_id: str, chunk_index: int, file: UploadFile = File(...)):
    """Upload a single chunk. Idempotent — re-uploading overwrites silently."""
    lock = _upload_locks.setdefault(upload_id, asyncio.Lock())
    async with lock:
        meta = _read_upload_meta(upload_id)
        if meta.get("completed"):
            raise HTTPException(status_code=400, detail="Upload session already completed")

        total_chunks = meta["totalChunks"]
        if chunk_index < 0 or chunk_index >= total_chunks:
            raise HTTPException(status_code=400, detail=f"chunk_index must be in [0, {total_chunks})")

        chunk_dir = _upload_session_dir(upload_id) / "chunks"
        chunk_path = chunk_dir / str(chunk_index)

        # Write chunk to disk
        size = await write_upload_to_path(file, chunk_path, meta["chunkSize"] + 1024 * 1024)

        # Update metadata
        received = set(meta.get("receivedChunks", []))
        received.add(chunk_index)
        meta["receivedChunks"] = sorted(received)
        _write_upload_meta(upload_id, meta)

    return {"ok": True, "receivedChunks": meta["receivedChunks"], "chunkSize": size}


@app.get("/api/uploads/{upload_id}")
async def get_upload_status(upload_id: str):
    """Return upload session status, including which chunks are received (for resume)."""
    meta = _read_upload_meta(upload_id)
    return {
        "uploadId": meta["uploadId"],
        "totalSize": meta["totalSize"],
        "chunkSize": meta["chunkSize"],
        "totalChunks": meta["totalChunks"],
        "receivedChunks": meta.get("receivedChunks", []),
        "completed": meta.get("completed", False),
    }


@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str):
    """Verify all chunks received, reassemble into source.bin, clean up chunks."""
    lock = _upload_locks.setdefault(upload_id, asyncio.Lock())
    async with lock:
        meta = _read_upload_meta(upload_id)
        if meta.get("completed"):
            raise HTTPException(status_code=400, detail="Upload session already completed")

        total_chunks = meta["totalChunks"]
        received = set(meta.get("receivedChunks", []))
        missing = set(range(total_chunks)) - received
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing chunks: {sorted(missing)}. Upload {len(missing)} more chunk(s) to complete.",
            )

        # Determine target path
        task_id = meta.get("taskId")
        if task_id:
            safe_task_id(task_id)  # validate
            target_path = task_source_path(task_id)
        else:
            target_path = _upload_session_dir(upload_id) / "assembled.bin"

        # Reassemble
        try:
            total_size = await run_in_threadpool(reassemble_chunks, upload_id, target_path)
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"Failed to reassemble chunks: {err}") from err

        # Mark completed and clean up chunk files
        meta["completed"] = True
        _write_upload_meta(upload_id, meta)

        # Remove chunk files to free disk space
        chunk_dir = _upload_session_dir(upload_id) / "chunks"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)

    # Clean up lock entry — upload is done, no more chunks expected
    _upload_locks.pop(upload_id, None)

    result = {"ok": True, "size": total_size}
    if task_id:
        result["url"] = task_source_url(task_id)
    return result


@app.post("/api/convert/to-pdf")
async def convert_to_pdf(file: UploadFile = File(...)):
    """Convert PPT/PPTX/DOC/DOCX to PDF using LibreOffice."""
    logger.info("Received conversion request for: %s", file.filename)

    if not shutil.which("soffice"):
        raise HTTPException(
            status_code=500,
            detail="LibreOffice (soffice) not found on server. Please install it to support Office conversion.",
        )

    filename = Path(file.filename or "upload").name
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".ppt", ".pptx", ".doc", ".docx"]:
        raise HTTPException(status_code=400, detail="Only .ppt, .pptx, .doc, and .docx files are supported.")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, filename)
            await write_upload_to_path(file, Path(input_path), MAX_REQUEST_BYTES)

            cmd = [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                temp_dir,
                input_path,
            ]

            logger.info("Running conversion command: %s", " ".join(cmd))
            result = await run_in_threadpool(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode != 0:
                logger.warning("Conversion failed: %s", result.stderr)
                raise HTTPException(status_code=500, detail=f"Conversion failed: {result.stderr}")

            pdfs = [f for f in os.listdir(temp_dir) if f.lower().endswith(".pdf")]
            if not pdfs:
                raise HTTPException(status_code=500, detail="PDF file not generated")

            pdf_path = os.path.join(temp_dir, pdfs[0])
            logger.info("Conversion successful, sending back: %s", pdf_path)

            with open(pdf_path, "rb") as f:
                pdf_content = await run_in_threadpool(f.read)

            return Response(content=pdf_content, media_type="application/pdf")

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="File conversion timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during conversion")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Background task manager for server-side OCR processing
# ---------------------------------------------------------------------------

class BackgroundTaskManager:
    """Manage background OCR processing tasks with progress tracking."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_flags: dict[str, asyncio.Event] = {}
        self._progress: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def start_processing(self, task_id: str, model_id: str, ocr_options: dict) -> None:
        async with self._lock:
            existing = self._tasks.get(task_id)
            if existing and not existing.done():
                raise HTTPException(status_code=409, detail="Task is already being processed")

            # Clean up any previous done task state
            if existing and existing.done():
                self.clear(task_id)

            cancel_flag = asyncio.Event()
            self._cancel_flags[task_id] = cancel_flag
            self._progress[task_id] = {
                "status": "queued",
                "currentBatchIndex": 0,
                "totalBatches": 0,
                "currentBatchLabel": "",
                "percent": 0,
                "startedAt": time.time(),
                "updatedAt": time.time(),
                "error": None,
            }
            self._tasks[task_id] = asyncio.create_task(
                process_task_background(task_id, model_id, ocr_options, cancel_flag)
            )

    def cancel_processing(self, task_id: str) -> bool:
        flag = self._cancel_flags.get(task_id)
        if flag and task_id in self._tasks and not self._tasks[task_id].done():
            flag.set()
            # Replace progress dict entirely (immutable update) to avoid
            # in-place mutation racing with SSE comparison
            old = self._progress.get(task_id, {})
            self._progress[task_id] = {**old, "status": "cancelling", "updatedAt": time.time()}
            return True
        return False

    def get_progress(self, task_id: str) -> dict | None:
        return self._progress.get(task_id)

    def set_progress(self, task_id: str, progress: dict) -> None:
        """Replace progress dict for a task (preferred over direct _progress access)."""
        self._progress[task_id] = progress

    def is_running(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        return task is not None and not task.done()

    def remove_done(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task and task.done():
            del self._tasks[task_id]
            self._cancel_flags.pop(task_id, None)
            self._progress.pop(task_id, None)

    def clear(self, task_id: str) -> None:
        """Force-clear all state for a task, even if it's still running."""
        self._tasks.pop(task_id, None)
        self._cancel_flags.pop(task_id, None)
        self._progress.pop(task_id, None)


task_manager = BackgroundTaskManager()


def _build_ocr_request_from_options(ocr_options: dict) -> "OCRRequest":
    """Build an OCRRequest from the options dict sent by the frontend."""
    return OCRRequest(
        useLayoutDetection=ocr_options.get("useLayoutDetection", True),
        useDocUnwarping=ocr_options.get("useDocUnwarping", False),
        useDocOrientationClassify=ocr_options.get("useDocOrientationClassify", False),
        useTextlineOrientation=ocr_options.get("useTextlineOrientation", False),
        useChartRecognition=ocr_options.get("useChartRecognition", False),
        useSealRecognition=ocr_options.get("useSealRecognition", True),
        formatBlockContent=ocr_options.get("formatBlockContent", True),
        showFormulaNumber=ocr_options.get("showFormulaNumber", True),
        markdownIgnoreLabels=ocr_options.get("markdownIgnoreLabels", []),
        layoutThreshold=ocr_options.get("layoutThreshold"),
        layoutNms=ocr_options.get("layoutNms"),
        layoutUnclipRatio=ocr_options.get("layoutUnclipRatio"),
        layoutMergeBboxesMode=ocr_options.get("layoutMergeBboxesMode"),
        repetitionPenalty=ocr_options.get("repetitionPenalty"),
        temperature=ocr_options.get("temperature"),
        topP=ocr_options.get("topP"),
        minPixels=ocr_options.get("minPixels"),
        maxPixels=ocr_options.get("maxPixels"),
        visualize=ocr_options.get("visualize"),
    )


async def process_task_background(
    task_id: str, model_id: str, ocr_options: dict, cancel_flag: asyncio.Event
) -> None:
    """Process all batches of a task sequentially in the background."""
    try:
        task_path = task_file_path(task_id)
        if not task_path.exists():
            logger.error("Task file not found: %s", task_path)
            task_manager.set_progress(task_id, {
                "status": "error", "error": "Task file not found",
                "updatedAt": time.time(), "percent": 0,
                "currentBatchIndex": 0, "totalBatches": 0, "currentBatchLabel": "",
            })
            return

        task = await run_in_threadpool(read_task_file, task_path)
        # Hydrate to load existing results from result.json
        # Without this, resuming a partially-completed task would start from
        # empty markdown, losing all previously-accumulated OCR data.
        # Use lite hydration first (fast — reads standalone .md file),
        # then load ocrResults on demand (needed for accumulation).
        # Skip images — they are merged incrementally and not needed for resume.
        task = hydrate_task_detail_lite(task_id, task)
        # Load ocrResults for accumulation — needed by background processing
        # but NOT images (those are huge and only merged at write time).
        result_path = task_result_path(task_id)
        if result_path.exists() and not task.get("ocrResults"):
            try:
                result_payload = read_json_file(result_path)
                if "ocrResults" in result_payload:
                    task["ocrResults"] = result_payload["ocrResults"]
            except (OSError, ValueError, json.JSONDecodeError) as err:
                logger.warning("Failed to load ocrResults for task %s: %s", task_id, err)
        batches = task.get("batches", [])
        if not batches:
            logger.warning("No batches found for task %s", task_id)
            task_manager.set_progress(task_id, {
                "status": "error", "error": "No batches to process",
                "updatedAt": time.time(), "percent": 0,
                "currentBatchIndex": 0, "totalBatches": 0, "currentBatchLabel": "",
            })
            return

        # Keep existing results intact — only initialize if truly missing
        if not task.get("markdown"):
            task["markdown"] = ""
        if not isinstance(task.get("images"), dict):
            task["images"] = {}
        if not isinstance(task.get("ocrResults"), list):
            task["ocrResults"] = []

        ocr_request = _build_ocr_request_from_options(ocr_options)
        total_batches = len(batches)

        task_manager.set_progress(task_id, {
            "status": "running",
            "currentBatchIndex": 0,
            "totalBatches": total_batches,
            "currentBatchLabel": batches[0].get("label", "") if batches else "",
            "percent": 0,
            "startedAt": time.time(),
            "updatedAt": time.time(),
            "error": None,
        })
        source_path = task_source_path(task_id)
        has_source = source_path.exists()

        for i, batch in enumerate(batches):
            if cancel_flag.is_set():
                task["status"] = "paused"
                await run_in_threadpool(write_task_bundle, task_id, task)
                task_manager.set_progress(task_id, {
                    "status": "cancelled",
                    "currentBatchIndex": i,
                    "totalBatches": total_batches,
                    "currentBatchLabel": batch.get("label", ""),
                    "percent": round(i / total_batches * 100),
                    "updatedAt": time.time(),
                    "error": None,
                })
                return

            if batch.get("status") == "completed":
                continue

            # Update progress
            task_manager.set_progress(task_id, {
                "status": "running",
                "currentBatchIndex": i,
                "totalBatches": total_batches,
                "currentBatchLabel": batch.get("label", ""),
                "percent": round(i / total_batches * 100),
                "updatedAt": time.time(),
                "error": None,
            })
            try:
                # Extract page range from source PDF or use entire source
                batch_file_path: Path | None = None
                start_page = batch.get("startPage")
                end_page = batch.get("endPage")

                if has_source and batch.get("fileType") == 0 and start_page and end_page:
                    batch_dir = task_dir_path(task_id) / "batches"
                    batch_dir.mkdir(parents=True, exist_ok=True)
                    output_path = batch_dir / f"batch_{i}_{start_page}_{end_page}.pdf"
                    batch_file_path = await run_in_threadpool(
                        extract_pdf_pages, source_path, start_page, end_page, output_path
                    )
                elif has_source and batch.get("fileType") == 1:
                    batch_file_path = source_path

                if batch_file_path is None or not batch_file_path.exists():
                    raise RuntimeError(f"Cannot prepare batch payload for batch {i}")

                # Check batch size for base64 services
                batch_size = batch_file_path.stat().st_size
                if model_id not in ("mineru", "glm-ocr") and batch_size > MAX_BATCH_BYTES:
                    max_mb = MAX_BATCH_BYTES / 1024 / 1024
                    raise RuntimeError(
                        f"Batch {i} ({batch_size / 1024 / 1024:.1f} MB) exceeds max batch size "
                        f"({max_mb:.0f} MB). Reduce pages per batch."
                    )

                # Call OCR service
                # Set fileType from batch metadata so MinerU gets the correct filename
                ocr_request.fileType = batch.get("fileType", 0)
                if model_id == "mineru":
                    result = await run_mineru_from_file(batch_file_path, ocr_request)
                elif model_id == "glm-ocr":
                    result = await run_glm_ocr_from_file(batch_file_path, ocr_request)
                else:
                    result = await run_ocr_from_file(batch_file_path, model_id, ocr_request)

                # Accumulate results
                batch_markdown = result.get("markdown", "")
                batch_images = result.get("images", {})

                batch["status"] = "completed"
                batch["markdown"] = batch_markdown

                # Append markdown
                existing_md = task.get("markdown", "")
                if existing_md and not existing_md.endswith("\n\n"):
                    existing_md += "\n\n"
                task["markdown"] = existing_md + batch_markdown + "\n\n"

                # Merge images
                task_images = task.get("images", {})
                if isinstance(task_images, dict) and isinstance(batch_images, dict):
                    task_images.update(batch_images)

                # Accumulate OCR results
                ocr_results = task.get("ocrResults", [])
                if not isinstance(ocr_results, list):
                    ocr_results = []
                normalized = normalize_ocr_json_results(result)
                for page_index, page_result in enumerate(normalized):
                    ocr_results.append(compact_ocr_json_result(page_result, batch, page_index))
                task["ocrResults"] = ocr_results

                # Accumulate contentList
                content_list = result.get("contentList")
                if isinstance(content_list, list):
                    if not isinstance(task.get("contentList"), list):
                        task["contentList"] = []
                    task["contentList"].extend(content_list)

                task["status"] = "processing"
                task["updatedAt"] = int(time.time() * 1000)
                await run_in_threadpool(write_task_bundle, task_id, task)

            except Exception as err:
                logger.exception("Error processing batch %d for task %s", i, task_id)
                batch["status"] = "error"
                task["status"] = "error"
                task["error"] = str(err)
                await run_in_threadpool(write_task_bundle, task_id, task)
                task_manager.set_progress(task_id, {
                    "status": "error",
                    "currentBatchIndex": i,
                    "totalBatches": total_batches,
                    "currentBatchLabel": batch.get("label", ""),
                    "percent": round(i / total_batches * 100),
                    "updatedAt": time.time(),
                    "error": str(err),
                })
                return

        # All batches done
        task["status"] = "completed"
        task["updatedAt"] = int(time.time() * 1000)
        await run_in_threadpool(write_task_bundle, task_id, task)
        task_manager.set_progress(task_id, {
            "status": "completed",
            "currentBatchIndex": total_batches,
            "totalBatches": total_batches,
            "currentBatchLabel": "",
            "percent": 100,
            "updatedAt": time.time(),
            "error": None,
        })
    except Exception as err:
        logger.exception("Fatal error in background processing for task %s", task_id)
        task_manager.set_progress(task_id, {
            "status": "error",
            "currentBatchIndex": 0,
            "totalBatches": 0,
            "currentBatchLabel": "",
            "percent": 0,
            "updatedAt": time.time(),
            "error": str(err),
        })
def normalize_ocr_json_results(result: dict) -> list:
    """Normalize OCR results to a flat list of page results."""
    layout_results = result.get("layoutParsingResults", [])
    if isinstance(layout_results, list):
        return layout_results
    return [result]


def compact_ocr_json_result(page_result: dict, batch: dict, page_index: int) -> dict:
    """Compact a single OCR page result for storage.

    Merges metadata into the top level so the frontend can access
    parser, ocrLines, pageImage etc. directly (same shape as the
    client-side compactOCRJsonResult in app.js).
    """
    compact = _strip_large_ocr_fields(page_result)
    compact["batchId"] = batch.get("id")
    compact["pageIndex"] = page_index
    compact["label"] = batch.get("label", "")
    if compact.get("parser") == "pp-ocrv6" and batch:
        compact["sourcePage"] = int(batch.get("startPage", 1)) + page_index
    return compact


def _strip_large_ocr_fields(value):
    """Recursively remove large image fields (inputImage, outputImages)
    to keep stored JSON small — mirrors the frontend stripLargeOCRFields."""
    if isinstance(value, list):
        return [_strip_large_ocr_fields(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _strip_large_ocr_fields(nested)
        for key, nested in value.items()
        if key not in ("inputImage", "outputImages")
    }


async def run_ocr_from_file(file_path: Path, model_id: str, ocr_request: "OCRRequest") -> dict:
    """Load a single batch file, base64-encode, and send to OCR service."""
    file_bytes = await run_in_threadpool(file_path.read_bytes)
    raw_input: RawOCRInput = file_bytes

    if model_id == "paddleocr-vl-1.6":
        return await run_ocr_request(ocr_request, raw_input)
    elif model_id == "pp-ocrv6":
        return await run_ppocrv6_request(ocr_request, raw_input)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")


async def run_mineru_from_file(file_path: Path, ocr_request: "OCRRequest") -> dict:
    """Stream a file from disk to MinerU without loading it all into memory.

    Uses a file-stream for the multipart upload so that even multi-GB PDFs
    are sent incrementally.  Page dimensions are obtained separately via
    fitz.open(path) which also streams from disk.
    """
    await acquire_ocr_slot("mineru", "MinerU service is not ready.")
    try:
        file_type = ocr_request.fileType
        # Default to PDF (0) if fileType not set — most background batches are PDFs
        if file_type is None:
            file_type = 0
        filename = "upload.pdf" if file_type == 0 else "upload.png"

        data_payload = {
            "return_md": "true",
            "return_images": "true",
            "return_content_list": "true",
            "formula_enable": str(ocr_request.useChartRecognition).lower(),
            "table_enable": "true",
            "image_analysis": str(ocr_request.useChartRecognition).lower(),
            "parse_method": "auto",
        }
        if ocr_request.useLayoutDetection:
            data_payload["backend"] = "hybrid-engine"
        else:
            data_payload["backend"] = "pipeline"

        logger.info("Streaming request to MinerU Service at %s/file_parse", MINERU_SERVICE_URL)
        timeout = PADDLE_REQUEST_TIMEOUT if PADDLE_REQUEST_TIMEOUT > 0 else None

        # Stream the file from disk via a file handle — avoids loading the
        # entire file into memory (critical for files >1 GB).
        file_handle = await run_in_threadpool(file_path.open, "rb")
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{MINERU_SERVICE_URL}/file_parse",
                    files={"files": (filename, file_handle, "application/octet-stream")},
                    data=data_payload,
                )

                if resp.status_code != 200:
                    logger.warning("MinerU Service Error (HTTP %s): %s", resp.status_code, resp.text)
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"Upstream MinerU error: {resp.text}",
                    )
        finally:
            file_handle.close()

        # Get dimensions from the file on disk (fitz.open streams from disk)
        dimensions = []
        try:
            import fitz
            if file_type == 0:
                doc = await run_in_threadpool(fitz.open, str(file_path))
                try:
                    for page in doc:
                        dimensions.append((page.rect.width, page.rect.height))
                finally:
                    doc.close()
            else:
                from PIL import Image
                img = await run_in_threadpool(Image.open, str(file_path))
                dimensions.append((img.width, img.height))
        except Exception as e:
            logger.warning("Failed to get MinerU page dimensions: %s", e)

        return parse_mineru_response(resp.json(), dimensions)
    finally:
        await release_ocr_slot()


@app.post("/api/tasks/{task_id}/process")
async def start_task_processing(task_id: str, request: ProcessRequest):
    """Start background OCR processing. Returns immediately."""
    task_path = task_file_path(task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    await task_manager.start_processing(task_id, request.modelId, request.ocrOptions)
    return {"ok": True, "status": "started"}


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task_processing(task_id: str):
    """Request cancellation of background processing."""
    if not task_manager.cancel_processing(task_id):
        raise HTTPException(status_code=404, detail="No running processing for this task")
    return {"ok": True}


@app.get("/api/tasks/{task_id}/progress")
async def task_progress_sse(task_id: str):
    """Server-Sent Events endpoint for real-time progress updates."""

    async def event_generator():
        last_progress_json = None
        max_idle = 3600  # 1 hour max
        last_send_time = time.time()
        start = time.time()
        while True:
            progress = task_manager.get_progress(task_id)
            progress_json = json.dumps(progress, sort_keys=True) if progress else None
            if progress_json != last_progress_json:
                yield f"data: {progress_json}\n\n"
                last_progress_json = progress_json
                last_send_time = time.time()
            elif time.time() - last_send_time > 15:
                # Keep-alive comment to prevent proxy/load-balancer timeout
                yield ": keep-alive\n\n"
                last_send_time = time.time()
            if not progress or progress.get("status") in ("completed", "error", "cancelled"):
                break
            if time.time() - start > max_idle:
                yield f"data: {json.dumps({'status': 'error', 'error': 'SSE connection timed out'})}\n\n"
                break
            await asyncio.sleep(0.5)
        # Cleanup done tasks
        task_manager.remove_done(task_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class OCRRequest(BaseModel):
    image: Optional[str] = None
    fileType: Optional[int] = None
    useLayoutDetection: bool = True
    useDocUnwarping: bool = False
    useDocOrientationClassify: bool = False
    useTextlineOrientation: bool = False
    useChartRecognition: bool = False
    useSealRecognition: bool = True
    formatBlockContent: bool = True
    showFormulaNumber: bool = True
    markdownIgnoreLabels: List[str] = Field(default_factory=list)
    layoutThreshold: Optional[float] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[float] = None
    layoutMergeBboxesMode: Optional[str] = None
    repetitionPenalty: Optional[float] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    minPixels: Optional[int] = None
    maxPixels: Optional[int] = None
    visualize: Optional[bool] = None


RawOCRInput = Union[bytes, str]


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_optional_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def parse_optional_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def parse_optional_string(value) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def parse_markdown_ignore_labels(value) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [text]


async def parse_ocr_input(request: Request) -> tuple[OCRRequest, RawOCRInput]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if not upload or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="Missing multipart field: file")

        file_bytes = await read_upload_bytes(upload, MAX_REQUEST_BYTES)
        ocr_request = OCRRequest(
            fileType=parse_optional_int(form.get("fileType")),
            useLayoutDetection=parse_bool(form.get("useLayoutDetection"), True),
            useDocUnwarping=parse_bool(form.get("useDocUnwarping"), False),
            useDocOrientationClassify=parse_bool(form.get("useDocOrientationClassify"), False),
            useTextlineOrientation=parse_bool(form.get("useTextlineOrientation"), False),
            useChartRecognition=parse_bool(form.get("useChartRecognition"), False),
            useSealRecognition=parse_bool(form.get("useSealRecognition"), True),
            formatBlockContent=parse_bool(form.get("formatBlockContent"), True),
            showFormulaNumber=parse_bool(form.get("showFormulaNumber"), True),
            markdownIgnoreLabels=parse_markdown_ignore_labels(form.get("markdownIgnoreLabels")),
            layoutThreshold=parse_optional_float(form.get("layoutThreshold")),
            layoutNms=parse_bool(form.get("layoutNms")) if form.get("layoutNms") is not None else None,
            layoutUnclipRatio=parse_optional_float(form.get("layoutUnclipRatio")),
            layoutMergeBboxesMode=parse_optional_string(form.get("layoutMergeBboxesMode")),
            repetitionPenalty=parse_optional_float(form.get("repetitionPenalty")),
            temperature=parse_optional_float(form.get("temperature")),
            topP=parse_optional_float(form.get("topP")),
            minPixels=parse_optional_int(form.get("minPixels")),
            maxPixels=parse_optional_int(form.get("maxPixels")),
            visualize=parse_bool(form.get("visualize")) if form.get("visualize") is not None else None,
        )
        return ocr_request, file_bytes

    body = await request.body()
    if MAX_REQUEST_BYTES > 0 and len(body) > MAX_REQUEST_BYTES:
        max_mb = MAX_REQUEST_BYTES / 1024 / 1024
        raise HTTPException(status_code=413, detail=f"Request body is too large. Max upload size is {max_mb:.0f} MB.")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as err:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from err
    ocr_request = OCRRequest(**payload)
    if not ocr_request.image:
        raise HTTPException(status_code=400, detail="Missing JSON field: image")
    return ocr_request, ocr_request.image


def normalize_raw_input_to_base64(raw_input: RawOCRInput) -> str:
    if isinstance(raw_input, bytes):
        return base64.b64encode(raw_input).decode("utf-8")
    if "base64," in raw_input:
        return raw_input.split("base64,")[1]
    return raw_input


def raw_input_to_bytes(raw_input: RawOCRInput) -> bytes:
    if isinstance(raw_input, bytes):
        return raw_input
    normalized = raw_input.split("base64,")[1] if "base64," in raw_input else raw_input
    try:
        return base64.b64decode(normalized, validate=True)
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid base64 input") from err


def prepare_service_input(ocr_request: OCRRequest, raw_input: RawOCRInput) -> tuple[str, int]:
    base64_data = normalize_raw_input_to_base64(raw_input)
    file_type = ocr_request.fileType

    if file_type is None:
        if isinstance(raw_input, bytes):
            if raw_input.startswith(b"%PDF-"):
                file_type = 0
                logger.info("Auto-detected PDF input")
            else:
                file_type = 1
                logger.info("Auto-detected Image input")
        elif base64_data.startswith("JVBERi0"):
            file_type = 0
            logger.info("Auto-detected PDF input")
        else:
            file_type = 1
            logger.info("Auto-detected Image input")

    if file_type == 1:
        try:
            img_bytes = raw_input_to_bytes(raw_input)
            img = Image.open(io.BytesIO(img_bytes))
            if img.format == "GIF":
                logger.info("GIF detected, converting to static JPEG for OCR")
                img.seek(0)
                rgb_img = img.convert("RGB")
                buffer = io.BytesIO()
                rgb_img.save(buffer, format="JPEG", quality=95)
                base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                logger.info("GIF conversion successful")
        except Exception as gif_err:
            logger.info("GIF conversion skipped: %s", gif_err)

    return base64_data, file_type


def build_pipeline_payload(request: OCRRequest, base64_data: str, file_type: int) -> dict:
    payload = {
        "file": base64_data,
        "fileType": file_type,
        "useLayoutDetection": request.useLayoutDetection,
        "useDocUnwarping": request.useDocUnwarping,
        "useDocOrientationClassify": request.useDocOrientationClassify,
        "useChartRecognition": request.useChartRecognition,
        "useSealRecognition": request.useSealRecognition,
        "formatBlockContent": request.formatBlockContent,
        "showFormulaNumber": request.showFormulaNumber,
        "prettifyMarkdown": True,
    }
    optional_params = [
        "markdownIgnoreLabels",
        "layoutThreshold",
        "layoutNms",
        "layoutUnclipRatio",
        "layoutMergeBboxesMode",
        "repetitionPenalty",
        "temperature",
        "topP",
        "minPixels",
        "maxPixels",
        "visualize",
    ]
    for param in optional_params:
        val = getattr(request, param)
        if val is not None:
            payload[param] = val
    return payload


def build_ppocr_payload(request: OCRRequest, base64_data: str, file_type: int) -> dict:
    payload = {
        "file": base64_data,
        "fileType": file_type,
        "useDocOrientationClassify": request.useDocOrientationClassify,
        "useDocUnwarping": request.useDocUnwarping,
        "useTextlineOrientation": request.useTextlineOrientation,
    }
    if request.visualize is not None:
        payload["visualize"] = request.visualize
    return payload


def parse_pipeline_response(data: dict, image_prefix: str = "") -> dict:
    if "result" not in data or "layoutParsingResults" not in data["result"]:
        logger.warning("Unexpected pipeline response format: %s", data)
        raise HTTPException(status_code=500, detail="Unexpected response format from Pipeline")

    results = data["result"]["layoutParsingResults"]
    full_markdown = ""
    all_images = {}

    for res in results:
        if "markdown" in res and "text" in res["markdown"]:
            md_text = res["markdown"]["text"]
            md_images = res["markdown"].get("images", {})
            if md_images:
                for img_path, img_base64 in md_images.items():
                    key = f"{image_prefix}_{img_path}" if image_prefix else img_path
                    all_images[key] = img_base64
            full_markdown += md_text + "\n\n"

    return {
        "markdown": full_markdown,
        "images": all_images,
        "layoutParsingResults": results,
    }


def as_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: as_jsonable(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    return value


def pick_indexed_value(values, index):
    if isinstance(values, list) and index < len(values):
        return as_jsonable(values[index])
    return None


def extract_ppocr_lines(pruned_result: dict) -> list[dict]:
    texts = pruned_result.get("rec_texts") if isinstance(pruned_result.get("rec_texts"), list) else []
    scores = pruned_result.get("rec_scores") if isinstance(pruned_result.get("rec_scores"), list) else []
    boxes = pruned_result.get("rec_boxes")
    polys = pruned_result.get("rec_polys")
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(polys, "tolist"):
        polys = polys.tolist()

    lines = []
    for index, text in enumerate(texts):
        line = {
            "text": str(text),
            "score": pick_indexed_value(scores, index),
        }
        box = pick_indexed_value(boxes, index)
        poly = pick_indexed_value(polys, index)
        if box is not None:
            line["box"] = box
        if poly is not None:
            line["poly"] = poly
        lines.append(line)
    return lines


def parse_ppocr_response(data: dict) -> dict:
    if "result" not in data or "ocrResults" not in data["result"]:
        logger.warning("Unexpected PP-OCR response format: %s", data)
        raise HTTPException(status_code=500, detail="Unexpected response format from PP-OCR service")

    pages = []
    full_markdown_parts = []
    for page_index, page_result in enumerate(data["result"]["ocrResults"]):
        pruned = page_result.get("prunedResult") if isinstance(page_result, dict) else {}
        if not isinstance(pruned, dict):
            pruned = {}
        pruned = as_jsonable(pruned)
        lines = extract_ppocr_lines(pruned)
        markdown_text = "\n".join(line["text"] for line in lines if line.get("text"))
        if markdown_text:
            full_markdown_parts.append(markdown_text)

        pages.append(
            {
                "model": PPOCR_V6_MODEL_NAME,
                "parser": "pp-ocrv6",
                "page_index": pruned.get("page_index", page_index),
                "pageImage": page_result.get("inputImage") if isinstance(page_result, dict) else None,
                "markdown": {
                    "text": markdown_text,
                    "images": {},
                },
                "ocrLines": lines,
                "prunedResult": pruned,
            }
        )

    return {
        "markdown": "\n\n".join(full_markdown_parts),
        "images": {},
        "layoutParsingResults": pages,
    }


async def acquire_ocr_slot(model_id: str, not_ready_message: str) -> None:
    global ocr_active_count
    await ocr_semaphore.acquire()
    try:
        async with model_runtime_lock:
            operation = model_runtime_operation
            if operation.get("state") == "switching":
                target = operation.get("targetModelId") or "requested model"
                raise HTTPException(status_code=409, detail=f"Model runtime is switching to {target}. Try again when it is ready.")
            runtime = await model_runtime_status(model_id)
            if not runtime["ready"]:
                raise HTTPException(status_code=503, detail=not_ready_message)
            ocr_active_count += 1
    except Exception:
        ocr_semaphore.release()
        raise


async def release_ocr_slot() -> None:
    global ocr_active_count
    async with model_runtime_lock:
        ocr_active_count = max(0, ocr_active_count - 1)
    ocr_semaphore.release()


async def run_ocr_request(ocr_request: OCRRequest, raw_input: RawOCRInput) -> dict:
    await acquire_ocr_slot(
        "paddleocr-vl-1.6",
        "PaddleOCR-VL service is not ready. Switch to this model and wait for it to become ready.",
    )
    try:
        base64_data, file_type = prepare_service_input(ocr_request, raw_input)
        payload = build_pipeline_payload(ocr_request, base64_data, file_type)

        logger.info("Sending request to Pipeline Service at %s", PADDLE_SERVICE_URL)
        timeout = PADDLE_REQUEST_TIMEOUT if PADDLE_REQUEST_TIMEOUT > 0 else None
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                PADDLE_SERVICE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code != 200:
                logger.warning("Service Error (HTTP %s): %s", resp.status_code, resp.text)
                if resp.status_code == 422:
                    logger.warning("Validation Error Details: %s", resp.json())
                raise HTTPException(status_code=resp.status_code, detail=f"Upstream error: {resp.text}")

            return parse_pipeline_response(resp.json())
    finally:
        await release_ocr_slot()


async def run_ppocrv6_request(ocr_request: OCRRequest, raw_input: RawOCRInput) -> dict:
    await acquire_ocr_slot(
        "pp-ocrv6",
        "PP-OCRv6 service is not ready. Switch to this model and wait for it to become ready.",
    )
    try:
        base64_data, file_type = prepare_service_input(ocr_request, raw_input)
        payload = build_ppocr_payload(ocr_request, base64_data, file_type)

        logger.info("Sending request to PP-OCR service at %s", PADDLE_OCR_SERVICE_URL)
        timeout = PADDLE_REQUEST_TIMEOUT if PADDLE_REQUEST_TIMEOUT > 0 else None
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                PADDLE_OCR_SERVICE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code != 200:
                logger.warning("PP-OCR Service Error (HTTP %s): %s", resp.status_code, resp.text)
                if resp.status_code == 422:
                    logger.warning("PP-OCR Validation Error Details: %s", resp.json())
                raise HTTPException(status_code=resp.status_code, detail=f"Upstream PP-OCR error: {resp.text}")

            return parse_ppocr_response(resp.json())
    finally:
        await release_ocr_slot()


def validate_proxy_input_size(raw_input: RawOCRInput) -> int:
    base64_data = normalize_raw_input_to_base64(raw_input)
    if MAX_REQUEST_BYTES > 0 and len(base64_data) > int(MAX_REQUEST_BYTES * 4 / 3) + 1024:
        max_mb = MAX_REQUEST_BYTES / 1024 / 1024
        raise HTTPException(status_code=413, detail=f"OCR input is too large. Max upload size is {max_mb:.0f} MB.")
    return len(base64_data)


@app.post("/api/paddleocr-vl-1.6")
async def proxy_paddleocr_vl(request: Request):
    """Proxy request to PaddleOCR-VL Pipeline Service."""
    try:
        ocr_request, raw_image = await parse_ocr_input(request)
        base64_size = validate_proxy_input_size(raw_image)
        logger.info("Received PaddleOCR-VL request. Base64 input size: %s bytes", base64_size)
        return await run_ocr_request(ocr_request, raw_image)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PaddleOCR-VL Proxy Error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pp-ocrv6")
async def proxy_ppocrv6(request: Request):
    """Proxy request to PP-OCRv6 OCR Pipeline Service."""
    try:
        ocr_request, raw_image = await parse_ocr_input(request)
        base64_size = validate_proxy_input_size(raw_image)
        logger.info("Received PP-OCRv6 request. Base64 input size: %s bytes", base64_size)
        return await run_ppocrv6_request(ocr_request, raw_image)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PP-OCRv6 Proxy Error")
        raise HTTPException(status_code=500, detail=str(e))


async def run_mineru_request(ocr_request: OCRRequest, raw_input: RawOCRInput) -> dict:
    """Proxy request to MinerU API Service (multipart /file_parse).

    Writes raw bytes to a temp file, then streams the upload from disk
    to avoid holding the entire file in memory twice.  Page dimensions
    are obtained via fitz.open(path) which also streams from disk.
    """
    await acquire_ocr_slot(
        "mineru",
        "MinerU service is not ready. Switch to this model and wait for it to become ready.",
    )
    try:
        file_bytes = raw_input_to_bytes(raw_input)
        file_type = ocr_request.fileType

        filename = "upload.pdf" if file_type == 0 else "upload.png"

        data_payload = {
            "return_md": "true",
            "return_images": "true",
            "return_content_list": "true",
            "formula_enable": str(ocr_request.useChartRecognition).lower(),
            "table_enable": "true",
            "image_analysis": str(ocr_request.useChartRecognition).lower(),
            "parse_method": "auto",
        }
        if ocr_request.useLayoutDetection:
            data_payload["backend"] = "hybrid-engine"
        else:
            data_payload["backend"] = "pipeline"

        logger.info("Sending request to MinerU Service at %s/file_parse", MINERU_SERVICE_URL)
        timeout = PADDLE_REQUEST_TIMEOUT if PADDLE_REQUEST_TIMEOUT > 0 else None

        # Write raw bytes to a temp file and stream from disk
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            await run_in_threadpool(tmp_path.write_bytes, file_bytes)
            # Free the raw bytes as soon as they are on disk
            del file_bytes

            with tmp_path.open("rb") as f:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{MINERU_SERVICE_URL}/file_parse",
                        files={"files": (filename, f, "application/octet-stream")},
                        data=data_payload,
                    )

                    if resp.status_code != 200:
                        logger.warning("MinerU Service Error (HTTP %s): %s", resp.status_code, resp.text)
                        raise HTTPException(status_code=resp.status_code, detail=f"Upstream MinerU error: {resp.text}")

            # Get dimensions from the temp file on disk (fitz.open streams from disk)
            dimensions = []
            try:
                import fitz
                if file_type == 0:
                    doc = await run_in_threadpool(fitz.open, str(tmp_path))
                    try:
                        for page in doc:
                            dimensions.append((page.rect.width, page.rect.height))
                    finally:
                        doc.close()
                else:
                    from PIL import Image
                    img = await run_in_threadpool(Image.open, str(tmp_path))
                    dimensions.append((img.width, img.height))
            except Exception as e:
                logger.warning("Failed to get MinerU page dimensions: %s", e)

            return parse_mineru_response(resp.json(), dimensions)
        finally:
            tmp_path.unlink(missing_ok=True)
    finally:
        await release_ocr_slot()


def parse_mineru_response(data: dict, dimensions: list = None) -> dict:
    """Convert MinerU /file_parse response to paddleocr-local format."""
    results = data.get("results") or {}
    full_markdown = ""
    all_images = {}
    layout_results = []
    all_content_list = []

    for doc_name, doc_result in results.items():
        if not isinstance(doc_result, dict):
            continue
        md_content = doc_result.get("md_content") or ""
        if md_content:
            full_markdown += md_content + "\n\n"
        images = doc_result.get("images") or {}
        for img_name, img_data in images.items():
            if img_data.startswith("data:"):
                all_images[img_name] = img_data.split(",", 1)[1] if "," in img_data else img_data
            else:
                all_images[img_name] = img_data
        content_list = doc_result.get("content_list") or []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            all_content_list.append({
                "type": item.get("type", "text"),
                "text": item.get("text", ""),
                "page_idx": item.get("page_idx"),
                "bbox": item.get("bbox"),
                "img_idx": item.get("img_idx"),
            })
        layout_results.append({
            "markdown": {"text": md_content, "images": images},
            "source": "mineru",
            "document": doc_name,
        })

    # Group content_list by page to create compatible layout structure
    pages_dict = {}
    for item in all_content_list:
        p_idx = item.get("page_idx", 0)
        if p_idx not in pages_dict:
            pages_dict[p_idx] = []
        
        # MinerU bbox is [x0, y0, x1, y1]
        bbox = item.get("bbox")
        if not bbox and item.get("type") == "text":
            # Just create a dummy bbox or skip if absolutely necessary, but we'll try to include it
            bbox = [0, 0, 100, 100]
            
        pages_dict[p_idx].append({
            "label": item.get("type", "text"),
            "bbox": bbox,
            "text": item.get("text", "")
        })

    pages = []
    if dimensions:
        for p_idx in range(len(dimensions)):
            w, h = dimensions[p_idx]
            pages.append({
                "width": w,
                "height": h,
                "parsing_res_list": pages_dict.get(p_idx, []),
                "parser": "mineru"
            })
    else:
        for p_idx, p_list in pages_dict.items():
            pages.append({
                "width": 1000, # default if no dimensions
                "height": 1000,
                "parsing_res_list": p_list,
                "parser": "mineru"
            })

    response = {
        "markdown": full_markdown,
        "images": all_images,
        "layoutParsingResults": layout_results,
        "pages": pages,
    }
    if all_content_list:
        response["contentList"] = all_content_list
    return response


@app.post("/api/mineru")
async def proxy_mineru(request: Request):
    """Proxy request to MinerU Document Parsing Service."""
    try:
        ocr_request, raw_image = await parse_ocr_input(request)
        base64_size = validate_proxy_input_size(raw_image)
        logger.info("Received MinerU request. Base64 input size: %s bytes", base64_size)
        return await run_mineru_request(ocr_request, raw_image)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("MinerU Proxy Error")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GLM-OCR (Ollama) Pipeline
# PP-OCRv6 layout detection → GLM-OCR text recognition → Post-processing
# ---------------------------------------------------------------------------

GLM_OCR_PROMPT = (
    "请只转写图片中清晰可见的文字，并输出 Markdown。"
    "保留原有换行、列表和表格结构；表格请使用 Markdown 或 HTML 表格。"
    "不要解释、总结、补全、翻译或编造图片中不存在的内容。"
    "跳过页眉、页脚和页码；如果没有可识别文字，只输出空字符串。"
)

GLM_OCR_MAX_IMAGE_HEIGHT = 1600
GLM_OCR_SEGMENT_OVERLAP = 80

# Labels to skip in layout detection
_LAYOUT_SKIP_LABELS = {"header", "footer", "footnote", "number"}
# Labels that must stay as solo regions (not merged with neighbors)
_LAYOUT_SOLO_LABELS = {"table", "figure", "figure_caption", "table_caption"}
_LAYOUT_MERGE_LABELS = {"text", "title", "list"}


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _split_tall_image(img: Image.Image) -> list[Image.Image]:
    """Split a tall image into overlapping segments for GLM-OCR."""
    w, h = img.size
    step = GLM_OCR_MAX_IMAGE_HEIGHT - GLM_OCR_SEGMENT_OVERLAP
    segments = []
    y = 0
    while y < h:
        bottom = min(y + GLM_OCR_MAX_IMAGE_HEIGHT, h)
        segments.append(img.crop((0, y, w, bottom)))
        y += step
        if bottom == h:
            break
    return segments


async def _detect_layout_via_ppocrv6(img: Image.Image) -> list[dict]:
    """Use PP-DocLayoutV3 layout detection service (running in paddleocr-ocr-api container).

    Returns list of {label, bbox} for detected layout regions (title, text, table, figure, etc.)
    """
    # The layout detection service runs on port 8081 in the OCR API container
    layout_url = os.getenv("PANDOCR_LAYOUT_DETECT_URL", "http://paddleocr-ocr-api:8081/layout-detect")

    b64 = _image_to_b64(img)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(layout_url, json={"image": b64})
            if resp.status_code != 200:
                logger.warning("[glm-ocr] Layout detection service returned %d", resp.status_code)
                return []
            data = resp.json()
    except Exception as err:
        logger.warning("[glm-ocr] Layout detection service unavailable: %s", err)
        return []

    regions = data.get("regions", [])
    # Filter out skip labels and ensure valid bboxes
    filtered = []
    for r in regions:
        label = r.get("label", "text")
        if label in _LAYOUT_SKIP_LABELS:
            continue
        bbox = r.get("bbox")
        if bbox and len(bbox) >= 4:
            filtered.append({"label": label, "bbox": bbox})

    # Sort top-to-bottom reading order
    filtered.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    logger.info("[glm-ocr] Layout detection: %d regions", len(filtered))
    return filtered


def _merge_layout_regions(raw_regions: list[dict]) -> list[list[dict]]:
    """Group adjacent text regions; solo regions (table/figure) stay separate."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for region in raw_regions:
        if region["label"] in _LAYOUT_SOLO_LABELS:
            if current:
                groups.append(current)
                current = []
            groups.append([region])
        elif region["label"] in _LAYOUT_MERGE_LABELS:
            current.append(region)
        else:
            # Unknown label: treat as solo
            if current:
                groups.append(current)
                current = []
            groups.append([region])
    if current:
        groups.append(current)
    return groups


def _group_bbox(regions: list[dict]) -> list[int]:
    x1 = min(r["bbox"][0] for r in regions)
    y1 = min(r["bbox"][1] for r in regions)
    x2 = max(r["bbox"][2] for r in regions)
    y2 = max(r["bbox"][3] for r in regions)
    return [x1, y1, x2, y2]


def _ollama_chat_payload(content: str, image_b64: str | None = None) -> dict:
    message = {"role": "user", "content": content}
    if image_b64 is not None:
        message["images"] = [image_b64]
    return {
        "model": OLLAMA_MODEL,
        "messages": [message],
        "stream": False,
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": 0,
        },
    }


async def _glm_ocr_single(image_b64: str) -> str:
    """Send a single image to Ollama GLM-OCR."""
    timeout = httpx.Timeout(PADDLE_REQUEST_TIMEOUT, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=_ollama_chat_payload(GLM_OCR_PROMPT, image_b64),
        )
        if resp.status_code != 200:
            detail = ""
            try:
                data = resp.json()
                detail = data.get("error", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            raise HTTPException(status_code=resp.status_code, detail=f"Ollama error: {detail}")
        result = resp.json()
        return result.get("message", {}).get("content", "")


# ── GLM-OCR Post-processing ─────────────────────────────────────

def _glm_postprocess(text: str) -> str:
    """Strip markdown fences, remove empty tables, dedup math, dedup lines."""
    text = re.sub(r'^```\w*\n?', '', text.strip())
    text = re.sub(r'\n?```$', '', text.strip())
    text = re.sub(r'(?m)^\s*```\w*\s*$', '', text)
    text = _glm_remove_empty_html_tables(text)
    text = _glm_remove_duplicate_display_math(text)
    text = _glm_dedup_lines(text)
    return text.strip()


def _glm_remove_empty_html_tables(text: str) -> str:
    empty_table = re.compile(
        r'<table\b[^>]*>\s*'
        r'(?:<tbody>\s*)?'
        r'<tr>\s*(?:<t[dh]\b[^>]*>\s*</t[dh]>\s*)+</tr>'
        r'\s*(?:</tbody>\s*)?'
        r'</table>',
        flags=re.IGNORECASE,
    )
    return empty_table.sub('', text)


def _glm_remove_duplicate_display_math(text: str) -> str:
    """Remove $$...$$ display math lines that duplicate $...$ inline math."""
    lines = text.split('\n')
    inline_contents = set()
    for line in lines:
        for m in re.finditer(r'\$([^$]+)\$', line):
            normalized = re.sub(r'\s+', '', m.group(1))
            inline_contents.add(normalized)
    result = []
    for line in lines:
        stripped = line.strip()
        m = re.match(r'^\$\$(.+)\$\$$', stripped)
        if m:
            normalized = re.sub(r'\s+', '', m.group(1))
            if normalized in inline_contents:
                continue
        result.append(line)
    return '\n'.join(result)


def _glm_dedup_lines(text: str) -> str:
    """Remove lines whose normalized content is a substring of any earlier line."""
    lines = text.split('\n')
    if len(lines) <= 1:
        return text
    result = [lines[0]]
    seen_norms = [re.sub(r'\s+', '', lines[0])]
    for line in lines[1:]:
        curr_norm = re.sub(r'\s+', '', line)
        if not curr_norm:
            result.append(line)
            continue
        is_dup = False
        for prev_norm in seen_norms:
            if curr_norm == prev_norm:
                is_dup = True
                break
            if len(curr_norm) > 2 and curr_norm in prev_norm:
                is_dup = True
                break
        if is_dup:
            continue
        result.append(line)
        seen_norms.append(curr_norm)
    return '\n'.join(result)


async def run_glm_ocr_from_file(file_path: Path, ocr_request: "OCRRequest") -> dict:
    """Run GLM-OCR pipeline: PP-OCRv6 layout detection → GLM-OCR text recognition → post-processing.

    PP-OCRv6 runs alongside GLM-OCR for layout detection (regions + coordinates).
    If PP-OCRv6 is unavailable, falls back to whole-image OCR with tall-image splitting.
    """
    await acquire_ocr_slot(
        "glm-ocr",
        "GLM-OCR (Ollama) is not ready. Ensure Ollama is running and the model is loaded.",
    )
    try:
        # Read the image/PDF — use fitz.open(path) for PDFs to stream from disk
        # instead of loading the entire file into memory.
        file_type = ocr_request.fileType

        # Convert to PIL Image(s)
        images: list[Image.Image] = []
        if file_type == 0:
            # PDF → render pages to images via fitz (streams from disk)
            import fitz
            doc = await run_in_threadpool(fitz.open, str(file_path))
            try:
                for page in doc:
                    pix = await run_in_threadpool(page.get_pixmap, matrix=fitz.Matrix(2.0, 2.0))
                    img = await run_in_threadpool(Image.frombytes, "RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
            finally:
                doc.close()
        else:
            # Single image — need bytes for PIL, but images are typically small
            file_bytes = await run_in_threadpool(file_path.read_bytes)
            img = await run_in_threadpool(Image.open, io.BytesIO(file_bytes))
            images.append(img.convert("RGB"))

        all_markdown = ""
        all_images = {}
        pages = []
        layout_results = []

        for page_idx, img in enumerate(images):
            w, h = img.size
            page_markdown = ""
            parsing_res_list = []

            # Try layout detection via PP-OCRv6 API
            raw_regions = []
            if ocr_request.useLayoutDetection:
                try:
                    raw_regions = await _detect_layout_via_ppocrv6(img)
                    if raw_regions:
                        logger.info("[glm-ocr] Page %d: %d layout regions detected", page_idx + 1, len(raw_regions))
                except Exception as err:
                    logger.warning("[glm-ocr] Layout detection failed, falling back to whole-image: %s", err)
                    raw_regions = []

            if raw_regions:
                # Layout detected: merge adjacent text regions, OCR each group
                groups = _merge_layout_regions(raw_regions)
                logger.info("[glm-ocr] Page %d: %d regions → %d groups", page_idx + 1, len(raw_regions), len(groups))

                for gi, group in enumerate(groups):
                    bbox = _group_bbox(group)
                    # Clamp bbox to image bounds
                    bbox = [
                        max(0, bbox[0]),
                        max(0, bbox[1]),
                        min(w, bbox[2]),
                        min(h, bbox[3]),
                    ]
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    cropped = img.crop(bbox)
                    seg_b64 = await run_in_threadpool(_image_to_b64, cropped)
                    text = await _glm_ocr_single(seg_b64)
                    text = _glm_postprocess(text)
                    if text:
                        page_markdown += text + "\n\n"
                    label = group[0]["label"] if len(group) == 1 else "text"
                    
                    parsing_res_list.append({
                        "label": label,
                        "bbox": bbox,
                        "text": text or "",
                    })
                    
                    layout_results.append({
                        "label": label,
                        "bbox": bbox,
                        "page_index": page_idx,
                        "markdown": {"text": text or "", "images": {}},
                    })
            else:
                # No layout detection: whole-image OCR with tall-image splitting
                if h > GLM_OCR_MAX_IMAGE_HEIGHT:
                    segments = await run_in_threadpool(_split_tall_image, img)
                    logger.info("[glm-ocr] Page %d: tall image (%dx%d) → %d segments", page_idx + 1, w, h, len(segments))
                else:
                    segments = [img]

                for seg in segments:
                    seg_b64 = await run_in_threadpool(_image_to_b64, seg)
                    text = await _glm_ocr_single(seg_b64)
                    text = _glm_postprocess(text)
                    if text:
                        page_markdown += text + "\n\n"

                parsing_res_list.append({
                    "label": "full_page",
                    "bbox": [0, 0, w, h],
                    "text": page_markdown,
                })

                layout_results.append({
                    "label": "full_page",
                    "page_index": page_idx,
                    "markdown": {"text": page_markdown, "images": {}},
                })

            all_markdown += page_markdown
            pages.append({
                "width": w,
                "height": h,
                "parsing_res_list": parsing_res_list,
                "parser": "glm-ocr"
            })

        return {
            "markdown": all_markdown,
            "images": all_images,
            "layoutParsingResults": layout_results,
            "pages": pages,
        }
    finally:
        await release_ocr_slot()


@app.post("/api/glm-ocr")
async def proxy_glm_ocr(request: Request):
    """Proxy request to GLM-OCR (Ollama) Document Parsing Service."""
    try:
        ocr_request, raw_image = await parse_ocr_input(request)
        base64_size = validate_proxy_input_size(raw_image)
        logger.info("Received GLM-OCR request. Base64 input size: %s bytes", base64_size)
        # For GLM-OCR, we work with the raw bytes directly, not base64
        # We need to write to a temp file and use run_glm_ocr_from_file
        file_bytes = raw_input_to_bytes(raw_image)
        suffix = ".pdf" if (ocr_request.fileType == 0 or file_bytes.startswith(b"%PDF-")) else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            await run_in_threadpool(tmp_path.write_bytes, file_bytes)
            return await run_glm_ocr_from_file(tmp_path, ocr_request)
        finally:
            tmp_path.unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("GLM-OCR Proxy Error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/glm-ocr/status")
async def glm_ocr_status():
    """Check Ollama status and GLM-OCR model availability."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            has_model = any(OLLAMA_MODEL in m for m in models)
            return {"online": True, "modelLoaded": has_model, "models": models}
    except Exception:
        return {"online": False, "modelLoaded": False, "models": []}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server. Target Pipeline: %s", PADDLE_SERVICE_URL)
    uvicorn.run(app, host=PANDOCR_HOST, port=PANDOCR_PORT)
