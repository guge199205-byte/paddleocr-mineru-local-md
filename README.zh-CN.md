# PaddleOCR Local - PaddleOCR-VL & PP-OCRv6 & MinerU WebUI

**语言 / Language**: 简体中文 | [English](README.en.md)

PaddleOCR Local 是一个面向 PaddleOCR-VL、PP-OCRv6 和 MinerU 的轻量 Web 前端。前端负责文件上传、队列、预览、模型切换和下载，后端 FastAPI 做静态文件服务、Office 转 PDF 和请求代理；OCR 推理由独立服务完成，NVIDIA 路线使用官方 Docker 服务，macOS Apple Silicon 路线使用本地 PaddleX/MLX 服务。

**支持四种模型：**

| 模型 | 用途 | 说明 |
|------|------|------|
| PaddleOCR-VL 1.6 | 文档解析 | 版面分析、表格、公式、印章识别 |
| PP-OCRv6 | 文字识别 | 轻量级纯文字 OCR |
| MinerU | 文档解析 | MinerU2.5-Pro-2605-1.2B，hybrid engine |
| GLM-OCR (Ollama) | 文字识别 | 智谱 GLM-OCR 模型，需配合 Ollama 运行 |

单 GPU 部署下，WebUI 模型选择器会自动管理容器启停，同一时间只有一个模型占用显存。

<img width="1920" height="945" alt="image" src="https://github.com/user-attachments/assets/85a247a0-c796-4a20-b596-1cc4148df964" />

## 一键部署

这个项目的默认目标是开源自部署：新用户尽量只运行一条命令，脚本自动检查环境、安装依赖、启动服务并打开 WebUI。

macOS Apple Silicon：

```bash
./macos-one-click.command
```

Windows + NVIDIA：

```powershell
.\windows-one-click.bat
```

部署前或失败后可先跑诊断：

```bash
make doctor
```

macOS 会走本地 PaddlePaddle + PaddleX + 可选 MLX-VLM；Windows/NVIDIA 会走 Docker Compose。默认只绑定本机地址，保持本地即开即用。

## 当前架构

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
       - NVIDIA: docker compose 中的 paddleocr-vl-api + paddleocr-ocr-api + paddleocr-vlm-server
       - macOS: 本地 paddlex --serve，可选 mlx_vlm.server
  -> MinerU service
       - NVIDIA: docker compose 中的 mineru-api (profile: mineru)
  -> GLM-OCR (Ollama)
       - 本地或 Docker 中的 Ollama 服务
```

NVIDIA Compose 保留 5 个核心服务 + 2 个可选服务：

- `pandocr-web`
- `paddleocr-vl-api`
- `paddleocr-ocr-api`
- `paddleocr-vlm-server`
- `mineru-api`（Docker profile `mineru`，需显式启用）
- `ollama`（Docker profile `glm-ocr`，需显式启用；也可使用已有的 Ollama 实例）

单 GPU Docker 部署默认只热加载一个模型。`pandocr-web` 常驻运行，并通过 Docker socket 按需启停模型容器：选择 `PaddleOCR-VL 1.6` 会启动 `paddleocr-vlm-server` + `paddleocr-vl-api` 并停止其他模型；选择 `PP-OCRv6` 或 `MinerU` 会反向切换；选择 `GLM-OCR` 不需要启停 Docker 容器，直接连接 Ollama 服务。顶部 UI 会实时轮询显示模型就绪、启动中、待启动或失败状态。

## 功能

- 支持图片、PDF、PPT/PPTX、DOC/DOCX 上传。
- 支持在 `PaddleOCR-VL 1.6` 文档解析、`PP-OCRv6` 文字识别、`MinerU` 文档解析和 `GLM-OCR` 文字识别之间自由切换；Docker 单 GPU 部署会按需启停模型，避免多个模型同时占用显存。
- 左侧"解析设置"面板会根据当前选中模型自动切换：PaddleOCR 系列显示版面检测、图表识别、文档矫正、印章识别等选项；MinerU 显示公式解析、表格解析、图片分析、解析方式等选项。
- WebUI 支持中文/英文一键切换并记住用户选择，翻译集中维护在 `static/i18n.js`，便于后续扩展更多语言。
- **OCR 结果翻译**：支持将 OCR 解析后的 Markdown 结果翻译为 20 种语言（简体中文、繁体中文、英语、日语、韩语、法语、德语、西班牙语、葡萄牙语、俄语、阿拉伯语、意大利语、荷兰语、波兰语、土耳其语、越南语、泰语、印尼语、马来语、印地语），需配置 `PANDOCR_TRANSLATE_API_URL` 和 `PANDOCR_TRANSLATE_API_KEY`（兼容 OpenAI API 格式）。
- PDF 按页发送给 PaddleOCR-VL，便于对齐官方在线解析结果并稳定保留每页原始 JSON。
- PP-OCRv6 结果使用接近官方的可视化文字层展示：左右页面对齐，上下/左右滚动和缩放同步，识别文字支持复制和纠正，同时保留原始 JSON。
- 解析任务会持久化到本机 `data/tasks/`，刷新页面后仍可查看历史任务，删除按钮会同步删除本地记录。
- Markdown 预览支持表格横向滚动、KaTeX 数学公式渲染、OCR 结果中的字面量 `\n` 换行修正。
- 支持解析选项：版面检测、图表识别、文档矫正、方向识别、印章识别、公式编号、Markdown 忽略标签等。
- 下载结果时会打包 Markdown 和 OCR 提取图片。
- **大文件支持**：超过 1GB 的大文件采用流式分批处理，避免内存溢出（OOM）；支持解析中断后从上次位置恢复继续处理。

## 部署方式

本项目支持两条部署路径，二者互不混用：

- **NVIDIA Docker 版本**：适合带 NVIDIA GPU 的 Linux/Windows Docker 环境，继续使用官方 PaddleOCR-VL Docker 服务。
- **macOS Apple Silicon 版本**：适合 Apple M1/M2/M3/M4 芯片，按官方 Apple Silicon 文档走本地 PaddlePaddle + PaddleX serving，可选 MLX-VLM 提速。

### 版本一：NVIDIA Docker

Windows + NVIDIA 用户推荐直接使用一键部署脚本：

```powershell
.\windows-one-click.bat
```

它会自动检查 Docker、识别 NVIDIA GPU、选择 `env.txt` 或 `env.docker`、拉取官方 PaddleOCR-VL 镜像、构建 `pandocr-web`、清理旧容器、创建所有模型容器但不会同时启动两个模型，然后启动 WebUI 并等待当前活跃模型健康检查。

常用一键部署参数：

```powershell
.\windows-one-click.bat -DryRun
.\windows-one-click.bat -GpuId 1
.\windows-one-click.bat -EnvFile env.docker
```

也可以继续使用手动部署流程：

先根据显卡型号选择环境文件：

| 显卡 | 推荐环境文件 | 镜像标签 |
| --- | --- | --- |
| RTX 30 系列 | `env.docker` | `latest-nvidia-gpu-offline` |
| RTX 40 系列 | `env.docker` | `latest-nvidia-gpu-offline` |
| RTX 50 系列 / Blackwell | `env.txt` | `latest-nvidia-gpu-sm120-offline` |

下面命令以 RTX 50 系列的 `env.txt` 为例；RTX 30/40 系列用户把命令里的 `env.txt` 换成 `env.docker` 即可。

**步骤 1：拉取 PaddleOCR 官方镜像并构建本地服务**

```bash
docker compose --env-file env.txt pull paddleocr-vlm-server paddleocr-vl-api
docker compose --env-file env.txt build paddleocr-ocr-api pandocr-web
```

**步骤 2：构建 MinerU 镜像（可选，需要 MinerU 时执行）**

```bash
# 使用国内镜像加速
docker build -t mineru:latest https://github.com/opendatalab/MinerU.git#master:docker/china

# 或使用国际源
# docker build -t mineru:latest https://github.com/opendatalab/MinerU.git#master:docker/global
```

> MinerU 镜像基于 vllm/vllm-openai，体积约 13GB，构建需 15-30 分钟（含下载模型权重）。

**步骤 3：创建并启动容器**

```bash
# 创建所有容器（不启动）
docker compose --env-file env.txt up -d --no-start

# 如果构建了 MinerU 镜像，也创建 mineru-api 容器
docker compose --env-file env.txt --profile mineru up -d --no-start mineru-api

# 只启动 WebUI（它会按需启停模型容器）
docker compose --env-file env.txt start pandocr-web
```

单 GPU 部署请保持这个 `up -d --no-start` 再 `start pandocr-web` 的顺序。不要直接执行普通的 `docker compose up -d`，否则可能同时热加载多个模型，造成显存被抢占。WebUI 打开后，通过右上角模型选择器切换模型即可；前端会调用 `/api/model-runtime/switch`，只启动当前选择的模型容器，停止非活跃模型容器，并让顶部运行状态实时同步真实容器状态。

访问：

- WebUI: http://localhost:18000（默认端口，可在 `env.txt` 中修改 `PANDOCR_PORT`）
- PaddleOCR-VL API health: http://localhost:8081/health，仅在 `PaddleOCR-VL 1.6` 为活跃模型时可用。
- PP-OCRv6 API health: http://localhost:8082/health，仅在 `PP-OCRv6` 为活跃模型时可用。
- MinerU API health: http://localhost:8083/health，仅在 `MinerU` 为活跃模型时可用。

**局域网访问：**

默认端口绑定到 `0.0.0.0`，局域网其他设备可通过 `http://<你的IP>:18000` 访问。`PANDOCR_CORS_ORIGINS` 需包含局域网访问地址。

Compose 中 `pandocr-web` 会挂载 `/var/run/docker.sock` 来启停本 compose 文件里的模型容器，这等同于具备 Docker 主机管理权限；不要在没有额外访问控制的情况下把 WebUI 暴露给不可信网络。

查看状态：

```bash
docker compose --env-file env.txt ps
curl http://localhost:18000/api/model-runtime
curl http://localhost:18000/api/models
```

常用环境变量：

`env.txt` 是当前 RTX 50 / Blackwell 推荐配置：

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

RTX 30/40 系列等非 Blackwell NVIDIA GPU 使用 `env.docker`，其中两个镜像标签都是 `latest-nvidia-gpu-offline`。

`PANDOCR_API_TOKEN` 为空时保持本机即开即用；如果要把 WebUI 暴露给反向代理、局域网或多人环境，请设置一个长随机 token。`PANDOCR_ENFORCE_ORIGIN_CHECK=1` 会拒绝未加入来源白名单的跨站 API 写请求，但它不能替代 token。前端会在 API 返回 401 时提示输入，并保存在浏览器本地。`PANDOCR_ENABLE_API_DOCS=1` 时才启用 `/docs` 和 `/redoc`，OpenAPI JSON 固定在 `/api/openapi.json`。

常用命令：

```bash
docker compose --env-file env.txt ps
docker compose --env-file env.txt logs -f pandocr-web
docker compose --env-file env.txt restart pandocr-web
docker compose --env-file env.txt down
```

### 版本二：macOS Apple Silicon

macOS Apple Silicon 按官方 PaddleOCR-VL 文档走手动部署，不使用 NVIDIA Docker Compose 镜像。官方依据：

- PaddleOCR-VL Apple Silicon Usage Tutorial: https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html
- PaddleOCR-VL Usage Tutorial: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
- PaddleX Serving Guide: https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_deploy/serving.html

支持芯片：

- Apple M1 / M2 / M3 / M4 系列芯片（arm64）
- 本项目脚本会检查 `Darwin + arm64`，因此 M1-M4 统一走同一套 Mac 本地部署路径。

Mac 默认启动官方 PaddleX 产线名：

```text
PaddleOCR-VL-1.6
```

不要使用裸 `PaddleOCR-VL` 作为 Mac 默认产线名；在当前 PaddleX 3.6.x 中，裸名对应旧版 v1 配置。`PaddleOCR-VL-1.6` 会使用 `PP-DocLayoutV3`、`PaddleOCR-VL-1.6-0.9B` 和 native PaddlePaddle 后端。

一键部署（推荐）：

```bash
./macos-one-click.command
```

也可以使用等价的 Make 命令：

```bash
make mac-one-click
```

这条一键命令会自动检查 Apple Silicon 环境、安装 macOS 依赖、默认启用 MLX-VLM 提速模式、启动 `mlx_vlm.server` / PaddleX API / PaddleOCR Local WebUI、执行健康检查，并自动打开 http://127.0.0.1:8000。

首次启动会下载 `PP-DocLayoutV3`、`PaddleOCR-VL-1.6-0.9B` 和 MLX 模型权重，耗时取决于网络和磁盘速度。模型缓存完成后，后续再次运行同一条命令会复用已安装环境和已启动服务。

高级手动启动：

```bash
make mac-setup
make mac-up
```

完成后访问：

- WebUI: http://127.0.0.1:8000
- PaddleOCR-VL API health: http://127.0.0.1:8081/health

测试、停止和查看日志：

```bash
make mac-test
make mac-down
make mac-logs
```

如果 native 模式太慢，可安装并启用官方 Apple Silicon 文档中的 MLX-VLM 路线：

```bash
make mac-setup-mlx
make mac-down
make mac-up-mlx
make mac-test-mlx
```

MLX 模式会启动三个本地服务：

- `mlx_vlm.server`: `127.0.0.1:8111`
- PaddleX 完整解析 API: `127.0.0.1:8081`
- PaddleOCR Local WebUI: `127.0.0.1:8000`

若 Hugging Face 下载较慢，可以设置 `HF_TOKEN` 提高 Hugging Face 的限流额度；模型缓存完成后后续启动会快很多。如果要改 MLX 端口，直接设置 `MLX_PORT` 即可；启动脚本会从模板生成 PaddleX 使用的配置。

常用环境变量：

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

端口被占用时，例如把 WebUI 改到 `18000`：

```bash
PANDOCR_PORT=18000 make mac-up
```

## MinerU 集成说明

本项目在 PaddleOCR 原版基础上集成了 [MinerU](https://github.com/opendatalab/MinerU) 文档解析服务。集成方式：

1. **后端** (`server.py`)：在 `MODEL_RUNTIME_CONFIG` 和 `model_catalog()` 中注册了 `mineru` 模型，新增 `/api/mineru` 代理路由，将 WebUI 请求转发为 MinerU 的 `multipart/form-data` 格式调用 `mineru-api` 的 `/file_parse` 端点，并将 MinerU 响应转换为 paddleocr-local 统一格式。

2. **前端** (`index.html` + `app.js`)：左侧"解析设置"面板根据当前选中模型动态切换 — PaddleOCR 系列显示版面检测/图表识别/文档矫正/印章识别等选项，MinerU 显示公式解析/表格解析/图片分析/解析方式等选项。

3. **Docker** (`docker-compose.yml`)：新增 `mineru-api` 服务，使用 `profiles: ["mineru"]` 控制不随默认 `docker compose up` 启动。需先构建 MinerU 镜像，再显式启用 profile 创建容器。

4. **模型切换**：与其他模型一致，WebUI 模型选择器切换到 MinerU 时，自动停止当前模型容器并启动 `mineru-api`，等待健康检查通过后标记为就绪。

## 主要接口

- `GET /`：WebUI 首页。
- `GET /api/models`：返回可用模型和对应代理入口。
- `GET /api/model-runtime`：返回当前活跃模型、就绪状态、容器状态和切换任务。
- `POST /api/model-runtime/switch`：Docker 模式下启动目标模型容器并停止非活跃模型容器。
- `GET /api/tasks`：读取本机持久化任务摘要列表，不返回大体积源文件和 OCR 结果。
- `GET /api/tasks/{task_id}`：读取一个任务的完整详情。
- `PUT /api/tasks/{task_id}`：保存一个任务到 `data/tasks/`；`task.json` 只保存轻量元数据，Markdown、OCR JSON、图片和 batch Markdown 拆到 `result.json`，摘要拆到 `summary.json`。
- `DELETE /api/tasks/{task_id}`：删除一个本地任务。
- `DELETE /api/tasks`：清空本地任务历史，只删除合法 task id 子目录，避免误删任务目录外文件。
- `POST /api/convert/to-pdf`：将 PPT/PPTX/DOC/DOCX 转为 PDF。
- `POST /api/paddleocr-vl-1.6`：代理 OCR 请求到 PaddleOCR-VL layout-parsing 服务。
- `POST /api/pp-ocrv6`：代理 OCR 请求到 PP-OCRv6 服务，返回页面图片、识别文字行、坐标框、置信度和原始 JSON。
- `POST /api/mineru`：代理 OCR 请求到 MinerU `/file_parse` 服务，返回 Markdown 和提取图片。
- `POST /api/glm-ocr`：代理 OCR 请求到 GLM-OCR (Ollama) 服务，使用 PP-OCRv6 版面检测 + GLM-OCR 文字识别。
- `POST /api/tasks/{task_id}/translate`：将任务的 OCR Markdown 结果翻译为目标语言，流式返回翻译进度。
- `GET /api/translate/config`：返回翻译功能是否已配置及使用的模型。
- `GET /api/openapi.json`：当前 WebUI 后端的 OpenAPI JSON；仓库里的 `paddle-layout-openapi.json` 是上游 Paddle layout-parsing 服务接口。

## 项目结构

```text
.
├── server.py                  # FastAPI 后端（含 MinerU / GLM-OCR / 翻译代理）
├── layout_detect_server.py    # PP-OCRv6 版面检测辅助服务
├── requirements.txt
├── requirements-macos.txt
├── requirements-macos-mlx.txt
├── macos-one-click.command
├── windows-one-click.bat
├── Dockerfile                 # pandocr-web 镜像
├── Dockerfile.ocr             # paddleocr-ocr-api 镜像
├── docker-compose.yml         # 含 mineru-api / ollama 可选服务
├── data/                      # 本地任务数据目录，默认不提交
├── model_cache_mineru/        # MinerU 模型缓存，默认不提交
├── env.txt                    # RTX 50 / Blackwell 环境变量
├── env.docker                 # RTX 30 / 40 环境变量
├── pipeline_config_ocr_v6.yaml
├── pipeline_config_vllm.yaml
├── pipeline_config_macos_mlx.template.yaml
├── scripts/                   # 部署辅助脚本
│   ├── windows-one-click.ps1
├── static/
│   ├── index.html             # 含 MinerU / GLM-OCR 专属设置面板
│   ├── app.js                 # 含 MinerU / GLM-OCR 模型切换和请求逻辑
│   ├── style.css
│   ├── i18n.js                # 中英文翻译
│   ├── latex_unicode.json     # LaTeX Unicode 映射
│   └── vendor/
│       ├── katex/             # KaTeX 数学公式渲染
│       └── pdfjs/             # PDF.js 预览（含 CJK CMap 支持）
├── QUICKSTART.md
├── webui-openapi.json
├── paddle-layout-openapi.json
├── DOCKER_DEPLOY.md
└── PROJECT_SUMMARY.md
```

## 本地开发

本地在 Docker 外运行 `server.py` 时，请设置 `PANDOCR_MODEL_CONTROL=none`，并自行启动模型服务。需要已有 PaddleOCR-VL 服务监听在 `http://localhost:8081/layout-parsing`；如需使用 PP-OCRv6，也需要启动 PaddleX OCR 服务监听在 `http://localhost:8082/ocr`，或设置 `PADDLE_OCR_SERVICE_URL`；如需使用 MinerU，需要启动 MinerU API 服务监听在 `http://localhost:8083`，或设置 `MINERU_SERVICE_URL`；如需使用 GLM-OCR，需要启动 Ollama 并拉取 `glm-ocr` 模型，或设置 `PANDOCR_OLLAMA_BASE_URL` 和 `PANDOCR_OLLAMA_MODEL`。

```bash
pip install -r requirements.txt
python server.py
```

然后打开 http://localhost:18000。

本地质量检查：

```bash
make check
```

## 与原版区别

本项目 Fork 自 [CHEN010325/paddleocr-local](https://github.com/CHEN010325/paddleocr-local)，在原版基础上增加了以下功能：

| 特性 | 原版 | 本 Fork |
|------|------|---------|
| OCR 模型 | PaddleOCR-VL 1.6、PP-OCRv6 | + MinerU 文档解析、GLM-OCR (Ollama) 文字识别 |
| 翻译 | 无 | 支持 20 种语言的 OCR 结果翻译（需配置 OpenAI 兼容 API） |
| 大文件 | 可能 OOM | >1GB 文件流式分批处理，支持中断恢复 |
| PDF 预览 | 基础预览 | 完整 CJK CMap 支持，中文/日文/韩文 PDF 不再乱码 |
| 模型缓存 | 单一目录 | 独立缓存目录（`model_cache_mineru/` 等），避免模型冲突 |
| 解析设置 | 固定面板 | 根据选中模型动态切换设置面板 |
| 局域网访问 | 仅绑定 127.0.0.1 | 默认绑定 0.0.0.0，支持局域网访问 |
| 端口 | 固定 8000 | 可通过 `PANDOCR_PORT` 环境变量自定义 |

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 百度飞桨 OCR 开源项目，提供了 PaddleOCR-VL 和 PP-OCRv6 模型
- [CHEN010325/paddleocr-local](https://github.com/CHEN010325/paddleocr-local) — 本项目 Fork 的上游仓库，提供了 WebUI 和 Docker 部署框架
- [MinerU](https://github.com/opendatalab/MinerU) — OpenDataLab 开源文档解析项目，MinerU2.5-Pro 模型
- [Ollama](https://github.com/ollama/ollama) — 本地大模型推理框架，用于运行 GLM-OCR
- [智谱 AI](https://www.zhipuai.cn/) — GLM-OCR 模型提供方
- [KaTeX](https://github.com/KaTeX/KaTeX) — 数学公式渲染
- [PDF.js](https://github.com/nicedoc/pdf.js) — PDF 预览渲染
