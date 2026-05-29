# PandOCR - PaddleOCR-VL 1.6 WebUI

PandOCR 是一个面向 PaddleOCR-VL 的轻量 Web 前端。当前项目固定部署 `PaddleOCR-VL-1.6-0.9B`，前端负责文件上传、队列、预览和下载，后端 FastAPI 只做静态文件服务、Office 转 PDF 和请求代理，OCR 推理由官方 PaddleOCR-VL Docker 镜像完成。

## 当前架构

```text
Browser
  -> pandocr-web:8000
       - FastAPI
       - static WebUI
       - Office to PDF conversion
       - PaddleOCR-VL request proxy
  -> paddleocr-vl-api:8080
       - PaddleX layout-parsing service
  -> paddleocr-vlm-server:8080
       - PaddleOCR-VL-1.6-0.9B VLLM inference
```

当前 Compose 只保留 3 个服务：

- `pandocr-web`
- `paddleocr-vl-api`
- `paddleocr-vlm-server`

项目不再包含 rerank/reranker 服务，也不再在 Web 容器里安装 Paddle/PaddleX。

## 功能

- 支持图片、PDF、PPT/PPTX、DOC/DOCX 上传。
- PDF 按页发送给 PaddleOCR-VL，便于对齐官方在线解析结果并稳定保留每页原始 JSON。
- 解析任务会持久化到本机 `data/tasks/`，刷新页面后仍可查看历史任务，删除按钮会同步删除本地记录。
- Markdown 预览支持表格横向滚动、KaTeX 数学公式渲染、OCR 结果中的字面量 `\n` 换行修正。
- 支持解析选项：版面检测、图表识别、文档矫正、方向识别、印章识别、公式编号、Markdown 忽略标签等。
- 下载结果时会打包 Markdown 和 OCR 提取图片。

## 快速启动

先根据显卡型号选择环境文件：

| 显卡 | 推荐环境文件 | 镜像标签 |
| --- | --- | --- |
| RTX 30 系列 | `env.docker` | `latest-nvidia-gpu-offline` |
| RTX 40 系列 | `env.docker` | `latest-nvidia-gpu-offline` |
| RTX 50 系列 / Blackwell | `env.txt` | `latest-nvidia-gpu-sm120-offline` |

下面命令以 RTX 50 系列的 `env.txt` 为例；RTX 30/40 系列用户把命令里的 `env.txt` 换成 `env.docker` 即可。

```powershell
docker compose --env-file env.txt pull
docker compose --env-file env.txt build pandocr-web
docker compose --env-file env.txt up -d
```

访问：

- WebUI: http://localhost:8000
- PaddleOCR-VL API health: http://localhost:8081/health

查看状态：

```powershell
docker compose --env-file env.txt ps
```

## 环境变量

`env.txt` 是当前 RTX 50 / Blackwell 推荐配置：

```text
API_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
VLM_BACKEND=vllm
VLM_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
PADDLEOCR_VL_MODEL_NAME=PaddleOCR-VL-1.6-0.9B
PADDLE_REQUEST_TIMEOUT=3600
```

RTX 30/40 系列等非 Blackwell NVIDIA GPU 使用 `env.docker`，其中两个镜像标签都是 `latest-nvidia-gpu-offline`。

## 主要接口

- `GET /`：WebUI 首页。
- `GET /api/models`：返回当前模型名。
- `GET /api/tasks`：读取本机持久化任务列表。
- `PUT /api/tasks/{task_id}`：保存一个任务到 `data/tasks/`。
- `DELETE /api/tasks/{task_id}`：删除一个本地任务。
- `DELETE /api/tasks`：清空本地任务历史。
- `POST /api/convert/to-pdf`：将 PPT/PPTX/DOC/DOCX 转为 PDF。
- `POST /api/paddleocr-vl-1.6`：代理 OCR 请求到 PaddleOCR-VL layout-parsing 服务。

## 项目结构

```text
.
├── server.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/                  # 本地任务数据目录，默认不提交
├── env.txt
├── env.docker
├── pipeline_config_vllm.yaml
├── static/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── vendor/katex/
├── QUICKSTART.md
├── DOCKER_DEPLOY.md
└── PROJECT_SUMMARY.md
```

## 常用命令

```powershell
docker compose --env-file env.txt ps
docker compose --env-file env.txt logs -f pandocr-web
docker compose --env-file env.txt restart pandocr-web
docker compose --env-file env.txt down
```

## 本地开发

本地运行 `server.py` 时，需要已有 PaddleOCR-VL 服务监听在 `http://localhost:8081/layout-parsing`。

```powershell
pip install -r requirements.txt
python server.py
```

然后打开 http://localhost:8000。
