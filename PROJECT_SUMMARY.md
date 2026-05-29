# 项目总结

## 定位

本项目是 `PaddleOCR-VL-1.6-0.9B` 的 WebUI 和代理层。它不在 Web 容器中直接运行 PaddleOCR 推理，而是通过 Docker Compose 调用官方 PaddleOCR-VL 服务。

## 运行架构

```text
Browser
  -> pandocr-web
     -> paddleocr-vl-api
        -> paddleocr-vlm-server
```

服务职责：

- `pandocr-web`：提供前端页面、FastAPI 代理、Office 转 PDF、结果格式整理。
- `paddleocr-vl-api`：官方 PaddleX layout-parsing 服务。
- `paddleocr-vlm-server`：官方 VLLM 推理服务，模型为 `PaddleOCR-VL-1.6-0.9B`。

已移除：

- `rerank-api`
- `reranker-server`
- `/api/rerank`
- `/api/services/*` 容器控制接口
- Web 容器中的 Docker SDK 和 Docker socket 挂载
- Web 容器中的 Paddle/PaddleX 安装步骤

## 核心接口

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | WebUI 首页 |
| `/api/models` | GET | 返回当前模型名 |
| `/api/convert/to-pdf` | POST | Office 文件转 PDF |
| `/api/paddleocr-vl-1.6` | POST | OCR 代理接口 |

## 前端处理流程

1. 上传图片、PDF 或 Office 文件。
2. Office 文件先请求 `/api/convert/to-pdf` 转成 PDF。
3. PDF 使用 PDF.js 生成缩略图，使用 PDF-lib 按 200 页一批切分请求体。
4. 请求 `/api/paddleocr-vl-1.6`。
5. 预览区使用 Marked + DOMPurify 渲染 Markdown。
6. 使用 KaTeX 渲染数学公式。
7. 将 OCR 结果中的字面量 `\n`、`\r\n`、`\t` 规范化为真实换行和制表符。
8. 下载时合并 Markdown 和提取图片。

## 关键配置

`env.txt` 用于 RTX 50 / Blackwell：

```text
API_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
VLM_BACKEND=vllm
VLM_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
PADDLEOCR_VL_MODEL_NAME=PaddleOCR-VL-1.6-0.9B
PADDLE_REQUEST_TIMEOUT=3600
```

`PADDLE_REQUEST_TIMEOUT` 建议保持较大值，因为 200 页 PDF 批处理可能需要较长时间。

## 主要文件

- `server.py`：FastAPI 后端。
- `static/app.js`：上传、队列、批处理、OCR 调用和下载逻辑。
- `static/index.html`：页面结构和前端依赖。
- `static/style.css`：界面样式、表格和公式预览样式。
- `static/vendor/katex/`：本地 KaTeX 运行时资源。
- `Dockerfile`：Web 容器镜像，只包含 Web、FastAPI、LibreOffice 依赖。
- `docker-compose.yml`：三服务编排。
- `pipeline_config_vllm.yaml`：PaddleOCR-VL API pipeline 配置。

## 验证建议

代码改动后至少执行：

```powershell
node --check static/app.js
python -m py_compile server.py
docker compose --env-file env.txt config --quiet
docker compose --env-file env.txt build pandocr-web
docker compose --env-file env.txt up -d --no-deps --force-recreate pandocr-web
curl http://localhost:8000/api/models
curl http://localhost:8081/health
```

涉及前端预览时，额外检查：

- 公式是否生成 `.katex` 节点。
- 表格是否可横向滚动。
- OCR 文本中的字面量 `\n` 是否变成真实换行。
