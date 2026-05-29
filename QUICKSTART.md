# 快速开始

## 1. 检查环境

```powershell
docker --version
nvidia-smi
```

当前机器如果是 RTX 50 / Blackwell，直接使用 `env.txt`。其他 NVIDIA GPU 可以改用 `env.docker` 或把 `env.txt` 里的两个镜像标签改为 `latest-nvidia-gpu-offline`。

## 2. 拉取并构建

```powershell
docker compose --env-file env.txt pull
docker compose --env-file env.txt build pandocr-web
```

`pandocr-web` 只构建 Web 服务，不包含 Paddle/PaddleX；PaddleOCR-VL 由官方 `paddleocr-vl-api` 和 `paddleocr-vlm-server` 镜像提供。

## 3. 启动服务

```powershell
docker compose --env-file env.txt up -d
```

首次启动 VLM 服务会加载模型，可能需要几分钟。

## 4. 验证

```powershell
docker compose --env-file env.txt ps
curl http://localhost:8000/api/models
curl http://localhost:8081/health
```

期望看到 3 个容器：

- `paddleocr-vlm-server`
- `paddleocr-vl-api`
- `pandocr-web`

`/api/models` 应返回 `PaddleOCR-VL-1.6-0.9B`。

## 5. 使用

打开 http://localhost:8000。

- 图片会直接作为图片请求提交。
- PDF 会按 200 页一批提交。
- PPT/PPTX/DOC/DOCX 会先由 `pandocr-web` 调 LibreOffice 转 PDF，再进入 PDF 流程。
- 结果区会渲染 Markdown、表格和 KaTeX 公式，并修正 OCR 结果里字面量 `\n` 导致的不换行问题。

## 常见问题

### 端口占用

修改 `docker-compose.yml` 中的端口映射，例如：

```yaml
ports:
  - "18000:8000"
```

### OCR 请求超时

大 PDF 批处理可能很慢，可以调大：

```text
PADDLE_REQUEST_TIMEOUT=7200
```

修改后重建或重启 `pandocr-web`：

```powershell
docker compose --env-file env.txt up -d --no-deps --force-recreate pandocr-web
```

### 前端改动没有生效

浏览器可能缓存了 `/static/app.js`。确认 `static/index.html` 中脚本版本号变化，或强制刷新页面。
