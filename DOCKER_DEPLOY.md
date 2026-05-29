# Docker 部署说明

## 服务组成

`docker-compose.yml` 当前只部署 3 个服务：

| 服务 | 作用 | 对外端口 |
| --- | --- | --- |
| `paddleocr-vlm-server` | VLLM 推理，加载 `PaddleOCR-VL-1.6-0.9B` | 无 |
| `paddleocr-vl-api` | PaddleX layout-parsing API | `8081:8080` |
| `pandocr-web` | WebUI、FastAPI 代理、Office 转 PDF | `8000:8000` |

rerank/reranker 服务已移除。Web 容器也不再挂载 Docker socket，不提供容器启停接口。

## 推荐配置

RTX 50 / Blackwell 使用 `env.txt`：

```text
API_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
VLM_BACKEND=vllm
VLM_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
PADDLEOCR_VL_MODEL_NAME=PaddleOCR-VL-1.6-0.9B
PADDLE_REQUEST_TIMEOUT=3600
```

非 Blackwell NVIDIA GPU 可以把两个镜像标签改为：

```text
latest-nvidia-gpu-offline
```

## 启动

```powershell
docker compose --env-file env.txt pull
docker compose --env-file env.txt build pandocr-web
docker compose --env-file env.txt up -d
```

## 健康检查

```powershell
docker compose --env-file env.txt ps
curl http://localhost:8000/api/models
curl http://localhost:8081/health
```

`/api/models` 应返回：

```json
{"data":[{"id":"PaddleOCR-VL-1.6-0.9B"}]}
```

## 重启 Web 服务

前端、FastAPI 或文档预览逻辑变更后，只需要重建并重启 `pandocr-web`：

```powershell
docker compose --env-file env.txt build pandocr-web
docker compose --env-file env.txt up -d --no-deps --force-recreate pandocr-web
```

如果只改了挂载的 `static/` 或 `server.py`，也可以直接重建/重启：

```powershell
docker compose --env-file env.txt up -d --no-deps --force-recreate pandocr-web
```

## 日志

```powershell
docker compose --env-file env.txt logs -f pandocr-web
docker compose --env-file env.txt logs -f paddleocr-vl-api
docker compose --env-file env.txt logs -f paddleocr-vlm-server
```

## 端口调整

修改 `docker-compose.yml`：

```yaml
pandocr-web:
  ports:
    - "18000:8000"

paddleocr-vl-api:
  ports:
    - "18081:8080"
```

## 数据和缓存

模型缓存通过目录挂载保留：

- `./model_cache:/home/paddleocr/.paddlex`
- `./model_cache_ocr:/home/paddleocr/.paddleocr`

这两个目录已加入 `.dockerignore`，不会被打进 `pandocr-web` 镜像构建上下文。

## 清理

```powershell
docker compose --env-file env.txt down
docker image prune
```

谨慎清理模型缓存目录；删除后下次启动会重新下载或加载模型资源。
