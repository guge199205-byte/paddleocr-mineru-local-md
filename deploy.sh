#!/bin/bash

set -e

echo "Deploying PandOCR..."

if [ ! -f "env.txt" ]; then
    echo "env.txt does not exist. Run build.sh or create the env file first."
    exit 1
fi

docker compose --env-file env.txt up -d

echo "Waiting for services..."
sleep 5

echo ""
echo "Service status:"
docker compose --env-file env.txt ps

echo ""
echo "Health checks:"

if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "pandocr-web (8000) OK"
else
    echo "pandocr-web (8000) not ready"
fi

if curl -f http://localhost:8081/health > /dev/null 2>&1; then
    echo "paddleocr-vl-api (8081) OK"
else
    echo "paddleocr-vl-api (8081) not ready"
fi

echo ""
echo "Done."
echo "WebUI: http://localhost:8000"
echo "API:   http://localhost:8081"
echo ""
echo "Useful commands:"
echo "  docker compose --env-file env.txt logs -f"
echo "  docker compose --env-file env.txt logs -f pandocr-web"
echo "  docker compose --env-file env.txt restart pandocr-web"
echo "  docker compose --env-file env.txt down"
