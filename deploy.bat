@echo off
REM Deploy PandOCR with Docker Compose (Windows)

echo Deploying PandOCR...

if not exist "env.txt" (
    echo env.txt does not exist. Run build.bat or create the env file first.
    pause
    exit /b 1
)

docker compose --env-file env.txt up -d

echo Waiting for services...
timeout /t 5 /nobreak >nul

echo.
echo Service status:
docker compose --env-file env.txt ps

echo.
echo Health checks:

curl -f http://localhost:8000/ >nul 2>&1
if not errorlevel 1 (
    echo pandocr-web ^(8000^) OK
) else (
    echo pandocr-web ^(8000^) not ready
)

curl -f http://localhost:8081/health >nul 2>&1
if not errorlevel 1 (
    echo paddleocr-vl-api ^(8081^) OK
) else (
    echo paddleocr-vl-api ^(8081^) not ready
)

echo.
echo Done.
echo WebUI: http://localhost:8000
echo API:   http://localhost:8081
echo.
echo Useful commands:
echo   docker compose --env-file env.txt logs -f
echo   docker compose --env-file env.txt logs -f pandocr-web
echo   docker compose --env-file env.txt restart pandocr-web
echo   docker compose --env-file env.txt down
echo.
pause
