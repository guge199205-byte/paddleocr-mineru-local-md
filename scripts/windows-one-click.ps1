param(
    [string]$EnvFile = "",
    [int]$GpuId = -1,
    [int]$TimeoutSeconds = 1800,
    [switch]$DryRun,
    [switch]$SkipPull,
    [switch]$SkipBuild,
    [switch]$SkipClean,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script:RuntimeEnv = ""
$script:DiagnosticsShown = $false
Set-Location $script:RepoRoot

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Invoke-Checked {
    param(
        [string]$File,
        [string[]]$Arguments,
        [string]$Description
    )

    Write-Section $Description
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Get-RequiredCommand {
    param([string]$Name, [string]$InstallHint)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $InstallHint"
    }
}

function Get-GpuList {
    $args = @(
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits"
    )
    $output = & nvidia-smi @args
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed. Please install/update the NVIDIA driver first."
    }

    $gpus = @()
    foreach ($line in $output) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $parts = $line -split ","
        if ($parts.Count -lt 4) {
            throw "Unexpected nvidia-smi output: $line"
        }

        $gpus += [pscustomobject]@{
            Index = [int]($parts[0].Trim())
            Name = $parts[1].Trim()
            TotalMiB = [int]([double]($parts[2].Trim()))
            FreeMiB = [int]([double]($parts[3].Trim()))
        }
    }

    if ($gpus.Count -eq 0) {
        throw "No NVIDIA GPU was detected by nvidia-smi."
    }

    return @($gpus)
}

function Test-IsBlackwellGpu {
    param([string]$Name)

    $normalized = $Name.ToLowerInvariant()
    return ($normalized -match "blackwell" -or $normalized -match "rtx\s+50(50|60|70|80|90)\b")
}

function Select-Gpu {
    param([object[]]$Gpus, [int]$RequestedGpuId)

    Write-Section "Detected NVIDIA GPUs"
    foreach ($gpu in $Gpus) {
        Write-Host ("GPU {0}: {1} | total={2} MiB free={3} MiB" -f $gpu.Index, $gpu.Name, $gpu.TotalMiB, $gpu.FreeMiB)
    }

    if ($RequestedGpuId -ge 0) {
        $requested = @($Gpus | Where-Object { $_.Index -eq $RequestedGpuId })
        if ($requested.Count -eq 0) {
            throw "Requested GPU $RequestedGpuId was not found."
        }
        return $requested[0]
    }

    return @($Gpus | Sort-Object -Property FreeMiB -Descending)[0]
}

function Resolve-BaseEnvFile {
    param([object]$Gpu, [string]$RequestedEnvFile)

    if (-not [string]::IsNullOrWhiteSpace($RequestedEnvFile)) {
        if ([System.IO.Path]::IsPathRooted($RequestedEnvFile)) {
            $path = $RequestedEnvFile
        }
        else {
            $path = Join-Path $script:RepoRoot $RequestedEnvFile
        }
        if (-not (Test-Path $path)) {
            throw "Env file not found: $RequestedEnvFile"
        }
        return (Resolve-Path $path).Path
    }

    if (Test-IsBlackwellGpu $Gpu.Name) {
        return (Resolve-Path (Join-Path $script:RepoRoot "env.txt")).Path
    }

    return (Resolve-Path (Join-Path $script:RepoRoot "env.docker")).Path
}

function Set-EnvLine {
    param(
        [string[]]$Lines,
        [string]$Key,
        [string]$Value
    )

    $updated = New-Object System.Collections.Generic.List[string]
    $found = $false
    $pattern = "^\s*" + [regex]::Escape($Key) + "\s*="

    foreach ($line in $Lines) {
        if ($line -match $pattern) {
            $updated.Add("$Key=$Value")
            $found = $true
        }
        else {
            $updated.Add($line)
        }
    }

    if (-not $found) {
        $updated.Add("$Key=$Value")
    }

    return [string[]]$updated.ToArray()
}

function New-RuntimeEnvFile {
    param([string]$BaseEnvFile, [object]$Gpu)

    $tmpDir = Join-Path $script:RepoRoot "tmp"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

    $runtimeEnv = Join-Path $tmpDir "windows-one-click.env"
    $lines = [string[]](Get-Content -Path $BaseEnvFile -Encoding UTF8)
    $lines = Set-EnvLine -Lines $lines -Key "PANDOCR_GPU_DEVICE_ID" -Value ([string]$Gpu.Index)
    Set-Content -Path $runtimeEnv -Value $lines -Encoding ASCII

    return (Resolve-Path $runtimeEnv).Path
}

function Get-ComposeArgs {
    param([string[]]$Arguments)
    return @("compose", "--env-file", $script:RuntimeEnv) + $Arguments
}

function Test-HttpOk {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    }
    catch {
        return $false
    }
}

function Get-ContainerStatus {
    param([string]$Name)

    $format = "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"
    $output = & docker inspect --format $format $Name 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($output)) {
        return "missing|none"
    }

    return $output.Trim()
}

function Show-Diagnostics {
    if ($script:DiagnosticsShown -or [string]::IsNullOrWhiteSpace($script:RuntimeEnv)) {
        return
    }

    $script:DiagnosticsShown = $true
    Write-Section "Service status"
    $statusArgs = Get-ComposeArgs @("ps", "-a")
    & docker @statusArgs

    foreach ($service in @("paddleocr-vlm-server", "paddleocr-vl-api", "pandocr-web")) {
        Write-Section "Recent logs: $service"
        $logArgs = Get-ComposeArgs @("logs", "--tail=160", $service)
        & docker @logArgs
    }
}

function Wait-ForServices {
    param([int]$Timeout)

    Write-Section "Waiting for services"
    $deadline = (Get-Date).AddSeconds($Timeout)
    $lastLine = ""

    while ((Get-Date) -lt $deadline) {
        $vlm = Get-ContainerStatus "paddleocr-vlm-server"
        $api = Get-ContainerStatus "paddleocr-vl-api"
        $web = Get-ContainerStatus "pandocr-web"
        $apiOk = Test-HttpOk "http://localhost:8081/health"
        $webOk = Test-HttpOk "http://localhost:8000/"

        if ($apiOk -and $webOk) {
            Write-Ok "All services are healthy."
            return
        }

        foreach ($status in @($vlm, $api, $web)) {
            if ($status -match "^exited\|") {
                Show-Diagnostics
                throw "A service exited before becoming healthy."
            }
        }

        $line = "vlm=$vlm api=$api web=$web apiHttp=$apiOk webHttp=$webOk"
        if ($line -ne $lastLine) {
            Write-Host $line
            $lastLine = $line
        }

        Start-Sleep -Seconds 15
    }

    Show-Diagnostics
    throw "Timed out after $Timeout seconds while waiting for services."
}

try {
    Write-Section "PandOCR Windows one-click deployment"
    Write-Host "Repository: $script:RepoRoot"

    Get-RequiredCommand -Name "docker" -InstallHint "Please install Docker Desktop and start it."
    Get-RequiredCommand -Name "nvidia-smi" -InstallHint "Please install/update the NVIDIA driver."

    Invoke-Checked -File "docker" -Arguments @("info", "--format", "{{.ServerVersion}}") -Description "Checking Docker Desktop"
    Invoke-Checked -File "docker" -Arguments @("compose", "version") -Description "Checking Docker Compose"

    $gpus = Get-GpuList
    $gpu = Select-Gpu -Gpus $gpus -RequestedGpuId $GpuId
    Write-Ok ("Selected GPU {0}: {1}" -f $gpu.Index, $gpu.Name)

    if ($gpu.TotalMiB -lt 8192) {
        throw "GPU $($gpu.Index) has only $($gpu.TotalMiB) MiB VRAM. PaddleOCR-VL requires at least 8192 MiB."
    }
    if ($gpu.FreeMiB -lt 6656) {
        throw "GPU $($gpu.Index) has only $($gpu.FreeMiB) MiB free VRAM. Close GPU-heavy apps or choose another GPU with -GpuId."
    }

    $baseEnv = Resolve-BaseEnvFile -Gpu $gpu -RequestedEnvFile $EnvFile
    $script:RuntimeEnv = New-RuntimeEnvFile -BaseEnvFile $baseEnv -Gpu $gpu
    Write-Ok "Base env: $baseEnv"
    Write-Ok "Runtime env: $script:RuntimeEnv"

    Invoke-Checked -File "docker" -Arguments (Get-ComposeArgs @("config", "--quiet")) -Description "Validating Docker Compose config"

    if ($DryRun) {
        Write-Section "Dry run complete"
        Write-Host "Selected GPU: $($gpu.Index) - $($gpu.Name)"
        Write-Host "Base env: $baseEnv"
        Write-Host "Runtime env: $script:RuntimeEnv"
        Write-Host "No images were pulled, built, or started."
        exit 0
    }

    if (-not $SkipPull) {
        Invoke-Checked -File "docker" -Arguments (Get-ComposeArgs @("pull", "paddleocr-vlm-server", "paddleocr-vl-api")) -Description "Pulling official PaddleOCR-VL images"
    }
    else {
        Write-Warn "Skipping image pull."
    }

    if (-not $SkipBuild) {
        Invoke-Checked -File "docker" -Arguments (Get-ComposeArgs @("build", "pandocr-web")) -Description "Building pandocr-web locally"
    }
    else {
        Write-Warn "Skipping pandocr-web build."
    }

    if (-not $SkipClean) {
        Invoke-Checked -File "docker" -Arguments (Get-ComposeArgs @("down", "--remove-orphans")) -Description "Clearing old containers"
    }
    else {
        Write-Warn "Skipping old-container cleanup."
    }

    Invoke-Checked -File "docker" -Arguments (Get-ComposeArgs @("run", "--rm", "--no-deps", "paddleocr-vlm-server", "nvidia-smi")) -Description "Checking Docker GPU access"
    Invoke-Checked -File "docker" -Arguments (Get-ComposeArgs @("up", "-d", "--force-recreate")) -Description "Starting PandOCR services"

    Wait-ForServices -Timeout $TimeoutSeconds

    Write-Section "Deployment complete"
    Write-Host "WebUI: http://localhost:8000"
    Write-Host "API health: http://localhost:8081/health"
    Write-Host "Useful logs: docker compose --env-file `"$script:RuntimeEnv`" logs -f"

    if (-not $NoOpen) {
        Start-Process "http://localhost:8000"
    }

    exit 0
}
catch {
    Write-Host ""
    Write-Host "[FAILED] $($_.Exception.Message)" -ForegroundColor Red
    Show-Diagnostics
    exit 1
}
