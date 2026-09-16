$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "找不到 python，请先安装 Python 3.10 或 3.11，并加入 PATH。"
}

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/create_windows_icon.py

if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist) { Remove-Item dist -Recurse -Force }

$env:PYINSTALLER_CONFIG_DIR = Join-Path $ProjectRoot ".pyinstaller"
$env:MPLCONFIGDIR = Join-Path $ProjectRoot ".build_cache\matplotlib"
New-Item -ItemType Directory -Force -Path $env:PYINSTALLER_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

python -m PyInstaller --clean --noconfirm AcademicAgent.windows.spec

Copy-Item .env.example dist\AcademicAgent\.env.example -Force
$archive = Join-Path $ProjectRoot "dist\AcademicAgent-windows.zip"
if (Test-Path $archive) { Remove-Item $archive -Force }
Compress-Archive -Path dist\AcademicAgent -DestinationPath $archive

Write-Host "Windows 构建完成：$archive"
