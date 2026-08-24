#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/trancy_tool/bin/python}"
SOURCE_ROOT="${VIDEO_AGENT_SOURCE_ROOT:-$PROJECT_DIR/../video_text_mutiplemodal_agent}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "找不到 Python：$PYTHON_BIN" >&2
  echo "可用 PYTHON_BIN=/path/to/python ./build_macos.sh 指定环境。" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import PyInstaller' >/dev/null 2>&1; then
  echo "当前环境没有 PyInstaller。请先执行：" >&2
  echo "  $PYTHON_BIN -m pip install -r $PROJECT_DIR/requirements-build.txt" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "找不到源项目：$SOURCE_ROOT" >&2
  exit 1
fi

cd "$PROJECT_DIR"
rm -rf build
# Finder 可能会在 dist 中自动创建 .DS_Store，直接删除整个目录会偶发
# 报错“Directory not empty”。逐项清理目录后复用它，避免影响打包结果。
mkdir -p dist
find "$PROJECT_DIR/dist" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# 把构建缓存放在项目内，便于 CI/受限环境构建，也避免写入用户目录。
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$PROJECT_DIR/.pyinstaller}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.build_cache/matplotlib}"
mkdir -p "$PYINSTALLER_CONFIG_DIR" "$MPLCONFIGDIR"

VIDEO_AGENT_SOURCE_ROOT="$SOURCE_ROOT" \
  "$PYTHON_BIN" -m PyInstaller \
  --clean \
  --noconfirm \
  AcademicAgent.spec

# BUNDLE 已经把 onedir 内容复制进 .app；交付时只保留 .app，删除重复目录和中间缓存。
rm -rf "$PROJECT_DIR/dist/AcademicAgent" \
       "$PROJECT_DIR/build" \
       "$PROJECT_DIR/.pyinstaller" \
       "$PROJECT_DIR/.build_cache"
cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/dist/.env.example"
if [[ -f "$PROJECT_DIR/.env" ]]; then
  # AcademicAgent.spec 已把真实配置封装进 .app；这里额外保留一份，方便
  # 开发者检查或在不重新打包时修改 dist 目录中的配置。
  cp "$PROJECT_DIR/.env" "$PROJECT_DIR/dist/.env"
  echo "已封装真实运行配置到 .app，并复制到：$PROJECT_DIR/dist/.env"
else
  echo "未找到真实 .env，仅复制 .env.example；云端模型需要用户另行配置。"
fi

echo
echo "构建完成：$PROJECT_DIR/dist/AcademicAgent.app"
echo "双击该 .app 即可启动。"
echo "如需把本地大模型一并打包，请使用 INCLUDE_LOCAL_MODELS=1；默认建议让用户在界面中选择模型目录。"
