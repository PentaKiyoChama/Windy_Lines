#!/bin/bash
# OST_WindyLines - Cross Platform Packaging Script
# macOS(.plugin) と Windows(.aex) を同梱した配布 ZIP を作成

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OST_WindyLines - Cross Platform Package${NC}"
echo -e "${BLUE}========================================${NC}\n"

BUILD_CONFIG="${1:-Release}"
WIN_INPUT_PATH="${2:-}"

if [ "$BUILD_CONFIG" != "Debug" ] && [ "$BUILD_CONFIG" != "Release" ]; then
    echo -e "${RED}✗${NC} 無効な構成です: $BUILD_CONFIG"
    echo "使い方: ./Mac/package_cross_platform.sh [Debug|Release] /path/to/OST_WindyLines.aex"
    exit 1
fi

if [ -z "$WIN_INPUT_PATH" ]; then
    echo -e "${RED}✗${NC} Windows 側入力のパスが未指定です。"
    echo ""
    echo "使い方:"
    echo "  ./Mac/package_cross_platform.sh Release /path/to/OST_WindyLines.aex"
    echo "  ./Mac/package_cross_platform.sh Release /path/to/windows_payload_dir"
    echo ""
    echo "例:"
    echo "  ./Mac/package_cross_platform.sh Release ~/Desktop/OST_WindyLines.aex"
    echo "  ./Mac/package_cross_platform.sh Release ~/Desktop/OST_WindyLines_win_payload"
    exit 1
fi

if [ ! -e "$WIN_INPUT_PATH" ]; then
    echo -e "${RED}✗${NC} Windows 側入力が見つかりません: $WIN_INPUT_PATH"
    exit 1
fi

WIN_MODE=""
if [ -d "$WIN_INPUT_PATH" ]; then
    WIN_MODE="dir"
elif [ -f "$WIN_INPUT_PATH" ]; then
    WIN_MODE="file"
else
    echo -e "${RED}✗${NC} Windows 側入力がファイル/フォルダとして解釈できません: $WIN_INPUT_PATH"
    exit 1
fi

if [ "$WIN_MODE" = "file" ] && [[ "$WIN_INPUT_PATH" != *.aex ]]; then
    echo -e "${YELLOW}[WARN]${NC} 指定ファイルの拡張子が .aex ではありません: $WIN_INPUT_PATH"
fi

MAC_PLUGIN_PATH="$SCRIPT_DIR/build/${BUILD_CONFIG}_signed/OST_WindyLines.plugin"
if [ ! -d "$MAC_PLUGIN_PATH" ]; then
    echo -e "${RED}✗${NC} macOS プラグインが見つかりません: $MAC_PLUGIN_PATH"
    echo "先に ./Mac/notarize_plugin.sh $BUILD_CONFIG を実行してください。"
    exit 1
fi

echo -e "${YELLOW}構成:${NC} $BUILD_CONFIG"
echo -e "${YELLOW}macOS:${NC} $MAC_PLUGIN_PATH"
echo -e "${YELLOW}Windows入力:${NC} $WIN_INPUT_PATH"
echo -e "${YELLOW}Windows入力種別:${NC} $WIN_MODE"
echo ""

TS="$(date +%Y%m%d_%H%M%S)"
PACKAGE_NAME="OST_WindyLines_${BUILD_CONFIG}_mac_win_${TS}"
OUT_DIR="$ROOT_DIR/dist"
OUT_ZIP="$OUT_DIR/${PACKAGE_NAME}.zip"
WORK_DIR="/tmp/ost_package_${TS}_$$"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/${PACKAGE_NAME}/macOS"
mkdir -p "$WORK_DIR/${PACKAGE_NAME}/Windows"
mkdir -p "$OUT_DIR"

echo -e "${YELLOW}[1/4] macOSプラグインをコピー中...${NC}"
COPYFILE_DISABLE=1 ditto --norsrc --noextattr --noacl "$MAC_PLUGIN_PATH" "$WORK_DIR/${PACKAGE_NAME}/macOS/OST_WindyLines.plugin"
xattr -cr "$WORK_DIR/${PACKAGE_NAME}/macOS/OST_WindyLines.plugin" || true
echo -e "${GREEN}✓${NC} 完了"

echo -e "${YELLOW}[2/4] Windowsプラグインをコピー中...${NC}"
if [ "$WIN_MODE" = "dir" ]; then
    cp -R "$WIN_INPUT_PATH"/. "$WORK_DIR/${PACKAGE_NAME}/Windows/"
    if ! find "$WORK_DIR/${PACKAGE_NAME}/Windows" -type f -name "*.aex" | grep -q .; then
        echo -e "${YELLOW}[WARN]${NC} Windows フォルダ内に .aex が見つかりませんでした。"
    fi
else
    cp "$WIN_INPUT_PATH" "$WORK_DIR/${PACKAGE_NAME}/Windows/OST_WindyLines.aex"
fi
echo -e "${GREEN}✓${NC} 完了"

echo -e "${YELLOW}[3/4] Readmeを生成中...${NC}"
cat > "$WORK_DIR/${PACKAGE_NAME}/README.txt" <<EOF
OST_WindyLines Cross Platform Package

Contents:
- macOS/OST_WindyLines.plugin
- Windows/... (aex と任意ファイル)

Install (macOS):
1. Copy OST_WindyLines.plugin to Adobe MediaCore plug-ins folder.

Install (Windows):
1. Windows フォルダ内の aex を Adobe plug-ins folder にコピーしてください。

Build config: $BUILD_CONFIG
Created at: $(date)
EOF
echo -e "${GREEN}✓${NC} 完了"

echo -e "${YELLOW}[4/4] ZIPを作成中...${NC}"
(
    cd "$WORK_DIR"
    COPYFILE_DISABLE=1 zip -r "$OUT_ZIP" "$PACKAGE_NAME" -x "*/.DS_Store"
)
echo -e "${GREEN}✓${NC} 完了"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}同梱パッケージ作成完了${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "出力ファイル:"
echo "  $OUT_ZIP"
echo ""
echo "サイズ: $(du -h "$OUT_ZIP" | cut -f1)"

rm -rf "$WORK_DIR"
