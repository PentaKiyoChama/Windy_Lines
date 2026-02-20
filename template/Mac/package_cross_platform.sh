#!/bin/bash
set -e

###############################################################################
#  __TPL_MATCH_NAME__ — クロスプラットフォームパッケージスクリプト
#  macOS (.plugin) + Windows (.aex) を同梱ZIPとして生成
###############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLUGIN_NAME="__TPL_MATCH_NAME__"

# バージョン（Version.hから自動取得）
VERSION=$(grep "VERSION_MAJOR" "$PROJECT_DIR/${PLUGIN_NAME}_Version.h" | head -1 | grep -o '[0-9]*')
VERSION="v${VERSION:-1}"

STAGE_DIR="/tmp/${PLUGIN_NAME}_package"
ZIP_NAME="${PLUGIN_NAME}_${VERSION}"

MAC_PLUGIN="$SCRIPT_DIR/build/Debug/${PLUGIN_NAME}.plugin"
WIN_AEX="$PROJECT_DIR/Win/x64/Release/${PLUGIN_NAME}.aex"

echo -e "${BLUE}=== ${PLUGIN_NAME} パッケージ作成 ===${NC}"

rm -rf "$STAGE_DIR"

# macOS パッケージ
if [ -d "$MAC_PLUGIN" ]; then
    echo -e "${YELLOW}macOS パッケージを作成中...${NC}"
    MAC_DIR="$STAGE_DIR/${ZIP_NAME}_macOS"
    mkdir -p "$MAC_DIR"
    COPYFILE_DISABLE=1 ditto --norsrc --noextattr --noacl "$MAC_PLUGIN" "$MAC_DIR/${PLUGIN_NAME}.plugin"
    [ -f "$PROJECT_DIR/INSTALL.txt" ] && cp "$PROJECT_DIR/INSTALL.txt" "$MAC_DIR/"
    [ -f "$PROJECT_DIR/README.txt" ] && cp "$PROJECT_DIR/README.txt" "$MAC_DIR/"
    
    pushd "$STAGE_DIR" > /dev/null
    COPYFILE_DISABLE=1 zip -r "${ZIP_NAME}_macOS.zip" "${ZIP_NAME}_macOS"
    popd > /dev/null
    
    cp "$STAGE_DIR/${ZIP_NAME}_macOS.zip" "$PROJECT_DIR/"
    echo -e "${GREEN}✅ macOS: ${ZIP_NAME}_macOS.zip${NC}"
else
    echo -e "${YELLOW}⚠ macOS ビルドが見つかりません: ${MAC_PLUGIN}${NC}"
fi

# Windows パッケージ
if [ -f "$WIN_AEX" ]; then
    echo -e "${YELLOW}Windows パッケージを作成中...${NC}"
    WIN_DIR="$STAGE_DIR/${ZIP_NAME}_Windows"
    mkdir -p "$WIN_DIR"
    cp "$WIN_AEX" "$WIN_DIR/"
    [ -f "$PROJECT_DIR/INSTALL.txt" ] && cp "$PROJECT_DIR/INSTALL.txt" "$WIN_DIR/"
    [ -f "$PROJECT_DIR/README.txt" ] && cp "$PROJECT_DIR/README.txt" "$WIN_DIR/"
    
    pushd "$STAGE_DIR" > /dev/null
    COPYFILE_DISABLE=1 zip -r "${ZIP_NAME}_Windows.zip" "${ZIP_NAME}_Windows"
    popd > /dev/null
    
    cp "$STAGE_DIR/${ZIP_NAME}_Windows.zip" "$PROJECT_DIR/"
    echo -e "${GREEN}✅ Windows: ${ZIP_NAME}_Windows.zip${NC}"
else
    echo -e "${YELLOW}⚠ Windows ビルドが見つかりません: ${WIN_AEX}${NC}"
fi

rm -rf "$STAGE_DIR"
echo -e "${GREEN}パッケージ作成完了${NC}"
