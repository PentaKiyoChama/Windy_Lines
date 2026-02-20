#!/bin/bash
set -e

###############################################################################
#  __TPL_MATCH_NAME__ — プラグインインストールスクリプト
#  Premiere Pro のプラグインフォルダにコピー
###############################################################################

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_NAME="__TPL_MATCH_NAME__"
BUILD_DIR="$SCRIPT_DIR/build/Debug"
PLUGIN_PATH="$BUILD_DIR/${PLUGIN_NAME}.plugin"
INSTALL_DIR="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"

echo -e "${BLUE}=== ${PLUGIN_NAME} インストール ===${NC}"

if [ ! -d "$PLUGIN_PATH" ]; then
    echo "エラー: ${PLUGIN_PATH} が見つかりません。先にビルドしてください。"
    exit 1
fi

echo "インストール先: ${INSTALL_DIR}"
sudo ditto "$PLUGIN_PATH" "$INSTALL_DIR/${PLUGIN_NAME}.plugin"

echo -e "${GREEN}✅ インストール完了${NC}"
echo "Premiere Pro を再起動してください。"
