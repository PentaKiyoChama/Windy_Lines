#!/bin/bash
set -e

###############################################################################
#  __TPL_MATCH_NAME__ — プラグイン署名スクリプト
#  
#  4ステップ方式:
#  1. /tmp にクリーンコピー → 2. codesign → 3. 署名検証 → 4. コピーバック
#  
#  /tmp にコピーする理由: xattr のクォランティン属性を回避するため
###############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/codesign_config.sh"

PLUGIN_NAME="__TPL_MATCH_NAME__"
BUILD_DIR="$SCRIPT_DIR/build/Debug"
PLUGIN_PATH="$BUILD_DIR/${PLUGIN_NAME}.plugin"
TMP_PLUGIN="/tmp/${PLUGIN_NAME}.plugin"

echo -e "${BLUE}=== ${PLUGIN_NAME} コード署名 ===${NC}"

# ステップ1: /tmp にクリーンコピー
echo -e "${YELLOW}[1/4] /tmp にクリーンコピー...${NC}"
rm -rf "$TMP_PLUGIN"
ditto "$PLUGIN_PATH" "$TMP_PLUGIN"
xattr -cr "$TMP_PLUGIN"

# ステップ2: 署名
echo -e "${YELLOW}[2/4] コード署名中...${NC}"
codesign --deep --force \
    --options runtime \
    --timestamp \
    --entitlements "$SCRIPT_DIR/${PLUGIN_NAME}.entitlements.plist" \
    --sign "$CODESIGN_IDENTITY" \
    "$TMP_PLUGIN"

# ステップ3: 検証
echo -e "${YELLOW}[3/4] 署名検証中...${NC}"
codesign --verify --verbose "$TMP_PLUGIN"
echo -e "${GREEN}署名検証OK${NC}"

# ステップ4: コピーバック
echo -e "${YELLOW}[4/4] ビルドディレクトリにコピー...${NC}"
rm -rf "$PLUGIN_PATH"
ditto "$TMP_PLUGIN" "$PLUGIN_PATH"

echo -e "${GREEN}✅ 署名完了: ${PLUGIN_PATH}${NC}"
