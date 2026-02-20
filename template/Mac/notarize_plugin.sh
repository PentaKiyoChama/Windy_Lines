#!/bin/bash
set -e

###############################################################################
#  __TPL_MATCH_NAME__ — Apple公証スクリプト
#  
#  前提: codesign_plugin.sh で署名済みであること
#  前提: `xcrun notarytool store-credentials AC_PASSWORD` でパスワード保存済み
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
ZIP_PATH="/tmp/${PLUGIN_NAME}_notarize.zip"

echo -e "${BLUE}=== ${PLUGIN_NAME} Apple公証 ===${NC}"

# ステップ1: 署名確認
echo -e "${YELLOW}[1/5] 署名状態を確認中...${NC}"
codesign --verify --verbose "$PLUGIN_PATH" || {
    echo -e "${RED}署名が無効です。先に codesign_plugin.sh を実行してください。${NC}"
    exit 1
}

# ステップ2: ZIP作成
echo -e "${YELLOW}[2/5] ZIP作成中...${NC}"
rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$PLUGIN_PATH" "$ZIP_PATH"

# ステップ3: 公証申請
echo -e "${YELLOW}[3/5] 公証申請中（5-10分かかります）...${NC}"
xcrun notarytool submit "$ZIP_PATH" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    --wait

# ステップ4: Staple
echo -e "${YELLOW}[4/5] Staple中...${NC}"
xcrun stapler staple "$PLUGIN_PATH"

# ステップ5: 最終検証
echo -e "${YELLOW}[5/5] 最終検証中...${NC}"
spctl --assess --type execute --verbose "$PLUGIN_PATH" 2>&1 || true

echo -e "${GREEN}✅ 公証完了: ${PLUGIN_PATH}${NC}"

# クリーンアップ
rm -f "$ZIP_PATH"
