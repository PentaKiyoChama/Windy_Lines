#!/bin/bash
# OST_WindyLines - Notarization Script
# Apple 公証 (Notarization) の実行

set -e

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 設定ファイルの読み込み
if [ ! -f "$SCRIPT_DIR/codesign_config.sh" ]; then
    echo -e "${RED}✗${NC} 設定ファイルが見つかりません。"
    echo "まず ./Mac/codesign_setup.sh を実行してください。"
    exit 1
fi

source "$SCRIPT_DIR/codesign_config.sh"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OST_WindyLines - Apple 公証${NC}"
echo -e "${BLUE}========================================${NC}\n"

# ビルド構成（Debug/Release）
BUILD_CONFIG="${1:-Release}"
if [ "$BUILD_CONFIG" != "Debug" ] && [ "$BUILD_CONFIG" != "Release" ]; then
    echo -e "${RED}✗${NC} 無効な構成です: $BUILD_CONFIG"
    echo "使い方: ./Mac/notarize_plugin.sh [Debug|Release]"
    exit 1
fi

PLUGIN_PATH="$SCRIPT_DIR/build/${BUILD_CONFIG}_signed/OST_WindyLines.plugin"
ZIP_PATH="/tmp/OST_WindyLines_Notarization.zip"

if [ ! -d "$PLUGIN_PATH" ]; then
    echo -e "${RED}✗${NC} プラグインが見つかりません: $PLUGIN_PATH"
    exit 1
fi

# /tmp にクリーンコピーして署名確認＆ZIP化（Finder xattr 問題回避）
WORK_DIR="/tmp/ost_notarize_$$"
PLUGIN_TMP="$WORK_DIR/OST_WindyLines.plugin"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
ditto --norsrc "$PLUGIN_PATH" "$PLUGIN_TMP"

# 署名確認
echo -e "${YELLOW}[1/5] 署名の確認...${NC}"
if ! codesign --verify --deep --strict "$PLUGIN_TMP" 2>/dev/null; then
    echo -e "${RED}✗${NC} プラグインが署名されていません。"
    echo "まず ./Mac/codesign_plugin.sh $BUILD_CONFIG を実行してください。"
    rm -rf "$WORK_DIR"
    exit 1
fi
echo -e "${GREEN}✓${NC} 署名済み\n"

# ZIP 化（/tmp 上のクリーンコピーから）
echo -e "${YELLOW}[2/5] ZIP アーカイブ作成中...${NC}"
rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$PLUGIN_TMP" "$ZIP_PATH"
echo -e "${GREEN}✓${NC} 作成完了: $ZIP_PATH"
echo "   サイズ: $(du -h "$ZIP_PATH" | cut -f1)\n"

# 公証申請
echo -e "${YELLOW}[3/5] Apple 公証の申請中...${NC}"
echo "   (数分かかる場合があります)"
echo ""

SUBMIT_OUTPUT=$(xcrun notarytool submit "$ZIP_PATH" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    --wait 2>&1)

echo "$SUBMIT_OUTPUT"
echo ""

# Submission IDの抽出
SUBMISSION_ID=$(echo "$SUBMIT_OUTPUT" | grep "id:" | head -1 | awk '{print $2}')

if [ -z "$SUBMISSION_ID" ]; then
    echo -e "${RED}✗${NC} 公証申請に失敗しました。"
    echo ""
    echo "エラーの可能性:"
    echo "  1. Keychain プロファイルが無効"
    echo "  2. Apple ID または Team ID が間違っている"
    echo "  3. App-Specific Password が無効"
    echo ""
    echo "再設定: ./Mac/codesign_setup.sh"
    exit 1
fi

# ステータス確認
echo -e "${YELLOW}[4/5] 公証ステータスの確認...${NC}"
xcrun notarytool info "$SUBMISSION_ID" \
    --keychain-profile "$KEYCHAIN_PROFILE"

echo ""

# 成功判定
if echo "$SUBMIT_OUTPUT" | grep -q "status: Accepted"; then
    echo -e "${GREEN}✓${NC} 公証成功！\n"
    
    # ステープル（/tmp コピーに適用）
    echo -e "${YELLOW}[5/5] 公証チケットのステープル中...${NC}"
    xcrun stapler staple "$PLUGIN_TMP"
    
    # 検証
    echo -e "${GREEN}✓${NC} ステープル完了\n"
    echo "検証:"
    xcrun stapler validate "$PLUGIN_TMP"
    
    # ステープル済みバンドルを元の場所に反映
    rm -rf "$PLUGIN_PATH"
    ditto "$PLUGIN_TMP" "$PLUGIN_PATH"
    rm -rf "$WORK_DIR"
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}公証完了！${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    echo "配布可能なプラグイン:"
    echo "  $PLUGIN_PATH"
    echo ""
    echo "このプラグインは macOS Gatekeeper を通過します。"
    echo "配布パッケージの作成:"
    echo "  ./Mac/create_installer.sh"
    
else
    rm -rf "$WORK_DIR"
    echo -e "${RED}✗${NC} 公証に失敗しました。"
    echo ""
    echo "詳細ログの確認:"
    echo "  xcrun notarytool log $SUBMISSION_ID \\"
    echo "    --apple-id $APPLE_ID \\"
    echo "    --team-id $TEAM_ID \\"
    echo "    --keychain-profile $KEYCHAIN_PROFILE"
    exit 1
fi

echo ""
