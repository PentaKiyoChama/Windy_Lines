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

PLUGIN_PATH="$SCRIPT_DIR/build/Debug/OST_WindyLines.plugin"
ZIP_PATH="$SCRIPT_DIR/build/OST_WindyLines_Notarization.zip"

if [ ! -d "$PLUGIN_PATH" ]; then
    echo -e "${RED}✗${NC} プラグインが見つかりません: $PLUGIN_PATH"
    exit 1
fi

# 署名確認
echo -e "${YELLOW}[1/5] 署名の確認...${NC}"
if ! codesign --verify --deep --strict "$PLUGIN_PATH" 2>/dev/null; then
    echo -e "${RED}✗${NC} プラグインが署名されていません。"
    echo "まず ./Mac/codesign_plugin.sh を実行してください。"
    exit 1
fi
echo -e "${GREEN}✓${NC} 署名済み\n"

# ZIP 化
echo -e "${YELLOW}[2/5] ZIP アーカイブ作成中...${NC}"
rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$PLUGIN_PATH" "$ZIP_PATH"
echo -e "${GREEN}✓${NC} 作成完了: $ZIP_PATH"
echo "   サイズ: $(du -h "$ZIP_PATH" | cut -f1)\n"

# 公証申請
echo -e "${YELLOW}[3/5] Apple 公証の申請中...${NC}"
echo "   (数分かかる場合があります)"
echo ""

SUBMIT_OUTPUT=$(xcrun notarytool submit "$ZIP_PATH" \
    --apple-id "$APPLE_ID" \
    --team-id "$TEAM_ID" \
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
    --apple-id "$APPLE_ID" \
    --team-id "$TEAM_ID" \
    --keychain-profile "$KEYCHAIN_PROFILE"

echo ""

# 成功判定
if echo "$SUBMIT_OUTPUT" | grep -q "status: Accepted"; then
    echo -e "${GREEN}✓${NC} 公証成功！\n"
    
    # ステープル
    echo -e "${YELLOW}[5/5] 公証チケットのステープル中...${NC}"
    xcrun stapler staple "$PLUGIN_PATH"
    
    # 検証
    echo -e "${GREEN}✓${NC} ステープル完了\n"
    echo "検証:"
    xcrun stapler validate "$PLUGIN_PATH"
    
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
