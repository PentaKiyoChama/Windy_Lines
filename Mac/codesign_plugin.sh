#!/bin/bash
# OST_WindyLines - Code Signing Script
# プラグインファイルに署名

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
echo -e "${BLUE}OST_WindyLines - プラグイン署名${NC}"
echo -e "${BLUE}========================================${NC}\n"

# プラグインファイルの検索
PLUGIN_PATH="$SCRIPT_DIR/build/Debug/OST_WindyLines.plugin"

if [ ! -d "$PLUGIN_PATH" ]; then
    echo -e "${RED}✗${NC} プラグインが見つかりません: $PLUGIN_PATH"
    echo "まず Xcode でビルドしてください。"
    exit 1
fi

echo -e "${YELLOW}対象:${NC} $PLUGIN_PATH"
echo -e "${YELLOW}証明書:${NC} $CODESIGN_IDENTITY"
echo ""

# 既存の署名を削除
echo -e "${YELLOW}[1/3] 既存の署名を削除中...${NC}"
codesign --remove-signature "$PLUGIN_PATH" 2>/dev/null || true
echo -e "${GREEN}✓${NC} 完了\n"

# 署名実行
echo -e "${YELLOW}[2/3] プラグインに署名中...${NC}"
codesign --deep --force --verify --verbose \
    --sign "$CODESIGN_IDENTITY" \
    --options runtime \
    --timestamp \
    "$PLUGIN_PATH"

echo -e "${GREEN}✓${NC} 署名完了\n"

# 署名の検証
echo -e "${YELLOW}[3/3] 署名の検証中...${NC}"
codesign --verify --deep --strict --verbose=2 "$PLUGIN_PATH"
spctl --assess --type execute --verbose "$PLUGIN_PATH" || true

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}署名完了！${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "署名情報:"
codesign -dv "$PLUGIN_PATH" 2>&1 | grep -E "(Authority|Identifier|TeamIdentifier)"

echo ""
echo "次のステップ:"
echo "  ./Mac/notarize_plugin.sh を実行して Apple 公証を取得"
echo ""
