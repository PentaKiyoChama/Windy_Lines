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

# ビルド構成（Debug/Release）
BUILD_CONFIG="${1:-Release}"
if [ "$BUILD_CONFIG" != "Debug" ] && [ "$BUILD_CONFIG" != "Release" ]; then
    echo -e "${RED}✗${NC} 無効な構成です: $BUILD_CONFIG"
    echo "使い方: ./Mac/codesign_plugin.sh [Debug|Release]"
    exit 1
fi

# Entitlements（必要なら codesign_config.sh 側で上書き可能）
ENTITLEMENTS_PATH="${ENTITLEMENTS_PATH:-$SCRIPT_DIR/OST_WindyLines.entitlements.plist}"

# ビルド成果物のパス
PLUGIN_BUILD_PATH="$SCRIPT_DIR/build/$BUILD_CONFIG/OST_WindyLines.plugin"
# 署名済み出力先
PLUGIN_SIGNED_PATH="$SCRIPT_DIR/build/${BUILD_CONFIG}_signed/OST_WindyLines.plugin"
# /tmp ステージング（Finder の xattr 再付与を完全回避）
WORK_DIR="/tmp/ost_codesign_$$"
PLUGIN_TMP="$WORK_DIR/OST_WindyLines.plugin"

if [ ! -d "$PLUGIN_BUILD_PATH" ]; then
    echo -e "${RED}✗${NC} プラグインが見つかりません: $PLUGIN_BUILD_PATH"
    echo "まず Xcode でビルドしてください。"
    exit 1
fi

echo -e "${YELLOW}対象:${NC} $PLUGIN_BUILD_PATH"
echo -e "${YELLOW}署名先:${NC} $PLUGIN_SIGNED_PATH"
echo -e "${YELLOW}証明書:${NC} $CODESIGN_IDENTITY"
echo -e "${YELLOW}構成:${NC} $BUILD_CONFIG"
echo -e "${YELLOW}Entitlements:${NC} $ENTITLEMENTS_PATH"
echo ""

# /tmp にクリーンコピー（Finder が監視しないのでxattr問題なし）
echo -e "${YELLOW}[1/4] /tmp にクリーンコピーを作成中...${NC}"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
ditto --norsrc "$PLUGIN_BUILD_PATH" "$PLUGIN_TMP"
codesign --remove-signature "$PLUGIN_TMP" 2>/dev/null || true
xattr -cr "$PLUGIN_TMP"
find "$PLUGIN_TMP" -exec xattr -c {} \; 2>/dev/null || true
find "$PLUGIN_TMP" -name '._*' -type f -delete 2>/dev/null || true
find "$PLUGIN_TMP" -name '.DS_Store' -type f -delete 2>/dev/null || true
echo -e "${GREEN}✓${NC} 完了"
echo ""

# 署名実行（/tmp 上で）
echo -e "${YELLOW}[2/4] プラグインに署名中...${NC}"
if [ -f "$ENTITLEMENTS_PATH" ]; then
    codesign --deep --force --verbose \
        --sign "$CODESIGN_IDENTITY" \
        --options runtime \
        --entitlements "$ENTITLEMENTS_PATH" \
        --timestamp \
        "$PLUGIN_TMP"
else
    echo -e "${YELLOW}[WARN]${NC} entitlements が見つかりません。Hardened Runtimeのみで署名します。"
    codesign --deep --force --verbose \
        --sign "$CODESIGN_IDENTITY" \
        --options runtime \
        --timestamp \
        "$PLUGIN_TMP"
fi
echo -e "${GREEN}✓${NC} 署名完了"
echo ""

# 署名の検証（/tmp 上で）
echo -e "${YELLOW}[3/4] 署名の検証中...${NC}"
codesign --verify --deep --strict --verbose=2 "$PLUGIN_TMP"
spctl --assess --type execute --verbose "$PLUGIN_TMP" || true
echo -e "${GREEN}✓${NC} 検証完了"
echo ""

# 署名済みバンドルをプロジェクトにコピー
echo -e "${YELLOW}[4/4] 署名済みプラグインをコピー中...${NC}"
rm -rf "$PLUGIN_SIGNED_PATH"
mkdir -p "$(dirname "$PLUGIN_SIGNED_PATH")"
ditto "$PLUGIN_TMP" "$PLUGIN_SIGNED_PATH"
rm -rf "$WORK_DIR"
echo -e "${GREEN}✓${NC} 完了"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}署名完了！${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "署名情報:"
codesign -dv "$PLUGIN_SIGNED_PATH" 2>&1 | grep -E "(Authority|Identifier|TeamIdentifier)"

echo ""
echo -e "${YELLOW}署名済みプラグイン:${NC}"
echo "  $PLUGIN_SIGNED_PATH"
echo ""
echo "次のステップ:"
echo "  ./Mac/notarize_plugin.sh $BUILD_CONFIG を実行して Apple 公証を取得"
echo ""
