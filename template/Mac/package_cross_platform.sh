#!/bin/bash
set -e

###############################################################################
#  __TPL_MATCH_NAME__ — 配布パッケージスクリプト
#  macOS単体 または macOS+Windows の配布 ZIP を作成
#  バージョンは __TPL_MATCH_NAME___Version.h から自動取得
###############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLUGIN_NAME="__TPL_MATCH_NAME__"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}${PLUGIN_NAME} - 配布パッケージ作成${NC}"
echo -e "${BLUE}========================================${NC}\n"

# ===== バージョン取得 =====
VERSION_H="$PROJECT_DIR/${PLUGIN_NAME}_Version.h"
if [ ! -f "$VERSION_H" ]; then
    echo -e "${RED}✗${NC} バージョンファイルが見つかりません: $VERSION_H"
    exit 1
fi

V_MAJOR=$(grep 'PUBLIC_VERSION_MAJOR' "$VERSION_H" | head -1 | awk '{print $3}')
V_MINOR=$(grep 'PUBLIC_VERSION_MINOR' "$VERSION_H" | head -1 | awk '{print $3}')
V_PATCH=$(grep 'PUBLIC_VERSION_PATCH' "$VERSION_H" | head -1 | awk '{print $3}')
VERSION="v${V_MAJOR}.${V_MINOR}.${V_PATCH}"

echo -e "${YELLOW}公開バージョン:${NC} $VERSION"

# ===== 引数解析 =====
BUILD_CONFIG="${1:-Release}"
WIN_INPUT_PATH="${2:-}"

if [ "$BUILD_CONFIG" != "Debug" ] && [ "$BUILD_CONFIG" != "Release" ]; then
    echo -e "${RED}✗${NC} 無効な構成です: $BUILD_CONFIG"
    echo ""
    echo "使い方:"
    echo "  ./Mac/package_cross_platform.sh [Debug|Release]                          # macOS のみ"
    echo "  ./Mac/package_cross_platform.sh [Debug|Release] /path/to/win_input       # macOS + Windows"
    exit 1
fi

# ===== macOS プラグイン確認 =====
MAC_PLUGIN_PATH="$SCRIPT_DIR/build/${BUILD_CONFIG}_signed/${PLUGIN_NAME}.plugin"
if [ ! -d "$MAC_PLUGIN_PATH" ]; then
    echo -e "${RED}✗${NC} macOS プラグインが見つかりません: $MAC_PLUGIN_PATH"
    echo "先に ./Mac/notarize_plugin.sh $BUILD_CONFIG を実行してください。"
    exit 1
fi

# ===== モード判定 =====
# __TPL_DIST_NAME_JP__ はテンプレート初期化時に日本語製品名に置換される
DIST_NAME="__TPL_DIST_NAME_JP__"
PACKAGE_NAME="${DIST_NAME}_${VERSION}"

if [ -z "$WIN_INPUT_PATH" ]; then
    MODE="mac_only"
    echo -e "${YELLOW}モード:${NC} macOS のみ"
else
    if [ ! -e "$WIN_INPUT_PATH" ]; then
        echo -e "${RED}✗${NC} Windows 側入力が見つかりません: $WIN_INPUT_PATH"
        exit 1
    fi
    MODE="cross_platform"
    echo -e "${YELLOW}モード:${NC} macOS + Windows"

    WIN_MODE=""
    if [ -d "$WIN_INPUT_PATH" ]; then
        WIN_MODE="dir"
    elif [ -f "$WIN_INPUT_PATH" ]; then
        WIN_MODE="file"
        if [[ "$WIN_INPUT_PATH" != *.aex ]]; then
            echo -e "${YELLOW}[WARN]${NC} 指定ファイルの拡張子が .aex ではありません: $WIN_INPUT_PATH"
        fi
    else
        echo -e "${RED}✗${NC} Windows 側入力がファイル/フォルダとして解釈できません: $WIN_INPUT_PATH"
        exit 1
    fi
fi

echo -e "${YELLOW}構成:${NC} $BUILD_CONFIG"
echo -e "${YELLOW}macOS:${NC} $MAC_PLUGIN_PATH"
if [ "$MODE" = "cross_platform" ]; then
    echo -e "${YELLOW}Windows入力:${NC} $WIN_INPUT_PATH"
fi
echo ""

OUT_DIR="$PROJECT_DIR/dist"
OUT_ZIP="$OUT_DIR/${PACKAGE_NAME}.zip"
WORK_DIR="/tmp/ost_package_$$"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/${PACKAGE_NAME}/mac"
mkdir -p "$OUT_DIR"

# ===== [1] macOS プラグインをコピー =====
TOTAL_STEPS=3
if [ "$MODE" = "cross_platform" ]; then
    TOTAL_STEPS=4
    mkdir -p "$WORK_DIR/${PACKAGE_NAME}/win"
fi

echo -e "${YELLOW}[1/${TOTAL_STEPS}] macOSプラグインをコピー中...${NC}"
COPYFILE_DISABLE=1 ditto --norsrc --noextattr --noacl "$MAC_PLUGIN_PATH" "$WORK_DIR/${PACKAGE_NAME}/mac/${PLUGIN_NAME}.plugin"
xattr -cr "$WORK_DIR/${PACKAGE_NAME}/mac/${PLUGIN_NAME}.plugin" 2>/dev/null || true
echo -e "${GREEN}✓${NC} 完了"

# ===== [2] Windows プラグインをコピー（クロスプラットフォーム時のみ） =====
STEP=2
if [ "$MODE" = "cross_platform" ]; then
    echo -e "${YELLOW}[${STEP}/${TOTAL_STEPS}] Windowsプラグインをコピー中...${NC}"
    if [ "$WIN_MODE" = "dir" ]; then
        cp -R "$WIN_INPUT_PATH"/. "$WORK_DIR/${PACKAGE_NAME}/win/"
        if ! find "$WORK_DIR/${PACKAGE_NAME}/win" -type f -name "*.aex" | grep -q .; then
            echo -e "${YELLOW}[WARN]${NC} Windows フォルダ内に .aex が見つかりませんでした。"
        fi
    else
        cp "$WIN_INPUT_PATH" "$WORK_DIR/${PACKAGE_NAME}/win/${PLUGIN_NAME}.aex"
    fi
    echo -e "${GREEN}✓${NC} 完了"
    STEP=3
fi

# ===== ドキュメントをコピー =====
echo -e "${YELLOW}[${STEP}/${TOTAL_STEPS}] ドキュメントをコピー中...${NC}"
DOCS_DIR="$PROJECT_DIR/docs"
for doc in "インストールガイド.txt" "README_プラグイン説明.txt"; do
    if [ -f "$DOCS_DIR/$doc" ]; then
        cp "$DOCS_DIR/$doc" "$WORK_DIR/${PACKAGE_NAME}/"
    fi
done
# fallback: ルートの INSTALL.txt / README.txt があればコピー
if [ ! -f "$WORK_DIR/${PACKAGE_NAME}/インストールガイド.txt" ]; then
    [ -f "$PROJECT_DIR/INSTALL.txt" ] && cp "$PROJECT_DIR/INSTALL.txt" "$WORK_DIR/${PACKAGE_NAME}/"
fi
if [ ! -f "$WORK_DIR/${PACKAGE_NAME}/README_プラグイン説明.txt" ]; then
    [ -f "$PROJECT_DIR/README.txt" ] && cp "$PROJECT_DIR/README.txt" "$WORK_DIR/${PACKAGE_NAME}/"
fi
echo -e "${GREEN}✓${NC} 完了"
STEP=$((STEP + 1))

# ===== ZIP 作成 =====
echo -e "${YELLOW}[${STEP}/${TOTAL_STEPS}] ZIPを作成中...${NC}"
rm -f "$OUT_ZIP"
(
    cd "$WORK_DIR"
    COPYFILE_DISABLE=1 zip -r "$OUT_ZIP" "$PACKAGE_NAME" -x "*/.DS_Store"
)
echo -e "${GREEN}✓${NC} 完了"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配布パッケージ作成完了${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "バージョン: ${VERSION}"
echo -e "出力ファイル:"
echo -e "  $OUT_ZIP"
echo ""
echo "サイズ: $(du -h "$OUT_ZIP" | cut -f1)"
echo ""
echo "内容:"
(cd "$WORK_DIR" && find "$PACKAGE_NAME" -type f | sort | while IFS= read -r f; do echo "  $f"; done)

rm -rf "$WORK_DIR"
