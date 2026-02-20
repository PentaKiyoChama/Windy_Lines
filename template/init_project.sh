#!/bin/bash
set -e

###############################################################################
#  Premiere Pro GPU Plugin Template — Project Initializer
#  
#  使い方:
#    ./init_project.sh
#
#  対話形式でプロジェクト名・表示名・カテゴリ等を入力し、
#  テンプレートファイル内のプレースホルダーを一括置換して
#  新しいプロジェクトフォルダを生成します。
###############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo -e "${BLUE}=== Premiere Pro GPU Plugin — プロジェクト初期化 ===${NC}"
echo ""

# ---- 1. 基本情報の入力 ----
read -p "プロジェクトプレフィックス (例: OST): " PROJECT_PREFIX
read -p "プラグインID名 (例: MyEffect, PascalCase): " PLUGIN_ID
read -p "日本語エフェクト名 (例: きらめくパーティクル): " EFFECT_NAME_JP
read -p "日本語カテゴリ名 (例: おしゃれテロップ): " CATEGORY_NAME_JP
read -p "英語Match Name (例: ${PROJECT_PREFIX}_${PLUGIN_ID}): " MATCH_NAME
MATCH_NAME=${MATCH_NAME:-"${PROJECT_PREFIX}_${PLUGIN_ID}"}
read -p "著作者名 (例: Kiyoto Nakamura): " AUTHOR_NAME
read -p "出力先ディレクトリ (例: /Users/you/Desktop/${MATCH_NAME}): " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-"$(dirname "$SCRIPT_DIR")/${MATCH_NAME}"}

# ベンダー名（ライセンスキャッシュ等で使用）
read -p "ベンダー名 (例: OshareTelop): " VENDOR_NAME
VENDOR_NAME=${VENDOR_NAME:-"$PROJECT_PREFIX"}

# 大文字バリエーション
UPPER_PREFIX=$(echo "$PROJECT_PREFIX" | tr '[:lower:]' '[:upper:]')
UPPER_PLUGIN=$(echo "$PLUGIN_ID" | sed 's/\([A-Z]\)/_\1/g' | tr '[:lower:]' '[:upper:]' | sed 's/^_//')

echo ""
echo -e "${YELLOW}--- 設定確認 ---${NC}"
echo "  プレフィックス: ${PROJECT_PREFIX}"
echo "  プラグインID:   ${PLUGIN_ID}"
echo "  Match Name:     ${MATCH_NAME}"
echo "  日本語名:       ${EFFECT_NAME_JP}"
echo "  カテゴリ:       ${CATEGORY_NAME_JP}"
echo "  著作者:         ${AUTHOR_NAME}"
echo "  ベンダー名:     ${VENDOR_NAME}"
echo "  出力先:         ${OUTPUT_DIR}"
echo ""
read -p "この設定で生成しますか？ (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo -e "${RED}キャンセルしました${NC}"
    exit 0
fi

# ---- 2. テンプレートのコピー ----
echo -e "${BLUE}[1/4] テンプレートをコピー中...${NC}"
mkdir -p "$OUTPUT_DIR"

# テンプレートファイルをコピー（init_project.sh自体と.gitは除外）
rsync -a --exclude='init_project.sh' --exclude='.git' --exclude='__pycache__' "$SCRIPT_DIR/" "$OUTPUT_DIR/"

# ---- 3. ファイル名のリネーム ----
echo -e "${BLUE}[2/4] ファイル名を置換中...${NC}"

# TEMPLATE_Plugin → 実際の名前にリネーム
find "$OUTPUT_DIR" -depth -name "*TEMPLATE_Plugin*" | while read f; do
    new=$(echo "$f" | sed "s/TEMPLATE_Plugin/${MATCH_NAME}/g")
    mv "$f" "$new"
done

# ---- 4. ファイル内容のプレースホルダー置換 ----
echo -e "${BLUE}[3/4] プレースホルダーを置換中...${NC}"

CURRENT_YEAR=$(date +%Y)

# テキストファイルのみ処理
find "$OUTPUT_DIR" -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cl" -o -name "*.metal" \
    -o -name "*.r" -o -name "*.rc" -o -name "*.rcp" -o -name "*.sh" -o -name "*.py" \
    -o -name "*.md" -o -name "*.txt" -o -name "*.plist" -o -name "*.pch" \
    -o -name "*.sln" -o -name "*.vcxproj" -o -name "*.vcxproj.filters" -o -name "*.i" \
    -o -name "*.json" -o -name "*.gitignore" -o -name "*.gitattributes" -o -name "*.tsv" \
    -o -name "*.entitlements" \) | while read f; do
    
    sed -i '' \
        -e "s/__TPL_PREFIX__/${PROJECT_PREFIX}/g" \
        -e "s/__TPL_PLUGIN_ID__/${PLUGIN_ID}/g" \
        -e "s/__TPL_MATCH_NAME__/${MATCH_NAME}/g" \
        -e "s/__TPL_UPPER_PREFIX__/${UPPER_PREFIX}/g" \
        -e "s/__TPL_UPPER_PLUGIN__/${UPPER_PLUGIN}/g" \
        -e "s/__TPL_EFFECT_NAME_JP__/${EFFECT_NAME_JP}/g" \
        -e "s/__TPL_CATEGORY_JP__/${CATEGORY_NAME_JP}/g" \
        -e "s/__TPL_AUTHOR__/${AUTHOR_NAME}/g" \
        -e "s/__TPL_VENDOR__/${VENDOR_NAME}/g" \
        -e "s/__TPL_YEAR__/${CURRENT_YEAR}/g" \
        "$f"
done

# ---- 5. 新しいGUIDを生成（vcxproj用） ----
echo -e "${BLUE}[4/4] プロジェクトGUIDを生成中...${NC}"
NEW_GUID=$(python3 -c "import uuid; print(str(uuid.uuid4()).upper())")
find "$OUTPUT_DIR" -type f \( -name "*.sln" -o -name "*.vcxproj" \) | while read f; do
    sed -i '' "s/__TPL_PROJECT_GUID__/${NEW_GUID}/g" "$f"
done

echo ""
echo -e "${GREEN}✅ プロジェクト '${MATCH_NAME}' を生成しました！${NC}"
echo -e "   場所: ${OUTPUT_DIR}"
echo ""
echo -e "${YELLOW}次のステップ:${NC}"
echo "  1. Premiere Pro SDK を配置（README_TEMPLATE.md 参照）"
echo "  2. Mac: Xcode プロジェクトを作成"
echo "  3. Win: ${MATCH_NAME}.sln を Visual Studio で開く"
echo "  4. パラメータを ${MATCH_NAME}_ParamNames.h で定義"
echo "  5. カーネルを ${MATCH_NAME}.cu / .cl に実装"
echo ""
