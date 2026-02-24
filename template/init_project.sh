#!/bin/bash
set -e

###############################################################################
#  Premiere Pro GPU Plugin Template — Project Initializer
#  
#  使い方:
#    cd template && chmod +x init_project.sh && ./init_project.sh
#
#  対話形式でプラグインIDを入力し、テンプレートファイル内の
#  プレースホルダーを一括置換して新しいプロジェクトフォルダを生成します。
###############################################################################

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

sed_in_place() {
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

echo ""
echo -e "${BLUE}=== Premiere Pro GPU Plugin — プロジェクト初期化 ===${NC}"
echo ""

# ---- 1. まずプラグインIDだけ入力（他はデフォルトを提示） ----
DEFAULT_PROJECT_PREFIX="OST"
DEFAULT_CATEGORY_JP="おしゃれテロップ"
DEFAULT_CATEGORY_ASCII="Oshare Telop"
DEFAULT_AUTHOR="Kiyoto Nakamura"
DEFAULT_VENDOR="OshareTelop"

# プラグインIDは必須入力（空なら再入力）
PLUGIN_ID=""
while [[ -z "$PLUGIN_ID" ]]; do
    read -p "プラグインID名 (例: MyEffect, PascalCase): " PLUGIN_ID
done

# 日本語名も必須入力（デフォルト不可能なため）
EFFECT_NAME_JP=""
while [[ -z "$EFFECT_NAME_JP" ]]; do
    read -p "日本語エフェクト名 (例: きらめくパーティクル): " EFFECT_NAME_JP
done

DEFAULT_MATCH_NAME="${DEFAULT_PROJECT_PREFIX}_${PLUGIN_ID}"
DEFAULT_OUTPUT_DIR="${HOME}/Desktop/${PLUGIN_ID}"

echo ""
echo -e "${YELLOW}--- デフォルト設定 ---${NC}"
echo "  プレフィックス: ${DEFAULT_PROJECT_PREFIX}"
echo "  プラグインID:   ${PLUGIN_ID}"
echo "  Match Name:     ${DEFAULT_MATCH_NAME}"
echo "  日本語名:       ${EFFECT_NAME_JP}"
echo "  カテゴリ:       ${DEFAULT_CATEGORY_JP}"
echo "  ASCIIカテゴリ:  ${DEFAULT_CATEGORY_ASCII}"
echo "  著作者:         ${DEFAULT_AUTHOR}"
echo "  ベンダー名:     ${DEFAULT_VENDOR}"
echo "  出力先:         ${DEFAULT_OUTPUT_DIR}"
echo ""
read -p "この設定で生成しますか？ (y/n): " QUICK_CONFIRM

REQUIRE_FINAL_CONFIRM=false
if [[ "$QUICK_CONFIRM" == "y" || "$QUICK_CONFIRM" == "Y" ]]; then
    PROJECT_PREFIX="$DEFAULT_PROJECT_PREFIX"
    MATCH_NAME="$DEFAULT_MATCH_NAME"
    # EFFECT_NAME_JP は既に入力済み
    CATEGORY_NAME_JP="$DEFAULT_CATEGORY_JP"
    CATEGORY_NAME_ASCII="$DEFAULT_CATEGORY_ASCII"
    AUTHOR_NAME="$DEFAULT_AUTHOR"
    VENDOR_NAME="$DEFAULT_VENDOR"
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
else
    echo ""
    echo -e "${YELLOW}--- 詳細設定を入力 ---${NC}"
    read -p "プロジェクトプレフィックス (デフォルト: ${DEFAULT_PROJECT_PREFIX}): " PROJECT_PREFIX
    PROJECT_PREFIX=${PROJECT_PREFIX:-"${DEFAULT_PROJECT_PREFIX}"}

    read -p "日本語エフェクト名 [現在: ${EFFECT_NAME_JP}]: " EFFECT_NAME_JP_NEW
    EFFECT_NAME_JP=${EFFECT_NAME_JP_NEW:-"${EFFECT_NAME_JP}"}

    read -p "日本語カテゴリ名 (例: おしゃれテロップ) [デフォルト: ${DEFAULT_CATEGORY_JP}]: " CATEGORY_NAME_JP
    CATEGORY_NAME_JP=${CATEGORY_NAME_JP:-"${DEFAULT_CATEGORY_JP}"}

    read -p "ASCII カテゴリ名 (.r 用) [デフォルト: ${DEFAULT_CATEGORY_ASCII}]: " CATEGORY_NAME_ASCII
    CATEGORY_NAME_ASCII=${CATEGORY_NAME_ASCII:-"${DEFAULT_CATEGORY_ASCII}"}

    read -p "英語Match Name (例: ${PROJECT_PREFIX}_${PLUGIN_ID}) [デフォルト: ${PROJECT_PREFIX}_${PLUGIN_ID}]: " MATCH_NAME
    MATCH_NAME=${MATCH_NAME:-"${PROJECT_PREFIX}_${PLUGIN_ID}"}

    read -p "著作者名 (例: Kiyoto Nakamura) [デフォルト: ${DEFAULT_AUTHOR}]: " AUTHOR_NAME
    AUTHOR_NAME=${AUTHOR_NAME:-"${DEFAULT_AUTHOR}"}

    read -p "ベンダー名 (例: OshareTelop) [デフォルト: ${DEFAULT_VENDOR}]: " VENDOR_NAME
    VENDOR_NAME=${VENDOR_NAME:-"${DEFAULT_VENDOR}"}

    read -p "出力先ディレクトリ (例: ${HOME}/Desktop/${MATCH_NAME}) [デフォルト: ${HOME}/Desktop/${MATCH_NAME}]: " OUTPUT_DIR
    OUTPUT_DIR=${OUTPUT_DIR:-"${HOME}/Desktop/${MATCH_NAME}"}
    REQUIRE_FINAL_CONFIRM=true
fi

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
echo "  ASCIIカテゴリ:  ${CATEGORY_NAME_ASCII}"
echo "  著作者:         ${AUTHOR_NAME}"
echo "  ベンダー名:     ${VENDOR_NAME}"
echo "  出力先:         ${OUTPUT_DIR}"
if [[ "$REQUIRE_FINAL_CONFIRM" == "true" ]]; then
    echo ""
    read -p "この設定で生成しますか？ (y/n): " CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
        echo -e "${RED}キャンセルしました${NC}"
        exit 0
    fi
fi

# 出力先の事前バリデーション（書き込み不可なら再入力）
while true; do
    if mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
        break
    fi
    echo -e "${RED}出力先に書き込めません: ${OUTPUT_DIR}${NC}"
    read -p "別の出力先を入力してください (空 Enter でデフォルトに戻す): " OUTPUT_DIR
    OUTPUT_DIR=${OUTPUT_DIR:-"${HOME}/Desktop/${MATCH_NAME}"}
done

# ---- 2. テンプレートのコピー ----
echo -e "${BLUE}[1/8] テンプレートをコピー中...${NC}"
mkdir -p "$OUTPUT_DIR"

# テンプレートファイルをコピー（init_project.sh自体と.gitは除外）
if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='init_project.sh' --exclude='.git' --exclude='__pycache__' "$SCRIPT_DIR/" "$OUTPUT_DIR/"
else
    cp -a "$SCRIPT_DIR/." "$OUTPUT_DIR/"
    rm -f "$OUTPUT_DIR/init_project.sh"
    rm -rf "$OUTPUT_DIR/.git" "$OUTPUT_DIR/__pycache__"
fi

# ---- 3. ファイル名のリネーム（深い順に処理） ----
echo -e "${BLUE}[2/8] ファイル名を置換中...${NC}"

# TEMPLATE_Plugin → 実際の名前にリネーム（ディレクトリ・ファイル両方）
find "$OUTPUT_DIR" -depth -name "*TEMPLATE_Plugin*" | while read f; do
    new=$(echo "$f" | sed "s/TEMPLATE_Plugin/${MATCH_NAME}/g")
    mv "$f" "$new"
done

# ---- 4. ファイル内容のプレースホルダー置換 ----
echo -e "${BLUE}[3/8] プレースホルダーを置換中...${NC}"

CURRENT_YEAR=$(date +%Y)

# MATCH_NAME からアンダースコアを除去したバージョン（bundle identifier用）
MATCH_NAME_NOUNDERSCORE=$(echo "$MATCH_NAME" | tr -d '_')
VENDOR_BUNDLE_SAFE=$(echo "$VENDOR_NAME" | tr -d ' ')
BUNDLE_ID="com.${VENDOR_BUNDLE_SAFE}.${MATCH_NAME_NOUNDERSCORE}"

# テキストファイルのみ処理（pbxproj, xcworkspacedata, strings も含む）
find "$OUTPUT_DIR" -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cl" -o -name "*.metal" \
    -o -name "*.r" -o -name "*.rc" -o -name "*.rcp" -o -name "*.sh" -o -name "*.py" \
    -o -name "*.md" -o -name "*.txt" -o -name "*.plist" -o -name "*.pch" \
    -o -name "*.sln" -o -name "*.vcxproj" -o -name "*.vcxproj.filters" -o -name "*.i" \
    -o -name "*.json" -o -name "*.gitignore" -o -name "*.gitattributes" -o -name "*.tsv" \
    -o -name "*.entitlements" \
    -o -name "*.pbxproj" -o -name "*.xcworkspacedata" -o -name "*.strings" \) | while read f; do
    
    sed_in_place \
        -e "s/__TPL_PREFIX__/${PROJECT_PREFIX}/g" \
        -e "s/__TPL_PLUGIN_ID__/${PLUGIN_ID}/g" \
        -e "s/__TPL_MATCH_NAME_NOUNDERSCORE__/${MATCH_NAME_NOUNDERSCORE}/g" \
        -e "s/__TPL_BUNDLE_ID__/${BUNDLE_ID}/g" \
        -e "s/__TPL_MATCH_NAME__/${MATCH_NAME}/g" \
        -e "s/__TPL_UPPER_PREFIX__/${UPPER_PREFIX}/g" \
        -e "s/__TPL_UPPER_PLUGIN__/${UPPER_PLUGIN}/g" \
        -e "s/__TPL_EFFECT_NAME_JP__/${EFFECT_NAME_JP}/g" \
        -e "s/__TPL_CATEGORY_JP__/${CATEGORY_NAME_JP}/g" \
        -e "s/__TPL_CATEGORY_ASCII__/${CATEGORY_NAME_ASCII}/g" \
        -e "s/__TPL_AUTHOR__/${AUTHOR_NAME}/g" \
        -e "s/__TPL_VENDOR__/${VENDOR_NAME}/g" \
        -e "s/__TPL_YEAR__/${CURRENT_YEAR}/g" \
        "$f"
done

# ---- 5. pbxproj 内の残存 TEMPLATE_Plugin 文字列を置換 ----
echo -e "${BLUE}[4/8] Xcode プロジェクト内のリファレンスを置換中...${NC}"
PBXPROJ="$OUTPUT_DIR/Mac/${MATCH_NAME}.xcodeproj/project.pbxproj"
if [[ -f "$PBXPROJ" ]]; then
    sed_in_place "s/TEMPLATE_Plugin/${MATCH_NAME}/g" "$PBXPROJ"
fi

# ---- 6. 新しいGUIDを生成（vcxproj用） ----
echo -e "${BLUE}[5/8] プロジェクトGUIDを生成中...${NC}"
NEW_GUID=""
if command -v python3 >/dev/null 2>&1; then
    NEW_GUID=$(python3 -c "import uuid; print(str(uuid.uuid4()).upper())" 2>/dev/null || true)
fi
if [[ -z "$NEW_GUID" ]] && command -v python >/dev/null 2>&1; then
    NEW_GUID=$(python -c "import uuid; print(str(uuid.uuid4()).upper())" 2>/dev/null || true)
fi
if [[ -z "$NEW_GUID" ]] && command -v powershell.exe >/dev/null 2>&1; then
    NEW_GUID=$(powershell.exe -NoProfile -Command "[guid]::NewGuid().ToString().ToUpper()" 2>/dev/null | tr -d '\r' || true)
fi
if [[ -z "$NEW_GUID" ]] && [[ -f /proc/sys/kernel/random/uuid ]]; then
    NEW_GUID=$(cat /proc/sys/kernel/random/uuid | tr '[:lower:]' '[:upper:]')
fi
if [[ -z "$NEW_GUID" ]]; then
    echo -e "${RED}GUID生成に必要なランタイムが見つかりませんでした${NC}"
    exit 1
fi
find "$OUTPUT_DIR" -type f \( -name "*.sln" -o -name "*.vcxproj" \) | while read f; do
    sed_in_place "s/__TPL_PROJECT_GUID__/${NEW_GUID}/g" "$f"
done

# ---- 7. SDK パスの自動検出＆設定 ----
echo -e "${BLUE}[6/8] SDK パスを設定中...${NC}"

# Premiere Pro SDK を自動検出（Desktop 上を探索）
PREMIERE_SDK_PATH=""
for candidate in "$HOME/Desktop/Premiere Pro"*"SDK" "$HOME/Desktop/Adobe Premiere"*"SDK" "$HOME/Documents/Premiere"*"SDK"; do
    if [[ -d "$candidate" ]]; then
        PREMIERE_SDK_PATH="$candidate"
        break
    fi
done

if [[ -z "$PREMIERE_SDK_PATH" ]]; then
    echo -e "${YELLOW}Premiere Pro SDK が自動検出されませんでした${NC}"
    read -p "Premiere Pro SDK のパス (空でスキップ): " PREMIERE_SDK_PATH
fi

# After Effects SDK を自動検出
AE_SDK_PATH=""
for candidate in "$HOME/Desktop/AfterEffects"*"SDK" "$HOME/Desktop/After Effects"*"SDK" "$HOME/Desktop/AfterEffectsSDK"*; do
    if [[ -d "$candidate" ]]; then
        AE_SDK_PATH="$candidate"
        break
    fi
done

if [[ -z "$AE_SDK_PATH" ]]; then
    echo -e "${YELLOW}After Effects SDK が自動検出されませんでした${NC}"
    read -p "After Effects SDK のパス (空でスキップ): " AE_SDK_PATH
fi

# 検出結果を表示
if [[ -n "$PREMIERE_SDK_PATH" ]]; then
    echo -e "${GREEN}✓${NC} Premiere SDK: ${PREMIERE_SDK_PATH}"
fi
if [[ -n "$AE_SDK_PATH" ]]; then
    echo -e "${GREEN}✓${NC} AE SDK:       ${AE_SDK_PATH}"
fi

# pbxproj 内の SDK パスプレースホルダーを置換
PBXPROJ="$OUTPUT_DIR/Mac/${MATCH_NAME}.xcodeproj/project.pbxproj"
if [[ -f "$PBXPROJ" ]]; then
    if [[ -n "$PREMIERE_SDK_PATH" ]]; then
        # sed でスラッシュを含むパスを扱うため区切り文字を | に変更
        sed_in_place "s|__TPL_PREMIERE_SDK_PATH__|${PREMIERE_SDK_PATH}|g" "$PBXPROJ"
    fi
    if [[ -n "$AE_SDK_PATH" ]]; then
        sed_in_place "s|__TPL_AE_SDK_PATH__|${AE_SDK_PATH}|g" "$PBXPROJ"
    fi
fi

# ---- 8. git init ----
echo -e "${BLUE}[7/8] Git リポジトリを初期化中...${NC}"

# 派生後ドキュメント配置（エージェント実装ガイドを docs/ に複製）
if [[ -f "$OUTPUT_DIR/AGENT_IMPLEMENTATION_GUIDE.md" ]]; then
    mkdir -p "$OUTPUT_DIR/docs"
    cp "$OUTPUT_DIR/AGENT_IMPLEMENTATION_GUIDE.md" "$OUTPUT_DIR/docs/AGENT_IMPLEMENTATION_GUIDE.md"
fi

(
    cd "$OUTPUT_DIR"
    git init -q
    git add -A
    git commit -q -m "Initial commit from template (${MATCH_NAME})"
)
echo -e "${GREEN}✓${NC} git init + initial commit 完了"

# ---- 完了 ----
echo -e "${BLUE}[8/8] 完了処理...${NC}"

echo ""
echo -e "${GREEN}✅ プロジェクト '${MATCH_NAME}' を生成しました！${NC}"
echo -e "   場所: ${OUTPUT_DIR}"
echo -e "   git: Initial commit 済み"
echo ""

# SDK パス未設定の警告
if [[ -z "$PREMIERE_SDK_PATH" || -z "$AE_SDK_PATH" ]]; then
    echo -e "${YELLOW}⚠ SDK パスが未設定の項目があります:${NC}"
    if [[ -z "$PREMIERE_SDK_PATH" ]]; then
        echo "  - PREMIERE_SDK_BASE_PATH → Xcode Build Settings で手動設定してください"
    fi
    if [[ -z "$AE_SDK_PATH" ]]; then
        echo "  - AE_SDK_BASE_PATH → Xcode Build Settings で手動設定してください"
    fi
    echo ""
fi

echo -e "${YELLOW}次のステップ:${NC}"
echo "  1. cd ${OUTPUT_DIR}"
echo "  2. README_TEMPLATE.md と docs/TEMPLATE_DEV_GUIDE.md を読む"
echo "  3. docs/AGENT_IMPLEMENTATION_GUIDE.md の必須実装を先に確認"
echo "  4. _ParamNames.h でパラメータ名を定義"
echo "  5. .h でパラメータ enum / 定数 / ProcAmpParams を定義"
echo "  6. .cu でCUDAカーネルを実装 → .cl に同期 → _CPU.cpp にフォールバック"
echo "  7. ウォーターマーク生成: python3 tools/generate_watermark_mask.py --font ... --out ${MATCH_NAME}_WatermarkMask.h"
echo "  8. Mac: cd Mac && open ${MATCH_NAME}.xcodeproj → ビルド → ./install_plugin.sh"
echo "  9. Win: ${MATCH_NAME}.sln → ビルド → プラグイン配置"
echo " 10. ライセンステスト: python3 tools/activate_license_cache.py"
echo ""
