#!/bin/bash
#
# Color Preset Verification Tool
# 色プリセット検証ツール
#
# Usage: bash verify_color_preset.sh <preset1_id> <preset2_id>
# Example: bash verify_color_preset.sh 3 36  # Compare Forest(3) with test_3(36)
#

if [ $# -ne 2 ]; then
    echo "使用方法: bash verify_color_preset.sh <preset1_id> <preset2_id>"
    echo "例: bash verify_color_preset.sh 3 36  # 森(3)とtest_3(36)を比較"
    exit 1
fi

PRESET1_ID=$1
PRESET2_ID=$2

echo "=== 色プリセット検証ツール ==="
echo ""
echo "比較対象: プリセット #$PRESET1_ID vs #$PRESET2_ID"
echo ""

# Check if files exist
if [ ! -f "color_presets.tsv" ]; then
    echo "❌ エラー: color_presets.tsv が見つかりません"
    exit 1
fi

if [ ! -f "OST_WindyLines_ColorPresets.h" ]; then
    echo "❌ エラー: OST_WindyLines_ColorPresets.h が見つかりません"
    echo "   python color_preset_converter.py を実行してください"
    exit 1
fi

# Get preset names from TSV
PRESET1_LINE=$(grep "^$PRESET1_ID	" color_presets.tsv)
PRESET2_LINE=$(grep "^$PRESET2_ID	" color_presets.tsv)

if [ -z "$PRESET1_LINE" ]; then
    echo "❌ エラー: プリセット #$PRESET1_ID がTSVファイルに見つかりません"
    exit 1
fi

if [ -z "$PRESET2_LINE" ]; then
    echo "❌ エラー: プリセット #$PRESET2_ID がTSVファイルに見つかりません"
    exit 1
fi

PRESET1_NAME=$(echo "$PRESET1_LINE" | cut -f2)
PRESET2_NAME=$(echo "$PRESET2_LINE" | cut -f2)
PRESET1_EN=$(echo "$PRESET1_LINE" | cut -f3)
PRESET2_EN=$(echo "$PRESET2_LINE" | cut -f3)

echo "プリセット1: #$PRESET1_ID - $PRESET1_NAME ($PRESET1_EN)"
echo "プリセット2: #$PRESET2_ID - $PRESET2_NAME ($PRESET2_EN)"
echo ""

# Compare TSV colors
echo "1. TSVファイルの色データ:"
echo ""
PRESET1_TSV_COLORS=$(echo "$PRESET1_LINE" | cut -f4-11)
PRESET2_TSV_COLORS=$(echo "$PRESET2_LINE" | cut -f4-11)

echo "【$PRESET1_NAME】:"
echo "$PRESET1_TSV_COLORS"
echo ""
echo "【$PRESET2_NAME】:"
echo "$PRESET2_TSV_COLORS"
echo ""

# Compare generated header colors
echo "2. OST_WindyLines_ColorPresets.h の色データ:"
echo ""

# Find color arrays in header
PRESET1_H_ARRAY=$(awk "/const PresetColor k$PRESET1_EN\[8\]/{getline; print; getline; print}" OST_WindyLines_ColorPresets.h | tr -d '\t\n ')
PRESET2_H_ARRAY=$(awk "/const PresetColor k$PRESET2_EN\[8\]/{getline; print; getline; print}" OST_WindyLines_ColorPresets.h | tr -d '\t\n ')

echo "【k$PRESET1_EN】:"
if [ -z "$PRESET1_H_ARRAY" ]; then
    echo "  ❌ ヘッダーに見つかりません"
else
    echo "  $PRESET1_H_ARRAY"
fi
echo ""

echo "【k$PRESET2_EN】:"
if [ -z "$PRESET2_H_ARRAY" ]; then
    echo "  ❌ ヘッダーに見つかりません"
else
    echo "  $PRESET2_H_ARRAY"
fi
echo ""

# Compare results
echo "3. 比較結果:"
echo ""

TSV_MATCH=0
HEADER_MATCH=0

if [ "$PRESET1_TSV_COLORS" = "$PRESET2_TSV_COLORS" ]; then
    echo "✅ TSVファイル: 2つのプリセットの色は完全に一致しています"
    TSV_MATCH=1
else
    echo "❌ TSVファイル: 2つのプリセットの色が異なります"
fi

if [ "$PRESET1_H_ARRAY" = "$PRESET2_H_ARRAY" ] && [ -n "$PRESET1_H_ARRAY" ] && [ -n "$PRESET2_H_ARRAY" ]; then
    echo "✅ ヘッダーファイル: 2つのプリセットの色は完全に一致しています"
    HEADER_MATCH=1
else
    echo "❌ ヘッダーファイル: 2つのプリセットの色が異なります"
fi

echo ""

# Recommendations
echo "=== 次のステップ ==="
echo ""

if [ $TSV_MATCH -eq 1 ] && [ $HEADER_MATCH -eq 1 ]; then
    echo "✅ TSVとヘッダーの両方で色が一致しています。"
    echo ""
    echo "After Effectsで色が異なって見える場合:"
    echo ""
    echo "【必須手順】"
    echo "1. プラグインを再ビルド:"
    echo "   cd OST_WindyLines"
    echo "   make clean && make"
    echo "   または"
    echo "   MSBuild OST_WindyLines.sln /t:Clean,Build /p:Configuration=Debug"
    echo ""
    echo "2. 再ビルドしたプラグインをAfter Effectsプラグインフォルダにコピー"
    echo ""
    echo "3. After Effectsを完全に終了して再起動"
    echo ""
    echo "4. 新しいコンポジションでテスト"
    echo ""
    echo "【重要】"
    echo "- After Effectsはプラグインをキャッシュします"
    echo "- 再起動せずにプラグインを入れ替えても反映されません"
    echo "- 既存のコンポジションではなく、新しいコンポジションでテストしてください"
elif [ $TSV_MATCH -eq 1 ] && [ $HEADER_MATCH -eq 0 ]; then
    echo "⚠️ TSVは一致していますが、ヘッダーは異なります。"
    echo ""
    echo "以下を実行してヘッダーを再生成してください:"
    echo "   python color_preset_converter.py"
elif [ $TSV_MATCH -eq 0 ]; then
    echo "⚠️ TSVファイルで色が異なります。"
    echo ""
    echo "これは正常です（異なるプリセットとして設計されている場合）。"
    echo "色を一致させたい場合は、color_presets.tsv を編集してください。"
fi

echo ""
