#!/bin/bash
# UTF-8 文字列を Shift_JIS のCヘキサ形式に変換

convert_to_sjis_hex() {
    local utf8_str="$1"
    local macro_name="$2"
    local sjis_hex=$(echo -n "$utf8_str" | iconv -f UTF-8 -t SHIFT_JIS 2>/dev/null | xxd -p | sed 's/\(..\)/\\x\1/g')
    echo "// \"$utf8_str\""
    echo "#define $macro_name \"$sjis_hex\""
    echo ""
}

echo "// Auto-generated Shift_JIS macros for Mac"
echo "// Generated on $(date)"
echo ""

# 基本設定
convert_to_sjis_hex "ライン数" "SJIS_LINE_COUNT"
convert_to_sjis_hex "寿命(フレーム)" "SJIS_LIFETIME"
convert_to_sjis_hex "移動距離(px)" "SJIS_TRAVEL"
convert_to_sjis_hex "ランダムシード" "SJIS_SEED"

# カラー設定
convert_to_sjis_hex "カラーモード" "SJIS_COLOR_MODE"
convert_to_sjis_hex "単色" "SJIS_SINGLE_COLOR"
convert_to_sjis_hex "プリセット" "SJIS_PRESET"
convert_to_sjis_hex "カスタム" "SJIS_CUSTOM"
convert_to_sjis_hex "色" "SJIS_COLOR"
convert_to_sjis_hex "カスタム1" "SJIS_CUSTOM_1"
convert_to_sjis_hex "カスタム2" "SJIS_CUSTOM_2"
convert_to_sjis_hex "カスタム3" "SJIS_CUSTOM_3"
convert_to_sjis_hex "カスタム4" "SJIS_CUSTOM_4"
convert_to_sjis_hex "カスタム5" "SJIS_CUSTOM_5"
convert_to_sjis_hex "カスタム6" "SJIS_CUSTOM_6"
convert_to_sjis_hex "カスタム7" "SJIS_CUSTOM_7"
convert_to_sjis_hex "カスタム8" "SJIS_CUSTOM_8"

# 外観
convert_to_sjis_hex "太さ(px)" "SJIS_THICKNESS"
convert_to_sjis_hex "長さ(px)" "SJIS_LENGTH"
convert_to_sjis_hex "端のスタイル" "SJIS_LINE_CAP"
convert_to_sjis_hex "角度" "SJIS_ANGLE"
convert_to_sjis_hex "アンチエイリアス" "SJIS_AA"
convert_to_sjis_hex "テールフェード" "SJIS_TAIL_FADE"

# 位置
convert_to_sjis_hex "風の起点" "SJIS_ORIGIN_MODE"
convert_to_sjis_hex "発生間隔(フレーム)" "SJIS_INTERVAL"
convert_to_sjis_hex "スポーンエリアX(%)" "SJIS_SPAWN_SCALE_X"
convert_to_sjis_hex "スポーンエリアY(%)" "SJIS_SPAWN_SCALE_Y"
convert_to_sjis_hex "スポーンエリア回転" "SJIS_SPAWN_ROTATION"
convert_to_sjis_hex "スポーンエリア表示" "SJIS_SHOW_SPAWN"
convert_to_sjis_hex "スポーンエリア色" "SJIS_SPAWN_COLOR"
convert_to_sjis_hex "オフセットX(px)" "SJIS_OFFSET_X"
convert_to_sjis_hex "オフセットY(px)" "SJIS_OFFSET_Y"

# アニメーション
convert_to_sjis_hex "アニメパターン" "SJIS_ANIM_PATTERN"
convert_to_sjis_hex "センターギャップ" "SJIS_CENTER_GAP"
convert_to_sjis_hex "イージング" "SJIS_EASING"
convert_to_sjis_hex "開始時間(フレーム)" "SJIS_START_TIME"
convert_to_sjis_hex "継続時間(フレーム)" "SJIS_DURATION"

# シャドウ
convert_to_sjis_hex "シャドウ" "SJIS_SHADOW"
convert_to_sjis_hex "シャドウ有効" "SJIS_SHADOW_ENABLE"
convert_to_sjis_hex "シャドウ色" "SJIS_SHADOW_COLOR"
convert_to_sjis_hex "シャドウX(px)" "SJIS_SHADOW_OFFSET_X"
convert_to_sjis_hex "シャドウY(px)" "SJIS_SHADOW_OFFSET_Y"
convert_to_sjis_hex "シャドウ不透明度" "SJIS_SHADOW_OPACITY"

# 詳細
convert_to_sjis_hex "エレメント非表示" "SJIS_HIDE_ELEMENT"
convert_to_sjis_hex "ブレンドモード" "SJIS_BLEND_MODE"
convert_to_sjis_hex "深度強度" "SJIS_DEPTH_STRENGTH"
convert_to_sjis_hex "アルファ閾値" "SJIS_ALPHA_THRESH"
convert_to_sjis_hex "モーションブラー" "SJIS_MOTION_BLUR"
convert_to_sjis_hex "ブラー品質" "SJIS_BLUR_SAMPLES"
convert_to_sjis_hex "ブラー強度" "SJIS_BLUR_STRENGTH"

# エフェクトプリセット
convert_to_sjis_hex "エフェクトプリセット" "SJIS_EFFECT_PRESET"
convert_to_sjis_hex "デフォルト" "SJIS_DEFAULT"

# ポップアップメニュー項目
convert_to_sjis_hex "単色|プリセット|カスタム" "SJIS_COLOR_MODE_MENU"
convert_to_sjis_hex "丸|四角|なし" "SJIS_LINE_CAP_MENU"
