# WATERMARK マスク生成手順（無料フォント）

対象文字列: `おしゃれテロップ・FREE MODE`

## 推奨フォント
- `Noto Sans JP`（SIL Open Font License 1.1）

## 1) 依存インストール
```bash
python3 -m pip install pillow
```

`pip` コマンドが見つからない場合は、以下を使用してください：
```bash
/Users/kiyotonakamura/Desktop/Windy_Lines/.venv/bin/python -m pip install pillow
```

## 2) ヘッダ生成
```bash
cd /Users/kiyotonakamura/Desktop/Windy_Lines
/Users/kiyotonakamura/Desktop/Windy_Lines/.venv/bin/python generate_watermark_mask.py \
  --font "/System/Library/Fonts/Hiragino Sans GB.ttc" \
  --font-size 24 \
  --stroke-width 2 \
  --out "OST_WindyLines_WatermarkMask.h"
```

### レトロPC風（少し荒い質感）
```bash
cd /Users/kiyotonakamura/Desktop/Windy_Lines
/Users/kiyotonakamura/Desktop/Windy_Lines/.venv/bin/python generate_watermark_mask.py \
  --font "/System/Library/Fonts/Hiragino Sans GB.ttc" \
  --font-size 24 \
  --stroke-width 1 \
  --hard-edge \
  --out "OST_WindyLines_WatermarkMask.h"
```

### 文字を変更する場合（`--text`）
```bash
cd /Users/kiyotonakamura/Desktop/Windy_Lines
/Users/kiyotonakamura/Desktop/Windy_Lines/.venv/bin/python generate_watermark_mask.py \
  --text "おしゃれテロップ・風を感じるライン" \
  --font "/System/Library/Fonts/Hiragino Sans GB.ttc" \
  --font-size 12 \
  --stroke-width 1 \
  --hard-edge \
  --out "OST_WindyLines_WatermarkMask.h"
```

### フォントサイズとは別に拡大率を指定する（`--scale`）
```bash
cd /Users/kiyotonakamura/Desktop/Windy_Lines
/Users/kiyotonakamura/Desktop/Windy_Lines/.venv/bin/python generate_watermark_mask.py \
  --text "おしゃれテロップ・風を感じるライン" \
  --font "/System/Library/Fonts/Hiragino Sans GB.ttc" \
  --font-size 24 \
  --scale 1.0 \
  --stroke-width 0 \
  --hard-edge \
  --out "OST_WindyLines_WatermarkMask.h"
```

`--scale 1.0` が等倍です。`2.0` なら2倍、`0.8` なら80%になります。

※ 文字列に空白や記号がある場合は、必ず `"..."` で囲ってください。

※ `--font` は実在するフォントファイルの絶対パスを指定してください。  
`/path/to/...` のままでは `FileNotFoundError` になります。

フォント候補を検索するコマンド:
```bash
find /System/Library/Fonts /Library/Fonts ~/Library/Fonts -type f \( -name "*.ttf" -o -name "*.otf" -o -name "*.ttc" \) 2>/dev/null | grep -Ei "Noto|Hiragino|Gothic|MPLUS|Zen"
```

## 3) 再ビルド
- Xcodeで `OST_WindyLines` をビルド

## パラメータ調整の目安
- 太くする: `--stroke-width 3`
- 少し小さく: `--font-size 22`
- 少し大きく: `--font-size 28`
- フォントサイズ据え置きで拡大: `--scale 1.5`
- フォントサイズ据え置きで縮小: `--scale 0.8`
- 輪郭のぼかし量: `--outline-blur 0.0`（0でぼかし無し）
- 2値化（ジャギー感）: `--binary-threshold 128`
- まとめて粗くする: `--hard-edge`

## 補足
- 生成先は `OST_WindyLines_WatermarkMask.h`（既存を上書き）
- C++側はこのヘッダを参照しているため、再生成だけで見た目を更新できます。
- 文字を変えるとマスクサイズ（幅・高さ）も自動で再計算されます。
