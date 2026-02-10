# プリセット管理 / Preset Management

このディレクトリには、エフェクトプリセットとカラープリセットの定義と変換ツールが含まれています。

This directory contains effect preset and color preset definitions and conversion tools.

## ファイル / Files

### エフェクトプリセット / Effect Presets

#### `presets.tsv`
エフェクトプリセット定義ファイル（TSV形式）

Effect preset definitions in TSV format containing:
- 16種類のプリセット（そよ風、雨、吹雪など）
- 各プリセットのパラメータ設定

#### `preset_converter.py`
TSVファイルからC++ヘッダーファイルを生成するスクリプト

Converts TSV to C++ header file (SDK_ProcAmp_Presets.h)

**使い方 / Usage:**
```bash
# デフォルト（同じディレクトリのpresets.tsvを使用）
python preset_converter.py

# または特定のTSVファイルを指定
python preset_converter.py path/to/presets.tsv
```

### カラープリセット / Color Presets

#### `color_presets.tsv`
カラープリセット定義ファイル（TSV形式）

Color preset definitions in TSV format containing:
- 16種類のカラーパレット（レインボー、森、サイバーなど）
- 各プリセットに8色のカラー定義

#### `color_preset_converter.py`
カラープリセットTSVからC++ヘッダーファイルを生成するスクリプト

Converts color preset TSV to C++ header file (SDK_ProcAmp_ColorPresets.h)

**使い方 / Usage:**
```bash
# デフォルト（同じディレクトリのcolor_presets.tsvを使用）
python color_preset_converter.py

# または特定のTSVファイルを指定
python color_preset_converter.py path/to/color_presets.tsv
```

## ワークフロー / Workflow

### エフェクトプリセット / Effect Presets
1. `presets.tsv` を編集してプリセットを追加・変更
2. `python preset_converter.py` を実行
3. プロジェクトルートに `SDK_ProcAmp_Presets.h` が生成される
4. プラグインをビルドして新しいプリセットを使用

### カラープリセット / Color Presets
1. `color_presets.tsv` を編集してカラーパレットを追加・変更
2. `python color_preset_converter.py` を実行
3. プロジェクトルートに `SDK_ProcAmp_ColorPresets.h` が生成される
4. プラグインをビルドして新しいカラープリセットを使用
