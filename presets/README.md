# プリセット管理 / Preset Management

このディレクトリには、エフェクトプリセットの定義と変換ツールが含まれています。

This directory contains effect preset definitions and conversion tools.

## ファイル / Files

### `presets.tsv`
エフェクトプリセット定義ファイル（TSV形式）

Effect preset definitions in TSV format containing:
- 16種類のプリセット（そよ風、雨、吹雪など）
- 各プリセットのパラメータ設定

**編集方法 / How to edit:**
- Excelやテキストエディタで編集可能
- タブ区切り形式を保持してください
- 編集後は `preset_converter.py` を実行して反映

### `preset_converter.py`
TSVファイルからC++ヘッダーファイルを生成するスクリプト

Converts TSV to C++ header file (SDK_ProcAmp_Presets.h)

**使い方 / Usage:**
```bash
# デフォルト（同じディレクトリのpresets.tsvを使用）
python preset_converter.py

# または特定のTSVファイルを指定
python preset_converter.py path/to/presets.tsv
```

## ワークフロー / Workflow

1. `presets.tsv` を編集してプリセットを追加・変更
2. `python preset_converter.py` を実行
3. プロジェクトルートに `SDK_ProcAmp_Presets.h` が生成される
4. プラグインをビルドして新しいプリセットを使用
