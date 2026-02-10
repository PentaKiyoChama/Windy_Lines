# Windy Lines - ファイル構成 / File Structure

このドキュメントは、リポジトリ内のファイル構成を説明します。

This document describes the file organization in the repository.

## ディレクトリ構造 / Directory Structure

### ルートディレクトリ / Root Directory
プラグインの主要ソースファイルが配置されています。
- `SDK_ProcAmp.*` - プラグインのメインソースコード（C++、HLSL、CUDA、OpenCLなど）
- `SDK_ProcAmp_*.h` / `SDK_ProcAmp_*.cpp` - ヘッダーファイルと実装ファイル
- `RCa18532`, `RCb18532` - リソースファイル
- `readme.txt` - DirectX関連のビルド手順
- `.gitattributes`, `.gitignore` - Git設定ファイル

### `docs/` ディレクトリ
ドキュメントファイルが整理されています（全12ファイル）。
- `GIT_SYNC_FIX.md` - Git同期問題の修正ドキュメント
- `SDK_ProcAmp_DevGuide.md` - 開発者ガイド
- `ANTIALIASING_*.md` - アンチエイリアシング関連のドキュメント
- `*_PLAN.md` - 最適化計画書
- `*_MEMO.md` - 実装メモ
- その他のドキュメント

### `scripts/` ディレクトリ
開発用のスクリプトとツールが整理されています（全10ファイル）。
- `*.py` - Pythonスクリプト（最適化適用、マクロ生成など）
- `*.sh` - シェルスクリプト（マクロ生成、検証など）
- `test_bezier` - テスト用実行ファイル

### `presets/` ディレクトリ
エフェクトプリセット関連のファイルが整理されています（全2ファイル）。
- `presets.tsv` - エフェクトプリセット定義データ（16種類のプリセット）
- `preset_converter.py` - TSVからC++コードを生成するスクリプト

### `LEGACY/` ディレクトリ
古いバックアップファイルや生成済みファイル、未使用のヘッダーファイルが保存されています（全4ファイル）。
- `SDK_ProcAmp.cu.backup_20260205_182402` - CUDAファイルのバックアップ
- `SDK_ProcAmp_Strings.h` - 未使用のプラグイン名定義ヘッダー
- `SDK_ProcAmp_Version.h` - 未使用のバージョン管理ヘッダー
- `presets_generated.cpp` - 生成済みのプリセットコード

### `Mac/` ディレクトリ
macOS用のビルド設定とXcodeプロジェクトファイル。

### `Win/` ディレクトリ
Windows用のビルド設定とVisual Studioプロジェクトファイル。

### `.vscode/` ディレクトリ
Visual Studio Codeのワークスペース設定。

## ファイル整理の利点 / Benefits of Organization

1. **見通しの改善** - ルートディレクトリがすっきりし、プロジェクトの構造が理解しやすくなりました
2. **メンテナンス性の向上** - 関連ファイルがグループ化され、更新や管理が容易になりました
3. **ドキュメントの発見** - すべてのドキュメントが `docs/` フォルダに集約されました
4. **スクリプトの管理** - 開発用ツールが `scripts/` フォルダに整理されました
5. **履歴の保存** - 古いファイルやバックアップが `LEGACY/` フォルダに保存されました

## クイックリンク / Quick Links

- [開発者ガイド](docs/SDK_ProcAmp_DevGuide.md)
- [Git同期修正](docs/GIT_SYNC_FIX.md)
- [アンチエイリアシング解説](docs/ANTIALIASING_README.md)
- [パフォーマンス最適化計画](docs/PERFORMANCE_OPTIMIZATION_PLAN.md)
