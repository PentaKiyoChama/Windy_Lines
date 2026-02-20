# Premiere Pro GPU Plugin Template

Adobe Premiere Pro 向け GPU ビデオフィルタープラグインの開発テンプレート。  
**Windy_Lines プロジェクト**の実戦知見をもとに構築。

---

## クイックスタート

```bash
cd template
chmod +x init_project.sh
./init_project.sh
```

対話形式でプロジェクト名等を入力すると、全プレースホルダーが自動置換された新プロジェクトが生成されます。

---

## ディレクトリ構成

```
template/
├── init_project.sh              ← プロジェクト初期化スクリプト
├── .gitignore
│
├── TEMPLATE_Plugin.h            ← メインヘッダー（パラメータID、定数、共有データ）
├── TEMPLATE_Plugin_Version.h    ← バージョン管理（ここだけ変更すればOK）
├── TEMPLATE_Plugin_Common.h     ← CPU/GPU共有ユーティリティ（static inline）
├── TEMPLATE_Plugin_ParamNames.h ← パラメータ名（UTF-8→Shift-JIS自動変換）
│
├── TEMPLATE_Plugin_CPU.cpp      ← CPUレンダラー（フォールバック）+ エントリーポイント
├── TEMPLATE_Plugin_GPU.cpp      ← GPUホストコントローラー（CUDA/Metal/OpenCLディスパッチ）
│
├── TEMPLATE_Plugin.cu           ← CUDA カーネル（Windows, リファレンス実装）
├── TEMPLATE_Plugin.cl           ← OpenCL カーネル（Mac/Windows共通）
├── TEMPLATE_Plugin.metal        ← Metal エントリ（.cl を #include するだけ）
│
├── TEMPLATE_Plugin.r            ← PiPL リソース定義
├── TEMPLATE_Plugin.rc           ← Windows リソースファイル
│
├── color_presets.tsv            ← カラープリセットマスターデータ
├── presets.tsv                  ← エフェクトプリセットマスターデータ
│
├── Mac/
│   ├── TEMPLATE_Plugin-Info.plist
│   ├── TEMPLATE_Plugin-Prefix.pch
│   ├── TEMPLATE_Plugin.entitlements.plist
│   ├── codesign_config.sh
│   ├── codesign_plugin.sh
│   ├── notarize_plugin.sh
│   ├── install_plugin.sh
│   ├── package_cross_platform.sh
│   └── BUILD_INSTALL.md
│
├── Win/
│   ├── TEMPLATE_Plugin.sln
│   ├── TEMPLATE_Plugin.i
│   ├── .gitattributes
│   └── .gitignore
│
├── tools/
│   ├── color_preset_converter.py    ← TSV → ColorPresets.h 自動生成
│   ├── preset_converter.py          ← TSV → Presets.h 自動生成
│   ├── patch_pipl_japanese.py       ← PiPL日本語パッチ
│   └── convert_r_encoding.py        ← .r ファイルエンコーディング変換
│
└── docs/
    └── TEMPLATE_DEV_GUIDE.md        ← 開発ガイド（本ファイル参照）
```

---

## アーキテクチャ概要

### レンダリングフロー

```
Premiere Pro
  → GPU利用可能？
    → YES: _GPU.cpp（ホスト）
      → Windows: CUDA (.cu)
      → Mac: Metal (.metal → .cl)
      → Fallback: OpenCL (.cl)
    → NO: _CPU.cpp（CPUフォールバック）
```

### ファイル間の依存関係

```
_Version.h  ─┐
_ParamNames.h─┤
              ├─→ _CPU.cpp  ←── _Common.h
   .h ────────┤
              ├─→ _GPU.cpp  ←── _Common.h
              │      ↓
              │   ┌──┴──────────┐
              │   .cu        .cl ←── .metal (#include)
              │   (Windows)  (Mac)
              │   
              ├─→ .r  (PiPL定義)
              └─→ .rc (Windowsリソース)
```

---

## 主要な設計パターン

### 1. 三重同期ルール（最重要）

新機能追加時は **必ず3箇所** に実装:

| # | ファイル | プラットフォーム | 備考 |
|---|---------|-----------------|------|
| 1 | `.cu` | Windows (CUDA) | **リファレンス実装** — 先にここでテスト |
| 2 | `.cl` | Mac (Metal/OpenCL) | Metal は .cl を `#include` するため実質1箇所 |
| 3 | `_CPU.cpp` | 全環境 | GPUが使えない場合のフォールバック |

共通ロジックは `_Common.h` に `static inline` で集約。ただし GPU カーネル内では同じ関数を手動で再定義する必要がある（`__device__` 修飾子の違い等）。

### 2. ProcAmpParams構造体の順序ルール

```
ProcAmpParams のフィールド順 = .cl カーネルの引数順 = .cu カーネルの引数順
```

**Metal はこの構造体をバッファとして丸ごとカーネルに渡す**ため、1つでもフィールドの順序がずれると描画が完全に壊れる。カーネルに渡さないパラメータは構造体末尾の「CPU-only section」に配置する。

### 3. パラメータ名のクロスプラットフォーム変換

```
ソースコード (UTF-8)
  ├── Windows: コンパイラオプション (/source-charset:utf-8 /execution-charset:shift_jis)
  └── Mac: ランタイム iconv (ParamNameConverter シングルトン)
```

`PARAM("日本語名")` マクロで自動変換。`_ParamNames.h` に全パラメータ名を定義。

### 4. TSV → C++ヘッダー自動生成

```
color_presets.tsv  → python tools/color_preset_converter.py  → _ColorPresets.h
presets.tsv        → python tools/preset_converter.py        → _Presets.h
```

デザイナーがC++不要でデータ編集可能。git diff が明確。

### 5. CPU-GPU間データ共有

GPUは `inClipTime`（メディア絶対時間）しか取得できない。CPUで取得した `clipStartFrame` を **SharedClipData** (static map + mutex) で共有。

---

## プラットフォーム固有の注意事項

### CUDA (.cu) vs OpenCL/Metal (.cl)

| 項目 | CUDA | OpenCL/Metal |
|------|------|-------------|
| 関数修飾子 | `__device__ __forceinline__` | なし or `inline` |
| カーネル修飾子 | `__global__` | `__kernel` |
| 数学関数 | `cosf`/`sinf`/`powf`/`fminf`/`fmaxf` | `cos`/`sin`/`pow`/`fmin`/`fmax` |
| パラメータ渡し | 個別カーネル引数 | Metal: バッファ(構造体), OpenCL: 個別引数 |

### Metal の .cl #include パターン

```cpp
// TEMPLATE_Plugin.metal（1行のみ）
#include "TEMPLATE_Plugin.cl"
```

Metal専用コードの二重管理を回避。メンテナンスが1箇所で済む。

### Windows エンコーディング

- `.cpp` ファイル: UTF-8（vcxproj で `/source-charset:utf-8` 指定）
- `.r` / `.rcp` ファイル: Shift-JIS（code_page 932）
- `.rc` ファイル: `#pragma code_page(932)`

---

## init_project.sh のプレースホルダー一覧

| プレースホルダー | 説明 | 例 |
|-----------------|------|-----|
| `__TPL_PREFIX__` | プロジェクトプレフィックス | `OST` |
| `__TPL_PLUGIN_ID__` | プラグインID (PascalCase) | `MyEffect` |
| `__TPL_MATCH_NAME__` | PiPL Match Name | `OST_MyEffect` |
| `__TPL_UPPER_PREFIX__` | 大文字プレフィックス | `OST` |
| `__TPL_UPPER_PLUGIN__` | 大文字プラグインID | `MY_EFFECT` |
| `__TPL_EFFECT_NAME_JP__` | 日本語エフェクト名 | `きらめくパーティクル` |
| `__TPL_CATEGORY_JP__` | 日本語カテゴリ名 | `おしゃれテロップ` |
| `__TPL_AUTHOR__` | 著作者名 | `Kiyoto Nakamura` |
| `__TPL_VENDOR__` | ベンダー名 | `OshareTelop` |
| `__TPL_YEAR__` | 年 | `2026` |
| `__TPL_PROJECT_GUID__` | VS プロジェクトGUID | 自動生成 |

---

## チェックリスト：新プロジェクト開始時

- [ ] `init_project.sh` でプロジェクト生成
- [ ] Premiere Pro SDK を配置（パスを確認）
- [ ] Mac: Xcode プロジェクト作成 or 既存をクローン
- [ ] Win: `.sln` を Visual Studio 2022 で開く
- [ ] `_ParamNames.h` でパラメータ名を定義
- [ ] `.h` でパラメータ enum と定数を定義
- [ ] `_CPU.cpp` で ParamsSetup と SmartRender を実装
- [ ] `.cu` でCUDAカーネルを実装（リファレンス）
- [ ] `.cl` でOpenCL/Metalカーネルを実装（.cuと同期）
- [ ] `_GPU.cpp` でパラメータ取得とカーネルディスパッチを実装
- [ ] `_Common.h` に共通ユーティリティを追加
- [ ] Mac: ビルド → インストール → テスト
- [ ] Win: ビルド → インストール → テスト
- [ ] カラープリセット使用時: `color_presets.tsv` を編集 → converter 実行
- [ ] PiPL 日本語対応: `tools/patch_pipl_japanese.py` を設定
- [ ] コード署名（Mac）: `codesign_config.sh` を設定
- [ ] 配布パッケージ: `package_cross_platform.sh`

---

## Windy_Lines から学んだ重要な教訓

1. **ProcAmpParams のフィールド順序ミス** → Mac の描画が完全に壊れた（v64 Skew追加時）
2. **OutIn イージングの中間点速度ゼロ** → 25% リニアブレンドで解決
3. **Premultiplied Alpha の統一** → Straight alpha との混在はグレーエッジの原因
4. **`clipTime` ベースのフレーム計算** → `seqTime` を使うとキャッシュ不整合
5. **Windows の `#pragma code_page(932)`** → .rc ファイルで必須
6. **CUDA ヘッダーの `MAJOR_VERSION` マクロ衝突** → `#pragma push_macro` で回避
7. **Metal カーネルは OpenCL を #include** → 二重管理回避の最善策
8. **TSV マスターデータ** → C++ を触らずにデータ変更が可能

---

## ライセンス

This template is based on the Adobe Premiere Pro SDK sample code.
Portions Copyright 2012 Adobe Systems Incorporated.
Used in accordance with the Adobe Developer SDK License.
