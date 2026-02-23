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

対話形式で **プラグインID** と **日本語エフェクト名** を入力すると、デフォルト設定で新プロジェクトを生成できます。  
最初の確認で `y` を選ぶと確認は1回だけ、`n` を選ぶと詳細入力後に最終確認が入ります。

---

## 前提条件（SDK のダウンロード）

テンプレートをビルドするには以下の SDK が必要です。

| SDK | 入手先 | 設定名 |
|-----|--------|--------|
| **Premiere Pro C++ SDK** | [Adobe Developer Console](https://developer.adobe.com/) → Downloads | `PREMIERE_SDK_BASE_PATH` |
| **After Effects SDK** | 同上 | `AE_SDK_BASE_PATH` |
| **CUDA Toolkit 12.x** (Win) | [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads) | `CUDA_SDK_BASE_PATH` |

### Mac — SDK 配置例

```
~/Desktop/Premiere Pro 26.0 C++ SDK/
  Examples/
    Headers/        ← PrSDK*.h, AE ヘッダー
    Utils/          ← PrGPUFilterModule.h
    Resources/      ← PiPLTool
    Projects/
      GPUVideoFilter/   ← ここにプロジェクトを配置すると相対パスが通る

~/Desktop/AfterEffectsSDK/
  Examples/
    Headers/        ← A.h, AEConfig.h, SPProps.h 等
    Util/
    Resources/
    GPUUtils/
```

### Win — 環境変数の設定

```
AE_SDK_BASE_PATH    = C:\AfterEffectsSDK
CUDA_SDK_BASE_PATH  = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
PREMSDKBUILDPATH    = <プロジェクト>\Win\Build
BOOST_BASE_PATH     = C:\boost
```

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
├── TEMPLATE_Plugin_License.h    ← ライセンス検証インターフェース
├── TEMPLATE_Plugin_WatermarkMask.h ← ウォーターマークビットマップ（要生成）
│
├── TEMPLATE_Plugin_CPU.cpp      ← CPUレンダラー（フォールバック）+ エントリーポイント + ライセンス実装
├── TEMPLATE_Plugin_GPU.cpp      ← GPUホストコントローラー（CUDA/Metal/OpenCLディスパッチ）+ ライセンスチェック
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
│   ├── TEMPLATE_Plugin.xcodeproj/  ← Xcodeプロジェクト（自動生成）
│   │   └── project.pbxproj
│   ├── TEMPLATE_Plugin-Info.plist
│   ├── TEMPLATE_Plugin-Prefix.pch
│   ├── TEMPLATE_Plugin.entitlements.plist
│   ├── en.lproj/InfoPlist.strings
│   ├── codesign_config.sh
│   ├── codesign_setup.sh       ← 署名初期セットアップ
│   ├── codesign_plugin.sh
│   ├── notarize_plugin.sh
│   ├── install_plugin.sh
│   ├── package_cross_platform.sh
│   └── BUILD_INSTALL.md
│
├── Win/
│   ├── TEMPLATE_Plugin.sln
│   ├── TEMPLATE_Plugin.vcxproj          ← Visual Studio プロジェクト
│   ├── TEMPLATE_Plugin.vcxproj.filters
│   ├── TEMPLATE_Plugin.i
│   ├── .gitattributes
│   └── .gitignore
│
├── tools/
│   ├── color_preset_converter.py    ← TSV → ColorPresets.h 自動生成
│   ├── preset_converter.py          ← TSV → Presets.h 自動生成
│   ├── patch_pipl_japanese.py       ← PiPL日本語パッチ
│   ├── convert_r_encoding.py        ← .r ファイルエンコーディング変換
│   ├── generate_watermark_mask.py   ← ウォーターマーク画像 → .h 生成ツール
│   └── activate_license_cache.py    ← ライセンスキャッシュ手動書き込みツール
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

## ライセンス認証システム

テンプレートには **OshareTelop 共通ライセンスシステム** が組み込まれています。  
1つのプラグインでアクティベートすれば、同じマシン上の全 OshareTelop プラグインがアンロックされます。

### 仕組み

| コンポーネント | 説明 |
|---|---|
| **キャッシュファイル** | `~/Library/Application Support/OshareTelop/license_cache_v1.txt` (Mac) / `%APPDATA%\OshareTelop\license_cache_v1.txt` (Win) |
| **Machine ID** | `DJB2(hostname\|platform\|sizeof(void*))` — プラグイン非依存 |
| **署名ソルト** | `OST_WL_2026_SALT_K9x3` — 全 OshareTelop プラグイン共通 |
| **バックグラウンドリフレッシュ** | TTL=600s、オフライン猶予=3600s |
| **ウォーターマーク** | 未認証時に CPU SmartRender で描画（387×41 ビットマップ） |

### ウォーターマーク生成

```bash
python3 tools/generate_watermark_mask.py \
  --text "Edit with" \
  --text2 "おしゃれテロップ・YourPlugin" \
  --font /path/to/NotoSansJP-Regular.ttf \
  --header-guard YOUR_UPPER_GUARD_H \
  --out YourPlugin_WatermarkMask.h
```

初期状態では 1×1 のダミーマスクがプレースホルダーとして入っています。  
リリース前に必ず上記コマンドで本物のウォーターマークを生成してください。

### 手動アクティベーション（テスト用）

```bash
python3 tools/activate_license_cache.py
```

### 主要関数（CPU.cpp 内）

- `IsLicenseAuthenticated()` — 現在の認証状態を返す（bool）
- `RefreshLicenseAuthenticatedState(force)` — キャッシュファイルの読み込みとバックグラウンドリフレッシュ
- `OpenActivationPage()` — ブラウザでアクティベーションページを開く

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
- [ ] Premiere Pro SDK / After Effects SDK をダウンロード＆配置
- [ ] Mac: Xcode プロジェクトの SDK パス設定（下記参照）
- [ ] Win: `.sln` を Visual Studio 2022 で開く＆環境変数確認
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
- [ ] コード署名（Mac）: `Mac/codesign_setup.sh` を実行
- [ ] 配布パッケージ: `Mac/package_cross_platform.sh`

---

## Mac — Xcode プロジェクト設定ガイド

### 生成済みの .xcodeproj を開く

```bash
cd <生成先>/Mac
open <MATCH_NAME>.xcodeproj
```

テンプレートの `project.pbxproj` には以下が事前設定されています:

- **Product Type**: Bundle (`.plugin`)
- **ソースファイル**: `_CPU.cpp`, `_GPU.cpp`（Obj-C++ モード）
- **フレームワーク**: Cocoa, Metal, OpenCL
- **リンカーフラグ**: `-liconv`（Shift-JIS 変換用）
- **Rez ビルドフェーズ**: `.r` ファイルの PiPL コンパイル
- **Run Script (Metal)**: `.metal` → `.metallib` のコンパイル＆バンドル内配置
- **Run Script (Deploy)**: ビルド後に MediaCore ディレクトリへ自動デプロイ

### SDK パスの設定（初回のみ）

Xcode プロジェクトを開いたら、**Project（ターゲットではない）のBuild Settings** で以下を確認・変更:

| 設定キー | デフォルト値（要変更） | 例 |
|----------|----------------------|-----|
| `PREMIERE_SDK_BASE_PATH` | `__TPL_PREMIERE_SDK_PATH__` | `/Users/you/Desktop/Premiere Pro 26.0 C++ SDK` |
| `AE_SDK_BASE_PATH` | `__TPL_AE_SDK_PATH__` | `/Users/you/Desktop/AfterEffectsSDK` |

> **ヒント**: Build Settings 画面で `PREMIERE` や `AE_SDK` で検索すると素早く見つかります。  
> User-Defined セクションに表示されます。

### Development Team の設定（署名が必要な場合）

1. ターゲットの **Signing & Capabilities** タブを開く
2. **Team** に自分の Apple Developer アカウントを選択
3. `CODE_SIGNING_ALLOWED` のデフォルトは `NO`（開発中は署名不要）

### ビルド & テスト

```bash
cd Mac
# Debugビルド（arm64）
xcodebuild clean -configuration Debug ARCHS=arm64
xcodebuild -configuration Debug ARCHS=arm64

# プラグインを MediaCore にインストール
./install_plugin.sh

# Premiere Pro を再起動してテスト
```

### Metal シェーダー Run Script の仕組み

`project.pbxproj` 内の Run Script フェーズが以下を実行:

1. `xcrun --find metal` で Metal コンパイラを検出
2. `.metal` → `.air`（中間表現） → `.metallib`（Metal ライブラリ）にコンパイル
3. 成果物をバンドル内の `Resources/MetalLib/` にコピー

```
Build Phases の順序:
  Sources → Frameworks → Resources → Rez → Metal Compile → Deploy
```

---

## Win — Visual Studio プロジェクト設定ガイド

### .sln を開く

Visual Studio 2022 で `Win/<MATCH_NAME>.sln` を開きます。  
`.vcxproj` には以下が事前設定されています:

- **構成**: Debug/Release × x64
- **出力**: `.aex` バンドル
- **CUDA ビルド**: カスタムビルドステップで `nvcc.exe` 呼び出し
- **OpenCL 文字列化**: `.cl` → プリプロセス → C文字列ヘッダー
- **PiPL 変換**: `.r` → PiPLTool → `patch_pipl_japanese.py` → `.rcp`
- **エンコーディング**: `/source-charset:utf-8 /execution-charset:shift_jis`

### 環境変数の確認

ビルド前にこれらの変数が定義されているか確認（`PreBuildEvent` で自動チェック）:

| 変数 | 用途 |
|------|------|
| `AE_SDK_BASE_PATH` | After Effects SDK のルート |
| `CUDA_SDK_BASE_PATH` | CUDA Toolkit のルート |
| `PREMSDKBUILDPATH` | ビルド出力先 |
| `BOOST_BASE_PATH` | Boost ライブラリ |

### プラグイン配置

ビルド成果物（`.aex`）を以下のディレクトリにコピー:

```
C:\Program Files\Adobe\Adobe Premiere Pro 2026\Plug-ins\Common\
```

### 注意: SDK 内での配置

`vcxproj` 内のヘッダー参照パス（`..\..\..\..\Headers` など）は、プロジェクトが **SDK 内の `Examples/Projects/GPUVideoFilter/` 配下** に配置されている前提です。SDK 外に配置する場合は `AdditionalIncludeDirectories` を環境変数ベースのパスに変更してください。

---

## コード署名 & 公証（Mac）

### 初期セットアップ

```bash
cd Mac
./codesign_setup.sh
```

このスクリプトが以下を対話形式で設定:
1. Apple Developer Program 登録状態の確認
2. Developer ID Application 証明書の検出
3. Team ID の取得
4. App-Specific Password の Keychain 保存
5. `codesign_config.sh` の自動生成

### 署名 → 公証 → インストール

```bash
./codesign_plugin.sh      # コード署名
./notarize_plugin.sh      # Apple 公証
./install_plugin.sh       # MediaCore にインストール
```

### クロスプラットフォーム配布

```bash
./package_cross_platform.sh    # Mac + Win を ZIP にまとめる
```

---

## エージェント向け：初期作業ガイド

> **AI エージェントがこの README を読み込んだ後、最初に行うべき作業手順。**
> ユーザーから「エフェクトの概要」を受け取ったら、以下の順序で実装を進める。

### Phase 1: パラメータ定義（コード変更 3ファイル）

1. **`_ParamNames.h`** — パラメータの日本語表示名を定義
   ```cpp
   constexpr const char* AMOUNT = "量";
   #define P_AMOUNT    PARAM(ParamNames::AMOUNT)
   ```

2. **`.h`（メインヘッダー）** — パラメータ enum、MIN/MAX/DFLT、ProcAmpParams 構造体を定義
   ```cpp
   enum { PARAM_AMOUNT = 1, PARAM_COUNT };
   #define AMOUNT_MIN 0.0f
   #define AMOUNT_MAX 100.0f
   #define AMOUNT_DFLT 50.0f
   ```

3. **`_CPU.cpp` の `ParamsSetup()`** — Premiere Pro にパラメータを登録
   ```cpp
   PF_ADD_FLOAT_SLIDER(P_AMOUNT, AMOUNT_MIN, AMOUNT_MAX, ...);
   ```

### Phase 2: GPU カーネル実装（3ファイル同期 — 最重要ルール）

**必ず 3箇所に同じロジックを実装する:**

4. **`.cu`（CUDA）** — リファレンス実装を先に書く
5. **`.cl`（OpenCL/Metal）** — `.cu` と同じロジック、関数名・数学関数を OpenCL 版に変換
6. **`_CPU.cpp` の `SmartRender()`** — GPU が使えない環境用フォールバック

### Phase 3: ホスト側接続（2ファイル）

7. **`_GPU.cpp`** — `ProcAmpParams` にフィールドを追加し、`Render()` でパラメータ取得→カーネルへ渡す
8. **`_CPU.cpp` の `SmartRender()`** — パラメータ取得してピクセルループ実行

### Phase 4: 検証

9. **ProcAmpParams の順序照合** — `.h` の構造体 / `.cu` カーネル引数 / `.cl` カーネル引数が完全一致か確認
10. Mac ビルド: `cd Mac && xcodebuild -configuration Debug ARCHS=arm64 && ./install_plugin.sh`
11. Premiere Pro で動作確認

### エージェントへの注意事項

- **`_Common.h` の `static inline` 関数**は CPU で使える。GPU カーネル内では同じロジックを `__device__`（CUDA）/ 修飾子なし（OpenCL）で再定義する必要がある
- **ProcAmpParams のフィールド順序を絶対に間違えない**（Metal がバッファ丸渡しのため、1つずれると描画が壊れる）
- **カーネルに渡さないパラメータ**は構造体末尾の CPU-only セクションに置く
- **数学関数**は CUDA (`cosf`) と OpenCL (`cos`) で名前が違う — DEV_GUIDE の対応表を参照
- **`.metal` は `.cl` を `#include` するだけ**なので、`.cl` だけ編集すれば Metal も対応完了

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
