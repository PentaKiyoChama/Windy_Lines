# __TPL_MATCH_NAME__ — 開発ガイド

Windy_Lines プロジェクトの開発メモ・知見を集約した開発ガイド。

---

## 1. GPU実装 チートシート

### カーネル関数を追加するとき

1. **`.cu` に `__device__ __forceinline__` で実装**（CUDA リファレンス）
2. **`.cl` に修飾子なし版を実装**（OpenCL/Metal）
3. **`_Common.h` に `static inline` 版を実装**（CPU）
4. GPU.cpp の `ProcAmpParams` にパラメータを追加（**末尾のCPU-onlyセクションの前**）
5. `.cu` と `.cl` のカーネル引数に同じ順序で追加

### パラメータを追加するとき

1. `_ParamNames.h` に日本語名を追加
2. `.h` の enum に ID を追加（末尾）
3. `.h` に MIN/MAX/DFLT の `#define` を追加
4. `_CPU.cpp` の `ParamsSetup()` に PF_ADD_xxx を追加
5. `_CPU.cpp` の `SmartRender()` でパラメータ取得
6. `_GPU.cpp` の `ProcAmpParams` にフィールド追加
7. `_GPU.cpp` の `Render()` でパラメータ取得 → params に格納
8. `.cu` / `.cl` のカーネル引数に追加

### CUDA ↔ OpenCL/Metal 関数対応表

| CUDA | OpenCL/Metal | Common.h (CPU) |
|------|-------------|----------------|
| `__device__ __forceinline__` | *(なし)* | `static inline` |
| `__global__` | `__kernel` | *(N/A)* |
| `cosf(x)` | `cos(x)` | `cosf(x)` |
| `sinf(x)` | `sin(x)` | `sinf(x)` |
| `powf(x,y)` | `pow(x,y)` | `powf(x,y)` |
| `fminf(x,y)` | `fmin(x,y)` | `fminf(x,y)` or `std::min` |
| `fmaxf(x,y)` | `fmax(x,y)` | `fmaxf(x,y)` or `std::max` |
| `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` | *(N/A)* |

---

## 2. ProcAmpParams 安全チェック手順

パラメータ追加/変更後に必ず確認:

```
1. ProcAmpParams のフィールド順を確認
2. .cl のカーネル引数順を確認
3. .cu のカーネル引数順を確認
4. 3つの順序が完全に一致しているか照合
5. CPU-onlyパラメータが末尾セクションにあるか確認
```

**確認コマンド例**（手動 diff）:
```bash
grep -E '^\s+(int|float|unsigned)\s+m' TEMPLATE_Plugin_GPU.cpp | head -20
grep 'inWidth\|inHeight\|inAmount' TEMPLATE_Plugin.cu
grep 'inWidth\|inHeight\|inAmount' TEMPLATE_Plugin.cl
```

---

## 3. パラメータ名 エンコーディング

### Windows ビルド設定 (vcxproj)

```xml
<AdditionalOptions>/source-charset:utf-8 /execution-charset:shift_jis %(AdditionalOptions)</AdditionalOptions>
```

### Mac ランタイム変換

`ParamNameConverter::ToShiftJIS()` が `iconv("UTF-8" → "SHIFT_JIS")` で変換。
結果はキャッシュされるため、同じ文字列の再変換コストはゼロ。

### 新しいパラメータ名を追加するパターン

```cpp
// 1. _ParamNames.h の namespace ParamNames に追加
constexpr const char* MY_PARAM = "マイパラメータ";

// 2. ショートカットマクロを追加
#define P_MY_PARAM    PARAM(ParamNames::MY_PARAM)

// 3. _CPU.cpp の ParamsSetup() で P_MY_PARAM を使用
PF_ADD_FLOAT_SLIDER(P_MY_PARAM, ...);
```

---

## 4. イージング関数の使い方

### CPU側

```cpp
#include "_Common.h"
float easedT = ApplyEasing(t, easingType);  // t: 0.0〜1.0, easingType: 0〜27
float derivative = ApplyEasingDerivative(t, easingType);  // モーションブラー用
```

### GPU側 (.cu / .cl)

各カーネルファイル内に同じ `ApplyEasing()` 関数が定義されている。
引数・戻り値は同一。

### イージングインデックス一覧

| Index | 名前 | 日本語 |
|-------|------|--------|
| 0 | Linear | リニア |
| 1 | Smoothstep | スムースステップ |
| 2 | Smootherstep | スムーサーステップ |
| 3 | Sine In | サインイン |
| 4 | Sine Out | サインアウト |
| 5 | Sine In-Out | サインインアウト |
| 6 | Sine Out-In | サインアウトイン |
| 7 | Quad In | 二次イン |
| 8 | Quad Out | 二次アウト |
| 9 | Quad In-Out | 二次インアウト |
| 10 | Quad Out-In | 二次アウトイン |
| 11 | Cubic In | 三次イン |
| 12 | Cubic Out | 三次アウト |
| 13 | Cubic In-Out | 三次インアウト |
| 14 | Cubic Out-In | 三次アウトイン |
| 15 | Circle In | サークルイン |
| 16 | Circle Out | サークルアウト |
| 17 | Circle In-Out | サークルインアウト |
| 18 | Circle Out-In | サークルアウトイン |
| 19 | Back In | バックイン |
| 20 | Back Out | バックアウト |
| 21 | Back In-Out | バックインアウト |
| 22 | Elastic In | エラスティックイン |
| 23 | Elastic Out | エラスティックアウト |
| 24 | Elastic In-Out | エラスティックインアウト |
| 25 | Bounce In | バウンスイン |
| 26 | Bounce Out | バウンスアウト |
| 27 | Bounce In-Out | バウンスインアウト |

---

## 5. カラープリセット システム

### 追加手順

1. `color_presets.tsv` にタブ区切りで行を追加（**末尾追加、既存ID変更禁止**）
2. `python tools/color_preset_converter.py` を実行
3. 生成された `_ColorPresets.h` をビルドに含める
4. `.h` で `#include "_ColorPresets.h"` が有効であることを確認

### TSV 形式

```
id	name	name_en	color1	color2	... color8
```

各色: `A,R,G,B` (0-255)

### 使わない場合

`color_presets.tsv` と `tools/color_preset_converter.py` を削除し、
`.h` から `#include "_ColorPresets.h"` を削除すればOK。

---

## 6. ビルド設定 重要ポイント

### Xcode (Mac)

| 設定 | 値 | 理由 |
|------|-----|------|
| `CFBundlePackageType` | `eFKT` | Premiere Pro プラグイン識別子 |
| `CFBundleSignature` | `FXTC` | Premiere Pro プラグイン署名 |
| `Wrapper Extension` | `plugin` | ファイル拡張子 |
| Entitlement | `com.apple.security.cs.disable-library-validation` | Premiere内ライブラリ読込許可 |

### Visual Studio (Windows)

| 設定 | 値 | 理由 |
|------|-----|------|
| `TargetExt` | `.aex` | Premiere Pro プラグイン拡張子 |
| `/source-charset:utf-8` | 必須 | UTF-8ソースの正しい読込 |
| `/execution-charset:shift_jis` | 必須 | 日本語パラメータ名の変換 |
| CUDA: `-arch=sm_75` | 推奨 | Turing以降対応 |
| CUDA: `-use_fast_math` | 推奨 | パフォーマンス最適化 |
| リンク | `cudart_static.lib`, `OpenCL.lib` | GPU依存ライブラリ |

### PiPL ビルドステップ (Windows)

```
.r → cl /EP (プリプロセス) → PiPLTool → .rcp → python patch_pipl_japanese.py → .rc
```

---

## 7. デバッグ Tips

### デバッグログ

```cpp
DebugLog("value = %f, mode = %d", amount, mode);
```

- Debug ビルドのみ有効（Release は no-op）
- 出力先: `/tmp/__TPL_MATCH_NAME___Log.txt` (Mac) / `C:\Temp\...` (Win)

### デバッグレンダーマーカー

`.h` で `ENABLE_DEBUG_RENDER_MARKERS 1` に設定すると、
画面左上隅に GPU/CPU 表示が出る。リリース時は必ず 0 に。

### クリップ時間のデバッグ

GPU で `inClipTime` が期待値と異なる場合:
1. SharedClipData のマップ内容をログ出力
2. CPU側の `clipStartFrame` 設定箇所を確認
3. `clipTime - seqTime` のオフセットキーが正しいか確認

---

## 9. 初回実装ステップバイステップ（エージェント / 初回開発者向け）

> `init_project.sh` でプロジェクト生成後、最初のビルド通過までの手順。  
> 「Hello World」的な最小エフェクト（画面を指定色でティント）を作る例。

### Step 1: パラメータ名を定義

**`_ParamNames.h`** を開き、既存のサンプルパラメータを確認。  
ここに追加するパラメータの日本語名を定義する。

```cpp
namespace ParamNames {
    constexpr const char* TINT_AMOUNT = "ティント量";
    constexpr const char* TINT_COLOR_R = "色 R";
    constexpr const char* TINT_COLOR_G = "色 G";
    constexpr const char* TINT_COLOR_B = "色 B";
}
#define P_TINT_AMOUNT    PARAM(ParamNames::TINT_AMOUNT)
#define P_TINT_COLOR_R   PARAM(ParamNames::TINT_COLOR_R)
// ...
```

### Step 2: メインヘッダーにパラメータ定義

**`.h`** の enum にパラメータ ID を追加（**Input の次から連番**）:

```cpp
enum {
    PARAM_INPUT = 0,
    PARAM_TINT_AMOUNT,
    PARAM_TINT_COLOR_R,
    PARAM_TINT_COLOR_G,
    PARAM_TINT_COLOR_B,
    PARAM_COUNT
};
```

MIN/MAX/DFLT マクロを追加:

```cpp
#define TINT_AMOUNT_MIN   0.0f
#define TINT_AMOUNT_MAX   100.0f
#define TINT_AMOUNT_DFLT  50.0f
```

### Step 3: CPU 側を実装

**`_CPU.cpp`**:

1. `ParamsSetup()` に `PF_ADD_FLOAT_SLIDER` で各パラメータを登録
2. `SmartRender()` でパラメータの値を取得し、ピクセルごとにティント処理

### Step 4: CUDA カーネルを実装（リファレンス）

**`.cu`** に `__global__ void TintKernel(...)` を記述。  
引数の順序を `ProcAmpParams` のフィールド順と**完全一致**させること。

### Step 5: OpenCL/Metal カーネルを実装

**`.cl`** に `__kernel void TintKernel(...)` を記述。  
`.cu` から:
- `cosf` → `cos`, `fminf` → `fmin` へ変換
- `__device__` 修飾子を除去
- その他は DEV_GUIDE セクション1 の対応表を参照

`.metal` は `.cl` を `#include` するだけなので**編集不要**。

### Step 6: GPU ホストを接続

**`_GPU.cpp`**:

1. `ProcAmpParams` 構造体にフィールドを追加（`.cu` / `.cl` のカーネル引数と同じ順序）
2. `Render()` 内のパラメータ取得部分で値を `params` に格納
3. カーネルディスパッチ呼び出し部分を確認

### Step 7: ビルド＆テスト

```bash
# Mac
cd Mac
xcodebuild clean -configuration Debug ARCHS=arm64
xcodebuild -configuration Debug ARCHS=arm64
./install_plugin.sh
# → Premiere Pro を再起動してエフェクト適用
```

### 最終チェック

```bash
# ProcAmpParams の順序照合（手動 diff）
grep -E '^\s+(int|float|unsigned)\s+m' *_GPU.cpp | head -20
grep -oE 'in[A-Z][a-zA-Z]+' *.cu | head -20
grep -oE 'in[A-Z][a-zA-Z]+' *.cl | head -20
```

3つの出力が同じ順序であれば OK。

---

## 10. 配布チェックリスト

- [ ] `ENABLE_DEBUG_RENDER_MARKERS` = 0
- [ ] `ENABLE_GPU_RENDERING` = 1
- [ ] DebugLog が Release で無効であることを確認
- [ ] Version.h のバージョン番号を更新
- [ ] Mac: コード署名済み
- [ ] Mac: Apple 公証済み
- [ ] Win: Release/x64 でビルド
- [ ] INSTALL.txt / README.txt を同梱
- [ ] `package_cross_platform.sh` で ZIP 生成
