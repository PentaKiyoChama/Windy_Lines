# CPU/GPU 共通コード統合 計画書・指示書

## 概要

`OST_WindyLines_CPU.cpp` と `OST_WindyLines_GPU.cpp` の間で重複している純粋関数を、
新規共有ヘッダ `OST_WindyLines_Common.h` に切り出して一元管理する。

**目標**: 約680行の重複コードを削減し、将来のバグ修正・機能追加を一箇所で完結させる。

---

## 基本方針

| 項目 | 方針 |
|------|------|
| 新規ファイル | `OST_WindyLines_Common.h`（ヘッダオンリー） |
| 関数修飾子 | `static inline` を基本とする（ODR違反回避、インライン最適化） |
| 依存関係 | `<cmath>` と `PrSDKTypes.h` のみ。SDK UI系ヘッダへの依存禁止 |
| テスト手順 | 各ステップごとにビルド → Premiere Pro で CPU/GPU 両方の描画一致を目視確認 |
| ロールバック | 各ステップは Git コミット単位。問題発生時は `git revert` で即座に戻せる |
| 対象外 | `NormalizePopupValue()`（CPU=1-based→0-based変換、GPU=0-basedクランプ。意味が異なるため統合不可）|

---

## フェーズ構成

```
Phase 1: 純粋数学関数の統合（リスク最小・効果最大）
  Step 1-1: ヘッダ作成 + HashUInt / Rand01 / DepthScale / EaseInOutSine
  Step 1-2: ApplyEasing（~200行）
  Step 1-3: ApplyEasingDerivative

Phase 2: SDF・ブレンド関数の統合（CPU専用→共有化）
  Step 2-1: SDFBox / SDFCapsule
  Step 2-2: BlendPremultiplied / BlendUnpremultiplied

Phase 3: リンケージ計算の共通関数化
  Step 3-1: ApplyLinkage() 関数の作成と適用

Phase 4: カラーパレット構築ロジック（Preset分岐のみ）
  Step 4-1: BuildPresetPalette() 関数の共通化
```

---

## Phase 1: 純粋数学関数の統合

### Step 1-1: ヘッダ作成 + 小関数4つ

**目的**: `OST_WindyLines_Common.h` を新規作成し、最もシンプルな関数から移行を開始する。

#### 1. 新規ファイル作成: `OST_WindyLines_Common.h`

```cpp
#ifndef OST_WINDYLINES_COMMON_H
#define OST_WINDYLINES_COMMON_H

// ========================================================================
// OST_WindyLines_Common.h
// CPU/GPU 共通ユーティリティ関数
// ========================================================================

#include "PrSDKTypes.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ========================================================================
// Hash / Random
// ========================================================================

static inline csSDK_uint32 HashUInt(csSDK_uint32 x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

static inline float Rand01(csSDK_uint32 x)
{
    return (HashUInt(x) & 0x00FFFFFF) / 16777215.0f;
}

// ========================================================================
// Depth
// ========================================================================

static inline float DepthScale(float depth, float strength)
{
    const float v = 1.0f - (1.0f - depth) * strength;
    return v < 0.05f ? 0.05f : v;
}

// ========================================================================
// Easing Helpers
// ========================================================================

static inline float EaseInOutSine(float t)
{
    return 0.5f * (1.0f - cosf((float)M_PI * t));
}

#endif // OST_WINDYLINES_COMMON_H
```

#### 2. CPU.cpp の変更

**場所**: [OST_WindyLines_CPU.cpp](OST_WindyLines_CPU.cpp#L18) の `#include` ブロック

- `#include "OST_WindyLines_WatermarkMask.h"` の直後に `#include "OST_WindyLines_Common.h"` を追加

**削除対象**（各関数の定義を丸ごと削除）:

| 関数 | 行番号 | 行数 |
|------|--------|------|
| `HashUInt()` | L1248-L1256 | 9行 |
| `Rand01()` | L1258-L1261 | 4行 |
| `EaseInOutSine()` | L1263-L1266 | 4行 |
| `DepthScale()` | L1268-L1273 | 6行 |

> **注意**: 削除は上から順に行うと行番号がずれるため、**下から順（DepthScale → EaseInOutSine → Rand01 → HashUInt）**に削除する。

#### 3. GPU.cpp の変更

**場所**: [OST_WindyLines_GPU.cpp](OST_WindyLines_GPU.cpp#L18) の `#include` ブロック

- `#include "OST_WindyLines_WatermarkMask.h"` の直後に `#include "OST_WindyLines_Common.h"` を追加

**削除対象**:

| 関数 | 行番号 | 行数 |
|------|--------|------|
| `HashUInt()` | L375-L383 | 9行 |
| `Rand01()` | L385-L388 | 4行 |
| `DepthScale()` | L390-L395 | 6行 |

> GPU.cpp には `EaseInOutSine()` は存在しない（GPU版 `ApplyEasing` case 5 でインライン展開されている）。

#### テスト手順

1. **Mac ビルド**: Xcode でクリーンビルド → コンパイルエラーがないことを確認
2. **Win ビルド**: Visual Studio でクリーンビルド → コンパイルエラーがないことを確認
3. **動作確認**: Premiere Pro で任意のクリップに WindyLines を適用
   - GPU レンダリングで線が正常に描画されることを確認
   - GPU を無効化（Mercury Playback Engine → Software Only）して CPU レンダリングを確認
   - 両者の描画が目視で一致していることを確認
4. **回帰テスト**: 線の深度（DepthScale）、ランダム配置（Rand01）が以前と同じ結果になることを確認
5. **コミット**: `git commit -m "Step 1-1: Move HashUInt/Rand01/DepthScale/EaseInOutSine to Common.h"`

---

### Step 1-2: ApplyEasing の統合

**目的**: 最大の重複関数（~200行）を共有化する。

#### 差異分析

| 項目 | CPU.cpp | GPU.cpp |
|------|---------|---------|
| 引数名 | `(float t, int easing)` | `(float t, int easingType)` |
| クランプ | 関数冒頭で `t` を `[0,1]` にクランプ | クランプなし |
| case 5 | `EaseInOutSine(t)` を呼び出し | `0.5f * (1.0f - cosf((float)M_PI * t))` をインライン |
| case 3 | `(float)M_PI * t * 0.5f` | `(float)M_PI * 0.5f * t`（乗算順序のみ差。結果は同一）|
| コメントスタイル | case 毎に別行コメント | case の末尾にインラインコメント |

→ **数学的に完全同一**。統合に際して CPU 側のクランプを残す（安全側）。

#### 1. `OST_WindyLines_Common.h` に追加

`EaseInOutSine()` の直後に以下を追加:

```cpp
// ========================================================================
// Easing Functions (28 types)
// ========================================================================

static inline float ApplyEasing(float t, int easingType)
{
    // Clamp to [0, 1] for safety (CPU version behavior)
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    switch (easingType)
    {
        case 0: return t; // Linear
        case 1: return t * t * (3.0f - 2.0f * t); // SmoothStep
        case 2: return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // SmootherStep
        case 3: return 1.0f - cosf((float)M_PI * t * 0.5f); // InSine
        case 4: return sinf((float)M_PI * t * 0.5f); // OutSine
        case 5: return EaseInOutSine(t); // InOutSine
        case 6: { // OutInSine
            if (t < 0.5f) {
                return 0.5f * ApplyEasing(t * 2.0f, 4);
            } else {
                return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 3);
            }
        }
        case 7: return t * t; // InQuad
        case 8: return 1.0f - (1.0f - t) * (1.0f - t); // OutQuad
        case 9: {
            const float u = t * 2.0f;
            if (u < 1.0f) { return 0.5f * u * u; }
            const float v = u - 1.0f;
            return 0.5f + 0.5f * (1.0f - (1.0f - v) * (1.0f - v));
        }
        case 10: { // OutInQuad
            if (t < 0.5f) {
                return 0.5f * ApplyEasing(t * 2.0f, 8);
            } else {
                return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 7);
            }
        }
        case 11: return t * t * t; // InCubic
        case 12: {
            const float u = 1.0f - t;
            return 1.0f - u * u * u; // OutCubic
        }
        case 13: {
            const float u = t * 2.0f;
            if (u < 1.0f) { return 0.5f * u * u * u; }
            const float v = u - 1.0f;
            return 0.5f + 0.5f * (1.0f - (1.0f - v) * (1.0f - v) * (1.0f - v));
        }
        case 14: { // OutInCubic
            if (t < 0.5f) {
                return 0.5f * ApplyEasing(t * 2.0f, 12);
            } else {
                return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 11);
            }
        }
        case 15: return 1.0f - sqrtf(1.0f - t * t); // InCirc
        case 16: { // OutCirc
            const float u = t - 1.0f;
            return sqrtf(1.0f - u * u);
        }
        case 17: { // InOutCirc
            const float u = t * 2.0f;
            if (u < 1.0f) {
                return 0.5f * (1.0f - sqrtf(1.0f - u * u));
            }
            const float v = u - 2.0f;
            return 0.5f * (sqrtf(1.0f - v * v) + 1.0f);
        }
        case 18: { // OutInCirc
            if (t < 0.5f) {
                return 0.5f * ApplyEasing(t * 2.0f, 16);
            } else {
                return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 15);
            }
        }
        case 19: { // InBack
            const float s = 1.70158f;
            return t * t * ((s + 1.0f) * t - s);
        }
        case 20: { // OutBack
            const float s = 1.70158f;
            const float u = t - 1.0f;
            return u * u * ((s + 1.0f) * u + s) + 1.0f;
        }
        case 21: { // InOutBack
            const float s = 1.70158f * 1.525f;
            const float u = t * 2.0f;
            if (u < 1.0f) {
                return 0.5f * u * u * ((s + 1.0f) * u - s);
            }
            const float v = u - 2.0f;
            return 0.5f * (v * v * ((s + 1.0f) * v + s) + 2.0f);
        }
        case 22: { // InElastic
            if (t == 0.0f) return 0.0f;
            if (t == 1.0f) return 1.0f;
            const float p = 0.3f;
            return -powf(2.0f, 10.0f * (t - 1.0f)) * sinf((t - 1.0f - p / 4.0f) * (2.0f * (float)M_PI) / p);
        }
        case 23: { // OutElastic
            if (t == 0.0f) return 0.0f;
            if (t == 1.0f) return 1.0f;
            const float p = 0.3f;
            return powf(2.0f, -10.0f * t) * sinf((t - p / 4.0f) * (2.0f * (float)M_PI) / p) + 1.0f;
        }
        case 24: { // InOutElastic
            if (t == 0.0f) return 0.0f;
            if (t == 1.0f) return 1.0f;
            const float p = 0.45f;
            const float s = p / 4.0f;
            const float u = t * 2.0f;
            if (u < 1.0f) {
                return -0.5f * powf(2.0f, 10.0f * (u - 1.0f)) * sinf((u - 1.0f - s) * (2.0f * (float)M_PI) / p);
            }
            return powf(2.0f, -10.0f * (u - 1.0f)) * sinf((u - 1.0f - s) * (2.0f * (float)M_PI) / p) * 0.5f + 1.0f;
        }
        case 25: { // InBounce
            const float u = 1.0f - t;
            float b;
            if (u < 1.0f / 2.75f) {
                b = 7.5625f * u * u;
            } else if (u < 2.0f / 2.75f) {
                const float v = u - 1.5f / 2.75f;
                b = 7.5625f * v * v + 0.75f;
            } else if (u < 2.5f / 2.75f) {
                const float v = u - 2.25f / 2.75f;
                b = 7.5625f * v * v + 0.9375f;
            } else {
                const float v = u - 2.625f / 2.75f;
                b = 7.5625f * v * v + 0.984375f;
            }
            return 1.0f - b;
        }
        case 26: { // OutBounce
            if (t < 1.0f / 2.75f) {
                return 7.5625f * t * t;
            } else if (t < 2.0f / 2.75f) {
                const float u = t - 1.5f / 2.75f;
                return 7.5625f * u * u + 0.75f;
            } else if (t < 2.5f / 2.75f) {
                const float u = t - 2.25f / 2.75f;
                return 7.5625f * u * u + 0.9375f;
            } else {
                const float u = t - 2.625f / 2.75f;
                return 7.5625f * u * u + 0.984375f;
            }
        }
        case 27: { // InOutBounce
            if (t < 0.5f) {
                const float u = 1.0f - t * 2.0f;
                float b;
                if (u < 1.0f / 2.75f) {
                    b = 7.5625f * u * u;
                } else if (u < 2.0f / 2.75f) {
                    const float v = u - 1.5f / 2.75f;
                    b = 7.5625f * v * v + 0.75f;
                } else if (u < 2.5f / 2.75f) {
                    const float v = u - 2.25f / 2.75f;
                    b = 7.5625f * v * v + 0.9375f;
                } else {
                    const float v = u - 2.625f / 2.75f;
                    b = 7.5625f * v * v + 0.984375f;
                }
                return (1.0f - b) * 0.5f;
            } else {
                const float u = t * 2.0f - 1.0f;
                float b;
                if (u < 1.0f / 2.75f) {
                    b = 7.5625f * u * u;
                } else if (u < 2.0f / 2.75f) {
                    const float v = u - 1.5f / 2.75f;
                    b = 7.5625f * v * v + 0.75f;
                } else if (u < 2.5f / 2.75f) {
                    const float v = u - 2.25f / 2.75f;
                    b = 7.5625f * v * v + 0.9375f;
                } else {
                    const float v = u - 2.625f / 2.75f;
                    b = 7.5625f * v * v + 0.984375f;
                }
                return b * 0.5f + 0.5f;
            }
        }
        default:
            return t;
    }
}
```

#### 2. CPU.cpp の変更

**削除対象**: `ApplyEasing()` 関数の定義全体
- 開始: L1319 `static float ApplyEasing(float t, int easing)` 
- 終了: L1533 の閉じ括弧 `}`
- 約215行

#### 3. GPU.cpp の変更

**削除対象**: `ApplyEasing()` 関数の定義全体
- 開始: L491 `static float ApplyEasing(float t, int easingType)`
- 終了: L679 の閉じ括弧 `}`
- 約189行

#### テスト手順

1. **ビルド**: Mac/Win 両方でクリーンビルド
2. **全イージング確認**: Premiere Pro で WindyLines を適用し、Travel Easing を全28種類（Linear ～ InOutBounce）順に切り替え
   - GPU: 各イージングで線の動きがスムーズで以前と同一
   - CPU (Software Only): 同様に確認
3. **再帰的イージング**: 特に OutIn 系（case 6, 10, 14, 18）が内部で `ApplyEasing()` を再帰呼び出しするため重点確認
4. **コミット**: `git commit -m "Step 1-2: Move ApplyEasing to Common.h"`

---

### Step 1-3: ApplyEasingDerivative の統合

**目的**: イージングの導関数（速度係数）を共有化する。

#### 差異分析

| 項目 | CPU.cpp (L1639-L1669) | GPU.cpp (L682-L710) |
|------|------------------------|----------------------|
| case 0 | `return 1.0f;` | `return 1.0f;` |
| case 1-27 | `default:` ブロックで一括処理 | 全28ケースを列挙して同じ処理 |
| ロジック | 数値微分 `(f(t+ε) - f(t-ε)) / (t2 - t1)` | 完全同一 |

→ **動作は完全同一**。CPU版の `default:` スタイルの方が簡潔でメンテナブル。

#### 1. `OST_WindyLines_Common.h` に追加

`ApplyEasing()` の直後に:

```cpp
// ========================================================================
// Easing Derivative (numerical differentiation)
// ========================================================================

static inline float ApplyEasingDerivative(float t, int easingType)
{
    const float epsilon = 0.001f;
    switch (easingType)
    {
        case 0: return 1.0f; // Linear: constant velocity
        default:
        {
            const float t1 = t > epsilon ? t - epsilon : 0.0f;
            const float t2 = t < 1.0f - epsilon ? t + epsilon : 1.0f;
            const float dt = t2 - t1;
            if (dt > 0.0f) {
                return (ApplyEasing(t2, easingType) - ApplyEasing(t1, easingType)) / dt;
            }
            return 1.0f;
        }
    }
}
```

#### 2. CPU.cpp の変更

**削除対象**: `ApplyEasingDerivative()` 関数の定義全体
- L1639-L1669（約31行）

#### 3. GPU.cpp の変更

**削除対象**: `ApplyEasingDerivative()` 関数の定義全体  
- L682-L710（約29行）

#### テスト手順

1. **ビルド**: Mac/Win クリーンビルド（エラーなし確認）
2. **速度連動確認**: Line Thickness Easing を InQuad や OutElastic 等に設定して、線の太さ変化が速度に正しく連動していることを確認
3. **コミット**: `git commit -m "Step 1-3: Move ApplyEasingDerivative to Common.h"`

---

## Phase 2: SDF・ブレンド関数の統合

### Step 2-1: SDFBox / SDFCapsule

**目的**: CPU 専用の SDF 関数を共有ヘッダへ移動する。  
GPU カーネル内（.cu/.cl/.metal）にも同等の実装があるが、将来ホスト側でプレビュー計算する場合に再利用可能。

#### 差異分析

現在 GPU.cpp にはこれらの関数が**存在しない**（GPU カーネル側に同等品あり）。  
CPU.cpp からの移動のみ。GPU.cpp への影響なし。

#### 1. `OST_WindyLines_Common.h` に追加

```cpp
// ========================================================================
// SDF (Signed Distance Field) Functions
// ========================================================================

static inline float SDFBox(float px, float py, float halfLen, float halfThick)
{
    const float dxBox = fabsf(px) - halfLen;
    const float dyBox = fabsf(py) - halfThick;
    const float ox = fmaxf(dxBox, 0.0f);
    const float oy = fmaxf(dyBox, 0.0f);
    const float outside = sqrtf(ox * ox + oy * oy);
    const float inside = fminf(fmaxf(dxBox, dyBox), 0.0f);
    return outside + inside;
}

static inline float SDFCapsule(float px, float py, float halfLen, float halfThick)
{
    const float ax = fabsf(px) - halfLen;
    const float qx = fmaxf(ax, 0.0f);
    return sqrtf(qx * qx + py * py) - halfThick;
}
```

#### 2. CPU.cpp の変更

**削除対象**: `SDFBox()` と `SDFCapsule()` の定義（コメント含む）
- `SDFBox`: L1018-L1031（約14行）
- `SDFCapsule`: L1039-L1049（約11行）
- 間のコメントブロックも含めて整理

#### テスト手順

1. **ビルド**: Mac/Win クリーンビルド
2. **CPU 描画確認**: Line Shape が Box と Capsule の両方で正常描画されることを確認
3. **GPU は変更なし**なので GPU 側のテストは不要（ただし念のため描画確認推奨）
4. **コミット**: `git commit -m "Step 2-1: Move SDFBox/SDFCapsule to Common.h"`

---

### Step 2-2: BlendPremultiplied / BlendUnpremultiplied

**目的**: CPU 専用の合成関数を共有ヘッダへ移動する。

#### 差異分析

現在 GPU.cpp にはこれらの関数が**存在しない**（GPU カーネル側に同等品あり）。  

> **注意**: CPU.cpp の `BlendPremultiplied` と `BlendUnpremultiplied` は**実装が完全に同一**（参照渡しで in/out する同じアルファ合成ロジック）。命名の違いは意図的なもの（呼び出し元で前処理が異なる）。両方とも移す。

#### 1. `OST_WindyLines_Common.h` に追加

```cpp
// ========================================================================
// Alpha Blending
// ========================================================================

static inline void BlendPremultiplied(
    float srcR, float srcG, float srcB, float srcA,
    float& dstR, float& dstG, float& dstB, float& dstA)
{
    const float invSrcA = 1.0f - srcA;
    const float outA = srcA + dstA * invSrcA;
    const float invOutA = 1.0f / fmaxf(outA, 1e-6f);
    dstR = (srcR * srcA + dstR * dstA * invSrcA) * invOutA;
    dstG = (srcG * srcA + dstG * dstA * invSrcA) * invOutA;
    dstB = (srcB * srcA + dstB * dstA * invSrcA) * invOutA;
    dstA = outA;
}

static inline void BlendUnpremultiplied(
    float srcR, float srcG, float srcB, float srcA,
    float& dstR, float& dstG, float& dstB, float& dstA)
{
    const float invSrcA = 1.0f - srcA;
    const float outA = srcA + dstA * invSrcA;
    const float invOutA = 1.0f / fmaxf(outA, 1e-6f);
    dstR = (srcR * srcA + dstR * dstA * invSrcA) * invOutA;
    dstG = (srcG * srcA + dstG * dstA * invSrcA) * invOutA;
    dstB = (srcB * srcA + dstB * dstA * invSrcA) * invOutA;
    dstA = outA;
}
```

#### 2. CPU.cpp の変更

**削除対象**: `BlendPremultiplied()` と `BlendUnpremultiplied()` の定義（コメント含む）
- `BlendPremultiplied`: L1057-L1083（約27行、コメント込み）
- `BlendUnpremultiplied`: L1091-L1112（約22行、コメント込み）

#### テスト手順

1. **ビルド**: Mac/Win クリーンビルド
2. **CPU アルファ合成確認**: 透明なクリップ上で WindyLines を適用し、線と背景のアルファ合成が正常であることを確認（特に半透明の線がにじまないこと）
3. **コミット**: `git commit -m "Step 2-2: Move BlendPremultiplied/BlendUnpremultiplied to Common.h"`

---

## Phase 3: リンケージ計算の共通関数化

### Step 3-1: ApplyLinkage() 関数の作成

**目的**: Length / Thickness / Travel の3つのリンケージ計算が CPU/GPU で行単位でほぼ同一のため、共通関数として切り出す。

#### 差異分析

| 項目 | CPU.cpp (L2815-L2847) | GPU.cpp (L1746-L1778) |
|------|------------------------|------------------------|
| ロジック | 完全同一 | 完全同一 |
| 変数名 | `lineLengthScaledFinal` | `lineLengthScaled` |
| 最小値制限 | `lineThicknessScaledFinal < 1.0f` → `1.0f` | `lineThicknessScaled_temp < 1.0f` → `1.0f` |
| dsScale 条件 | `LINKAGE_MODE_OFF` の時のみ dsScale を掛ける | 同一 |

#### 1. `OST_WindyLines_Common.h` に追加

```cpp
// ========================================================================
// Linkage Calculation
// ========================================================================

/**
 * Apply linkage mode for a single dimension (Length, Thickness, or Travel).
 * When linkage is OFF, applies dsScale to the user input value.
 * When linkage is ON (WIDTH or HEIGHT), uses bounds * rate directly.
 *
 * @param userValue       Original user-specified value (full-resolution pixels)
 * @param linkageMode     LINKAGE_MODE_OFF / LINKAGE_MODE_WIDTH / LINKAGE_MODE_HEIGHT
 * @param linkageRate     Rate multiplier (0.0 - N)
 * @param boundsWidth     Alpha bounds width (downsampled pixels)
 * @param boundsHeight    Alpha bounds height (downsampled pixels)
 * @param dsScale         Downsample scale factor
 * @return Final value in downsampled pixel space
 */
static inline float ApplyLinkageValue(
    float userValue, int linkageMode, float linkageRate,
    float boundsWidth, float boundsHeight, float dsScale)
{
    float finalValue = userValue;
    if (linkageMode == LINKAGE_MODE_WIDTH) {
        finalValue = boundsWidth * linkageRate;
    } else if (linkageMode == LINKAGE_MODE_HEIGHT) {
        finalValue = boundsHeight * linkageRate;
    }
    // Apply dsScale only when linkage is OFF (user input is full-resolution)
    return (linkageMode == LINKAGE_MODE_OFF) ? (finalValue * dsScale) : finalValue;
}
```

> **前提**: `LINKAGE_MODE_OFF`, `LINKAGE_MODE_WIDTH`, `LINKAGE_MODE_HEIGHT` は `OST_WindyLines.h` で定義済みであること。`OST_WindyLines_Common.h` は `OST_WindyLines.h` の後に include されるため参照可能。

#### 2. CPU.cpp の変更

**場所**: L2815-L2847 付近のリンケージ計算ブロック

**変更前**（概要、約33行のリンケージ分岐 + dsScale 適用）:
```cpp
float finalLineLength = lineLength;
float finalLineThickness = lineThickness;
float finalLineTravel = lineTravel;
// ... Length/Thickness/Travel の各 if-else 分岐 (18行) ...
const float lineLengthScaledFinal = (lengthLinkage == LINKAGE_MODE_OFF) ? ...
const float lineThicknessScaledFinal_temp = ...
const float lineThicknessScaledFinal = lineThicknessScaledFinal_temp < 1.0f ? 1.0f : ...
const float lineTravelScaled = ...
```

**変更後**:
```cpp
const float lineLengthScaledFinal = ApplyLinkageValue(
    lineLength, lengthLinkage, lengthLinkageRate,
    alphaBoundsWidthSafe, alphaBoundsHeightSafe, dsScale);
const float lineThicknessScaledFinal_temp = ApplyLinkageValue(
    lineThickness, thicknessLinkage, thicknessLinkageRate,
    alphaBoundsWidthSafe, alphaBoundsHeightSafe, dsScale);
const float lineThicknessScaledFinal = lineThicknessScaledFinal_temp < 1.0f ? 1.0f : lineThicknessScaledFinal_temp;
const float lineTravelScaled = ApplyLinkageValue(
    lineTravel, travelLinkage, travelLinkageRate,
    alphaBoundsWidthSafe, alphaBoundsHeightSafe, dsScale);
```

#### 3. GPU.cpp の変更

**場所**: L1746-L1778 付近の同等ブロック

**変更後**（同様のパターン）:
```cpp
const float lineLengthScaled = ApplyLinkageValue(
    lineLength, lengthLinkage, lengthLinkageRate,
    alphaBoundsWidthSafe, alphaBoundsHeightSafe, dsScale);
const float lineThicknessScaled_temp = ApplyLinkageValue(
    lineThickness, thicknessLinkage, thicknessLinkageRate,
    alphaBoundsWidthSafe, alphaBoundsHeightSafe, dsScale);
const float lineThicknessScaled = lineThicknessScaled_temp < 1.0f ? 1.0f : lineThicknessScaled_temp;
const float lineTravelScaled = ApplyLinkageValue(
    lineTravel, travelLinkage, travelLinkageRate,
    alphaBoundsWidthSafe, alphaBoundsHeightSafe, dsScale);
```

#### テスト手順

1. **ビルド**: Mac/Win クリーンビルド
2. **リンケージ OFF テスト**: Length/Thickness/Travel Linkage を全て OFF の状態で、各値を手動変更 → 以前と同じ動作
3. **リンケージ WIDTH テスト**: Length Linkage = Width にして、クリップの横幅に連動して線の長さが変化することを確認
4. **リンケージ HEIGHT テスト**: Thickness Linkage = Height で同様に確認
5. **最小値テスト**: Thickness を極小値（ほぼ0）にして、`< 1.0f → 1.0f` の最小保証が効いていることを確認
6. **CPU/GPU 一致確認**: 同一設定で GPU/CPU を切り替えて描画が一致することを確認
7. **コミット**: `git commit -m "Step 3-1: Extract ApplyLinkageValue to Common.h"`

---

## Phase 4: カラーパレット構築ロジック（Preset 分岐のみ）

### Step 4-1: BuildPresetPalette() の共通化

**目的**: Preset モード時のパレット構築ロジックが CPU/GPU で完全同一のため、共通関数化する。

#### 差異分析

| 分岐 | CPU と GPU の違い |
|------|-------------------|
| **Single** | CPU: `params[...]->u.cd.value` で PF_Pixel 取得 / GPU: `GetParam()` → `PrParam` 型分岐（3パターン） → **統合不可** |
| **Custom** | CPU: `params[...]->u.cd.value` / GPU: `GetParam()` → PrParam 型分岐 → **統合不可** |
| **Preset** | CPU: `GetPresetPalette(idx+1)` → `preset[i].r/255.0f` / GPU: **完全同一** → **統合可能** |

> Single と Custom は パラメータ取得API が根本的に異なるため統合不可。  
> Preset 分岐のみ共通化する。

#### 1. `OST_WindyLines_Common.h` に追加

```cpp
// ========================================================================
// Color Palette Helpers
// ========================================================================

/**
 * Fill colorPalette[8][3] with preset colors.
 * @param paletteOut  Output array [8][3] (RGB normalized 0.0-1.0)
 * @param presetIndex 0-based preset index (internally converted to 1-based for GetPresetPalette)
 */
static inline void BuildPresetPalette(float paletteOut[][3], int presetIndex)
{
    const PresetColor* preset = GetPresetPalette(presetIndex + 1);  // GetPresetPalette expects 1-based
    if (preset)
    {
        for (int i = 0; i < 8; ++i)
        {
            paletteOut[i][0] = preset[i].r / 255.0f;
            paletteOut[i][1] = preset[i].g / 255.0f;
            paletteOut[i][2] = preset[i].b / 255.0f;
        }
    }
}
```

> **前提**: `PresetColor` 構造体と `GetPresetPalette()` は `OST_WindyLines_ColorPresets.h`（`OST_WindyLines.h` 経由でインクルード）で定義済み。

#### 2. CPU.cpp の変更

**場所**: L2599-L2608 の Preset 分岐

**変更前**:
```cpp
else  // Preset (colorMode == 2, 0-based)
{
    const PresetColor* preset = GetPresetPalette(presetIndex + 1);
    DebugLog(...);
    for (int i = 0; i < 8; ++i)
    {
        colorPalette[i][0] = preset[i].r / 255.0f;
        colorPalette[i][1] = preset[i].g / 255.0f;
        colorPalette[i][2] = preset[i].b / 255.0f;
    }
    DebugLog(...);
}
```

**変更後**:
```cpp
else  // Preset (colorMode == 2, 0-based)
{
    BuildPresetPalette(colorPalette, presetIndex);
    DebugLog("[ColorPreset] Preset mode: Loading preset #%d, Color[0]: R=%.2f G=%.2f B=%.2f",
        presetIndex + 1, colorPalette[0][0], colorPalette[0][1], colorPalette[0][2]);
}
```

#### 3. GPU.cpp の変更

**場所**: L1263-L1278 の Preset 分岐（同パターン）

**変更後**:
```cpp
else  // Preset (colorMode == 2, 0-based)
{
    BuildPresetPalette(colorPalette, presetIndex);
    DebugLog("[GPU ColorPreset] Preset mode: Loading preset #%d, Color[0]: R=%.2f G=%.2f B=%.2f",
        presetIndex + 1, colorPalette[0][0], colorPalette[0][1], colorPalette[0][2]);
}
```

#### テスト手順

1. **ビルド**: Mac/Win クリーンビルド
2. **プリセット切替テスト**: Color Preset ドロップダウンで全プリセット（Rainbow, Pastel, Forest 等）を順に切り替え → 色が正しく反映されることを確認
3. **Single / Custom モード**: プリセット以外のモードが以前と同じ動作であることを確認（変更していないが回帰テスト）
4. **CPU/GPU 一致**: 同じプリセットで GPU/CPU を切り替えて色が一致
5. **コミット**: `git commit -m "Step 4-1: Extract BuildPresetPalette to Common.h"`

---

## 統合対象外としたもの（将来検討）

| 項目 | 理由 |
|------|------|
| **NormalizePopupValue()** | CPU版は1-based→0-based変換、GPU版は0-basedクランプ。意味が異なるため統合不可 |
| **アニメーションパターン分岐** | CPU は `animPattern == 2/3`（1-based 由来）、GPU は `animPattern == 1/2`（0-based）。値の解釈が異なり、関数抽出すると引数に「オフセット補正」が必要になり可読性が低下 |
| **タイルビニング** | CPU は `std::vector<std::vector<int>>` タイルマップ、GPU は `std::unordered_map<uint64_t, std::vector<int>>` スパースマップ。データ構造が異なりロジック共通化が困難 |
| **線の生成コアループ** | アルゴリズムは同一だが、パラメータ取得API（`params[]->u.fs_d.value` vs `GetParam().mFloat64`）の差が全行に染みており、テンプレート化のコストが共通化メリットを上回る |
| **ウォーターマーク合成** | CPU は PF_Pixel16 ベース、GPU はホスト側で float 配列操作。データ型が根本的に異なる |

---

## 最終チェックリスト

全フェーズ完了後のリグレッションテスト:

- [ ] Mac Xcode ビルド成功
- [ ] Win Visual Studio ビルド成功  
- [ ] GPU レンダリング: 全エフェクトプリセット正常描画
- [ ] CPU レンダリング: 全エフェクトプリセット正常描画
- [ ] 全28イージング: GPU/CPU で動作一致
- [ ] リンケージ (OFF/WIDTH/HEIGHT): GPU/CPU で動作一致
- [ ] カラープリセット: 全プリセット GPU/CPU で色一致
- [ ] Line Shape (Box/Capsule): CPU で正常描画
- [ ] アルファ合成: 透明背景で CPU の線がにじまない
- [ ] パフォーマンス: 体感的な速度低下がないこと

---

## 想定される削減行数

| ステップ | 削除行数(推定) | 対象 |
|---------|--------------|------|
| Step 1-1 | ~23行 (CPU) + ~19行 (GPU) = **42行** | HashUInt, Rand01, DepthScale, EaseInOutSine |
| Step 1-2 | ~215行 (CPU) + ~189行 (GPU) = **404行** | ApplyEasing |
| Step 1-3 | ~31行 (CPU) + ~29行 (GPU) = **60行** | ApplyEasingDerivative |
| Step 2-1 | ~25行 (CPU) = **25行** | SDFBox, SDFCapsule |
| Step 2-2 | ~49行 (CPU) = **49行** | BlendPremultiplied, BlendUnpremultiplied |
| Step 3-1 | ~33行 (CPU) + ~33行 (GPU) = **66行** (リファクタ) | Linkage計算 |
| Step 4-1 | ~10行 (CPU) + ~10行 (GPU) = **20行** (リファクタ) | BuildPresetPalette |
| **合計** | **~666行削減** + Common.h ~350行追加 = **純減 ~316行** | |

> Common.h への集約により、将来のイージング追加・バグ修正が1ファイルで完結する保守性メリットが最大の価値。
