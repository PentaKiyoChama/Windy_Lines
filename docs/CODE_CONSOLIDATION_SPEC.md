# CPU/GPU 共通コード統合 仕様書

## 1. 文書情報

- 文書名: CPU/GPU 共通コード統合 仕様書
- 対象プロジェクト: OST_WindyLines
- 対象実装: `OST_WindyLines_Common.h` 導入による CPU/GPU 重複コード統合
- 対応フェーズ: Phase 1〜4（実装済み）

---

## 2. 目的

`OST_WindyLines_CPU.cpp` と `OST_WindyLines_GPU.cpp` に重複していた純粋処理を共通化し、
以下を達成することを目的とする。

- 同一ロジックの二重保守を解消
- 修正時の差分漏れ・実装乖離を防止
- 将来機能追加時の変更点を最小化

---

## 3. 適用範囲

### 3.1 対象ファイル

- `OST_WindyLines_Common.h`（新規共通ヘッダ）
- `OST_WindyLines_CPU.cpp`
- `OST_WindyLines_GPU.cpp`

### 3.2 対象機能

- 擬似乱数/深度補正
- Easing 本体/導関数
- SDF（Box/Capsule）
- Alpha Blend（Premultiplied / Unpremultiplied）
- Linkage 計算（Length/Thickness/Travel）
- Preset パレット構築

### 3.3 対象外

- `NormalizePopupValue()` の共通化（CPU/GPUで意味が異なる）
- Single/Custom 色取得の共通化（CPU/GPUでパラメータ取得 API が異なる）
- GPU カーネル（`.cu`, `.cl`, `.metal`）の統合

---

## 4. 設計方針

- 共通処理は `static inline` 関数として `OST_WindyLines_Common.h` に配置する
- 依存は最小化し、ヘッダ単体で完結する純粋関数中心とする
- 既存挙動互換を優先し、関数仕様変更は行わない
- 呼び出し側で必要な最小値ガード（例: thickness >= 1.0f）は維持する

---

## 5. 関数仕様（共通ヘッダ）

## 5.1 基本ユーティリティ

### `csSDK_uint32 HashUInt(csSDK_uint32 x)`
- 入力値をハッシュ化し `0..2^32-1` の整数を返す
- CPU/GPU で同一アルゴリズム

### `float Rand01(csSDK_uint32 x)`
- `HashUInt` を用いて `0.0..1.0` の疑似乱数を返す

### `float DepthScale(float depth, float strength)`
- 深度に応じたスケールを返す
- 最小値は `0.05f` を下限とする

---

## 5.2 Easing

### `float EaseInOutSine(float t)`
- InOutSine の共通補助関数

### `float ApplyEasing(float t, int easingType)`
- 28 種類の Easing を実装
- 入力 `t` は内部で `[0,1]` にクランプ
- CPU/GPU で同一結果を返すこと

### `float ApplyEasingDerivative(float t, int easingType)`
- `ApplyEasing` の数値微分を返す
- `easingType==0` は `1.0f` を返す

---

## 5.3 SDF

### `float SDFBox(float px, float py, float halfLen, float halfThick)`
- Box 形状の signed distance を返す

### `float SDFCapsule(float px, float py, float halfLen, float halfThick)`
- Capsule 形状の signed distance を返す

---

## 5.4 Blend

### `void BlendPremultiplied(...)`
- Premultiplied alpha の over 合成を行う
- `outA` 0 除算回避として `fmaxf(outA, 1e-6f)` を使用

### `void BlendUnpremultiplied(...)`
- Unpremultiplied 前提の合成処理
- 実装ロジックは既存挙動互換

---

## 5.5 Linkage

### `float ApplyLinkageValue(float userValue, int linkageMode, float linkageRate, float boundsWidth, float boundsHeight, float dsScale)`

- モード別に最終値を返す
  - `LINKAGE_MODE_OFF`: `userValue * dsScale`
  - `LINKAGE_MODE_WIDTH`: `boundsWidth * linkageRate`
  - `LINKAGE_MODE_HEIGHT`: `boundsHeight * linkageRate`
- CPU/GPU の既存分岐ロジックと等価であること

---

## 5.6 Color Preset

### `void BuildPresetPalette(float paletteOut[8][3], int presetIndex)`

- Preset（0-based index）から 8 色 RGB 正規化配列を構築
- `GetPresetPalette(presetIndex + 1)` を内部使用
- `preset == nullptr` の場合は何も書き換えない

---

## 6. 実装要件

## 6.1 CPU側

- `OST_WindyLines_CPU.cpp` の重複関数定義を削除し、共通関数呼び出しに置換
- Linkage 計算は `ApplyLinkageValue` を使用
- Preset 分岐は `BuildPresetPalette` を使用

## 6.2 GPU側

- `OST_WindyLines_GPU.cpp` の重複関数定義を削除し、共通関数呼び出しに置換
- Linkage 計算は `ApplyLinkageValue` を使用
- Preset 分岐は `BuildPresetPalette` を使用

---

## 7. 互換性要件

- 既存プロジェクトを開いた際の見た目差分が発生しないこと
- CPU/GPU 切替で描画差分が実用上無視できること
- 既存プリセットの色再現性を維持すること
- Linkage OFF/WIDTH/HEIGHT の挙動互換を維持すること

---

## 8. 受け入れ基準

以下を満たした場合、本仕様の実装は受け入れ可能とする。

- Mac/Win のビルド成功
- CPU/GPU 描画の一致
- Easing 回帰なし
- Linkage 回帰なし
- Preset 回帰なし
- 起動時・適用時クラッシュなし

検証手順は以下を参照:
- `docs/CONSOLIDATION_VALIDATION_CHECKLIST.md`

---

## 9. 変更履歴

- v1.0: Phase 1〜4 の実装反映版を初版として作成
