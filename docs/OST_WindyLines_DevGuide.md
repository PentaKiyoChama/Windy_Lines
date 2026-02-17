# OST_WindyLines 開発ガイド

## ステータス: 安定版 (2026-02-17)

`allowMidPlay` パラメータ、線のアニメーション（頭が伸び尾が縮む）、Wind Originが正しく動作しています。CPU-GPU間のデータ共有により、クリップの先頭から線が徐々に出現する機能が実現されました。

v64以降: Skew（シアー変形）パラメータを追加、OutInイージングの中間停止問題を修正、Mac Metal/OpenCLの描画不具合を修正。

---

## 互換性ポリシー（2026-02-17 追記）

### リリース可（既存プロジェクト互換を維持）

- パラメータを**末尾に追加**する
- Effect Preset / Color Preset の項目を**末尾に追加**する
- 既存パラメータのデフォルト値を微調整する（意味・型を変えない範囲）

### リリース注意（既存プロジェクト破壊リスク）

- 既存パラメータIDの**順序入れ替え**
- 既存パラメータの**削除**
- 既存パラメータの**型変更**や**意味変更**
- `AE_Effect_Match_Name` の変更（既存クリップ紐付けが切れる）

### 最小回帰テスト（リリース前）

1. 既存プロジェクトを開く
2. クリップ上の既存設定が崩れていないことを確認
3. 再生・保存・再オープンで設定保持を確認

---

## プリセット拡張ルール（2026-02-17 追記）

### Effect Preset を増やす場合

- `presets.tsv` に**末尾追加**
- `preset_converter.py` を実行して `OST_WindyLines_Presets.h` を再生成
- `kEffectPresetCount` と UI メニューは生成結果から自動反映される

### Color Preset を増やす場合

- `color_presets.tsv` に**IDを増分で末尾追加**（既存IDは変更しない）
- `color_preset_converter.py` を実行して `OST_WindyLines_ColorPresets.h` を再生成
- `kColorPresetCount` / `kUnifiedPresetCount` / `GetUnifiedPresetMenuString()` / `UnifiedIndexToColorModeAndPreset()` の整合を確認

### 禁止事項

- 既存プリセットIDの再採番
- 既存プリセットの並び替え
- 既存プリセット名と中身の不一致変更（同じIDに別の色を割り当てる）

---

## ライセンスキャッシュ（Windows）補足（2026-02-17 追記）

### 実パス

- 基本: `%APPDATA%\\OshareTelop\\license_cache_v1.txt`
- フォールバック: `%USERPROFILE%\\AppData\\Roaming\\OshareTelop\\license_cache_v1.txt`

### K-1 テスト時の推奨コマンド

```powershell
# キャッシュ削除（ファイル/誤ディレクトリ両対応）
Remove-Item "$env:APPDATA\OshareTelop\license_cache_v1.txt" -Recurse -Force -ErrorAction SilentlyContinue

# キャッシュ確認
Get-Content "$env:APPDATA\OshareTelop\license_cache_v1.txt"

# 失敗時（ファイル種別確認）
Get-Item "$env:APPDATA\OshareTelop\license_cache_v1.txt" | Select-Object FullName, PSIsContainer
```

> 補足: Windows 側でキャッシュ保存処理のパス切り出しは修正済み（`/` と `\\` の両方を考慮）。

---

## 解決した主要な問題

### 問題
GPU側では `inClipTime`（メディアソース絶対時間）しか取得できず、「クリップの先頭」を検出できない。

### 解決策
CPU側で取得した `clipStartFrame` を、静的なグローバルマップでGPU側と共有。

### 実装

```
OST_WindyLines.h:
  SharedClipData 構造体（静的マップ + mutex）

OST_WindyLines_CPU.cpp:
  PF_UtilitySuite::GetClipStart() で clipStartFrame を取得
  SharedClipData::SetClipStart() で共有マップに保存

OST_WindyLines_GPU.cpp:
  SharedClipData::GetClipStart() で clipStartFrame を取得
  frameIndex = mediaFrameIndex - clipStartFrame（0から始まる）
```

---

## 技術的詳細

### 時間の計算

```cpp
// GPU側
mediaFrameIndex = clipTime / ticksPerFrame;  // 例: 215910
clipStartFrame = SharedClipData::GetClipStart();  // 例: 215860
frameIndex = mediaFrameIndex - clipStartFrame;  // 例: 50 (クリップ先頭から50フレーム)
timeFrames = frameIndex * speed;
```

### allowMidPlay の動作

- **ON**: スキップロジックなし。線はアニメーションの任意の位置で出現。
- **OFF**: `timeFrames < startFrame` の場合、その線をスキップ。クリップ先頭から徐々に線が出現。

### キャッシュとの一貫性

- `frameIndex` は `clipTime` から派生（`clipStartFrame` は定数）
- 同じ `clipTime` に対して常に同じ `frameIndex` が計算される
- キャッシュの問題なし

---

## 線のアニメーション

### 「頭が伸び、尾が縮む」方式

線は4つのフェーズでアニメーションします：

1. **頭が伸び始める**（尾は固定）
2. **全長に達する**
3. **尾が縮み始める**（頭は固定）
4. **消える**

### 実装

```cpp
// 前半（t = 0.0 ～ 0.5）：尾が固定、頭が伸びる
if (t <= 0.5f) {
    tailPosX = currentTravelPos;              // 尾は現在の移動位置
    headPosX = tailPosX + maxLen * extendT;   // 頭が伸びる
    currentLength = maxLen * extendT;         // 長さ 0 → maxLen
}
// 後半（t = 0.5 ～ 1.0）：頭が固定、尾が縮む
else {
    headPosX = currentTravelPos + maxLen;     // 頭は移動位置 + 最大長
    tailPosX = headPosX - maxLen * (1-retractT);  // 尾が縮む
    currentLength = maxLen * (1-retractT);    // 長さ maxLen → 0
}
```

### 移動距離の計算

- **総移動距離**: `travelRange + maxLen`
- **開始位置**: `-0.5*travel - maxLen`（左側で完全に隠れている）
- **終了位置**: `+0.5*travel`（右側に完全に通過）

これにより、線が画面外から現れて画面外に消えます。

---

## Wind Origin（線の発生位置）

### 目的

文字などのオブジェクトに対する「風の雰囲気」を調整します。

### 3つのモード

- **Center（中心）**: 線が対象にまとわりつく（左側から現れ、右側で消える）
- **Forward（前方）**: 線が対象から出ていく（対象から風が出る）
- **Backward（後方）**: 線が対象に向かってくる（対象に風があたる）

### 実装の重要ポイント

```cpp
// ❌ 間違い：個々の線のアニメーションに originOffset を混ぜる
const float tailStartPos = -0.5f * travelScaled - maxLen + originOffset;  // NG

// ✅ 正しい：線の発生位置（centerX/Y）に originOffset を適用
const float centerX = alphaCenterX + (rx - 0.5f) * alphaBoundsWidthSafe 
                    + originOffset * lineCos;  // 線の角度方向に適用
const float centerY = alphaCenterY + (ry - 0.5f) * alphaBoundsHeightSafe 
                    + originOffset * lineSin;  // Y軸にも効くように
```

### なぜ lineCos/lineSin を使うか

線は様々な角度（0度、45度、90度など）で表示されます。`originOffset` を線の角度方向に適用することで、全ての角度で正しく動作します。

---

## 主要ファイル

| ファイル | 役割 |
|---------|------|
| `OST_WindyLines.h` | パラメータ定義、SharedClipData 構造体 |
| `OST_WindyLines_CPU.cpp` | CPUレンダリング、clipStartFrame の取得と共有 |
| `OST_WindyLines_GPU.cpp` | GPUレンダリング（DirectX/CUDA/OpenCL/Metal）、ProcAmp2Params構造体 |
| `OST_WindyLines.cu` | CUDAカーネル（Windowsリファレンス実装） |
| `OST_WindyLines.cl` | OpenCL/Metalカーネル（Macはこれを共有） |
| `OST_WindyLines.metal` | Metal エントリポイント（.clをinclude） |
| `OST_WindyLines_Common.h` | 共有イージング関数（CPUで使用） |
| `OST_WindyLines_Notes.json` | AI向け技術ノート |

---

## デバッグ

### ログファイル
- 出力先: `OST_WindyLines_Debug.log`（同ディレクトリ）

### ログフォーマット
```
[GPU] allow=0 mediaF=215910 clipStart=215860 relF=50 seed=1 count=100
[CPU] clipTime=0 frame=0 clipStart=215860 trackStart=0 seed=1 count=100
[SKIP] i=0 timeFrames=25.0 startF=2.6 skip=0
```

### 確認ポイント
- `clipStart` が正しく取得されているか
- `relF`（相対フレーム）が0から始まっているか
- `skip` が期待通りに動作しているか

---

## テスト手順

1. **基本動作確認**
   - クリップを左詰めに配置
   - `allowMidPlay=OFF` に設定
   - フレーム0から再生
   - 線が徐々に出現することを確認

2. **他クリップとの独立性**
   - 同列に複数のクリップを配置
   - 各クリップで独立して線が出現することを確認

3. **シーク動作**
   - 途中のフレームにシーク
   - 乱れなく表示されることを確認

4. **パラメータ変更**
   - `allowMidPlay` をON/OFFに切り替え
   - 即座に動作が変わることを確認

---

## エージェント利用の最適化

### 効率的な依頼方法

1. **問題を具体的に説明**
   - 「乱れる」→「クリップ移動後、フレーム0で途中から線が表示される」
   - 再現手順を明記

2. **デバッグログを活用**
   - `OST_WindyLines_Debug.log` の関連部分を共有
   - `clipStart` と `relF` の値を確認

3. **OST_WindyLines_Notes.json を参照**
   - AI向けの技術ノートが記載されています
   - 「なぜこの実装になったか」が説明されています

### やってはいけないこと

- `seqTime` をフレーム計算に使う（キャッシュ不整合の原因）
- インスタンス状態でクリップを追跡（マルチスレッド競合）
- `mNodeID` を信頼できるキーとして使う（頻繁に変わる）
- `originOffset` を線のアニメーションロジックに混ぜる（発生位置のみに適用）

---

## 最近の修正履歴

### 2026-02-17: Skew（シアー変形）パラメータの追加

**機能**: 線を平行四辺形に変形する「シアー量」パラメータ

**実装詳細**:
- 変形式: `px -= skew * py`（回転後のローカル座標空間で適用）
- 範囲: -2.0 ～ +2.0（デフォルト: 0.0）
- enum位置: `OST_WINDYLINES_LINE_SKEW`（LINE_ANGLE=27 と LINE_CAP=29 の間、位置28）
- 全5箇所のSDF評価ポイントにシアー適用（線本体、影、モーションブラー）
- CPU/CUDA/OpenCL/Metal全バックエンドに実装済み

**注意事項**:
- CUDAは個別カーネル引数として `inLineSkew` を渡す
- OpenCLは `clSetKernelArg` インデックス65で渡す
- Metalは `ProcAmp2Params` 構造体経由（mLineSkewフィールド）

---

### 2026-02-17: OutInイージングの中間停止問題修正

**問題**: OutIn系イージング（SineOutIn, QuadOutIn, CubicOutIn, CircOutIn）の中間地点でアニメーションが完全に停止する瞬間があった

**原因**: piecewise OutIn実装で t=0.5 付近の導関数が0になり、速度が一瞬ゼロになる

**解決策**: 25%リニアブレンド（k=0.25）
```cpp
const float k = 0.25f;
return outIn * (1.0f - k) + t * k;
```
OutInカーブを75%、リニアを25%でブレンドすることで、中間の死点を解消しつつOutInの特性（速→遅→速）を維持。

**対象**: `OST_WindyLines_Common.h` の case 6, 10, 14, 18

---

### 2026-02-17: Mac Metal/OpenCL描画不具合の修正

**問題**: Mac環境（Metal/OpenCL）で線が一切描画されない

**原因**: `ProcAmp2Params` 構造体と `.cl` カーネル引数の順序不一致（2つの問題）
1. `mLineSkew` が構造体では位置21（mLineCapの後）だったが、.clカーネルでは末尾（位置60）→ 位置21以降の全パラメータがずれる
2. `mMotionBlurType` が構造体に存在するが `.cl` カーネルには存在しない → さらに1つ分ずれる

**解決策**:
- `mLineSkew` を `mMotionBlurVelocity` の後（カーネル引数の末尾）に移動
- `mMotionBlurType` をCPU専用セクションに移動

**重要な教訓**: Metalは `ProcAmp2Params` 構造体をバッファとして丸ごとカーネルに渡すため、構造体のフィールド順序は `.cl` カーネルの引数順序と**完全に一致**していなければならない。CUDAは個別引数を使うため影響を受けない。

---

### 2026-01-20: 配色システムの実装

**機能**: 線ごとに異なる色を設定可能に

**3つのモード**:
1. **Single（単色）**: すべての線が同じ色
2. **Preset（プリセット）**: 33種類の配色プリセットから選択
3. **Custom（カスタム）**: ユーザーが8色を自由に設定

**実装詳細**:
- 各線に `colorIndex = i % 8` で色インデックスを割り当て
- GPU: cbufferに `mPaletteR/G/B[8]` を追加
- CPU: VUYA形式のパレット配列を使用
- シェーダー: `GetPaletteColor(colorIndex)` で色を取得

**プリセット例**:
Rainbow, Pastel, Forest, Cyber, Sakura, Ocean, Sunset, Neon, Gold, Monochrome...

---

### 2026-01-20: 線のアニメーションとWind Originの完成

**問題1**: 「頭が伸び、尾が縮む」の実装が不正確
- ユーザー要件: 移動=0の場合、尾は完全に固定され、頭のみが伸縮するべき
- 解決: 前半/後半で独立した頭・尾の計算ロジックを実装

**問題2**: Wind Originが個々の線のアニメーションと混在
- ユーザー要件: 「全体の雰囲気」として線の発生位置を調整
- 解決: `originOffset` を `centerX/centerY` にのみ適用、`segCenterX` から分離

**問題3**: Wind OriginがY軸に効いていない
- 原因: X軸にのみ `originOffset` を加算
- 解決: 線の角度方向に適用（`originOffset * lineCos`, `originOffset * lineSin`）

---

## Metal/OpenCL構造体アライメント注意事項

### ProcAmp2Params 構造体とカーネル引数の同期

Metalレンダリングパスでは、`ProcAmp2Params` 構造体がバッファ（インデックス5）としてカーネルに渡される。このため:

1. **構造体のフィールド順序** = `.cl` カーネルの引数順序（完全一致必須）
2. CUDAは個別引数なので影響を受けない
3. OpenCLは `clSetKernelArg` で個別インデックス指定なので影響を受けない
4. **パラメータ追加時**: 構造体と `.cl` カーネルの**両方**に同じ順序で追加すること

### CPU専用パラメータの扱い

カーネルに渡す必要のないパラメータ（例: `mMotionBlurType`）は、構造体の末尾に「CPU-only section」として配置する。カーネル引数セクションの中に混ぜてはならない。

---

## 今後の改善候補

1. **SharedClipData のクリーンアップ**
   - 古いエントリを定期的に削除
   - メモリリーク防止

2. **デバッグログの改善**
   - タイムスタンプ追加
   - フレーム番号でフィルタリング

3. **パフォーマンス最適化**
   - タイルベースのビニング実装済み
   - アルファバウンド検出実装済み
