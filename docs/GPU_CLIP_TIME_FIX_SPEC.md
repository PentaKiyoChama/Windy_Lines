# GPU クリップ相対時間計算 — 仕様書

## 概要

GPUレンダラーにおいて、線の描画開始時間がクリップの時間ではなくシーケンスの時間に依存していた問題を修正。
クリップをタイムライン上のどの位置に配置しても、クリップ先頭から常に同じアニメーションが再生されるようになった。

## 背景

### CPU vs GPU の時間API

| API | 値の意味 | 例（25fps, クリップがseq 10秒目に配置, メディアは6分目から開始） |
|-----|---------|--------------------------------------------------------------|
| **CPU** `in_data->current_time` | クリップ相対時間（先頭=0） | `0` |
| **GPU** `inRenderParams->inClipTime` | メディアの絶対時間 | `914414307206400`（≈frame 215798） |
| **GPU** `inRenderParams->inSequenceTime` | シーケンス上の位置 | `42378336000`（= 10秒 × ticksPerFrame） |
| **GPU** `inRenderParams->inRenderTicksPerFrame` | 1フレームの時間解像度 | `4237833600`（25fps） |

- CPUは `current_time=0` で直接クリップ先頭を知れるが、GPUにはそのAPIがない
- `inClipTime` は「メディアファイル内の絶対時間」であり、クリップ位置とは無関係
- `inClipTime - inSequenceTime` は定数でないことが確認済み（フレームごとに微妙に変動）

### 試行錯誤の経緯

| # | アプローチ | 結果 | 原因 |
|---|-----------|------|------|
| 1 | `clipTime / ticksPerFrame` | ✗ | clipTimeは絶対メディア時間（frame 215798） |
| 2 | VideoSegmentSuiteノード入力走査 | ✗ | mNodeID(EffectImpl)のinputは0個 |
| 3 | `clipTime - seqTime` をキャッシュ | ✗ | GPUインスタンスが毎フレーム再作成される＋値が不定 |
| 4 | `seqTime / ticksPerFrame` | △ | クリップがseq先頭にあれば正しいが、移動すると破綻 |
| **5** | **ツリーBFS → ClipNode → TrackItemStartAsTicks** | **✓** | **最終採用** |

---

## 実装仕様

### 1. クリップ開始位置の取得（VideoSegmentSuite BFS走査）

**ファイル**: `OST_WindyLines_GPU.cpp` L919〜L1092

**アルゴリズム**:

```
1. mVideoSegmentSuite->AcquireVideoSegmentsID(mTimelineID) でセグメント一覧取得
2. 各セグメントのうち seqTime を含むものを選択（segStart <= seqTime < segEnd）
3. セグメントのルートノードを取得
4. BFS（幅優先探索）で全ノードを走査:
   a. 各ノードの GetNodeOperatorCount() でオペレーター数を取得
   b. AcquireOperatorNodeID() で各オペレーターを取得
   c. ハッシュ比較: memcmp(&opHash, &myNodeHash, sizeof(prPluginID))
   d. フォールバック: FilterMatchName に "WindyLines" を含むか
   e. マッチした場合、親ノード（ClipNode）から TrackItemStartAsTicks を読み取り
5. GetNodeInputCount() / AcquireInputNodeID() で子ノードを辿りBFSを継続
```

**ノードツリー構造**（Premiere Pro内部）:

```
Compositor
  ├─ ClipNode (TrackItemStartAsTicks あり)
  │    ├─ [operator] EffectImpl (= 我々のプラグイン, mNodeID)
  │    ├─ [operator] EffectImpl (= 他のエフェクト)
  │    └─ [input] MediaNode
  └─ ClipNode
       ├─ ...
```

**重要な知見**:
- エフェクトはClipNodeの **operator** であり **input** ではない
- `AcquireOperatorNodeID()` で取得したIDは参照カウント付きの新IDで、`mNodeID` と数値一致しない
- ノードハッシュ（`prPluginID`）の比較で正しくマッチできる
- FilterMatchNameによるフォールバック検索も実装（複数インスタンス時の安全策）

**フレームインデックス計算**:

```cpp
frameIndex = (seqTime - trackItemStart) / ticksPerFrame
```

- `seqTime`: 現在のシーケンス上の位置
- `trackItemStart`: ClipNodeから取得したクリップの開始位置（シーケンス時間）
- 結果: クリップ先頭で `frameIndex = 0`、以降インクリメント

### 2. アニメーションロジック統一（CPU = GPU）

**ファイル**: `OST_WindyLines_GPU.cpp` L1733〜L1767

**修正前（GPU独自）**:
```cpp
// effectiveTimeを先にStart Timeで相殺 → 全ラインのstartFrameを再計算
effectiveTime = timeFrames - startTimeFrames;
timeSinceLineStart = effectiveTime - startFrame;
age = fmodf(timeSinceLineStart, period);
```

**修正後（CPUと同一）**:
```cpp
// 先にageを計算し、cycleStartFrameでStart Time判定
age = fmodf(timeFrames - startFrame, period);
if (age < 0.0f) age += period;
cycleStartFrame = timeFrames - age;
if (cycleStartFrame < startTimeFrames) continue;   // Start Time前のサイクルはスキップ
if (endTimeFrames > 0 && cycleStartFrame >= endTimeFrames) continue;  // End Time後もスキップ
```

**違いの影響**:

| 条件 | 旧GPU（effectiveTime方式） | 新GPU/CPU（cycleStartFrame方式） |
|------|--------------------------|-------------------------------|
| Start Time = -300, frame 0 | effectiveTime = 300 → 既に進行中 | cycleStartFrame >= -300 → 正常出現 |
| Start Time = 0, frame 0 | effectiveTime = 0 → 一部の線が即出現 | cycleStartFrame < 0 → 全スキップ（空白） |
| Start Time = 0, frame 5 | effectiveTime = 5 | startFrame < 5 の線のみ出現（自然な立ち上がり） |

### 3. hideElement + 線0本の早期リターン修正

**ファイル**: `OST_WindyLines_GPU.cpp` L2421

**修正前**:
```cpp
if (lineData.empty()) {
    return suiteError_NoError;  // 元画像をそのまま返す
}
```

**修正後**:
```cpp
if (lineData.empty() && !hideElement) {
    return suiteError_NoError;  // hideElementがOFFの時のみパススルー
}
```

**理由**: `hideElement=ON` 時は、GPUカーネルが `pixel = (float4)(0,0,0,0)` で元画像を透明化する。
線が0本でもカーネル実行が必要。

---

## パラメータ仕様

### Start Time（開始時間）

| 項目 | 値 |
|------|-----|
| パラメータID | `OST_WINDYLINES_LINE_START_TIME` |
| 単位 | フレーム |
| デフォルト | `-300` |
| 範囲 | `-36000` 〜 `36000` |
| スライダー範囲 | `-300` 〜 `300` |

**動作**:
- **負の値**: クリップ開始前からアニメーションが走っていたように振る舞う。デフォルト-300で、クリップ先頭では既に線が定常的に出現している状態
- **0**: クリップ先頭で完全に空白。1-数フレーム後から徐々に線が出現（period依存）
- **正の値**: 指定フレームまで完全に空白。その後から出現開始

### Duration（持続時間）

| 項目 | 値 |
|------|-----|
| パラメータID | `OST_WINDYLINES_LINE_DURATION` |
| 単位 | フレーム |
| 動作 | `0` = 無限、`> 0` = Start Time から Duration フレーム後に新サイクル停止 |

---

## 制限事項・既知の課題

1. **ツリー走査のパフォーマンス**: 毎フレームBFS走査を行う（キャッシュ無効化問題のため）。セグメント数が多い複雑なシーケンスではオーバーヘッドが生じる可能性がある
2. **複数インスタンス**: 同一クリップに複数のWindyLinesエフェクトが適用されている場合、FilterMatchNameフォールバック検索で誤ったClipNodeにマッチする可能性がある（ハッシュ比較が優先されるため通常は発生しない）
3. **Windows未テスト**: VideoSegmentSuite走査はプラットフォーム非依存のAPI（PrGPUFilterBase提供）だが、Windows環境での動作は未検証

---

## テスト手順

### 基本テスト
1. クリップをシーケンス先頭（0秒）に配置 → Start Time = 0 → クリップ先頭でアニメーションが空白から始まる ✓
2. クリップをシーケンス途中（例: 10秒目）に移動 → クリップ先頭は引き続き同じ空白→出現アニメーション ✓
3. Start Time = -300 → クリップ先頭で既に線が定常出現 ✓

### hideElementテスト
4. hideElement = ON, Start Time = 0 → クリップ先頭の空白フレームでも元映像が透明 ✓
5. hideElement = ON, Start Time = -300 → 通常通り透明 ✓

### デバッグ確認
- ログファイル: `/tmp/OST_WindyLines_Log.txt`
- `GPU CLIP FOUND: ... TrackItemStart=X (frame Y)` → ClipNode検出成功
- `GPU TIME DEBUG: ... frameIndex=Z found=1` → クリップ相対フレーム計算成功
- `GPU CLIP WARN:` → ClipNode未検出（fallback to 0）

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `OST_WindyLines_GPU.cpp` | VideoSegmentSuite BFS走査、frameIndex計算、アニメーションロジック統一、hideElement修正 |

**変更行数**: 約170行追加、80行削除（ネット+90行）
