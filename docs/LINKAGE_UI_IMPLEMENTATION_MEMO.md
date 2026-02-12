# 連動設定UIの実装メモ (Linkage Settings UI Implementation Memo)

## English Summary

This implementation reorganizes the linkage settings UI for Thickness, Length, and Travel Distance parameters:

**Key Changes:**
1. **Parameter Relocation**: Moved linkage settings from a separate topic group to directly above their corresponding parameters
2. **Conditional Display**: Implemented logic to show/hide parameters based on linkage mode:
   - When "Off": Show actual value (px), hide linkage rate (%)
   - When "Linked to Width/Height": Hide actual value, show linkage rate (%)
3. **Files Modified**: `OST_WindyLines.h` (parameter indices), `OST_WindyLines_CPU.cpp` (UI setup and visibility logic)

**Testing Required**: Full UI testing in After Effects/Premiere Pro to verify parameter ordering and visibility toggling.

---

## 概要 (Overview)

太さ(Thickness)、長さ(Length)、移動距離(Travel Distance)の3つのパラメータに対して、連動設定(Linkage Settings)を各パラメータの直前に配置し、連動状態に応じて実数値または連動率を表示/非表示する機能を実装しました。

## 変更内容 (Changes Made)

### 1. パラメータの並び順変更 (Parameter Order Reorganization)

**変更前 (Before):**
```
基本設定:
  - 線の数
  - 寿命
  - インターバル
  - 移動距離 ← 離れた場所にあった
  - イージング

外観:
  - 太さ ← 離れた場所にあった
  - 長さ ← 離れた場所にあった
  - 角度
  - ...

連動設定 (最後にまとめてあった):
  - 長さの連動
  - 長さ連動率(%)
  - 太さの連動
  - 太さ連動率(%)
  - 移動距離の連動
  - 移動距離連動率(%)
```

**変更後 (After):**
```
基本設定:
  - 線の数
  - 寿命
  - インターバル
  - 移動距離の連動 ← 追加
  - 移動距離連動率(%) ← 追加（条件付き表示）
  - 移動距離 ← 移動（条件付き表示）
  - イージング

外観:
  - 太さの連動 ← 追加
  - 太さ連動率(%) ← 追加（条件付き表示）
  - 太さ ← 移動（条件付き表示）
  - 長さの連動 ← 追加
  - 長さ連動率(%) ← 追加（条件付き表示）
  - 長さ ← 移動（条件付き表示）
  - 角度
  - ...

連動設定トピック: 削除
```

### 2. 条件付き表示ロジック (Conditional Visibility Logic)

各連動設定について、以下のロジックを実装:

| 連動モード | 実数値パラメータ | 連動率(%) | 
|-----------|-----------------|-----------|
| オフ (1) | 表示 ✓ | 非表示 ✗ |
| 要素の幅 (2) | 非表示 ✗ | 表示 ✓ |
| 要素の高さ (3) | 非表示 ✗ | 表示 ✓ |

**実装箇所:**
- `UpdatePseudoGroupVisibility()` 関数内に条件分岐を追加
- 各連動パラメータ変更時に `USER_CHANGED_PARAM` ハンドラが自動的にUIを更新

### 3. パラメータインデックスの更新 (Parameter Index Updates)

`OST_WindyLines.h` の enum を以下のように変更:

```cpp
// 旧: Basic Settings (3-7)
OST_WINDYLINES_LINE_COUNT,           // 3
OST_WINDYLINES_LINE_LIFETIME,        // 4
OST_WINDYLINES_LINE_INTERVAL,        // 5
OST_WINDYLINES_LINE_TRAVEL,          // 6 ← 旧位置
OST_WINDYLINES_LINE_EASING,          // 7

// 新: Basic Settings (3-9) - 連動設定を挿入
OST_WINDYLINES_LINE_COUNT,           // 3
OST_WINDYLINES_LINE_LIFETIME,        // 4
OST_WINDYLINES_LINE_INTERVAL,        // 5
OST_WINDYLINES_TRAVEL_LINKAGE,       // 6 ← 新規追加
OST_WINDYLINES_TRAVEL_LINKAGE_RATE,  // 7 ← 新規追加
OST_WINDYLINES_LINE_TRAVEL,          // 8 ← 新位置(+2)
OST_WINDYLINES_LINE_EASING,          // 9 ← 新位置(+2)
```

```cpp
// 旧: Appearance (19-23)
OST_WINDYLINES_LINE_THICKNESS,       // 19 ← 旧位置
OST_WINDYLINES_LINE_LENGTH,          // 20 ← 旧位置
OST_WINDYLINES_LINE_ANGLE,           // 21
OST_WINDYLINES_LINE_CAP,             // 22
OST_WINDYLINES_LINE_TAIL_FADE,       // 23

// 新: Appearance (21-29) - 連動設定を挿入
OST_WINDYLINES_THICKNESS_LINKAGE,        // 21 ← 新規追加
OST_WINDYLINES_THICKNESS_LINKAGE_RATE,   // 22 ← 新規追加
OST_WINDYLINES_LINE_THICKNESS,           // 23 ← 新位置(+4)
OST_WINDYLINES_LENGTH_LINKAGE,           // 24 ← 新規追加
OST_WINDYLINES_LENGTH_LINKAGE_RATE,      // 25 ← 新規追加
OST_WINDYLINES_LINE_LENGTH,              // 26 ← 新位置(+6)
OST_WINDYLINES_LINE_ANGLE,               // 27 ← 新位置(+6)
OST_WINDYLINES_LINE_CAP,                 // 28 ← 新位置(+6)
OST_WINDYLINES_LINE_TAIL_FADE,           // 29 ← 新位置(+6)
```

**重要:** 以降のすべてのパラメータインデックスが+4シフトされています。最終的な総パラメータ数は68個(0-67)です。旧Linkageトピックグループ(ヘッダーとエンドマーカー)が削除されたため、実質的にパラメータの総数は変わっていません。

### 4. 変更されたファイル (Modified Files)

1. **OST_WindyLines.h**
   - enum OST_WindyLines_Param のインデックスを更新
   - 連動パラメータを適切な位置に移動
   - 旧Linkageトピックグループの定義を削除

2. **OST_WindyLines_CPU.cpp**
   - `ParamsSetup()` 関数内でパラメータ定義順序を変更:
     - TRAVEL_LINKAGE/TRAVEL_LINKAGE_RATE を LINE_TRAVEL の直前に移動
     - THICKNESS_LINKAGE/THICKNESS_LINKAGE_RATE を LINE_THICKNESS の直前に移動
     - LENGTH_LINKAGE/LENGTH_LINKAGE_RATE を LINE_LENGTH の直前に移動
     - 旧Linkageトピックグループのコードを削除
   - `UpdatePseudoGroupVisibility()` 関数に条件付き表示ロジックを追加
   - `PF_Cmd_USER_CHANGED_PARAM` ハンドラに連動パラメータ変更の検出を追加
   - 各連動パラメータに `PF_ParamFlag_SUPERVISE` フラグを追加

## ユーザー体験 (User Experience)

### 使用例 1: 太さを要素の幅に連動させる

1. 「太さの連動」を「オフ」→「要素の幅」に変更
2. 自動的に:
   - 「太さ(px)」パラメータが非表示になる
   - 「太さ連動率(%)」パラメータが表示される
3. 「太さ連動率(%)」で幅に対する比率を調整（例: 5% = 要素幅の5%）

### 使用例 2: 長さを固定値に戻す

1. 「長さの連動」を「要素の高さ」→「オフ」に変更
2. 自動的に:
   - 「長さ連動率(%)」パラメータが非表示になる
   - 「長さ(px)」パラメータが表示される
3. 「長さ(px)」で固定ピクセル値を設定（例: 50.0px）

## 技術的な詳細 (Technical Details)

### 条件付き表示の実装

```cpp
// UpdatePseudoGroupVisibility() 内
const int thicknessLinkage = params[OST_WINDYLINES_THICKNESS_LINKAGE]->u.pd.value;
const bool thicknessIsLinked = (thicknessLinkage == 2 || thicknessLinkage == 3);
setVisible(OST_WINDYLINES_LINE_THICKNESS, !thicknessIsLinked);
setVisible(OST_WINDYLINES_THICKNESS_LINKAGE_RATE, thicknessIsLinked);
```

### 変更検出とUI更新

```cpp
// PF_Cmd_USER_CHANGED_PARAM ハンドラ内
if (changedExtra && (changedExtra->param_index == OST_WINDYLINES_THICKNESS_LINKAGE ||
                     changedExtra->param_index == OST_WINDYLINES_LENGTH_LINKAGE ||
                     changedExtra->param_index == OST_WINDYLINES_TRAVEL_LINKAGE))
{
    UpdatePseudoGroupVisibility(in_data, params);
    out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
}
```

## テスト手順 (Testing Instructions)

### 必須テスト項目

1. **パラメータ順序の確認**
   - [ ] After Effects/Premiere Proでプラグインを読み込む
   - [ ] 基本設定セクションで「移動距離の連動」が「移動距離」の直前にあることを確認
   - [ ] 外観セクションで「太さの連動」が「太さ」の直前、「長さの連動」が「長さ」の直前にあることを確認

2. **表示/非表示の動作確認**
   - [ ] 太さの連動: オフ → 要素の幅 → 要素の高さ → オフ と切り替え
     - オフ時: 「太さ(px)」表示、「太さ連動率(%)」非表示
     - 幅/高さ時: 「太さ(px)」非表示、「太さ連動率(%)」表示
   - [ ] 長さの連動: 同様の切り替えテスト
   - [ ] 移動距離の連動: 同様の切り替えテスト

3. **レンダリング動作確認**
   - [ ] 連動オフ時: 固定値でレンダリングが正常に動作
   - [ ] 連動オン時: 要素サイズに応じて動的に値が変化
   - [ ] 連動率の変更が即座にレンダリングに反映される

4. **既存プロジェクトとの互換性**
   - [ ] 旧バージョンで作成したプロジェクトを開く
   - [ ] パラメータ値が正しく保持されている
   - [ ] レンダリング結果に変化がない

### エッジケーステスト

- [ ] 連動モードを素早く連続で切り替える
- [ ] キーフレームアニメーション中の連動モード変更
- [ ] プリセット適用時の連動設定の動作
- [ ] エフェクトのコピー＆ペースト
- [ ] アンドゥ/リドゥ操作

## 既知の制限事項 (Known Limitations)

1. **パラメータインデックス変更の影響**
   - この変更は破壊的変更です。旧バージョンとの完全な互換性は保証されません
   - ただし、パラメータ名ベースのマッピングにより、多くのケースで互換性が維持されるはずです

2. **ビルド要件**
   - Adobe After Effects SDK が必要
   - Windows: Visual Studio
   - Mac: Xcode

## 今後の拡張予定 (Future Enhancements)

- [ ] 連動率の範囲を動的に調整（現在は固定範囲）
- [ ] カスタム連動式のサポート
- [ ] 複数パラメータの連鎖連動

## トラブルシューティング (Troubleshooting)

### Q: パラメータが表示されない

A: `UpdatePseudoGroupVisibility()` が正しく呼び出されているか確認してください。`PF_Cmd_UPDATE_PARAMS_UI` と `PF_Cmd_USER_CHANGED_PARAM` の両方で呼び出されます。

### Q: 連動率が機能しない

A: レンダリング部分(`OST_WindyLines_CPU.cpp` と `OST_WindyLines_GPU.cpp`)で正しい linkage パラメータを参照していることを確認してください。

### Q: 既存プロジェクトで値がおかしい

A: パラメータインデックスが変更されているため、プロジェクトファイルの再保存が必要な場合があります。

## 参考情報 (References)

- Adobe After Effects SDK Documentation
- `OST_WindyLines_ParamNames.h` - パラメータ名の定義
- `OST_WindyLines_Notes.json` - プロジェクト全体の注意事項

## 作成日・作成者 (Created)

- 日付: 2026-02-08
- 作成者: GitHub Copilot Implementation Agent
- バージョン: v64 (Linkage UI Reorganization)
