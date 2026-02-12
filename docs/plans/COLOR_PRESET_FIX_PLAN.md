# カラープリセット修正＆統合計画

> 作成日: 2026-02-10  
> ステータス: 計画段階（レビュー待ち）

---

## 概要

2つの修正を行う：

1. **バグ修正**: カラープリセットを選択しても色が反映されない問題の修正
2. **UX改善**: 「カラーモード」パラメータを廃止し、カラープリセットドロップダウンに統合

---

## フェーズ1: バグ修正 — カラーが適用されない問題

### 根本原因

**GPU側の `NormalizePopupValue` 関数の条件判定順序が逆**

| ファイル | 判定順序 | 結果 |
|---|---|---|
| OST_WindyLines_GPU.cpp (L685-696) | `0-based` チェック → `1-based` チェック | **間違い** |
| OST_WindyLines_CPU.cpp (L61-73, L1696-1707) | `1-based` チェック → `0-based` チェック | **正しい** |

#### GPU版NormalizePopupValue（現在 — バグあり）
```cpp
static int NormalizePopupValue(int value, int maxValue)
{
    if (value >= 0 && value < maxValue)      // ← 0-basedを先にチェック
        return value;
    if (value >= 1 && value <= maxValue)      // ← 1-basedが後
        return value - 1;
    return 0;
}
```

#### 問題の影響（具体例）

Premiere Proのポップアップは **1-based** の値を送る。

| ユーザー操作 | Premiere送信値 | GPU結果 | 期待結果 | 状態 |
|---|---|---|---|---|
| 「単色」を選択 | colorMode=1 | **1** (Preset分岐) | 0 (Single分岐) | **❌ ずれ** |
| 「プリセット」を選択 | colorMode=2 | **2** (Custom分岐) | 1 (Preset分岐) | **❌ ずれ** |
| 「カスタム」を選択 | colorMode=3 | **2** (Custom分岐) | 2 (Custom分岐) | ✅ 偶然一致 |

**結果**: GPUレンダリング時にプリセット選択してもCustomカラーが使われ、プリセットの色が適用されない。

#### なぜCPUでは正常に動くか

CPU版Render関数内のラムダ `normalizePopup` (L1696) は正しい順序で判定している：
```cpp
auto normalizePopup = [](int value, int maxValue) {
    if (value >= 1 && value <= maxValue)  // ← 1-basedを先にチェック ✅
        return value - 1;
    // ...
};
```

### 修正箇所

#### 修正1-A: GPU.cpp の NormalizePopupValue 修正
- **ファイル**: `OST_WindyLines_GPU.cpp` L685-696
- **変更**: 条件の判定順序を CPU版と同じに修正

```cpp
// 修正後
static int NormalizePopupValue(int value, int maxValue)
{
    if (value >= 1 && value <= maxValue)      // 1-based を先にチェック
        return value - 1;
    if (value >= 0 && value < maxValue)       // 0-based は後
        return value;
    return 0;
}
```

> ⚠️ **影響範囲**: この関数は GPU.cpp 内のすべてのポップアップパラメータに使われている（blendMode, animPattern, colorMode, presetIndex, lineCap, spawnSource, originMode, easing, linkage×3）。修正により**全GPU側ポップアップの値が正しくなる**。  
> 他のポップアップが「偶然正しく動いていた」場合はそちらが壊れる可能性がある。ただし CPU版は既にこの順序で正常動作しているため、GPU版も同じにするのが正しい。

#### 修正1-B: GPU.cpp の presetIndex maxValue 更新
- **ファイル**: `OST_WindyLines_GPU.cpp` L1144
- **変更**: ハードコード `33` → `kColorPresetCount` を使用

```cpp
// 修正前
const int presetIndex = NormalizePopupParam(GetParam(OST_WINDYLINES_COLOR_PRESET, ...), 33);

// 修正後
const int presetIndex = NormalizePopupParam(GetParam(OST_WINDYLINES_COLOR_PRESET, ...), kColorPresetCount);
```

#### 修正1-C: デバッグログ追加（GPU.cpp）
- カラーモード・プリセットインデックスの値をログ出力
- パレット構築後の色値をログ出力
- CPU版には既にDebugLogがあるので、GPU版にも同等のものを追加

```cpp
// パレット構築前
DebugLog("[GPU ColorPreset] colorMode=%d, presetIndex=%d", colorMode, presetIndex);

// Preset分岐内
DebugLog("[GPU ColorPreset] Loaded preset #%d, Color[0]: R=%.2f G=%.2f B=%.2f", 
    presetIndex + 1, colorPalette[0][0], colorPalette[0][1], colorPalette[0][2]);
```

### 検証方法

1. ビルド後、Premiere Proでエフェクトを適用
2. カラーモード「プリセット」を選択
3. 各プリセット（レインボー、桜、サイバー等）を切り替え
4. ログ出力で `colorMode=1` (Preset), `presetIndex=0` (Rainbow) 等を確認
5. GPU/CPU 両方のレンダリングパスで色が一致することを確認

---

## フェーズ2: UX改善 — カラーモード統合

### 現状のUI構造
```
カラーモード:     [単色 ▼]          ← OST_WINDYLINES_COLOR_MODE (ポップアップ)
色：単色:         [■ ホワイト]       ← OST_WINDYLINES_LINE_COLOR (カラーピッカー)
色：プリセット:    [レインボー ▼]     ← OST_WINDYLINES_COLOR_PRESET (ポップアップ)
カスタム1～8:     [■][■][■]...       ← OST_WINDYLINES_CUSTOM_COLOR_1～8
```

### 変更後のUI構造
```
色：プリセット:    [単色 ▼]          ← OST_WINDYLINES_COLOR_PRESET (統合ポップアップ)
色：単色:         [■ ホワイト]       ← OST_WINDYLINES_LINE_COLOR (カラーピッカー)
カスタム1～8:     [■][■][■]...       ← OST_WINDYLINES_CUSTOM_COLOR_1～8
```

### 統合後のドロップダウン内容

| インデックス (1-based) | 表示名 | 旧colorMode相当 |
|---|---|---|
| 1 | 単色 | COLOR_MODE_SINGLE |
| 2 | カスタム | COLOR_MODE_CUSTOM |
| 3 | レインボー | Preset #1 |
| 4 | パステルレインボー | Preset #2 |
| 5 | 森 | Preset #3 |
| ... | ... | ... |
| N+2 | (最後のプリセット) | Preset #N |

→ 0-based変換後: `0=単色, 1=カスタム, 2+=プリセット（旧index = current - 2）`

### 修正ファイル一覧と変更内容

#### 2-A: color_preset_converter.py
- メニュー文字列の先頭に「単色」「カスタム」を追加する生成ロジック
- `generate_cpp_header()` に `kColorPresetMenuString` 定数を追加
  - 形式: `"単色|カスタム|レインボー|パステルレインボー|森|..."`
- `kColorPresetCount` は従来通り（プリセットの数のみ）。別途 `kColorMenuTotalCount = kColorPresetCount + 2` を生成

#### 2-B: OST_WindyLines_ColorPresets.h （自動生成を更新）
- `kColorPresetMenuString` — 統合メニュー文字列
- `kColorMenuTotalCount` — メニュー項目の総数（単色 + カスタム + プリセット数）
- `kColorMenuPresetOffset = 2` — プリセットのオフセット（最初のプリセットのメニュー内0-basedインデックス）

#### 2-C: OST_WindyLines.h
- `ColorMode` enum を以下に変更:
```cpp
// 統合カラーメニュー (0-based, NormalizePopup後)
// 0 = 単色, 1 = カスタム, 2+ = プリセット（プリセットインデックス = value - 2）
constexpr int COLOR_MENU_SINGLE = 0;
constexpr int COLOR_MENU_CUSTOM = 1;
constexpr int COLOR_MENU_PRESET_OFFSET = 2;
```
- `OST_WINDYLINES_COLOR_MODE` パラメータの enum 定義（残すか廃止か後述）
- `EffectPreset` 構造体: `colorMode` と `colorPreset` を 1つの `colorSelection` に統合
- `COLOR_MODE_DFLT` → `COLOR_SELECTION_DFLT = 1` （単色がデフォルト、1-based）

#### 2-D: OST_WindyLines_ParamNames.h
- `COLOR_MODE` 関連の名前定義を削除（または非表示に変更）
- `COLOR_PRESET` の名前を `"色の設定"` 等に変更
- `COLOR_MODE_MENU` の削除
- `COLOR_PRESET_MENU` は自動生成の `kColorPresetMenuString` を参照するように変更

#### 2-E: OST_WindyLines_GPU.cpp
- `OST_WINDYLINES_COLOR_MODE` の読み取りを廃止
- 統合パラメータの読み取りロジック:
```cpp
const int colorSelection = NormalizePopupParam(
    GetParam(OST_WINDYLINES_COLOR_PRESET, ...), kColorMenuTotalCount);

if (colorSelection == COLOR_MENU_SINGLE) {
    // 単色モード（旧 colorMode == 0）
} else if (colorSelection == COLOR_MENU_CUSTOM) {
    // カスタムモード（旧 colorMode == 2）
} else {
    // プリセットモード（旧 colorMode == 1）
    const int presetIndex = colorSelection - COLOR_MENU_PRESET_OFFSET;
    const PresetColor* preset = GetPresetPalette(presetIndex + 1);
    // ...
}
```

#### 2-F: OST_WindyLines_CPU.cpp

**ParamsSetup (L1115-1170)**:
- `PF_ADD_POPUP` for `OST_WINDYLINES_COLOR_MODE` を削除
- `OST_WINDYLINES_COLOR_PRESET` のポップアップに統合メニュー文字列を使用
- ラベル数を `kColorMenuTotalCount` に変更

**UpdatePseudoGroupVisibility (L770-825)**:
- `colorMode == 3` の判定を `colorSelection == COLOR_MENU_CUSTOM + 1` (1-based) に変更
- カラーピッカーの表示/非表示: 単色選択時のみ表示

**Render関数 (L1717-1795)**:
- `colorMode` と `presetIndex` を統合値から計算:
```cpp
const int colorSelection = normalizePopup(
    params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value, kColorMenuTotalCount);

if (colorSelection == COLOR_MENU_SINGLE) {
    // 単色
} else if (colorSelection == COLOR_MENU_CUSTOM) {
    // カスタム
} else {
    // プリセット
    const int presetIndex = colorSelection - COLOR_MENU_PRESET_OFFSET;
    const PresetColor* preset = GetPresetPalette(presetIndex + 1);
}
```

**自動切替ロジック (L2796-2856)**:
- `COLOR_PRESET変更→COLOR_MODEをPresetに設定` のロジック削除（統合されたため不要）
- `LINE_COLOR変更→COLOR_MODEをSingleに設定` → `LINE_COLOR変更→COLOR_PRESETを「単色」(1)に設定` に変更

**PF_Cmd_USER_CHANGED_PARAM (イベント処理)**:
- `OST_WINDYLINES_COLOR_PRESET` 変更時、選択値に応じてカスタムカラーの表示/非表示を切り替え

#### 2-G: OST_WindyLines_Presets.h & preset_converter.py
- `EffectPreset` の `colorMode` と `colorPreset` を統合
  - 旧: `colorMode=2, colorPreset=5` (プリセットモード, 5番目のプリセット)
  - 新: `colorSelection = 5 + 2 = 7` (オフセット2を加算)
  - 旧: `colorMode=1` (単色) → 新: `colorSelection = 1`
  - 旧: `colorMode=3` (カスタム) → 新: `colorSelection = 2`
- `presets.tsv` のカラム更新（colorMode + colorPreset → colorSelection）

#### 2-H: presets.tsv
- `colorMode` と `colorPreset` を `colorSelection` に統合

---

## パラメータインデックスの扱い

### 重要: OST_WINDYLINES_COLOR_MODE は廃止か隠しパラメータか

Premiere Proのパラメータはインデックスで永続化されるため、**既存プロジェクトとの後方互換性** の問題がある。

#### 選択肢A: COLOR_MODE を隠しパラメータにする（推奨）
- `OST_WINDYLINES_COLOR_MODE` の enum 位置はそのまま維持
- ParamsSetup で `PF_ParamFlag_INVISIBLE` を設定して非表示にする
- デフォルト値を COLOR_MODE_PRESET (2) に固定
- 新規プロジェクトでは常に COLOR_PRESET が統合メニューとして機能
- 既存プロジェクトの場合: 読み込み時に旧 COLOR_MODE + COLOR_PRESET から新しい統合値を計算するマイグレーションロジックを追加

#### 選択肢B: パラメータインデックスを変更する（非推奨）
- 既存プロジェクトのパラメータが壊れる
- VERSION_MAJOR を上げる必要がある

**→ 選択肢A を推奨**

### マイグレーションロジック（選択肢A）

```cpp
// PF_Cmd_SEQUENCE_SETUP or initial render:
const int legacyColorMode = params[OST_WINDYLINES_COLOR_MODE]->u.pd.value;
const int legacyPreset = params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value;

// 旧プロジェクト判定: COLOR_MODE が非Preset (≠2) の場合はマイグレーション
if (legacyColorMode == COLOR_MODE_SINGLE) {
    // 単色 → 統合メニュー index 1
    params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value = 1;
} else if (legacyColorMode == COLOR_MODE_CUSTOM) {
    // カスタム → 統合メニュー index 2  
    params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value = 2;
} else {
    // プリセット → 統合メニュー index = legacyPreset + 2
    params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value = legacyPreset + 2;
}
// マイグレーション後、COLOR_MODE を固定値に
params[OST_WINDYLINES_COLOR_MODE]->u.pd.value = COLOR_MODE_PRESET;
```

---

## 実装順序

### Step 1: フェーズ1のバグ修正（先にやる）
1. GPU.cpp の `NormalizePopupValue` 修正 (修正1-A)
2. GPU.cpp の `presetIndex` maxValue 修正 (修正1-B)
3. デバッグログ追加 (修正1-C)
4. ビルド・テスト

### Step 2: フェーズ2の準備
1. `color_preset_converter.py` に統合メニュー生成を追加 (2-A)
2. Python実行して `OST_WindyLines_ColorPresets.h` を再生成 (2-B)
3. `OST_WindyLines.h` の定義更新 (2-C)
4. `OST_WindyLines_ParamNames.h` の更新 (2-D)

### Step 3: フェーズ2の実装
1. `OST_WindyLines_CPU.cpp` の修正 (2-F)
   - ParamsSetup
   - UpdatePseudoGroupVisibility
   - Render
   - 自動切替ロジック
2. `OST_WindyLines_GPU.cpp` の修正 (2-E)
3. ビルド確認

### Step 4: エフェクトプリセット統合
1. `preset_converter.py` / `presets.tsv` の更新 (2-G, 2-H)
2. `OST_WindyLines_Presets.h` の再生成
3. マイグレーションロジックの実装

### Step 5: テストと検証
1. 新規エフェクトでの動作確認
2. プリセット切替の動作確認
3. カスタムカラー表示/非表示の確認
4. エフェクトプリセットからの色設定反映確認
5. ログ出力の確認
6. （可能であれば）旧プロジェクト読み込みテスト

---

## リスクと注意事項

| リスク | 対策 |
|---|---|
| NormalizePopupValue修正で他のポップアップパラメータが壊れる可能性 | CPU版は既にこの順序で正常動作。全パラメータの動作確認を行う |
| 旧プロジェクトとの後方互換性 | COLOR_MODEを隠しパラメータとして残し、マイグレーションロジックを実装 |
| エフェクトプリセットのcolorMode/colorPresetフィールド変更 | presets.tsvの値を一括変換、converter.pyで対応 |
| GPU/CPU両方のレンダリングパスの同期 | 同じロジック構造を維持、デバッグログで値を比較 |
| Mac/Metal/OpenCLへの影響 | カラー選択はホスト側で完結。カーネルに変更なし |

---

## ファイル変更マトリクス

| ファイル | フェーズ1 | フェーズ2 |
|---|---|---|
| OST_WindyLines_GPU.cpp | ✅ NormalizePopupValue修正, maxValue修正, ログ追加 | ✅ 統合ロジック |
| OST_WindyLines_CPU.cpp | — | ✅ ParamsSetup, Render, UI制御, 切替ロジック |
| OST_WindyLines.h | — | ✅ enum/define変更, EffectPreset構造体 |
| OST_WindyLines_ParamNames.h | — | ✅ 名前定義更新 |
| color_preset_converter.py | — | ✅ 統合メニュー生成 |
| OST_WindyLines_ColorPresets.h | — | ✅ 自動再生成 |
| preset_converter.py | — | ✅ colorSelection対応 |
| presets.tsv | — | ✅ カラム統合 |
| OST_WindyLines_Presets.h | — | ✅ 自動再生成 |
| OST_WindyLines.cu | — | — (変更不要) |
| OST_WindyLines.cl | — | — (変更不要) |
| OST_WindyLines.hlsl | — | — (変更不要) |
