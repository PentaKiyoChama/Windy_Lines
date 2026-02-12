# ステップ2 実装とテストガイド

## 概要

このドキュメントは、配色カラープリセットTSVシステムの**ステップ2（新システム有効化）**の実装とテスト方法を詳細に説明します。

**ステップ2の目的**: 新しいヘッダーファイルを有効化し、既存の定義をコメントアウトすることで、新システムに移行する。既存定義はコメントとして残すことで、問題があればすぐに戻せる状態を維持する。

---

## 📋 実装内容

### 変更したファイル

1. **OST_WindyLines.h** - 新ヘッダーを有効化、既存定義をコメントアウト

### 実装の詳細

#### OST_WindyLines.h の変更

**変更前（Step 1の状態）**:
```cpp
#if 0  // まだ有効化しない
#include "OST_WindyLines_ColorPresets.h"
#endif

struct PresetColor {
    unsigned char a, r, g, b;
};

namespace ColorPresets {
    const PresetColor kRainbow[8] = { ... };
    // ... 全33プリセット
}

inline const PresetColor* GetPresetPalette(int presetIndex) {
    // ... 実装
}
```

**変更後（Step 2の状態）**:
```cpp
// 新しい色プリセットシステム（有効化）
#include "OST_WindyLines_ColorPresets.h"

/*
struct PresetColor {
    unsigned char a, r, g, b;
};
*/

/*
namespace ColorPresets {
    const PresetColor kRainbow[8] = { ... };
    // ... 全33プリセット
}
*/

/*
inline const PresetColor* GetPresetPalette(int presetIndex) {
    // ... 実装
}
*/
```

**重要なポイント**:
- `#if 0`を削除してincludeを有効化
- 既存の定義をすべて`/* */`でコメントアウト
- コメントアウトした定義は残す（Step 3で削除）
- 新ヘッダーから提供される定義のみが使用される

---

## 🧪 テスト手順

### テスト1: ビルドテスト（最重要）

#### 目的
新ヘッダーを使用した状態でビルドが成功することを確認する。

#### 手順

**Windows (Visual Studio)**:
```bash
cd /home/runner/work/Windy_Lines/Windy_Lines

# クリーンビルド
MSBuild OST_WindyLines.sln /t:Clean
MSBuild OST_WindyLines.sln /t:Build /p:Configuration=Debug

# ビルドログを確認
# エラー: 0
# 警告: 0（または既存の警告のみ）
```

**Mac (Xcode)**:
```bash
# クリーンビルド
xcodebuild clean -project OST_WindyLines.xcodeproj -configuration Debug
xcodebuild build -project OST_WindyLines.xcodeproj -configuration Debug

# ビルド結果を確認
# ** BUILD SUCCEEDED **
```

**Make**:
```bash
make clean
make

# エラーがないことを確認
echo $?  # 0 であるべき
```

#### 期待される結果
- ✅ ビルドエラーなし
- ✅ リンクエラーなし
- ✅ 新しいwarningなし
- ✅ プラグインファイルが正常に生成される

#### トラブルシューティング
**エラー: "redefinition of struct PresetColor"**
- 原因: 既存の定義がコメントアウトされていない
- 解決: OST_WindyLines.hの既存定義を確認し、すべて`/* */`で囲まれているか確認

**エラー: "OST_WindyLines_ColorPresets.h: No such file"**
- 原因: ヘッダーファイルが生成されていない
- 解決: `python color_preset_converter.py`を実行

---

### テスト2: プリセット数の確認

#### 目的
新システムで全プリセットが正しく認識されることを確認する。

#### 手順
```bash
cd /home/runner/work/Windy_Lines/Windy_Lines

# 生成されたヘッダーのプリセット数を確認
echo "新ヘッダーのプリセット数:"
grep -c "const PresetColor k" OST_WindyLines_ColorPresets.h

# 期待値: 33 または 34（ユーザーが追加した場合）

# GetPresetPalette関数のケース数を確認
echo "GetPresetPalette関数のケース数:"
grep -c "case.*return ColorPresets::" OST_WindyLines_ColorPresets.h

# 期待値: 33 または 34
```

#### 期待される結果
- ✅ プリセット数が正しい（33または34）
- ✅ GetPresetPalette関数のケース数が一致

---

### テスト3: 色データの検証

#### 目的
新システムで生成される色データが既存と完全に一致することを確認する。

#### 手順
```bash
cd /home/runner/work/Windy_Lines/Windy_Lines

# レインボープリセットの最初の色を確認
echo "レインボープリセット（新）:"
grep -A 1 "const PresetColor kRainbow\[8\]" OST_WindyLines_ColorPresets.h | tail -1 | grep -o "{[^}]*}" | head -1
# 期待値: {255, 255, 0, 0}

# モノクロプリセットの最後の色を確認
echo "モノクロプリセット（新）:"
grep -A 2 "const PresetColor kMonochrome\[8\]" OST_WindyLines_ColorPresets.h | tail -1 | grep -o "{[^}]*}" | tail -1
# 期待値: {255, 0, 0, 0}
```

#### 期待される結果
- ✅ レインボーの最初の色: {255, 255, 0, 0}
- ✅ モノクロの最後の色: {255, 0, 0, 0}
- ✅ すべての色データが期待通り

---

### テスト4: After Effects動作確認（重要）

#### 目的
実際のAfter Effectsで新システムが正常に動作することを確認する。

#### 手順
1. ビルドしたプラグインをAfter Effectsのプラグインディレクトリにコピー
2. After Effectsを起動
3. 新規コンポジション作成
4. 平面レイヤー作成
5. OST_WindyLinesエフェクトを適用

#### 確認項目
- [ ] エフェクトが正常に読み込まれる
- [ ] 色プリセット選択UIが表示される
- [ ] 全プリセット（33または34個）が表示される
- [ ] プリセット切り替えが動作する
- [ ] 各プリセットの色が正しい:
  - [ ] レインボー: 虹色（赤、オレンジ、黄、緑、青、藍、紫、マゼンタ）
  - [ ] パステル: 淡い虹色
  - [ ] 森: 緑系
  - [ ] サイバー: 青/紫系
  - [ ] モノクロ: 白→灰→黒
- [ ] レンダリングが正常に動作する
- [ ] クラッシュやエラーがない

#### 期待される結果
- ✅ すべてのプリセットが正常に動作
- ✅ 色が既存バージョンと一致
- ✅ UIが正常に動作
- ✅ エラーなし

---

### テスト5: 既存プロジェクトの互換性確認

#### 目的
既存の.aepファイルが新システムでも正常に動作することを確認する。

#### 手順
1. OST_WindyLinesを使用している既存の.aepファイルを開く
2. 以下を確認:
   - [ ] プロジェクトが正常に開ける
   - [ ] エラーメッセージがない
   - [ ] 色プリセット設定が維持されている
   - [ ] 色が変わっていない
   - [ ] レンダリング結果が一致する

#### テスト方法（ピクセル完全一致確認）
```bash
# 1. 既存バージョンでレンダリング（事前準備）
#    → output_old.png

# 2. 新バージョンでレンダリング
#    → output_new.png

# 3. ピクセル単位で比較
# ImageMagick等を使用
compare -metric AE output_old.png output_new.png difference.png
# 期待値: 0 (完全一致)
```

#### 期待される結果
- ✅ 既存プロジェクトが開ける
- ✅ 色プリセット設定が維持される
- ✅ レンダリング結果がピクセル完全一致
- ✅ パフォーマンス劣化なし

---

### テスト6: パフォーマンステスト

#### 目的
新システムでパフォーマンスが劣化していないことを確認する。

#### 手順
```bash
# After Effectsでレンダリング時間を測定

# 1. 複雑なエフェクトを含むコンポジションを用意
# 2. 既存バージョンでレンダリング時間を測定
# 3. 新バージョンでレンダリング時間を測定
# 4. 比較
```

#### 期待される結果
- ✅ レンダリング時間が同等（±5%以内）
- ✅ メモリ使用量が同等
- ✅ CPUリソース使用量が同等

---

## ✅ テスト完了チェックリスト

以下の全項目が✅になったら、ステップ2完了です。

### ビルド
- [ ] クリーンビルドが成功
- [ ] ビルドエラーなし
- [ ] リンクエラーなし
- [ ] 新しいwarningなし
- [ ] プラグインファイルが生成される

### データ検証
- [ ] プリセット数が正しい（33または34）
- [ ] 色データが既存と一致
- [ ] GetPresetPalette関数が動作

### 動作確認
- [ ] After Effectsでロードされる
- [ ] 全プリセットが表示される
- [ ] プリセット切り替えが動作
- [ ] 色が正しい
- [ ] レンダリングが正常

### 互換性
- [ ] 既存プロジェクトが開ける
- [ ] 色プリセット設定が維持される
- [ ] レンダリング結果が一致
- [ ] パフォーマンス劣化なし

---

## 🎯 テスト実行コマンドまとめ

```bash
#!/bin/bash
# ステップ2 テストスクリプト

cd /home/runner/work/Windy_Lines/Windy_Lines

echo "=== ステップ2 テスト開始 ==="
echo ""

echo "✓ テスト1: ファイル変更の確認"
echo "  新ヘッダーinclude確認:"
if grep -q "^#include \"OST_WindyLines_ColorPresets.h\"" OST_WindyLines.h; then
    echo "    ✅ 新ヘッダーが有効化されています"
else
    echo "    ❌ 新ヘッダーが有効化されていません"
fi

echo "  既存定義のコメントアウト確認:"
if grep -q "^/\* *$" OST_WindyLines.h | head -1; then
    echo "    ✅ 既存定義がコメントアウトされています"
else
    echo "    ⚠️  既存定義の状態を確認してください"
fi
echo ""

echo "✓ テスト2: プリセット数確認"
PRESET_COUNT=$(grep -c "const PresetColor k" OST_WindyLines_ColorPresets.h)
echo "  新ヘッダーのプリセット数: ${PRESET_COUNT}"
if [ "$PRESET_COUNT" -ge 33 ]; then
    echo "    ✅ プリセット数が正しい"
else
    echo "    ❌ プリセット数が不足: ${PRESET_COUNT}"
fi
echo ""

echo "✓ テスト3: 色データ検証"
echo "  レインボープリセット（新）:"
RAINBOW_COLOR=$(grep -A 1 "const PresetColor kRainbow\[8\]" OST_WindyLines_ColorPresets.h | tail -1 | grep -o "{[^}]*}" | head -1)
echo "    ${RAINBOW_COLOR}"
if [[ "$RAINBOW_COLOR" == *"255, 255, 0, 0"* ]]; then
    echo "    ✅ 色データが正しい"
else
    echo "    ❌ 色データが不正"
fi
echo ""

echo "=== ステップ2 基本テスト完了 ==="
echo ""
echo "📋 次の確認項目（手動）:"
echo "  1. ビルドテスト（必須）"
echo "  2. After Effectsでの動作確認（必須）"
echo "  3. 既存プロジェクトの互換性確認（必須）"
echo "  4. パフォーマンステスト（推奨）"
echo ""
echo "詳細は STEP2_IMPLEMENTATION_AND_TEST.md を参照してください"
```

---

## 📊 テスト結果の記録

テスト実行後、以下の表に結果を記録してください。

| テスト項目 | 結果 | 備考 |
|-----------|------|------|
| ビルド成功 | ✅ / ❌ | |
| プリセット数 | ✅ / ❌ | |
| 色データ一致 | ✅ / ❌ | |
| AE起動 | ✅ / ❌ | |
| プリセット表示 | ✅ / ❌ | |
| 色の正確性 | ✅ / ❌ | |
| レンダリング | ✅ / ❌ | |
| 既存プロジェクト | ✅ / ❌ | |
| パフォーマンス | ✅ / ❌ | |

---

## 🚨 トラブルシューティング

### エラー: "redefinition of struct PresetColor"

**原因**: 既存の定義がコメントアウトされていない、または新旧両方が有効

**解決**:
```bash
# 既存定義がコメントアウトされているか確認
grep -A 3 "struct PresetColor" OST_WindyLines.h

# コメントアウトされていない場合、手動で /* */ を追加
```

### エラー: "GetPresetPalette undefined reference"

**原因**: GetPresetPalette関数がコメントアウトされている

**解決**: OST_WindyLines_ColorPresets.hに関数が定義されているか確認
```bash
grep "GetPresetPalette" OST_WindyLines_ColorPresets.h
```

### 警告: After Effectsで色が違う

**原因**: TSVファイルと既存定義が一致していない可能性

**解決**:
```bash
# TSVファイルを再確認
python color_preset_converter.py

# 既存コードと比較
# コメントアウトされた既存定義と生成されたヘッダーを比較
```

### エラー: ビルドは成功するが、実行時にクラッシュ

**原因**: プリセットIDのミスマッチ or メモリ破損

**解決**:
```bash
# プリセットIDを確認
grep "case.*return ColorPresets::" OST_WindyLines_ColorPresets.h

# enum定義と一致しているか確認
```

---

## ✅ ステップ2完了の確認

以下をすべて確認できたら、ステップ2は完了です：

1. ✅ 新ヘッダーが有効化されている
2. ✅ 既存定義がすべてコメントアウトされている
3. ✅ ビルドが成功する
4. ✅ After Effectsで正常に動作する
5. ✅ 全プリセットが表示される
6. ✅ 色が既存バージョンと一致する
7. ✅ 既存プロジェクトが正常に開ける
8. ✅ レンダリング結果がピクセル完全一致
9. ✅ パフォーマンス劣化なし

**本番への影響**: ⚠️ **低** - 新システムを使用するが、内容は既存と同じ

**ロールバック方法**: 簡単 - `git checkout HEAD~1 -- OST_WindyLines.h`または手動で`#include`をコメントアウト

**次のステップ**: ステップ3（既存定義の完全削除）へ進む

---

**作成日**: 2026-02-09  
**対象**: ステップ2実装とテスト  
**ステータス**: 実装完了・テスト準備完了
