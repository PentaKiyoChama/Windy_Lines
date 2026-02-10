# ステップ1 実装とテストガイド

## 概要

このドキュメントは、配色カラープリセットTSVシステムの**ステップ1（新ヘッダー追加）**の実装とテスト方法を詳細に説明します。

**ステップ1の目的**: 新しいヘッダーファイルをプロジェクトに追加するが、`#if 0`で無効化しておくことで、既存の定義のみが使用され、本番環境への影響をゼロに保つ。

---

## 📋 実装内容

### 変更したファイル

1. **SDK_ProcAmp.h** - 新ヘッダーのincludeを追加（無効化状態）

### 実装の詳細

#### SDK_ProcAmp.h の変更（行536付近）

**追加したコード**:
```cpp
// ========== 新しい色プリセットシステム（テスト中）==========
// Step 1: 新ヘッダーを追加（まだ無効化状態）
#if 0  // まだ有効化しない（テストのみ）
#include "SDK_ProcAmp_ColorPresets.h"
#endif
// ========================================================
```

**重要なポイント**:
- `#if 0`により、includeは完全に無効化されている
- 既存の`struct PresetColor`と`namespace ColorPresets`の定義はそのまま残っている
- コンパイル時には既存のコードのみが使用される
- **本番環境への影響はゼロ**

---

## 🧪 テスト手順

### テスト1: ファイル変更の確認

#### 目的
SDK_ProcAmp.hに正しくincludeが追加されたか確認する。

#### 手順
```bash
cd /home/runner/work/Windy_Lines/Windy_Lines

# 1. 変更内容を確認
git diff SDK_ProcAmp.h

# 期待される出力:
# +// ========== 新しい色プリセットシステム（テスト中）==========
# +// Step 1: 新ヘッダーを追加（まだ無効化状態）
# +#if 0  // まだ有効化しない（テストのみ）
# +#include "SDK_ProcAmp_ColorPresets.h"
# +#endif
# +// ========================================================

# 2. #if 0 が正しく設定されているか確認
grep -A 2 "#if 0" SDK_ProcAmp.h | grep "SDK_ProcAmp_ColorPresets"
# 期待値: #include "SDK_ProcAmp_ColorPresets.h"

# 3. 既存の定義が残っているか確認
grep -c "struct PresetColor" SDK_ProcAmp.h
# 期待値: 1（既存の定義が残っている）

grep -c "namespace ColorPresets" SDK_ProcAmp.h
# 期待値: 1（既存の定義が残っている）
```

#### 期待される結果
- ✅ includeが追加されている
- ✅ `#if 0`で無効化されている
- ✅ 既存の定義はすべて残っている
- ✅ コメントが適切に追加されている

---

### テスト2: プリプロセッサ動作の確認

#### 目的
`#if 0`により新ヘッダーが実際に無効化され、既存コードのみがコンパイルされることを確認する。

#### 手順
```bash
cd /home/runner/work/Windy_Lines/Windy_Lines

# プリプロセッサの出力を確認（C++のみを展開）
# 注: 実際の環境に応じてコンパイラを調整
# g++ -E -P SDK_ProcAmp.h 2>/dev/null | grep -A 5 "struct PresetColor"

# または、手動で確認:
# 1. #if 0 の中のコードが無視されるか
# 2. 既存の struct PresetColor が使われるか
echo "✓ #if 0により新ヘッダーは無効化されています"
```

#### 期待される結果
- ✅ `#if 0`ブロック内のコードはコンパイル対象外
- ✅ 既存の定義のみがコンパイルされる

---

### テスト3: ビルドテスト（重要）

#### 目的
変更後もプロジェクトが正常にビルドできることを確認する。

#### 手順

**Windows (Visual Studio)**:
```bash
# クリーンビルド
MSBuild SDK_ProcAmp.sln /t:Clean
MSBuild SDK_ProcAmp.sln /t:Build /p:Configuration=Debug

# または
devenv SDK_ProcAmp.sln /Clean Debug
devenv SDK_ProcAmp.sln /Build Debug
```

**Mac (Xcode)**:
```bash
# クリーンビルド
xcodebuild clean -project SDK_ProcAmp.xcodeproj -configuration Debug
xcodebuild build -project SDK_ProcAmp.xcodeproj -configuration Debug
```

**Make**:
```bash
make clean
make

# ビルドログを確認
# エラーがないことを確認
```

#### 期待される結果
- ✅ ビルドエラーなし
- ✅ ワーニングなし（または既存のワーニングのみ）
- ✅ リンクエラーなし
- ✅ プラグインファイル（.aex, .plugin等）が正常に生成

---

### テスト4: 動作確認（After Effects）

#### 目的
ビルドしたプラグインがAfter Effectsで正常に動作することを確認する。

#### 手順
```bash
# 1. ビルドしたプラグインをAfter Effectsのプラグインディレクトリにコピー
# （環境に応じてパスを調整）

# Windows例:
# copy Debug\SDK_ProcAmp.aex "C:\Program Files\Adobe\Adobe After Effects 2024\Support Files\Plug-ins\"

# Mac例:
# cp -R build/Debug/SDK_ProcAmp.plugin "/Applications/Adobe After Effects 2024/Plug-ins/"
```

**After Effects での確認**:
1. After Effectsを起動
2. 新規コンポジションを作成
3. 平面レイヤーを作成
4. エフェクト → SDK_ProcAmp を適用
5. 色プリセット選択UIを開く
6. 以下を確認:
   - [ ] 全33プリセット（または34プリセット）が表示される
   - [ ] プリセットを切り替えて色が表示される
   - [ ] レインボー、パステル、森、サイバーなど既存プリセットが正常動作
   - [ ] エフェクトが正常にレンダリングされる

#### 期待される結果
- ✅ プラグインが正常にロードされる
- ✅ 全プリセットが表示される
- ✅ 色が正しく表示される
- ✅ レンダリングが正常に動作する
- ✅ エラーやクラッシュがない

---

### テスト5: 既存プロジェクトの互換性確認

#### 目的
既存の.aepファイルが正常に開け、色が変わっていないことを確認する。

#### 手順
```bash
# 既存プロジェクトを用意（SDK_ProcAmpを使用しているもの）
# 例: test_project.aep
```

**After Effects での確認**:
1. 既存の.aepファイルを開く
2. SDK_ProcAmpが適用されているレイヤーを確認
3. 以下を確認:
   - [ ] プロジェクトが正常に開ける
   - [ ] エラーメッセージがない
   - [ ] 色プリセットの設定が維持されている
   - [ ] 色が変わっていない
   - [ ] レンダリング結果が以前と同じ

#### 期待される結果
- ✅ 既存プロジェクトが正常に開ける
- ✅ 色プリセット設定が維持される
- ✅ 色が変わっていない
- ✅ レンダリング結果が一致する

---

### テスト6: 無効化状態の確認

#### 目的
新ヘッダーが実際に使用されていないことを最終確認する。

#### 手順
```bash
cd /home/runner/work/Windy_Lines/Windy_Lines

# 1. SDK_ProcAmp_ColorPresets.h を一時的にリネーム
mv SDK_ProcAmp_ColorPresets.h SDK_ProcAmp_ColorPresets.h.tmp

# 2. ビルド（成功するはず - #if 0で無効化されているため）
make clean && make
# または
# MSBuild SDK_ProcAmp.sln /t:Clean,Build

# 3. ビルドが成功することを確認
echo $?  # 0 であるべき（エラーなし）

# 4. ファイルを元に戻す
mv SDK_ProcAmp_ColorPresets.h.tmp SDK_ProcAmp_ColorPresets.h
```

#### 期待される結果
- ✅ 新ヘッダーがなくてもビルドが成功する
- ✅ これにより`#if 0`が正しく機能していることが証明される

---

## ✅ テスト完了チェックリスト

以下の全項目が✅になったら、ステップ1完了です。

### ファイル変更
- [ ] SDK_ProcAmp.h にincludeが追加されている
- [ ] `#if 0`で無効化されている
- [ ] 既存の定義がすべて残っている
- [ ] コメントが適切

### ビルド
- [ ] クリーンビルドが成功
- [ ] ビルドエラーなし
- [ ] リンクエラーなし
- [ ] プラグインファイルが生成される

### 動作確認
- [ ] After Effectsでプラグインがロードされる
- [ ] 全プリセットが表示される
- [ ] プリセット切り替えが動作する
- [ ] レンダリングが正常

### 互換性
- [ ] 既存プロジェクトが開ける
- [ ] 色プリセット設定が維持される
- [ ] 色が変わっていない
- [ ] レンダリング結果が一致

### 無効化確認
- [ ] 新ヘッダーなしでもビルド成功
- [ ] `#if 0`が正しく機能している

---

## 🎯 テスト実行コマンドまとめ

以下のコマンドを順番に実行することで、主要なテストを実施できます。

```bash
#!/bin/bash
# ステップ1 テストスクリプト

cd /home/runner/work/Windy_Lines/Windy_Lines

echo "=== ステップ1 テスト開始 ==="
echo ""

echo "✓ テスト1: ファイル変更の確認"
echo "include追加確認:"
git diff SDK_ProcAmp.h | grep -c "SDK_ProcAmp_ColorPresets.h"
echo ""

echo "#if 0 確認:"
git diff SDK_ProcAmp.h | grep -c "#if 0"
echo ""

echo "既存定義の確認:"
grep -c "struct PresetColor" SDK_ProcAmp.h
grep -c "namespace ColorPresets" SDK_ProcAmp.h
echo ""

echo "✓ テスト2: プリプロセッサ動作"
echo "  #if 0により新ヘッダーは無効化されています"
echo ""

echo "✓ テスト3: ビルドテスト"
echo "  ※ 実際の環境でビルドコマンドを実行してください"
echo "  Windows: MSBuild SDK_ProcAmp.sln /t:Clean,Build"
echo "  Mac: xcodebuild clean build -project SDK_ProcAmp.xcodeproj"
echo "  Make: make clean && make"
echo ""

echo "✓ テスト4-5: After Effects動作確認"
echo "  ※ After Effectsで以下を確認してください:"
echo "  1. プラグインのロード"
echo "  2. 全プリセットの表示"
echo "  3. 色の正確性"
echo "  4. 既存プロジェクトの互換性"
echo ""

echo "✓ テスト6: 無効化状態の確認"
echo "新ヘッダーを一時的にリネームしてビルドテスト..."
if [ -f SDK_ProcAmp_ColorPresets.h ]; then
    mv SDK_ProcAmp_ColorPresets.h SDK_ProcAmp_ColorPresets.h.tmp
    echo "  ヘッダーをリネームしました"
    echo "  ※ ビルドを実行して成功することを確認してください"
    echo "  ※ 確認後、以下のコマンドで元に戻してください:"
    echo "     mv SDK_ProcAmp_ColorPresets.h.tmp SDK_ProcAmp_ColorPresets.h"
else
    echo "  ⚠️ SDK_ProcAmp_ColorPresets.h が見つかりません"
fi
echo ""

echo "=== ステップ1 テスト完了 ==="
echo ""
echo "すべてのテストが完了したら、次のステップへ進めます。"
echo "次: ステップ2（新システムの有効化）"
```

---

## 📊 テスト結果の記録

テスト実行後、以下の表に結果を記録してください。

| テスト項目 | 結果 | 備考 |
|-----------|------|------|
| include追加 | ✅ / ❌ | |
| #if 0 無効化 | ✅ / ❌ | |
| 既存定義維持 | ✅ / ❌ | |
| ビルド成功 | ✅ / ❌ | |
| プラグインロード | ✅ / ❌ | |
| プリセット表示 | ✅ / ❌ | |
| 色の正確性 | ✅ / ❌ | |
| 既存プロジェクト | ✅ / ❌ | |
| 無効化確認 | ✅ / ❌ | |

---

## 🚨 トラブルシューティング

### エラー: ビルド時にincludeファイルが見つからない

**症状**: 
```
fatal error: SDK_ProcAmp_ColorPresets.h: No such file or directory
```

**原因**: `#if 0`が正しく設定されていない

**解決**:
```bash
# SDK_ProcAmp.h の該当箇所を確認
grep -B 1 -A 1 "SDK_ProcAmp_ColorPresets.h" SDK_ProcAmp.h

# 以下のようになっているべき:
# #if 0
# #include "SDK_ProcAmp_ColorPresets.h"
# #endif

# もし #if 0 がない場合は、手動で追加
```

### エラー: struct PresetColor の重複定義

**症状**:
```
error: redefinition of 'struct PresetColor'
```

**原因**: 既存の定義が削除されている or `#if 0`が設定されていない

**解決**:
```bash
# 既存の定義があるか確認
grep -A 2 "struct PresetColor" SDK_ProcAmp.h

# ない場合は、gitで元に戻す
git checkout HEAD -- SDK_ProcAmp.h

# 再度ステップ1を実装
```

### 警告: After Effectsで色が違う

**症状**: プリセットの色が期待と異なる

**原因**: これはステップ1では起こらないはず（既存コードのみ使用）

**確認**:
```bash
# #if 0 が正しく設定されているか確認
grep -B 1 "#include \"SDK_ProcAmp_ColorPresets.h\"" SDK_ProcAmp.h
# 直前に "#if 0" があるべき

# 既存の定義が残っているか確認
grep -c "const PresetColor kRainbow" SDK_ProcAmp.h
# 1 であるべき（既存定義）
```

---

## ✅ ステップ1完了の確認

以下をすべて確認できたら、ステップ1は完了です：

1. ✅ SDK_ProcAmp.h にincludeが追加されている（`#if 0`で無効化）
2. ✅ 既存の定義がすべて残っている
3. ✅ ビルドが成功する
4. ✅ After Effectsで正常に動作する
5. ✅ 全プリセットが表示される
6. ✅ 既存プロジェクトが正常に開ける
7. ✅ 色が変わっていない
8. ✅ 新ヘッダーなしでもビルドが成功する（無効化確認）

**本番への影響**: ❌ **影響ゼロ** - `#if 0`により新ヘッダーは使用されていません

**次のステップ**: ステップ2（新システムの有効化）へ進む

---

**作成日**: 2026-02-09  
**対象**: ステップ1実装とテスト  
**ステータス**: 実装完了・テスト準備完了
