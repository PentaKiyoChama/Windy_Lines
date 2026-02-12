# 配色カラープリセットTSV対応 - 実装ガイド

**対象**: ローカルエージェント / 実装担当者  
**前提**: COLOR_PRESET_TSV_VERIFICATION.md の検証完了

---

## 📋 実装タスク一覧

### Phase 1: 基本実装（必須）

#### Task 1.1: 色プリセットTSVファイルの作成
**ファイル**: `color_presets.tsv`

**アクション**:
1. `OST_WindyLines.h` の行542-708から33個の色プリセット定義を抽出
2. 以下のTSVフォーマットで保存：

```tsv
id	name	name_en	color1	color2	color3	color4	color5	color6	color7	color8
1	レインボー	Rainbow	255,255,0,0	255,255,128,0	...
```

**データソース**: `OST_WindyLines.h` 行542-708
- 各プリセット: `const PresetColor kXXX[8] = { ... };`
- 33個のプリセット × 8色 = 264色のデータ

**手動入力の代替案**:
- `extract_color_presets_POC.py` スクリプトを改良して自動抽出
- または手動でコピー＆ペースト＆整形

**検証**:
```bash
# TSVの構造を確認
head color_presets.tsv
wc -l color_presets.tsv  # 34行（ヘッダー1行 + データ33行）
```

---

#### Task 1.2: 色プリセット変換スクリプトの実装
**ファイル**: `color_preset_converter.py`

**アクション**:
1. `color_preset_converter_POC.py` をベースに本実装を作成
2. 以下の機能を追加：
   - エラーハンドリングの強化
   - カラー値のバリデーション（0-255範囲チェック）
   - プリセットID重複チェック
   - 8色すべてが存在するかチェック

**必須機能**:
```python
def parse_tsv(filepath):
    """TSVをパースしてプリセット辞書のリストを返す"""
    
def validate_preset(preset):
    """プリセットデータの妥当性を検証"""
    # ID範囲チェック (1-33)
    # 色数チェック (8色)
    # ARGB範囲チェック (0-255)
    
def format_preset_cpp(preset):
    """C++配列初期化子を生成"""
    
def generate_lookup_function(presets):
    """GetPresetPalette() switch-case関数を生成"""
    
def generate_cpp_header(presets):
    """完全なヘッダーファイルを生成"""
```

**出力**: `OST_WindyLines_ColorPresets.h`

**検証**:
```bash
python color_preset_converter.py
# 出力確認
cat OST_WindyLines_ColorPresets.h | head -50
```

---

#### Task 1.3: OST_WindyLines.h の修正
**ファイル**: `OST_WindyLines.h`

**アクション**:
1. 行537-748の色プリセット定義を削除（約211行）
   - `struct PresetColor { ... };` から
   - `inline const PresetColor* GetPresetPalette(int presetIndex) { ... }` まで

2. 削除した箇所に以下を追加：
```cpp
// Color presets (auto-generated from color_presets.tsv)
#include "OST_WindyLines_ColorPresets.h"
```

**重要**: 
- `#ifndef __cplusplus` より前に追加（C++コードのみで有効）
- enum定義（`enum ColorPreset`）は**削除しない**（そのまま残す）
- `GetPresetPalette()`関数は新ヘッダーで提供されるので削除

**変更前**:
```cpp
// 行537-748
struct PresetColor { ... };
namespace ColorPresets { ... }
inline const PresetColor* GetPresetPalette(...) { ... }
```

**変更後**:
```cpp
// 色プリセット定義（自動生成）
#include "OST_WindyLines_ColorPresets.h"
```

**検証**:
```bash
# includeが正しく追加されたか確認
grep "OST_WindyLines_ColorPresets.h" OST_WindyLines.h

# 古い定義が削除されたか確認
grep "const PresetColor kRainbow" OST_WindyLines.h  # 見つからないはず
```

---

### Phase 2: ビルドとテスト（必須）

#### Task 2.1: ビルドテスト
**アクション**:
1. プロジェクトをクリーンビルド
2. コンパイルエラーがないことを確認
3. リンクエラーがないことを確認

**確認ポイント**:
- `OST_WindyLines_ColorPresets.h` がincludeパスに含まれているか
- `PresetColor` 構造体の定義が重複していないか
- `GetPresetPalette()` 関数が正しく解決されるか

**コマンド例**:
```bash
# Windows (Visual Studio)
MSBuild OST_WindyLines.sln /t:Clean,Build /p:Configuration=Debug

# Mac (Xcode)
xcodebuild clean build -project OST_WindyLines.xcodeproj -configuration Debug
```

---

#### Task 2.2: 機能テスト
**アクション**:
1. After Effectsで効果を起動
2. 色プリセット選択UIが正常に動作するか確認
3. 全33プリセットを順番に選択して色が正しく表示されるか確認

**テスト項目**:
- [ ] プリセット1: レインボー - 赤、オレンジ、黄、緑、青、藍、紫、マゼンタ
- [ ] プリセット2: パステルレインボー - 淡い虹色
- [ ] ... (33個すべて)

**視覚的確認**:
- スクリーンショットを撮影して、既存の出力と比較
- 色が変わっていないことを確認

---

#### Task 2.3: リグレッションテスト
**アクション**:
1. 既存のプロジェクトファイル（.aep）を開く
2. 色プリセットが変更されていないか確認
3. レンダリング出力が以前と同じか確認

**重要**: 既存ユーザーのプロジェクトが壊れないように！

---

### Phase 3: 自動化（推奨）

#### Task 3.1: Git Pre-commit Hook
**ファイル**: `.git/hooks/pre-commit`

**アクション**:
```bash
#!/bin/bash
# TSVが変更されたら自動的にヘッダーを再生成

if git diff --cached --name-only | grep -q "color_presets.tsv"; then
    echo "🔄 color_presets.tsv changed, regenerating header..."
    python color_preset_converter.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Header regenerated successfully"
        git add OST_WindyLines_ColorPresets.h
    else
        echo "✗ Error regenerating header"
        exit 1
    fi
fi
```

**権限設定**:
```bash
chmod +x .git/hooks/pre-commit
```

---

#### Task 3.2: CI/CD統合
**ファイル**: `.github/workflows/verify-presets.yml` (例)

**アクション**:
```yaml
name: Verify Color Presets

on:
  push:
    paths:
      - 'color_presets.tsv'
  pull_request:
    paths:
      - 'color_presets.tsv'

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.x'
      
      - name: Regenerate header
        run: python color_preset_converter.py
      
      - name: Check if header is up-to-date
        run: |
          if ! git diff --quiet OST_WindyLines_ColorPresets.h; then
            echo "Error: OST_WindyLines_ColorPresets.h is not up-to-date!"
            echo "Please run: python color_preset_converter.py"
            exit 1
          fi
```

---

### Phase 4: ドキュメント（必須）

#### Task 4.1: README更新
**ファイル**: `README.md` または新規 `COLOR_PRESETS_README.md`

**追加内容**:
```markdown
## 色プリセットの編集方法

### 前提条件
- Python 3.x

### 手順

1. `color_presets.tsv` を編集
   - エクセル、Googleスプレッドシート、テキストエディタで編集可能
   - TSV形式（タブ区切り）を維持すること

2. 変換スクリプトを実行
   ```bash
   python color_preset_converter.py
   ```

3. 生成されたヘッダーを確認
   ```bash
   cat OST_WindyLines_ColorPresets.h
   ```

4. ビルドしてテスト

### TSVフォーマット

| 列名 | 説明 | 例 |
|------|------|-----|
| id | プリセットID (1-33) | 1 |
| name | 日本語名 | レインボー |
| name_en | 英語識別子 | Rainbow |
| color1-8 | 色 (a,r,g,b) | 255,255,0,0 |

### 新しいプリセットの追加

1. TSVに新しい行を追加（ID 34以降）
2. OST_WindyLines.h の `enum ColorPreset` に新しい定数を追加
3. 変換スクリプトを実行
4. ビルド
```

---

#### Task 4.2: 開発者ガイド更新
**ファイル**: `OST_WindyLines_DevGuide.md`

**追加セクション**:
```markdown
## 色プリセットシステム

### アーキテクチャ
色プリセットはTSV→C++の自動生成システムで管理されています。

### ファイル構成
- `color_presets.tsv` - マスターデータ（手動編集）
- `color_preset_converter.py` - 変換スクリプト
- `OST_WindyLines_ColorPresets.h` - 自動生成（編集禁止）
- `OST_WindyLines.h` - プリセット利用側

### 編集ワークフロー
（省略 - README参照）
```

---

## ✅ 完了チェックリスト

### Phase 1: 基本実装
- [ ] `color_presets.tsv` 作成完了（33プリセット）
- [ ] `color_preset_converter.py` 実装完了
- [ ] `OST_WindyLines_ColorPresets.h` 生成成功
- [ ] `OST_WindyLines.h` 修正完了（include追加、ハードコード削除）

### Phase 2: ビルドとテスト
- [ ] クリーンビルド成功
- [ ] コンパイルエラーなし
- [ ] リンクエラーなし
- [ ] 全33プリセットの表示確認
- [ ] 既存プロジェクトのリグレッションテスト

### Phase 3: 自動化（オプション）
- [ ] Git pre-commit hook 設定
- [ ] CI/CD統合（該当する場合）

### Phase 4: ドキュメント
- [ ] README更新
- [ ] 開発者ガイド更新
- [ ] TSVファイルにコメント追加（必要に応じて）

---

## 🚨 トラブルシューティング

### ビルドエラー: "PresetColor redefinition"
**原因**: `PresetColor`構造体が重複定義されている

**解決策**: 
- `OST_WindyLines.h`内の古い`PresetColor`定義を削除
- `OST_WindyLines_ColorPresets.h`のみで定義されるようにする

---

### ビルドエラー: "GetPresetPalette undefined reference"
**原因**: ヘッダーがincludeされていない

**解決策**:
- `OST_WindyLines.h`に`#include "OST_WindyLines_ColorPresets.h"`を追加
- includeパスが正しく設定されているか確認

---

### 色が正しく表示されない
**原因**: TSVのARGB値が間違っている

**解決策**:
- TSVファイルの色値を確認（0-255範囲）
- 変換スクリプトを再実行
- リビルド

---

### TSV編集後に変更が反映されない
**原因**: ヘッダーファイルが再生成されていない

**解決策**:
```bash
python color_preset_converter.py
# ビルドシステムのキャッシュをクリア
make clean && make
```

---

## 📞 サポート

質問や問題がある場合:
1. このドキュメントのトラブルシューティングセクションを確認
2. `COLOR_PRESET_TSV_VERIFICATION.md`の検証レポートを参照
3. 既存の`preset_converter.py`実装を参考にする

---

**最終更新**: 2026-02-09  
**作成者**: GitHub Copilot Workspace Agent
