# 新しい色プリセットの追加方法

このガイドでは、新しい色プリセットを追加する正しい方法を説明します。

## 📋 手順（3ステップ）

### 1. TSVファイルに新しいプリセットを追加

`color_presets.tsv`ファイルを開いて、最後に新しい行を追加します。

**重要な注意事項**:
- **タブ文字**で区切る必要があります（スペースではありません）
- 各フィールドは **タブ文字 (`\t`)** で区切ります
- ファイルは **改行で終わる** 必要があります

**フォーマット**:
```
ID[TAB]日本語名[TAB]英語名[TAB]色1[TAB]色2[TAB]...[TAB]色8
```

**例**: 「春の桜」プリセットを追加する場合
```tsv
35	春の桜	SpringSakura	255,255,192,203	255,255,182,193	255,255,218,185	255,255,182,193	255,255,192,203	255,255,182,193	255,255,218,185	255,238,203,173
```

**色のフォーマット**: `A,R,G,B` （Alpha, Red, Green, Blue）
- 各値は 0-255 の範囲
- カンマで区切る
- 各プリセットには **8色** が必要

### 2. 変換スクリプトを実行

ターミナルで以下のコマンドを実行します：

```bash
python color_preset_converter.py
```

**出力例**:
```
Reading color presets from: color_presets.tsv
Parsed 35 color presets
✓ Generated: OST_WindyLines_ColorPresets.h
✓ Total color presets: 35
✓ Total colors: 280

Preset Summary:
  [1] レインボー (Rainbow)
  ...
  [35] 春の桜 (SpringSakura)
```

このスクリプトが自動的に以下を生成します：
- `enum ColorPreset` に `COLOR_PRESET_SPRINGSAKURA` が追加
- `namespace ColorPresets` に `kSpringSakura[8]` 配列が追加  
- `GetPresetPalette()` 関数に `case 35` が追加

### 3. プラグインをビルド

プロジェクトをリビルドします：

**Windows**:
```bash
MSBuild OST_WindyLines.sln /t:Clean,Build /p:Configuration=Debug
```

**Mac**:
```bash
xcodebuild clean build -project OST_WindyLines.xcodeproj -configuration Debug
```

**Make**:
```bash
make clean && make
```

### 4. After Effectsで確認

1. 新しくビルドされたプラグインをAfter Effectsのプラグインフォルダにコピー
2. After Effectsを再起動
3. OST_WindyLinesエフェクトを適用
4. 色プリセット選択UIで新しいプリセット「春の桜」を確認
5. 選択して表示を確認

---

## ⚠️ よくある問題と解決方法

### 問題1: 新しいプリセットが表示されない

**症状**: TSVに追加して`python color_preset_converter.py`を実行したが、プリセットが表示されない

**原因**: TSVファイルが改行で終わっていない

**解決方法**:

#### オプションA: Pythonで修正（推奨）
```bash
python3 << 'EOF'
with open('color_presets.tsv', 'r', encoding='utf-8') as f:
    content = f.read()
if not content.endswith('\n'):
    content += '\n'
with open('color_presets.tsv', 'w', encoding='utf-8') as f:
    f.write(content)
print("Fixed!")
EOF
```

#### オプションB: テキストエディタで修正
1. VSCodeやその他のエディタで`color_presets.tsv`を開く
2. ファイルの最後の行の後にカーソルを置く
3. Enterキーを押して新しい空行を追加
4. ファイルを保存

#### オプションC: コマンドラインで修正
```bash
# 末尾に改行を追加
echo >> color_presets.tsv
```

**修正後**: 再度`python color_preset_converter.py`を実行してください

### 問題2: タブ文字の代わりにスペースを使用

**症状**: 変換エラーが発生する

**解決方法**:
- エディタでタブ文字を表示する設定を有効にする
- VSCode: "View" > "Render Whitespace"を有効化
- タブ文字が正しく使われているか確認

### 問題3: 色の値が範囲外

**症状**: エラーメッセージ "Color values must be 0-255"

**解決方法**:
- すべての色の値が0-255の範囲にあることを確認
- ARGBフォーマット（Alpha, Red, Green, Blue）を使用

### 問題4: 色の数が8色ではない

**症状**: エラーメッセージ "Missing column in TSV"

**解決方法**:
- 各プリセットには正確に8色が必要です
- `color1`, `color2`, ..., `color8`のすべての列があることを確認

---

## 💡 ベストプラクティス

### 1. エディタの設定

**VSCode**:
```json
{
  "files.insertFinalNewline": true,
  "files.trimTrailingWhitespace": false,
  "[tsv]": {
    "editor.insertSpaces": false,
    "editor.tabSize": 4
  }
}
```

この設定により：
- ファイルの最後に自動的に改行が追加される
- タブ文字が保持される

### 2. TSVの検証

追加後、以下のコマンドで検証できます：

```bash
# 行数を確認（ヘッダー1行 + データ行）
wc -l color_presets.tsv

# 最後の行を確認
tail -1 color_presets.tsv

# 末尾が改行で終わっているか確認
tail -c 1 color_presets.tsv | od -An -tx1
# 出力が " 0a" なら OK（改行 = 0x0a）
```

### 3. 変換の確認

```bash
# 変換を実行
python color_preset_converter.py

# 生成されたヘッダーで新しいプリセットを確認
grep "COLOR_PRESET_SPRINGSAKURA" OST_WindyLines_ColorPresets.h
grep "kSpringSakura" OST_WindyLines_ColorPresets.h
```

---

## 📝 テンプレート

新しいプリセットを追加する際は、このテンプレートを使用してください：

```tsv
[次のID]	[日本語名]	[英語名（PascalCase）]	255,R1,G1,B1	255,R2,G2,B2	255,R3,G3,B3	255,R4,G4,B4	255,R5,G5,B5	255,R6,G6,B6	255,R7,G7,B7	255,R8,G8,B8
```

**注意**:
- `[次のID]`: 現在の最大ID + 1
- `[日本語名]`: UIに表示される日本語の名前
- `[英語名（PascalCase）]`: 変数名として使用される英語名（例: `SpringSakura`）
- 各色は `255,R,G,B` 形式（Alphaは通常255）

---

## 🎨 色の選び方のヒント

### グラデーション
連続した8色でスムーズなグラデーションを作成：
```tsv
35	青緑グラデ	BlueGreen	255,0,255,255	255,0,238,221	255,0,221,204	255,0,204,170	255,0,187,153	255,0,170,136	255,0,153,119	255,0,136,102
```

### コントラスト
対照的な色を交互に：
```tsv
36	炎と氷	FireAndIce	255,255,69,0	255,0,191,255	255,255,140,0	255,135,206,250	255,255,0,0	255,173,216,230	255,220,20,60	255,176,224,230
```

### テーマ
特定のテーマやムードに基づいて：
```tsv
37	秋の森	AutumnForest	255,139,69,19	255,160,82,45	255,210,105,30	255,244,164,96	255,222,184,135	255,245,222,179	255,255,228,181	255,255,239,213
```

---

## 🔧 トラブルシューティング

### デバッグ手順

1. **TSVファイルを確認**:
   ```bash
   cat -A color_presets.tsv | tail -3
   ```
   - タブは `^I` として表示されます
   - 改行は `$` として表示されます

2. **変換スクリプトを実行**:
   ```bash
   python color_preset_converter.py 2>&1 | tee conversion.log
   ```

3. **生成されたヘッダーを確認**:
   ```bash
   grep -n "COLOR_PRESET_" OST_WindyLines_ColorPresets.h | tail -5
   ```

4. **ビルドログを確認**:
   - コンパイルエラーがないか確認
   - 警告メッセージに注意

5. **After Effectsのログを確認**:
   - プラグインが正しくロードされているか確認

---

## 📞 サポート

問題が解決しない場合：

1. 上記のトラブルシューティング手順を試してください
2. エラーメッセージの全文を記録してください
3. 使用しているツールとバージョンを確認してください:
   - Python: `python --version`
   - エディタとバージョン
   - OS

---

**最終更新**: 2026-02-09  
**対象バージョン**: 完全自動化システム（ecf51b1以降）
