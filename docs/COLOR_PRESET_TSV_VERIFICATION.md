# 配色カラープリセットTSV対応 - 検証レポート

**日付**: 2026-02-09  
**ステータス**: ✅ 実装可能（検証完了）

---

## 📋 目的

エフェクトプリセットと同様に、配色カラープリセットもTSV、Pythonなどを使って編集・自動適用できるようにする実装の可能性を検証する。

---

## ✅ 検証結果サマリー

**結論**: 配色カラープリセットのTSV管理は**完全に実装可能**です。

既存のエフェクトプリセットシステムと同様の仕組みで、以下が実現できます：
1. TSVファイルで色プリセットを管理
2. Pythonスクリプトで自動的にC++ヘッダーファイルを生成
3. ビルドプロセスへの統合
4. プリセット名・色情報の一元管理

---

## 🔍 既存システムの分析

### 1. エフェクトプリセットの現在の実装

#### ファイル構成
```
presets.tsv                    # エフェクトプリセットのデータ（TSV形式）
preset_converter.py            # TSV → C++変換スクリプト
OST_WindyLines_Presets.h         # 自動生成されたプリセット定義（C++）
OST_WindyLines.h                 # プリセット利用側のメインヘッダー
```

#### 動作フロー
```
1. presets.tsv を編集
   ↓
2. preset_converter.py を実行
   ↓
3. OST_WindyLines_Presets.h が自動生成
   ↓
4. ビルド時に自動的に反映
```

### 2. 色プリセットの現在の実装

#### 現在の問題点
色プリセットは現在、`OST_WindyLines.h` 内に**ハードコーディング**されています：

**場所**: `OST_WindyLines.h` 行542-708

**構造**:
```cpp
// 色の構造体定義
struct PresetColor {
    unsigned char a, r, g, b;
};

// プリセット定義（33種類 × 8色）
namespace ColorPresets {
    const PresetColor kRainbow[8] = {
        {255, 255, 0, 0}, {255, 255, 128, 0}, ...
    };
    const PresetColor kPastel[8] = { ... };
    // ... 33個のプリセット
}

// プリセット選択関数
inline const PresetColor* GetPresetPalette(int presetIndex) {
    switch (presetIndex) {
        case COLOR_PRESET_RAINBOW: return ColorPresets::kRainbow;
        // ... 33 cases
    }
}
```

**プリセット一覧** (33種類):
1. Rainbow (レインボー)
2. Rainbow Pastel (パステルレインボー)
3. Forest (森)
4. Cyber (サイバー)
5. Hazard (警告)
6. Sakura (桜)
7. Desert (砂漠)
8. Star Dust (星屑)
9. Wakaba (若葉)
10. Danger Zone (危険地帯)
11. Yoen (妖艶)
12. Sokai (爽快)
13. Dreamy Wind (夢幻の風)
14. Sunset (夕焼け)
15. Ocean (海)
16. Autumn (秋)
17. Snow (雪)
18. Deep Sea (深海)
19. Morning Dew (朝露)
20. Night Sky (夜空)
21. Flame (炎)
22. Earth (大地)
23. Jewel (宝石)
24. Pastel 2 (パステル2)
25. City Night (夜の街)
26. Moonlight (月光)
27. Dazzling Light (眩光)
28. Neon Blast (ネオンブラスト)
29. Toxic Swamp (毒沼)
30. Cosmic Storm (宇宙嵐)
31. Lava Flow (溶岩流)
32. Gold (金)
33. Monochrome (モノクロ)

各プリセットは**8色**で構成されています。

---

## 💡 提案実装プラン

### アーキテクチャ概要

エフェクトプリセットと完全に並列な構造を採用：

```
color_presets.tsv              # NEW: 色プリセットデータ（TSV形式）
color_preset_converter.py      # NEW: TSV → C++変換スクリプト
OST_WindyLines_ColorPresets.h     # NEW: 自動生成される色プリセット定義
OST_WindyLines.h                  # MODIFY: 既存のハードコード削除、新ヘッダーをinclude
```

---

## 📝 詳細設計

### 1. TSVファイル設計: `color_presets.tsv`

#### フォーマット
```tsv
id	name	name_en	color1	color2	color3	color4	color5	color6	color7	color8
1	レインボー	Rainbow	255,255,0,0	255,255,128,0	255,255,255,0	255,0,255,0	255,0,0,255	255,74,0,130	255,140,0,255	255,255,0,255
2	パステルレインボー	Rainbow Pastel	255,255,178,178	255,255,217,178	255,255,255,178	255,178,255,178	255,178,178,255	255,217,178,255	255,255,178,255	255,255,204,255
...
```

#### 列の説明
- `id`: 色プリセットID（1-33、`COLOR_PRESET_XXX` enumに対応）
- `name`: 日本語名
- `name_en`: 英語名（C++識別子用）
- `color1-color8`: 各色の ARGB 値（カンマ区切り: `a,r,g,b`）

#### 利点
- **視認性**: エクセル等で編集可能
- **バージョン管理**: git diff で変更点が明確
- **拡張性**: 新しいプリセットの追加が容易
- **メンテナンス性**: 色の微調整が簡単
- **ドキュメント**: TSV自体がプリセット一覧のドキュメントとなる

---

### 2. 変換スクリプト: `color_preset_converter.py`

#### 機能
`preset_converter.py` と同様の構造で実装：

```python
#!/usr/bin/env python3
"""
Color Preset Converter: TSV to C++ Color Preset Arrays
Usage: python color_preset_converter.py [color_presets.tsv]
Output: OST_WindyLines_ColorPresets.h
"""

def parse_color_tsv(filepath):
    """TSVファイルをパースして色プリセットリストを返す"""
    # TSV読み込み
    # 各行を辞書形式で返す
    
def parse_argb(color_str):
    """'255,255,0,0' → (255, 255, 0, 0) に変換"""
    # カンマ区切りをパース
    
def format_preset_cpp(preset):
    """プリセット辞書をC++配列初期化子に変換"""
    # 例:
    # const PresetColor kRainbow[8] = {
    #     {255, 255, 0, 0}, {255, 255, 128, 0}, ...
    # };
    
def generate_lookup_function(presets):
    """GetPresetPalette() 関数を生成"""
    # switch-case文を自動生成
    
def generate_cpp_header(presets):
    """完全なヘッダーファイルを生成"""
    # Header guard
    # struct PresetColor定義
    # namespace ColorPresets { ... }
    # GetPresetPalette()関数
```

#### 出力例
```cpp
// Auto-generated by color_preset_converter.py - DO NOT EDIT MANUALLY
// Edit color_presets.tsv and run color_preset_converter.py to regenerate

#ifndef OST_WINDYLINES_COLOR_PRESETS_H
#define OST_WINDYLINES_COLOR_PRESETS_H

struct PresetColor {
    unsigned char a, r, g, b;
};

namespace ColorPresets {
    // Rainbow (レインボー)
    const PresetColor kRainbow[8] = {
        {255, 255, 0, 0}, {255, 255, 128, 0}, ...
    };
    
    // ... 全33プリセット
}

// Preset color lookup table
inline const PresetColor* GetPresetPalette(int presetIndex) {
    switch (presetIndex) {
        case 1: return ColorPresets::kRainbow;        // COLOR_PRESET_RAINBOW
        case 2: return ColorPresets::kPastel;         // COLOR_PRESET_RAINBOW_PASTEL
        // ... 全33ケース
        default: return ColorPresets::kRainbow;
    }
}

#endif // OST_WINDYLINES_COLOR_PRESETS_H
```

---

### 3. ヘッダー統合: `OST_WindyLines.h` の修正

#### Before（現在）
```cpp
// OST_WindyLines.h

struct PresetColor {
    unsigned char a, r, g, b;
};

namespace ColorPresets {
    const PresetColor kRainbow[8] = { ... };  // 200行以上のハードコード
    // ...
}

inline const PresetColor* GetPresetPalette(int presetIndex) { ... }
```

#### After（提案）
```cpp
// OST_WindyLines.h

// 色プリセット定義（自動生成）
#include "OST_WindyLines_ColorPresets.h"

// 残りのコードはそのまま
```

#### 変更箇所
- **削除**: 行537-748（約211行）
- **追加**: `#include "OST_WindyLines_ColorPresets.h"` 1行
- **影響**: 既存コードとの互換性100%維持

---

## 🔄 ワークフロー設計

### 開発者の作業手順

#### 1. 新しい色プリセットを追加する場合

```bash
# 1. TSVファイルを編集
vim color_presets.tsv
# 新しい行を追加: ID 34, 名前, 8色の値

# 2. 変換スクリプトを実行
python color_preset_converter.py

# 3. 自動的に OST_WindyLines_ColorPresets.h が更新される

# 4. OST_WindyLines.h の enum ColorPreset に新しい定数を追加
# enum ColorPreset {
#     ...
#     COLOR_PRESET_NEW_THEME = 34,  // 追加
#     COLOR_PRESET_COUNT
# };

# 5. ビルド
# プロジェクトをビルドすれば自動反映
```

#### 2. 既存プリセットの色を変更する場合

```bash
# 1. TSVファイルで対象プリセットの色を編集
vim color_presets.tsv

# 2. 変換スクリプトを実行
python color_preset_converter.py

# 3. ビルド（完了！）
```

### 自動化オプション

#### Git Pre-commit Hook
TSVが変更されたら自動的にヘッダーを再生成：

```bash
#!/bin/bash
# .git/hooks/pre-commit

if git diff --cached --name-only | grep -q "color_presets.tsv"; then
    echo "color_presets.tsv changed, regenerating header..."
    python color_preset_converter.py
    git add OST_WindyLines_ColorPresets.h
fi
```

#### ビルドスクリプト統合
CMake/Makefileに統合して、TSVが更新されたら自動再生成：

```cmake
# CMakeLists.txt (例)
add_custom_command(
    OUTPUT OST_WindyLines_ColorPresets.h
    COMMAND python color_preset_converter.py
    DEPENDS color_presets.tsv
    COMMENT "Generating color presets header from TSV"
)
```

---

## 📊 実装難易度と作業見積もり

### 難易度: ⭐⭐☆☆☆ (簡単)

理由：
- 既存の `preset_converter.py` をベースにできる
- アーキテクチャは既に確立されている
- 影響範囲が明確（3ファイル）

### 作業見積もり

| タスク | 難易度 | 時間 |
|--------|--------|------|
| color_presets.tsv作成（33プリセット分のデータ入力） | ⭐⭐ | 1-2時間 |
| color_preset_converter.py実装 | ⭐⭐ | 2-3時間 |
| OST_WindyLines.h修正（ハードコード削除、include追加） | ⭐ | 30分 |
| 動作確認・テスト | ⭐⭐ | 1-2時間 |
| ドキュメント作成 | ⭐ | 1時間 |
| **合計** | | **6-9時間** |

---

## 🎯 メリット

### 1. メンテナンス性の向上
- 色の編集がTSVファイルで完結
- コードを触らずにデザイナーでも編集可能
- エクセル等でビジュアル編集可能

### 2. バージョン管理の改善
- git diff で色の変更が明確に見える
- コンフリクトが起きにくい
- 変更履歴の追跡が容易

### 3. 拡張性
- 新しいプリセットの追加が容易
- プリセット数の変更に柔軟に対応
- 将来的な機能追加（プリセットのカテゴリ分けなど）が容易

### 4. 一貫性
- エフェクトプリセットと同じワークフロー
- 開発者の学習コストゼロ
- コードベースの統一性向上

### 5. 品質向上
- 手作業のコーディングミスを削減
- 自動生成により一貫したフォーマット
- TSVでのバリデーションが可能

---

## ⚠️ 注意事項

### 1. 互換性の維持
- 既存の `COLOR_PRESET_XXX` enum値は変更しない
- プリセットIDの順序を変更しない
- 既存プロジェクトのプリセット選択が壊れないようにする

### 2. ビルドプロセス
- TSV変更後は必ず `color_preset_converter.py` を実行
- CI/CDで自動チェックを追加推奨
- ビルド前にヘッダーが最新か確認

### 3. 色の値の範囲
- ARGB値は0-255の範囲を厳守
- TSVパース時にバリデーション追加推奨

---

## 📖 参考実装例

### TSVからのデータ抽出（Python）

```python
import csv

def parse_tsv(filepath):
    """Parse color preset TSV file"""
    presets = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            preset = {
                'id': int(row['id']),
                'name': row['name'],
                'name_en': row['name_en'],
                'colors': []
            }
            for i in range(1, 9):
                color_str = row[f'color{i}']
                a, r, g, b = map(int, color_str.split(','))
                preset['colors'].append((a, r, g, b))
            presets.append(preset)
    return presets
```

### C++配列生成

```python
def format_preset_cpp(preset):
    """Format preset as C++ array initializer"""
    name_en = preset['name_en'].replace(' ', '')
    name_jp = preset['name']
    
    cpp = f'\t// {name_jp} ({name_en})\n'
    cpp += f'\tconst PresetColor k{name_en}[8] = {{\n'
    
    colors = []
    for a, r, g, b in preset['colors']:
        colors.append(f'{{{a}, {r}, {g}, {b}}}')
    
    # 4 colors per line
    for i in range(0, 8, 4):
        line = ', '.join(colors[i:i+4])
        cpp += f'\t\t{line}'
        if i + 4 < 8:
            cpp += ',\n'
        else:
            cpp += '\n'
    
    cpp += '\t};\n'
    return cpp
```

---

## 🚀 次のステップ（実装フェーズ）

### Phase 1: 基本実装（優先度：高）
1. ✅ **検証完了** - 実装可能性の確認
2. ⬜ 現在のハードコードされた色プリセットをTSVに抽出
3. ⬜ `color_preset_converter.py` の実装
4. ⬜ `OST_WindyLines.h` の修正（include追加）

### Phase 2: 統合とテスト（優先度：高）
5. ⬜ 既存のビルドプロセスでのテスト
6. ⬜ 全33プリセットの視覚的確認
7. ⬜ エフェクトプリセットとの連携テスト

### Phase 3: 自動化（優先度：中）
8. ⬜ Git pre-commit hook 追加
9. ⬜ CI/CDへの統合
10. ⬜ ビルドスクリプトへの統合

### Phase 4: ドキュメント（優先度：中）
11. ⬜ README更新（色プリセット編集方法）
12. ⬜ 開発者ガイド更新
13. ⬜ サンプルTSVとコメント追加

---

## 📚 関連ファイル

### 既存ファイル（参考用）
- `presets.tsv` - エフェクトプリセットのTSV実装例
- `preset_converter.py` - TSV→C++変換の実装例
- `OST_WindyLines_Presets.h` - 自動生成ヘッダーの例

### 新規作成予定ファイル
- `color_presets.tsv` - 色プリセットデータ
- `color_preset_converter.py` - 色プリセット変換スクリプト
- `OST_WindyLines_ColorPresets.h` - 自動生成される色プリセットヘッダー

### 修正予定ファイル
- `OST_WindyLines.h` - ハードコード削除、include追加

---

## ✅ 検証結論

**配色カラープリセットのTSV対応は完全に実装可能です。**

既存のエフェクトプリセットシステムと同じアーキテクチャを採用することで：
- 低リスク・低工数で実装可能
- 高いメンテナンス性を実現
- 将来の拡張に対応可能
- 開発者の学習コストゼロ

**推奨**: 即座に実装フェーズに移行可能です。

---

## 📝 補足: 初期TSVデータの作成方法

### 自動抽出スクリプト

現在の `OST_WindyLines.h` から色データを自動抽出するスクリプトも作成可能：

```python
# extract_color_presets.py
"""
Extract existing color presets from OST_WindyLines.h and generate TSV
"""

import re

def extract_presets_from_header(header_path):
    """Parse OST_WindyLines.h and extract color preset definitions"""
    # 正規表現でプリセット定義を抽出
    # const PresetColor kXXX[8] = { ... };
    
def write_tsv(presets, output_path):
    """Write presets to TSV file"""
    # TSV形式で出力
```

このスクリプトにより、手作業でのデータ入力を不要にできます。

---

**作成者**: GitHub Copilot  
**レビュー**: ローカルエージェントによる実装前の最終確認推奨
