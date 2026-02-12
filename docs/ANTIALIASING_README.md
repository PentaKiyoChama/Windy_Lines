# Anti-Aliasing Verification - Investigation Summary

## Quick Answer / 簡単な答え

**Question:** When increasing the anti-aliasing parameter, lines appear thicker. Is this expected?

**質問:** アンチエイリアスパラメータを増やすと、線が太く見える。これは期待される動作か？

**Answer / 答え:** ✓ **YES, this is expected behavior. No code changes needed.**

**はい、これは期待される動作です。コード変更は不要です。**

---

## Documentation / ドキュメント

### For Users / ユーザー向け

**Japanese / 日本語:**
- 📄 **[アンチエイリアスと線の太さについて.md](./アンチエイリアスと線の太さについて.md)**
  - わかりやすい日本語の説明
  - 推奨設定値
  - なぜこれが正常なのか

**English:**
- 📄 **[ANTIALIASING_ANALYSIS.md](./ANTIALIASING_ANALYSIS.md)**
  - Comprehensive technical documentation (bilingual)
  - Detailed explanations with examples
  - User recommendations

### For Developers / 開発者向け

- 🔧 **[OST_WindyLines_Notes.json](./OST_WindyLines_Notes.json)**
  - See section: `ANTIALIASING_VISUAL_THICKNESS_EFFECT`
  - Technical implementation details
  - Code locations
  - Troubleshooting guide

- 🐍 **[verify_antialiasing.py](./verify_antialiasing.py)**
  - Mathematical verification script
  - Run: `python3 verify_antialiasing.py`
  - Produces numerical analysis tables

---

## Quick Summary / 概要

### Why Lines Appear Thicker / なぜ線が太く見えるか

**Technical reason / 技術的理由:**
1. Core line thickness (`halfThick`) **never changes**
2. Anti-aliasing adds a **fade-out zone** of width `aa` pixels
3. Human vision **integrates semi-transparent pixels**
4. Result: Lines appear **visually thicker**

**日本語:**
1. 線の核となる太さ（`halfThick`）は**変わらない**
2. アンチエイリアスは幅`aa`ピクセルの**フェードアウト領域**を追加
3. 人間の視覚は**半透明のピクセルを統合**
4. 結果: 線が**視覚的に太く見える**

### Numerical Example / 数値例

For 10-pixel line (halfThick = 5.0):

| aa value | Visual Width | Increase |
|----------|--------------|----------|
| 0.0      | 10.00 px     | 0%       |
| 1.0      | 11.00 px     | 10%      |
| 2.0      | 12.00 px     | 20%      |
| 5.0      | 15.00 px     | 50%      |

---

## User Recommendations / 推奨設定

| Use Case | Recommended aa | Notes |
|----------|---------------|-------|
| **Default** | 1.0 | Best balance for most cases |
| **Thin lines** | 0.5 - 1.0 | When size precision matters |
| **Normal use** | 1.0 - 2.0 | Good smoothness |
| **Artistic** | 3.0 - 5.0 | Maximum smoothness |
| **No AA** | 0.0 | Sharp edges (may appear jagged) |

| 用途 | 推奨aa値 | 備考 |
|------|---------|------|
| **デフォルト** | 1.0 | ほとんどの場合に最適 |
| **細い線** | 0.5 - 1.0 | サイズの正確性が重要 |
| **通常の使用** | 1.0 - 2.0 | 適度な滑らかさ |
| **アーティスティック** | 3.0 - 5.0 | 最大限の滑らかさ |
| **AAなし** | 0.0 | シャープ（ジャギーあり） |

---

## Why This is Normal / なぜこれが正常か

### Universal Graphics Principle / 普遍的なグラフィックスの原理

**All anti-aliasing systems have this characteristic:**
- OpenGL MSAA
- Font rendering (FreeType, DirectWrite)
- Image scaling (Bicubic, Lanczos)

**すべてのアンチエイリアスシステムがこの特性を持つ:**
- OpenGL MSAA
- フォントレンダリング（FreeType、DirectWrite）
- 画像スケーリング（Bicubic、Lanczos）

### The Trade-off / トレードオフ

```
Sharp Edges          ←→          Smooth Edges
シャープな縁         ←→          滑らかな縁

Precise Size                     Visual Thickness
正確なサイズ                      視覚的な太さ

Jagged                           Smooth
ジャギー                         滑らか
```

**This cannot be eliminated. Users choose via `aa` parameter.**

**これは排除できない。ユーザーは`aa`パラメータで選択する。**

---

## Code Implementation / コード実装

### Algorithm / アルゴリズム

```cuda
// Smoothstep anti-aliasing
float tt = saturate((dist - aa) / (0.0f - aa));
float coverage = tt * tt * (3.0f - 2.0f * tt);
```

### Locations / 場所

- CUDA: `OST_WindyLines.cu` lines 504-507, 539-542
- OpenCL/Metal: `OST_WindyLines.cl` lines 527-528, 555-556
- CPU: `OST_WindyLines_CPU.cpp` lines 2600-2602, 2625-2627
- Host: `OST_WindyLines_GPU.cpp` line 2015
- Parameters: `OST_WindyLines.h` lines 329-333

---

## Verification / 検証

### Run Verification Script / 検証スクリプトを実行

```bash
cd /path/to/Windy_Lines
python3 verify_antialiasing.py
```

**Output includes:**
- Visual width calculations for different aa values
- Coverage analysis at various distances
- Mathematical proof of expected behavior

**出力内容:**
- 異なるaa値での視覚的な幅の計算
- 様々な距離でのカバレッジ分析
- 期待される動作の数学的証明

---

## Conclusion / 結論

✓ **This is working as designed** / **これは設計通りに動作している**

✓ **No code changes needed** / **コード変更は不要**

✓ **Verified with mathematical proof** / **数学的証明により検証済み**

✓ **Matches industry standards** / **業界標準に適合**

---

## Investigation Date / 調査日

**Date / 日付:** 2026-02-08

**Status / ステータス:** COMPLETED / 完了

**Result / 結果:** Issue resolved through documentation / ドキュメント化により解決

---

## Quick Links / クイックリンク

- [Japanese Summary / 日本語サマリー](./アンチエイリアスと線の太さについて.md)
- [Technical Analysis / 技術分析](./ANTIALIASING_ANALYSIS.md)
- [Verification Script / 検証スクリプト](./verify_antialiasing.py)
- [Implementation Notes / 実装ノート](./OST_WindyLines_Notes.json) (see `ANTIALIASING_VISUAL_THICKNESS_EFFECT`)
