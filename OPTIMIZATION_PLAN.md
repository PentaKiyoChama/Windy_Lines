# SDK_ProcAmp 包括的リファクタリング計画書

## 概要
SDK_ProcAmp Wind Lines Effect v63 の包括的最適化  
**規模:** 約7,000行（CPU:2,405行、GPU:2,473行、CUDA:871行、OpenCL:734行、Header:688行）  
**目標:** CPU軽量化 → 共通化 → 最適化 → 品質改善（4フェーズ・21タスク）

---

## 全体構成

| フェーズ | 目的 | 対象ファイル | 期間目安 | 進捗 |
|---------|------|-------------|---------|------|
| **Phase 1** | CPU軽量化（即効性） | SDK_ProcAmp_CPU.cpp | 1週間 | 🔄 進行中 |
| **Phase 2** | 共通コード抽出 | 全ファイル | 1-2週間 | 📝 未着手 |
| **Phase 3** | パフォーマンス最適化 | CPU.cpp, .cu, .cl | 2週間 | 📝 未着手 |
| **Phase 4** | 品質・保守性改善 | 全ファイル | 1週間 | 📝 未着手 |

---

## Phase 1: CPU軽量化（最優先）

| # | タスク | 対象箇所 | 期待効果 | 状態 |
|---|--------|---------|---------|------|
| **1-1** | 不変値のループ外移動 | Renderループ (L1140-2280) | ~10%高速化 | ✅ 完了 |
| **1-2** | 早期リターン追加 | バウンディングボックス | ~20%高速化 | ✅ 完了 |
| **1-3** | `coverage < 0.001f` スキップ | ブレンド処理前 | ~5%高速化 | ⏳ 次に実施 |
| **1-4** | `halfThick < 0.5f` 極小線スキップ | 線ループ先頭 | 可変 | 📝 予定 |
| **1-5** | モーションブラー最適化 | サンプル数動的調整 | ~15%高速化 | 📝 予定 |
| **1-6** | powf → 乗算展開 | ApplyEasing関数 | イージング50%+ | ✅ 完了 |
| **1-7** | effectiveAA事前計算 | 6箇所の重複削減 | 微小改善 | ✅ 完了 |

**Phase 1 進捗:** 4/7 完了 (57%)

---

## Phase 2: 共通コード抽出

| # | タスク | 重複箇所 | 効果 | 状態 |
|---|--------|---------|------|------|
| **2-1** | SDF計算を共通関数化 | CPU(6箇所), CUDA(3), OpenCL(3) | -300行 | 📝 予定 |
| **2-2** | ブレンド処理を関数化 | 4モード × 3実装 = 12箇所 | 保守性向上 | 📝 予定 |
| **2-3** | ユーティリティ関数を共通ヘッダへ | saturate, DepthScale, smoothstep | 重複削除 | 📝 予定 |
| **2-4** | イージング関数の統一 | CPU:10種 → 28種に拡張 | 動作一致 | 📝 予定 |
| **2-5** | デバッグログ関数の共通化 | CPU, GPU両方 | -50行 | 📝 予定 |

**Phase 2 進捗:** 0/5 完了 (0%)

---

## Phase 3: パフォーマンス最適化

| # | タスク | 詳細 | 効果 | 状態 |
|---|--------|------|------|------|
| **3-1** | イージングLUT導入 | 28種×256サンプル=28KB | ~15%高速化 | 📝 予定 |
| **3-2** | 三角関数LUT導入 | sin/cos 256サンプル=2KB | ~5%高速化 | 📝 予定 |
| **3-3** | smoothstep LUT導入 | 256サンプル=1KB | ~3%高速化 | 📝 予定 |
| **3-4** | SIMD化（SSE/AVX） | SDF計算、色変換 | ~30%高速化 | 📝 予定 |
| **3-5** | メモリレイアウト最適化 | LineDerived → SOA形式 | キャッシュ改善 | 📝 予定 |

**Phase 3 進捗:** 0/5 完了 (0%)

---

## Phase 4: 品質・保守性改善

| # | タスク | 詳細 | 状態 |
|---|--------|------|------|
| **4-1** | マジックナンバー定数化 | 0.05f, 0.6f等に名前付け | 📝 予定 |
| **4-2** | 変数命名改善 | a, t, dを意味ある名前に | 📝 予定 |
| **4-3** | コメント追加 | SDF計算、イージングの数学的説明 | 📝 予定 |
| **4-4** | #ifdef整理 | デバッグコード、プラットフォーム分岐 | 📝 予定 |
| **4-5** | Notes.json更新 | リファクタリング内容を文書化 | 📝 予定 |
| **4-6** | 3実装の最終同期確認 | CPU=CUDA=OpenCL出力一致テスト | 📝 予定 |

**Phase 4 進捗:** 0/6 完了 (0%)

---

## 全体進捗サマリー

- **完了:** 4タスク (19%)
- **進行中:** Phase 1
- **累計期待改善:** Phase 1完了時点で 50-60% の高速化
- **次のタスク:** 1-3 (coverage スキップ)

---

## 🔧 重要な技術情報（引継ぎ必読）

### ⚠️ ファイルエンコーディング
**超重要:** すべての`.cpp`ファイルは **cp932 (Shift-JIS)** でエンコードされています。

- ❌ **UTF-8で保存すると日本語コメントが文字化けします**
- ✅ PowerShellでの編集: **必ず `-Encoding Default`** を使用
- ✅ VS Codeの場合: 右下の「UTF-8」→「エンコード指定で再オープン」→「Japanese (Shift JIS)」

**ファイル編集例:**
```powershell
# 正しい（cp932で保存）
Get-Content SDK_ProcAmp_CPU.cpp -Raw -Encoding Default | ... | Set-Content SDK_ProcAmp_CPU.cpp -Encoding Default -NoNewline

# 間違い（UTF-8で保存してしまう）
Get-Content SDK_ProcAmp_CPU.cpp | ... | Set-Content SDK_ProcAmp_CPU.cpp
```

### 🔴 GPU強制無効化設定（現在テスト中）
現在、CPUテストのため**GPUを強制的に失敗**させています：

- **ファイル:** `SDK_ProcAmp_GPU.cpp`
- **行:** 807
- **現在のコード:**
```cpp
return suiteError_Fail;  // GPU強制失敗（CPU最適化テスト用）
```
- **Phase 1完了後:** この行をコメントアウトまたは削除してGPUを再有効化

### 📁 ビルド環境

**プロジェクト構造:**
```
Windy_Lines/
├── SDK_ProcAmp_CPU.cpp    (2,428行) - CPU実装（現在最適化中）
├── SDK_ProcAmp_GPU.cpp    (2,473行) - GPU統合レイヤー
├── SDK_ProcAmp.cu         (871行)   - CUDA実装
├── SDK_ProcAmp.cl         (734行)   - OpenCL実装
├── SDK_ProcAmp.h          (688行)   - 共通ヘッダー
├── OPTIMIZATION_PLAN.md   (本ファイル)
└── Win/
    └── SDK_ProcAmp.vcxproj
```

**ビルド設定:**
- IDE: Visual Studio 2022 Community
- Platform: x64
- Configuration: Debug
- SDK: Windows 10 SDK
- CUDA: v13.1

**プラグイン出力先:**
```
Debug:   G:\アドビ関連\Adobe Premiere Pro 2026\PlugIns\Common\MyPlugins\SDK_ProcAmp.aex
Release: （未設定）
```

---

## 🛠️ ビルド手順（詳細）

### 1. ビルド前の準備
```powershell
# 1. Premiere Proを完全終了（重要！）
Get-Process | Where-Object {$_.Name -like "*Premiere*"} | Stop-Process -Force

# 2. ビルドディレクトリに移動
cd "c:\Users\Owner\Desktop\Premiere_Pro_24.0_C_Win_SDK\Premiere_Pro_24.0_C++_Win_SDK\Premiere_Pro_24.0_SDK\Examples\Projects\GPUVideoFilter\Windy_Lines\Win"
```

### 2. ビルド実行
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" SDK_ProcAmp.vcxproj /p:Configuration=Debug /p:Platform=x64 /t:Build
```

### 3. ビルド結果の確認
✅ **成功の場合:**
```
ビルドに成功しました。
    67 個の警告
    0 エラー
```

❌ **失敗の場合（LNK1104）:**
```
LINK : fatal error LNK1104: ファイル 'G:\アドビ関連\Adobe Premiere Pro 2026\PlugIns\Common\MyPlugins\SDK_ProcAmp.aex' を開くことができません。
```
→ **原因:** Premiere Proがプラグインをロック中  
→ **解決:** Premiere Proを完全終了してリビルド

---

## 🧪 テスト手順

### Premiere Proでの動作確認
1. **Premiere Pro 2026を起動**
2. **新規プロジェクト作成** (1920x1080推奨)
3. **エフェクトを追加:**
   - エフェクトパネル → ビデオエフェクト → 「SDK ProcAmp」を検索
   - クリップにドラッグ&ドロップ
4. **パラメータ調整例:**
   ```
   Line Count: 100
   Line Thickness: 5.0
   Line Length: 200
   Motion Blur Samples: 8
   ```
5. **プレビュー再生:**
   - スペースキーで再生
   - タスクマネージャーでCPU使用率を確認

### パフォーマンス測定方法
**ベンチマーク設定:**
```
解像度: 1920x1080
Line Count: 100
Motion Blur Samples: 8
テスト時間: 10秒のクリップを3回レンダリング
```

**測定項目:**
- プレビューFPS
- フルレンダリング時間
- CPU使用率

---

## ⏳ 次のタスク実装ガイド (タスク1-3)

### タスク1-3: `coverage < 0.001f` スキップ

**目的:** ほぼ透明（coverage < 0.001f）なピクセルのブレンド処理をスキップして5%高速化

**実装ファイル:** `SDK_ProcAmp_CPU.cpp`

**実装箇所:** 行1900-2200付近（ブレンド処理の直前、複数箇所）

#### 実装方法

1. **coverage計算の直後**に以下を追加:
```cpp
// coverageの計算
float coverage = ...;

// 追加: 微小なcoverageをスキップ
if (coverage < 0.001f) {
    continue;
}
```

2. **対象箇所（推定6-10箇所）:**
   - メインラインレンダリング（3-4箇所）
   - シャドウレンダリング（2-3箇所）
   - アンチエイリアシング処理（1-2箇所）

3. **検索方法:**
```powershell
# coverageが計算される箇所を検索
grep_search -query "coverage =" -isRegexp false -includePattern "SDK_ProcAmp_CPU.cpp"
```

4. **注意点:**
   - ブレンド処理（Normal/Add/Multiply/Screen）の**直前**に配置
   - 既に `if (coverage > 0.0f)` がある場合はその中に配置

#### 期待される結果
- 透明度の高いピクセルのスキップ → ~5%高速化
- 特にLine Countが多い場合に効果大

---

## 📝 Phase 1残タスク概要

### タスク1-4: `halfThick < 0.5f` 極小線スキップ

**実装箇所:** 線ループの先頭（各線を処理する前）

**実装例:**
```cpp
for (int i = 0; i < lineCount; i++) {
    const LineDerived& ld = lines[i];
    
    // 追加: 極小線をスキップ
    if (ld.halfThick < 0.5f) {
        continue;
    }
    
    // ... 線の処理 ...
}
```

**期待効果:** 線が細い場合に高速化（可変）

---

### タスク1-5: モーションブラー最適化

**実装箇所:** motionBlurSamples処理ループ

**実装方針:**
1. 線の速度を計算:
```cpp
const float lineSpeed = sqrtf(ld.velocityX * ld.velocityX + ld.velocityY * ld.velocityY);
```

2. 速度に応じてサンプル数を調整:
```cpp
const float speedThreshold = 5.0f;  // ピクセル/フレーム
const int effectiveSamples = (lineSpeed < speedThreshold) ? 1 : motionBlurSamples;

for (int s = 0; s < effectiveSamples; s++) {
    // モーションブラーサンプリング
}
```

**期待効果:** ~15%高速化（モーションブラー有効時）

---

## 🔄 Phase 2以降への移行

### Phase 1完了チェックリスト
- [ ] タスク1-3完了（coverage スキップ）
- [ ] タスク1-4完了（極小線スキップ）
- [ ] タスク1-5完了（モーションブラー最適化）
- [ ] Premiere Proで動作確認
- [ ] パフォーマンスベンチマーク実施
- [ ] GPU強制失敗を解除（SDK_ProcAmp_GPU.cpp:807）
- [ ] GPUモードで動作確認
- [ ] Phase 2計画の詳細化

### Phase 2への準備

**Phase 2の主な課題:**
1. **3プラットフォーム同期:**
   - CPU、CUDA、OpenCLの出力完全一致が必須
   - テスト方法の確立が重要

2. **SDF計算の共通化:**
   - CPU: 6箇所（円形、カプセル形状）
   - CUDA: 3箇所
   - OpenCL: 3箇所
   - 合計12箇所を1つの共通関数に

3. **ブレンド処理の統一:**
   - Normal, Add, Multiply, Screen × 3実装 = 12箇所
   - 保守性の大幅向上

---

## 🐛 トラブルシューティング

### よくある問題と解決方法

#### 1. ビルドエラー: LNK1104
```
LINK : fatal error LNK1104: ファイル 'G:\...\SDK_ProcAmp.aex' を開くことができません。
```
**原因:** Premiere Proがプラグインをロック中  
**解決:**
```powershell
# Premiere Proを強制終了
Get-Process | Where-Object {$_.Name -like "*Premiere*"} | Stop-Process -Force

# リビルド
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" SDK_ProcAmp.vcxproj /p:Configuration=Debug /p:Platform=x64 /t:Build
```

#### 2. 文字化け
**原因:** UTF-8でファイルを保存した  
**解決:**
```powershell
# cp932 (Shift-JIS) で保存し直す
$content = Get-Content SDK_ProcAmp_CPU.cpp -Raw -Encoding UTF8
$content | Set-Content SDK_ProcAmp_CPU.cpp -Encoding Default -NoNewline
```

#### 3. プラグインが表示されない
**原因:** 出力先パスが間違っている  
**確認:**
```powershell
# プラグインファイルの存在確認
Test-Path "G:\アドビ関連\Adobe Premiere Pro 2026\PlugIns\Common\MyPlugins\SDK_ProcAmp.aex"
```

#### 4. コンパイルエラー: 変数名の重複
**例:** `dx`/`dy` がすでに定義されている  
**解決:** 別の変数名を使用（例: `skipDx`/`skipDy`）

#### 5. GPU版が動かない
**原因:** SDK_ProcAmp_GPU.cpp:807 で強制失敗させている  
**解決:**
```cpp
// 変更前
return suiteError_Fail;

// 変更後（コメントアウト）
// return suiteError_Fail;  // Phase 1テスト用の強制失敗
```

---

## 📞 引継ぎサマリー

### 現在の状態
| 項目 | 値 |
|------|-----|
| **Phase** | Phase 1 (CPU軽量化) |
| **進捗** | 4/7タスク完了 (57%) |
| **次のタスク** | 1-3: `coverage < 0.001f` スキップ |
| **最終ビルド** | 2026年2月5日 |
| **動作確認** | Premiere Pro 2026で確認済み |
| **GPU状態** | 強制無効化中（CPU最適化のため） |

### 作業環境
```
OS: Windows 11
IDE: Visual Studio 2022 Community
Premiere Pro: 2026
ワークスペース: c:\Users\Owner\Desktop\Premiere_Pro_24.0_C_Win_SDK\...
出力先: G:\アドビ関連\Adobe Premiere Pro 2026\PlugIns\Common\MyPlugins\
```

### 次のエージェントへの指示
1. **まず `OPTIMIZATION_PLAN.md` を熟読**
2. **タスク1-3から開始**（coverage < 0.001f スキップ）
3. **必ずcp932エンコーディングで編集**
4. **ビルド前にPremiere Proを終了**
5. **各タスク完了後、このファイルの進捗を更新**

---

## 📚 関連ドキュメント

- `GPU_IMPLEMENTATION_MEMO.md` - GPU実装メモ
- `SDK_ProcAmp_DevGuide.md` - 開発ガイド
- `SDK_ProcAmp_Notes.json` - 実装ノート（JSON形式）
- `readme.txt` - プロジェクト概要

---

**実装日:** 2026年2月5日

### 変更内容
1. `LineDerived`構造体に2つのフィールドを追加 (行165-166):
   - `float depthAlpha;` - 深度フェード用のアルファ値
   - `float invDenom;` - テールフェード計算用の逆数

2. ライン生成時に事前計算を追加 (行1685付近)

3. レンダリングループで事前計算値を使用:
   - `depthAlpha`計算を `ld.depthAlpha` に置換
   - 全ての `/ sdenom` と `/ denom` を `* ld.invDenom` に置換

### 追加修正
- `NormalizePopupValue`関数の条件判定順序を修正（合成モードずれ解消）

---

## ✅ Step 2: 早期スキップ最適化 (完了)

**実装日:** 2026年2月5日

### 実装内容
メインレンダリングループ（行1820-1829付近）に早期スキップ判定を追加。

### 追加コード
```cpp
// === STEP 2: Early skip optimization ===
const float skipDx = (x + 0.5f) - ld.centerX;
const float skipDy = (y + 0.5f) - ld.centerY;
const float shadowMargin = shadowEnable ? fmaxf(fabsf(shadowOffsetX), fabsf(shadowOffsetY)) : 0.0f;
const float margin = ld.halfThick + lineAAScaled + shadowMargin;
const float skipPx = skipDx * ld.cosA + skipDy * ld.sinA - ld.segCenterX;
const float skipPy = -skipDx * ld.sinA + skipDy * ld.cosA;

if (fabsf(skipPx) > ld.halfLen + margin && fabsf(skipPy) > margin)
{
    continue;  // Skip this line - pixel is too far away
}
// === END STEP 2 ===
```

---

## ⏳ Step 3: powf → 乗算展開 (次に実施)

### 目的
ApplyEasing関数内のpowf呼び出しを乗算に展開してイージング計算を高速化

### 対象ファイル
`SDK_ProcAmp_CPU.cpp`

### 変更箇所: 2箇所

---

#### 変更1: InOutQuad (行305付近)

**oldString:**
```cpp
		case 5: // OutQuad
			return 1.0f - (1.0f - t) * (1.0f - t);
		case 6: // InOutQuad
			return t < 0.5f ? 2.0f * t * t : 1.0f - powf(-2.0f * t + 2.0f, 2.0f) * 0.5f;
		case 7: // InCubic
			return t * t * t;
```

**newString:**
```cpp
		case 5: // OutQuad
			return 1.0f - (1.0f - t) * (1.0f - t);
		case 6: // InOutQuad
		{
			const float u = 2.0f - 2.0f * t;
			return t < 0.5f ? 2.0f * t * t : 1.0f - u * u * 0.5f;
		}
		case 7: // InCubic
			return t * t * t;
```

---

#### 変更2: InOutCubic (行313付近)

**oldString:**
```cpp
		case 8: // OutCubic
		{
			const float u = 1.0f - t;
			return 1.0f - u * u * u;
		}
		case 9: // InOutCubic
			return t < 0.5f ? 4.0f * t * t * t : 1.0f - powf(-2.0f * t + 2.0f, 3.0f) * 0.5f;
		default:
			return t;
```

**newString:**
```cpp
		case 8: // OutCubic
		{
			const float u = 1.0f - t;
			return 1.0f - u * u * u;
		}
		case 9: // InOutCubic
		{
			const float u = 2.0f - 2.0f * t;
			return t < 0.5f ? 4.0f * t * t * t : 1.0f - u * u * u * 0.5f;
		}
		default:
			return t;
```

### 期待される改善
- イージング計算の50%以上高速化
- powf関数呼び出しのオーバーヘッド削減

---

## 📝 Step 4: effectiveAA事前計算 (予定)

### 目的
ループ内で毎回評価される条件を外に出す

### 変更箇所

#### 変更1: effectiveAA変数を追加 (行1290の直後)

**oldString:**
```cpp
		const float lineAAScaled = lineAA * dsScale;
```

**newString:**
```cpp
		const float lineAAScaled = lineAA * dsScale;
		const float effectiveAA = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;
```

#### 変更2-7: 6箇所の置換

以下の6箇所で置換を実行:

| 行番号 | oldString | newString |
|--------|-----------|-----------|
| 1881 | `const float saa = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;` | `const float saa = effectiveAA;` |
| 1921 | `const float saa = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;` | `const float saa = effectiveAA;` |
| 1956 | `const float saa = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;` | `const float saa = effectiveAA;` |
| 2036 | `const float aa = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;` | `const float aa = effectiveAA;` |
| 2074 | `const float aa = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;` | `const float aa = effectiveAA;` |
| 2109 | `const float aa = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;` | `const float aa = effectiveAA;` |

### 期待される改善
- 条件分岐の削減（微小だが累積効果あり）

---

## 📝 Step 5: lineCap分岐最適化 (検討中)

### 目的
ピクセル毎の `if (lineCap == 0)` 分岐をループ外に出す

### 方針
SDF計算を共通inline関数化してコンパイラ最適化に委ねる

### 期待される改善
5-15%の高速化

---

## GPU関連

### GPU再有効化手順
**ファイル:** `SDK_ProcAmp_GPU.cpp`  
**行:** 807  

```cpp
// 変更前（GPU無効）
return suiteError_Fail;

// 変更後（GPU有効）
// return suiteError_Fail;  // コメントアウト
```

---

## ビルドコマンド

```powershell
cd "c:\Users\Owner\Desktop\Premiere_Pro_24.0_C_Win_SDK\Premiere_Pro_24.0_C++_Win_SDK\Premiere_Pro_24.0_SDK\Examples\Projects\GPUVideoFilter\Windy_Lines\Win"
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" SDK_ProcAmp.vcxproj /p:Configuration=Debug /p:Platform=x64 /t:Build
```

---

## 実装順序チェックリスト

- [x] Step 1: 事前計算（depthAlpha, invDenom）
- [x] Step 2: 早期スキップ（バウンディングボックス）
- [ ] Step 3: powf → 乗算展開
- [ ] Step 4: effectiveAA事前計算
- [ ] Step 5: lineCap分岐最適化
- [ ] GPU再有効化

---

## 注意事項

1. **ファイルエンコーディング:** `cp932`（Shift-JIS）- UTF-8ではない
2. **変数名衝突注意:** `dx`/`dy`は他の場所で使用されているため、早期スキップでは`skipDx`/`skipDy`を使用
3. **ビルド前にPremiere Pro終了:** ファイルがロックされるとLNK1104エラー
