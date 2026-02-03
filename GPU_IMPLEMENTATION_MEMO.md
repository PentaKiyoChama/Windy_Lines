# GPU実装メモ - 必ず3箇所を同期すること

## 🚨 重要: 機能追加・変更時の必須作業

新機能を追加する際は、**必ず以下の3箇所すべて**に実装してください：

### 1. CPU実装
**ファイル**: `SDK_ProcAmp_CPU.cpp`
- 用途: CPUフォールバック、GPUが使えない環境
- 注意: GPU実装と完全に同じ結果を出力する必要あり

### 2. CUDA実装 (Windows)
**ファイル**: `SDK_ProcAmp.cu`
- 用途: Windows環境でのGPU処理
- 行数: 674行 (v51時点)
- 注意: これが**リファレンス実装**

### 3. OpenCL/Metal実装 (Mac)
**ファイル**: `SDK_ProcAmp.cl` (実体)
**ファイル**: `SDK_ProcAmp.metal` (単に`#include "SDK_ProcAmp.cl"`するだけ)
- 用途: Mac環境でのGPU処理
- 行数: 511行 (v51時点)
- 注意: MetalはOpenCLを流用（メンテナンスが1箇所で済む）

---

## 実装の流れ (推奨)

### ステップ1: CUDA実装 (基準)
1. `SDK_ProcAmp.cu` に新機能を実装
2. Windows環境でビルド・テスト
3. 動作確認完了

### ステップ2: OpenCL/Metal実装
1. `SDK_ProcAmp.cl` に同じロジックを移植
   - `__device__` → (何もつけない)
   - `__forceinline__` → `inline` (省略可)
   - `cosf/sinf/powf` → `cos/sin/pow`
2. Mac環境でビルド・テスト (arm64)
3. 動作確認完了

### ステップ3: CPU実装
1. `SDK_ProcAmp_CPU.cpp` に同じロジックを実装
2. GPU実装と結果が一致するかテスト

---

## パラメータ追加時のチェックリスト

新しいパラメータを追加する場合：

- [ ] `SDK_ProcAmp.h` にパラメータID定義を追加
- [ ] `SDK_ProcAmp_GPU.cpp` のホスト側コードにパラメータ取得処理を追加
- [ ] `SDK_ProcAmp.cu` (CUDA) にパラメータを追加
- [ ] `SDK_ProcAmp.cl` (OpenCL/Metal) にパラメータを追加
- [ ] `SDK_ProcAmp_CPU.cpp` にパラメータを追加
- [ ] 3つの実装すべてで同じ結果が出ることを確認

---

## 既知の違い

### 関数修飾子
| CUDA | OpenCL/Metal |
|------|--------------|
| `__device__ __forceinline__` | (何もつけない) or `inline` |
| `__global__` | `__kernel` |

### 数学関数
| CUDA | OpenCL/Metal |
|------|--------------|
| `cosf/sinf/tanf` | `cos/sin/tan` |
| `powf/sqrtf` | `pow/sqrt` |
| `fmaxf/fminf` | `fmax/fmin` |

### 型名
| CUDA | OpenCL/Metal |
|------|--------------|
| `unsigned int` | `uint` (推奨) |
| `float4` | `float4` (同じ) |

---

## Metal特有の注意点

### 1. Atomic操作
```cpp
// CUDA
atomicAdd(&counter[0], 1);

// OpenCL/Metal
#if GF_DEVICE_TARGET_METAL
    atomic_fetch_add_explicit((device atomic_uint*)&counter[0], 1, memory_order_relaxed);
#else
    atomic_add(&counter[0], 1);
#endif
```

### 2. Buffer初期化
- Metal実装では、バッファは必ず初期化データ付きで作成
- `nullptr`でバッファを作成すると描画されない

---

## 現在の実装状況 (v51)

### 実装済み機能
✅ 基本線描画
✅ Easing (10種類)
✅ Shadow (影付き)
✅ Blend Mode (4種類: Back/Front/Back&Front/Alpha)
✅ Motion Blur
✅ Spawn Area (出現範囲スケール)
✅ Focus/Depth (深度フォーカス)
✅ Start Time/Duration (clipTime-based計算)

### CPU/GPU同期状態
✅ 全機能で3実装が同期済み

---

## トラブルシューティング

### 「CUDAで動くのにMetalで動かない」場合
1. `SDK_ProcAmp.cl` に該当機能が移植されているか確認
2. 関数名・型名の違いを確認（上記の表参照）
3. Atomic操作をMetal形式に変換しているか確認
4. バッファ初期化でnullptrを使っていないか確認

### 「CPUとGPUで結果が違う」場合
1. 浮動小数点演算の順序が違う可能性
2. frameIndex計算ロジックを確認

---

## ビルド手順

### Mac (arm64) - Xcodeで手動ビルド

1. **プロジェクトを開く**
   - `SDK_ProcAmp/Mac/SDK_ProcAmp.xcodeproj` をXcodeで開く

2. **ビルド設定を確認**
   - Scheme: `SDK_ProcAmp`
   - Configuration: `Debug` または `Release`
   - Architecture: `arm64`

3. **ビルド実行**
   - `Cmd + B` または メニュー → Product → Build

4. **プラグインをインストール**
   - ビルド成功後、生成された `SDK_ProcAmp.plugin` を以下にコピー:
   ```
   /Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/
   ```

5. **Premiere Proを再起動**して動作確認

---

## 最終更新
- 日付: 2026-01-27
- バージョン: v51
- 状態: Start Time/Duration修正完了、全機能動作中
