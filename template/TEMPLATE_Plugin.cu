/*
 * __TPL_MATCH_NAME__.cu — CUDA Kernel (Windows リファレンス実装)
 *
 * ★★★ 三重同期ルール ★★★
 * このファイルに関数を追加/変更したら、必ず以下も同期すること:
 *   1. __TPL_MATCH_NAME__.cl    (OpenCL/Metal)
 *   2. __TPL_MATCH_NAME___Common.h (CPU共有)
 *
 * CUDA固有の注意:
 *   - 関数修飾子: __device__ __forceinline__
 *   - 数学関数:   cosf/sinf/powf/fminf/fmaxf  (float suffix必須)
 *   - カーネル:   GF_KERNEL_FUNCTION マクロ
 */

#ifndef __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___CU
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___CU

#define ENABLE_DEBUG_RENDER_MARKERS 0

#if __CUDACC_VER_MAJOR__ >= 9
    #include <cuda_fp16.h>
#endif

#include "PrGPU/KernelSupport/KernelCore.h"
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE

// ===================== ユーティリティ関数 =====================
// Common.h と同じロジック。CUDA用に __device__ 修飾子付き。

__device__ __forceinline__ unsigned int HashUInt(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float Rand01(unsigned int x)
{
    return (float)(HashUInt(x) & 0x00FFFFFF) / 16777215.0f;
}

__device__ __forceinline__ float EaseInOutSine(float t)
{
    return 0.5f * (1.0f - cosf(3.14159265f * t));
}

// イージング関数（Common.hと同期すること）
__device__ __forceinline__ float ApplyEasing(float t, int easing)
{
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    if (easing == 0) return t;                                            // Linear
    else if (easing == 1) return t * t * (3.0f - 2.0f * t);              // Smoothstep
    else if (easing == 2) return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // Smootherstep
    else if (easing == 3) return 1.0f - cosf(3.14159265f * t * 0.5f);    // Sine In
    else if (easing == 4) return sinf(3.14159265f * t * 0.5f);           // Sine Out
    else if (easing == 5) return EaseInOutSine(t);                        // Sine InOut
    // ... 以下同様に case 6〜27 を実装（Common.h参照）
    return t;
}


/*******************************************************************/
/*  メインカーネル                                                  */
/*  GF_KERNEL_FUNCTION マクロで定義（SDK提供）                       */
/*  引数の順序 = ProcAmpParams構造体のフィールド順序                 */
/*******************************************************************/
GF_KERNEL_FUNCTION(PluginKernel,
    // ---- 入出力バッファ ----
    ((GF_PTR(GF_PIXEL_FLOAT))(inBuf))
    ((GF_PTR(GF_PIXEL_FLOAT))(outBuf)),
    // ---- パラメータ（ProcAmpParamsの順序と一致させること） ----
    ((int)(inWidth))
    ((int)(inHeight))
    ((float)(inAmount))
    ((float)(inColorR))
    ((float)(inColorG))
    ((float)(inColorB))
    ((int)(inMode)),
    // ---- ディスパッチ次元 ----
    ((uint2)(inXY)(KERNEL_XY)))
{
    int x = inXY.x;
    int y = inXY.y;
    if (x >= inWidth || y >= inHeight) return;

    // 入力ピクセル読み込み
    GF_PIXEL_FLOAT pixel = GF_LOAD_PIXEL(inBuf, x, y);

    // ========================================
    // TODO: ここにGPUエフェクトロジックを実装
    // ========================================

    // サンプル: amountに応じて色を混ぜる
    float t = inAmount / 100.0f;
    pixel.red   = pixel.red   * (1.0f - t) + inColorR * t * pixel.alpha;
    pixel.green = pixel.green * (1.0f - t) + inColorG * t * pixel.alpha;
    pixel.blue  = pixel.blue  * (1.0f - t) + inColorB * t * pixel.alpha;

    // 出力に書き込み
    GF_STORE_PIXEL(outBuf, x, y, pixel);
}

#endif // GF_DEVICE_TARGET_DEVICE


/*******************************************************************/
/*  ホストエントリーポイント（GPU.cpp から呼ばれる）                 */
/*******************************************************************/

// ProcAmpParams 構造体の定義をインクルード
// （GPU.cppで定義されているものと同一）

extern "C" void PluginKernel_CUDA(
    float const* inBuf,
    float* outBuf,
    unsigned int outPitch,
    int is16f,
    const void* params)
{
    // TODO: カーネルランチの実装
    // dim3 blockDim(16, 16);
    // dim3 gridDim((width + 15) / 16, (height + 15) / 16);
    // PluginKernel<<<gridDim, blockDim>>>(...);
}

#endif // __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___CU
