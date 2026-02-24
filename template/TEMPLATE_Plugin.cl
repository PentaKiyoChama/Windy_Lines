/*
 * __TPL_MATCH_NAME__.cl — OpenCL / Metal 共有カーネル
 *
 * ★★★ 三重同期ルール ★★★
 * このファイルに関数を追加/変更したら、必ず以下も同期すること:
 *   1. __TPL_MATCH_NAME__.cu      (CUDA)
 *   2. __TPL_MATCH_NAME___Common.h (CPU共有)
 *
 * OpenCL/Metal 差異:
 *   - 関数修飾子: なし or inline（__device__ は不要）
 *   - 数学関数:   cos/sin/pow（f接尾辞なし）
 *   - カーネル:   GF_KERNEL_FUNCTION マクロ（SDK共通）
 *
 * Metal の重要ルール:
 *   ProcAmpParams 構造体のフィールド順序 = このカーネルの引数順序
 *   (Metal はバッファ経由で構造体を丸渡しするため)
 */

#ifndef SDK_PROC_AMP
#define SDK_PROC_AMP

#define ENABLE_DEBUG_RENDER_MARKERS 0

#include "PrGPU/KernelSupport/KernelCore.h"
#include "PrGPU/KernelSupport/KernelMemory.h"

// Metal 固有のインクルード
#if GF_DEVICE_TARGET_METAL
#include <metal_stdlib>
#include <metal_atomic>
    using namespace metal;
#endif

#if GF_DEVICE_TARGET_DEVICE

// ===================== ユーティリティ関数 =====================
// Common.h と同じロジック。__device__ なし版。
// cos/sin/pow を使う（cosf/sinf/powf ではなく）

uint HashUInt(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

float Rand01(uint x)
{
    return (float)(HashUInt(x) & 0x00FFFFFF) / 16777215.0f;
}

float EaseInOutSine(float t)
{
    return 0.5f * (1.0f - cos(3.14159265f * t));
}

// イージング関数（Common.hと同期すること）
float ApplyEasing(float t, int easing)
{
    t = fmin(fmax(t, 0.0f), 1.0f);
    if (easing == 0) return t;                                           // Linear
    else if (easing == 1) return t * t * (3.0f - 2.0f * t);             // Smoothstep
    else if (easing == 2) return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // Smootherstep
    else if (easing == 3) return 1.0f - cos(3.14159265f * t * 0.5f);    // Sine In
    else if (easing == 4) return sin(3.14159265f * t * 0.5f);           // Sine Out
    else if (easing == 5) return EaseInOutSine(t);                       // Sine InOut
    // ... 以下同様に case 6〜27 を実装（Common.h参照）
    return t;
}


/*******************************************************************/
/*  メインカーネル                                                  */
/*  引数の順序 = ProcAmpParams構造体のフィールド順序（Metal必須）     */
/*******************************************************************/
GF_KERNEL_FUNCTION(PluginKernel,
    // ---- 入出力バッファ ----
    ((GF_PTR(float4))(inBuf))
    ((GF_PTR(float4))(outBuf)),
    ((int)(inPitch))
    ((int)(outPitch))
    ((int)(in16f))
    // ---- パラメータ（ProcAmpParamsの順序と一致） ----
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

    float4 pixel = ReadFloat4(inBuf, y * inPitch + x, !!in16f);

    // ========================================
    // TODO: ここにGPUエフェクトロジックを実装
    // .cu と同じロジックを実装すること
    // ========================================

    // サンプル: amountに応じて色を混ぜる
    float t = inAmount / 100.0f;
    pixel.z = pixel.z * (1.0f - t) + inColorR * t * pixel.w;
    pixel.y = pixel.y * (1.0f - t) + inColorG * t * pixel.w;
    pixel.x = pixel.x * (1.0f - t) + inColorB * t * pixel.w;

    WriteFloat4(pixel, outBuf, y * outPitch + x, !!in16f);
}

#endif // GF_DEVICE_TARGET_DEVICE
#endif // SDK_PROC_AMP
