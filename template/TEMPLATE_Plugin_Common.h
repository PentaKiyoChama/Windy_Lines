/*******************************************************************/
/*                                                                 */
/*  __TPL_MATCH_NAME__ - CPU/GPU Shared Utilities                  */
/*  ヘッダーオンリー: static inline 関数のみ                        */
/*                                                                 */
/*  依存: PrSDKTypes.h, <cmath>                                    */
/*                                                                 */
/*  NOTE: このファイルはCPU側(_CPU.cpp)で #include される。           */
/*        GPU側(.cu, .cl)では同じ関数を手動で再実装する必要がある。   */
/*        → GPU側では __device__ 修飾子やcos/cosf等の差異があるため  */
/*                                                                 */
/*  関数を追加したら必ず .cu と .cl にも同等の実装を追加すること！    */
/*  (三重同期ルール)                                                */
/*                                                                 */
/*******************************************************************/

#ifndef __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___COMMON_H
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___COMMON_H

#include "PrSDKTypes.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===================== 乱数ユーティリティ =====================
// GPU側にも同じ実装が必要（.cu: __device__ __forceinline__, .cl: 修飾子なし）

static inline csSDK_uint32 HashUInt(csSDK_uint32 x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

static inline float Rand01(csSDK_uint32 x)
{
    return (HashUInt(x) & 0x00FFFFFF) / 16777215.0f;
}


// ===================== イージング関数 =====================
// 28種類（Windy_Linesプロジェクトから継承）
// GPU側にも同じ実装が必要

static inline float EaseInOutSine(float t)
{
    return 0.5f * (1.0f - cosf((float)M_PI * t));
}

static inline float ApplyEasing(float t, int easingType)
{
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    switch (easingType)
    {
        case 0: return t;                                                    // Linear
        case 1: return t * t * (3.0f - 2.0f * t);                           // Smoothstep
        case 2: return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);        // Smootherstep
        case 3: return 1.0f - cosf((float)M_PI * t * 0.5f);                 // Sine In
        case 4: return sinf((float)M_PI * t * 0.5f);                        // Sine Out
        case 5: return EaseInOutSine(t);                                     // Sine In-Out
        case 6: // Sine Out-In
        {
            if (t < 0.5f) return 0.5f * sinf((float)M_PI * t);
            else return 0.5f + 0.5f * (1.0f - cosf((float)M_PI * (t - 0.5f)));
        }
        case 7:  return t * t;                                               // Quad In
        case 8:  return 1.0f - (1.0f - t) * (1.0f - t);                     // Quad Out
        case 9:  // Quad In-Out
        {
            if (t < 0.5f) return 2.0f * t * t;
            float u = -2.0f * t + 2.0f;
            return 1.0f - u * u * 0.5f;
        }
        case 10: // Quad Out-In
        {
            float lin = t;
            float eased;
            if (t < 0.5f) { float u = 2.0f * t; eased = 0.5f * (1.0f - (1.0f - u) * (1.0f - u)); }
            else { float u = 2.0f * (t - 0.5f); eased = 0.5f + 0.5f * u * u; }
            return eased * 0.75f + lin * 0.25f;
        }
        case 11: return t * t * t;                                           // Cubic In
        case 12: { float u = 1.0f - t; return 1.0f - u * u * u; }           // Cubic Out
        case 13: // Cubic In-Out
        {
            if (t < 0.5f) return 4.0f * t * t * t;
            float u = -2.0f * t + 2.0f;
            return 1.0f - u * u * u * 0.5f;
        }
        case 14: // Cubic Out-In
        {
            float lin = t;
            float eased;
            if (t < 0.5f) { float u = 2.0f * t; float v = 1.0f - u; eased = 0.5f * (1.0f - v * v * v); }
            else { float u = 2.0f * (t - 0.5f); eased = 0.5f + 0.5f * u * u * u; }
            return eased * 0.75f + lin * 0.25f;
        }
        case 15: return 1.0f - sqrtf(1.0f - t * t);                         // Circle In
        case 16: return sqrtf(1.0f - (t - 1.0f) * (t - 1.0f));              // Circle Out
        case 17: // Circle In-Out
        {
            if (t < 0.5f) return (1.0f - sqrtf(1.0f - 4.0f * t * t)) * 0.5f;
            float u = -2.0f * t + 2.0f;
            return (sqrtf(1.0f - u * u) + 1.0f) * 0.5f;
        }
        case 18: // Circle Out-In
        {
            float lin = t;
            float eased;
            if (t < 0.5f) { float u = 2.0f * t; eased = 0.5f * sqrtf(1.0f - (u - 1.0f) * (u - 1.0f)); }
            else { float u = 2.0f * (t - 0.5f); eased = 0.5f + 0.5f * (1.0f - sqrtf(1.0f - u * u)); }
            return eased * 0.75f + lin * 0.25f;
        }
        case 19: return t * t * (2.70158f * t - 1.70158f);                   // Back In
        case 20: { float u = t - 1.0f; return 1.0f + u * u * (2.70158f * u + 1.70158f); } // Back Out
        case 21: // Back In-Out
        {
            float c = 1.70158f * 1.525f;
            if (t < 0.5f) { float u = 2.0f * t; return u * u * ((c + 1.0f) * u - c) * 0.5f; }
            float u = 2.0f * t - 2.0f;
            return (u * u * ((c + 1.0f) * u + c) + 2.0f) * 0.5f;
        }
        case 22: // Elastic In
        {
            if (t <= 0.0f) return 0.0f;
            if (t >= 1.0f) return 1.0f;
            return -powf(2.0f, 10.0f * t - 10.0f) * sinf((t * 10.0f - 10.75f) * (2.0f * (float)M_PI / 3.0f));
        }
        case 23: // Elastic Out
        {
            if (t <= 0.0f) return 0.0f;
            if (t >= 1.0f) return 1.0f;
            return powf(2.0f, -10.0f * t) * sinf((t * 10.0f - 0.75f) * (2.0f * (float)M_PI / 3.0f)) + 1.0f;
        }
        case 24: // Elastic In-Out
        {
            if (t <= 0.0f) return 0.0f;
            if (t >= 1.0f) return 1.0f;
            float c = (2.0f * (float)M_PI) / 4.5f;
            if (t < 0.5f) return -(powf(2.0f, 20.0f * t - 10.0f) * sinf((20.0f * t - 11.125f) * c)) * 0.5f;
            return (powf(2.0f, -20.0f * t + 10.0f) * sinf((20.0f * t - 11.125f) * c)) * 0.5f + 1.0f;
        }
        case 25: // Bounce In
        {
            float u = 1.0f - t, b;
            if (u < 1.0f / 2.75f) b = 7.5625f * u * u;
            else if (u < 2.0f / 2.75f) { u -= 1.5f / 2.75f; b = 7.5625f * u * u + 0.75f; }
            else if (u < 2.5f / 2.75f) { u -= 2.25f / 2.75f; b = 7.5625f * u * u + 0.9375f; }
            else { u -= 2.625f / 2.75f; b = 7.5625f * u * u + 0.984375f; }
            return 1.0f - b;
        }
        case 26: // Bounce Out
        {
            float u = t, b;
            if (u < 1.0f / 2.75f) b = 7.5625f * u * u;
            else if (u < 2.0f / 2.75f) { u -= 1.5f / 2.75f; b = 7.5625f * u * u + 0.75f; }
            else if (u < 2.5f / 2.75f) { u -= 2.25f / 2.75f; b = 7.5625f * u * u + 0.9375f; }
            else { u -= 2.625f / 2.75f; b = 7.5625f * u * u + 0.984375f; }
            return b;
        }
        case 27: // Bounce In-Out
        {
            float result;
            if (t < 0.5f)
            {
                float u = 1.0f - 2.0f * t, b;
                if (u < 1.0f / 2.75f) b = 7.5625f * u * u;
                else if (u < 2.0f / 2.75f) { u -= 1.5f / 2.75f; b = 7.5625f * u * u + 0.75f; }
                else if (u < 2.5f / 2.75f) { u -= 2.25f / 2.75f; b = 7.5625f * u * u + 0.9375f; }
                else { u -= 2.625f / 2.75f; b = 7.5625f * u * u + 0.984375f; }
                result = (1.0f - b) * 0.5f;
            }
            else
            {
                float u = 2.0f * t - 1.0f, b;
                if (u < 1.0f / 2.75f) b = 7.5625f * u * u;
                else if (u < 2.0f / 2.75f) { u -= 1.5f / 2.75f; b = 7.5625f * u * u + 0.75f; }
                else if (u < 2.5f / 2.75f) { u -= 2.25f / 2.75f; b = 7.5625f * u * u + 0.9375f; }
                else { u -= 2.625f / 2.75f; b = 7.5625f * u * u + 0.984375f; }
                result = b * 0.5f + 0.5f;
            }
            return result;
        }
        default: return t;
    }
}

// イージングの数値微分（モーションブラー等で使用）
static inline float ApplyEasingDerivative(float t, int easingType, float epsilon = 0.001f)
{
    float t0 = t - epsilon;
    float t1 = t + epsilon;
    if (t0 < 0.0f) t0 = 0.0f;
    if (t1 > 1.0f) t1 = 1.0f;
    float dt = t1 - t0;
    if (dt < 0.0001f) return 0.0f;
    return (ApplyEasing(t1, easingType) - ApplyEasing(t0, easingType)) / dt;
}


// ===================== ブレンドユーティリティ =====================

static inline void BlendPremultiplied(
    float srcR, float srcG, float srcB, float srcA,
    float& dstR, float& dstG, float& dstB, float& dstA)
{
    dstR = srcR + dstR * (1.0f - srcA);
    dstG = srcG + dstG * (1.0f - srcA);
    dstB = srcB + dstB * (1.0f - srcA);
    dstA = srcA + dstA * (1.0f - srcA);
}


// ===================== カラープリセットユーティリティ =====================
// カラープリセットを使う場合は、ColorPresets.h を #include した上でこの関数を使う
// 使わない場合はこのセクションを削除してOK

/*
static inline void BuildPresetPalette(
    int presetIndex, int numColors,
    float* outR, float* outG, float* outB, float* outA)
{
    // ColorPresets::GetPresetPalette(presetIndex, ...) を呼び出す
    // 実装は ColorPresets.h の定義に依存
}
*/

#endif // __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___COMMON_H
