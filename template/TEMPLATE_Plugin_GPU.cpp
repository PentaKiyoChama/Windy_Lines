/*******************************************************************/
/*                                                                 */
/*  __TPL_MATCH_NAME__ - GPU Host Controller                       */
/*  CUDA (Windows) / Metal+OpenCL (Mac) ディスパッチ                */
/*                                                                 */
/*  Copyright (c) __TPL_YEAR__ __TPL_AUTHOR__. All rights reserved.*/
/*                                                                 */
/*  ### 重要な設計ルール ###                                        */
/*  1. ProcAmpParams構造体のフィールド順 = .clカーネルの引数順       */
/*     (Metalはバッファ経由で渡すため順序一致が必須)                 */
/*  2. 新パラメータ追加時は .cu, .cl 両方に追加すること              */
/*  3. CPUで取得した情報はSharedClipDataで共有                      */
/*                                                                 */
/*******************************************************************/

#include "__TPL_MATCH_NAME__.h"
#include "__TPL_MATCH_NAME___Version.h"
#include "__TPL_MATCH_NAME___Common.h"
#include "PrGPUFilterModule.h"

#if _WIN32
    #include "__TPL_MATCH_NAME__.cl.h"        // OpenCL文字列（cl.exe /P で生成）
    #include <CL/cl.h>
#else
    #include <OpenCL/cl.h>
#endif

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <cstring>
#include <cstdint>
#include <climits>
#include <unordered_map>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <vector>

// === License verification: shared with CPU (single source of truth) ===
#include "__TPL_MATCH_NAME___License.h"

static bool IsGpuLicenseAuthenticated()
{
    RefreshLicenseAuthenticatedState(false);
    return IsLicenseAuthenticated();
}

// ========== プラットフォーム分岐 ==========
#if _WIN32
    #define HAS_CUDA    1
    #define HAS_METAL   0
#else
    #define HAS_CUDA    0
    #define HAS_METAL   1
#endif

// ========== CUDA ヘッダー（Windowsのみ） ==========
#if _WIN32
    // CUDAヘッダーとSDKのマクロ衝突を回避
    #if defined(MAJOR_VERSION)
        #pragma push_macro("MAJOR_VERSION")
        #undef MAJOR_VERSION
        #define __TPL_RESTORE_MAJOR 1
    #endif
    #if defined(MINOR_VERSION)
        #pragma push_macro("MINOR_VERSION")
        #undef MINOR_VERSION
        #define __TPL_RESTORE_MINOR 1
    #endif
    #if defined(PATCH_LEVEL)
        #pragma push_macro("PATCH_LEVEL")
        #undef PATCH_LEVEL
        #define __TPL_RESTORE_PATCH 1
    #endif
    #include <cuda_runtime.h>
    #if defined(__TPL_RESTORE_PATCH)
        #pragma pop_macro("PATCH_LEVEL")
    #endif
    #if defined(__TPL_RESTORE_MINOR)
        #pragma pop_macro("MINOR_VERSION")
    #endif
    #if defined(__TPL_RESTORE_MAJOR)
        #pragma pop_macro("MAJOR_VERSION")
    #endif
#endif


/*******************************************************************/
/*  ProcAmpParams — カーネルに渡すパラメータ構造体                   */
/*                                                                  */
/*  ★★★ 最重要ルール ★★★                                      */
/*  このstructのフィールド順序は、.clカーネルの引数順序と              */
/*  完全に一致しなければならない。                                    */
/*  Metalはこの構造体をバッファとして丸ごとカーネルに渡すため、        */
/*  順序が1つでもずれると描画が完全に壊れる。                         */
/*                                                                  */
/*  カーネルに渡さないパラメータは末尾の「CPU-only section」に        */
/*  配置すること。                                                   */
/*******************************************************************/
typedef struct
{
    // ---- GPU Kernel Parameters (順序 = .cl カーネル引数順) ----
    int     mWidth;
    int     mHeight;
    float   mAmount;
    float   mColorR;
    float   mColorG;
    float   mColorB;
    int     mMode;

    // ---- CPU-only section (カーネルに渡さないもの) ----
    // ここに追加してもカーネル側には影響しない

} ProcAmpParams;


/*******************************************************************/
/*  CUDA 外部関数宣言（Windowsのみ）                                 */
/*******************************************************************/
#if HAS_CUDA
extern "C" void PluginKernel_CUDA(
    float const* inBuf,
    float* outBuf,
    unsigned int outPitch,
    int is16f,
    const ProcAmpParams* params);
#endif


/*******************************************************************/
/*  GPU フィルタークラス                                             */
/*******************************************************************/
static const int kMaxDevices = 8;

class GPUFilter : public PrGPUFilterBase
{
public:
    prSuiteError InitializeDefaultParams(PrGPUFilterInstance* ioInstanceData)
    {
        return suiteError_NoError;
    }

    prSuiteError GetFrameDependencies(
        const PrGPUFilterRenderParams* inRenderParams,
        csSDK_int32* ioQueryIndex,
        PrGPUFilterFrameDependency* outFrameDependency)
    {
        // 現在のフレームのみ必要（時間依存なし）
        return suiteError_NoError;
    }

    prSuiteError Render(
        const PrGPUFilterRenderParams* inRenderParams,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        PPixHand* outFrame)
    {
        prSuiteError result = suiteError_NoError;

        // パラメータ構造体の構築
        ProcAmpParams params = {};
        params.mWidth  = inRenderParams->inRenderWidth;
        params.mHeight = inRenderParams->inRenderHeight;

        // TODO: パラメータの取得
        // GetParam() で各パラメータを取得し params に格納
        // 例:
        // float amount = 0;
        // GetParam(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_AMOUNT, amount);
        // params.mAmount = amount;

        params.mAmount = 50.0f;  // テスト値
        params.mColorR = 1.0f;
        params.mColorG = 1.0f;
        params.mColorB = 1.0f;
        params.mMode = 1;

        // 入力フレームの取得
        if (inFrameCount < 1) return suiteError_InvalidParms;

        // GPU バッファの取得
        void* inBuffer = nullptr;
        void* outBuffer = nullptr;
        csSDK_int32 outPitch = 0;
        int is16f = 0;

        // TODO: PPixSuite でバッファ取得
        // mDeviceIndex でデバイスを識別
        // GetGPUPPixData() 等で実際のGPUポインタを取得

        // --- License check (shared cache with CPU) ---
        const bool isLicensed = IsGpuLicenseAuthenticated();

        // ========== GPUカーネル ディスパッチ ==========
#if HAS_CUDA
        if (inRenderParams->inRenderGPUType == PrGPURenderType_CUDA)
        {
            PluginKernel_CUDA(
                (const float*)inBuffer,
                (float*)outBuffer,
                outPitch,
                is16f,
                &params);
        }
#endif

#if HAS_METAL
        if (inRenderParams->inRenderGPUType == PrGPURenderType_Metal)
        {
            // Metal パイプラインキャッシュ
            // static void* sPipelineCache[kMaxDevices] = {};
            // MetalDevicePtr device = ...

            // TODO: Metal カーネルのディスパッチ
            // 1. .metal をコンパイル（パイプラインキャッシュに保存）
            // 2. ProcAmpParams をバッファにコピー
            // 3. setBuffer() でバッファを設定
            // 4. dispatchThreadgroups() で実行
        }
#endif

        if (inRenderParams->inRenderGPUType == PrGPURenderType_OpenCL)
        {
            // OpenCL フォールバック
            // TODO: OpenCL カーネルのディスパッチ
            // 1. clCreateProgramWithSource() で .cl をコンパイル
            // 2. clSetKernelArg() で個別にパラメータを設定
            // 3. clEnqueueNDRangeKernel() で実行
        }

        return result;
    }
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<GPUFilter>)
