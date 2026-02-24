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
    #include "__TPL_MATCH_NAME__.cl.h"
    #include <CL/cl.h>
#else
    #include <OpenCL/cl.h>
    #include <Metal/Metal.h>
    #include <dlfcn.h>
#endif

#include <fstream>
#include <string>

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
/*  ProcAmpParams — カーネルに渡すパラメータ                         */
/*  注意: .cl/.cu の引数順序と同期すること                            */
/*******************************************************************/
typedef struct
{
    int     mPitch;
    int     m16f;
    int     mWidth;
    int     mHeight;
    float   mAmount;
    float   mColorR;
    float   mColorG;
    float   mColorB;
    int     mMode;
} ProcAmpParams;


static int NormalizePopupValue(int value, int maxValue)
{
    if (value >= 1 && value <= maxValue)
    {
        return value - 1;
    }
    if (value >= 0 && value < maxValue)
    {
        return value;
    }
    return 0;
}

static void DecodePackedColor32(csSDK_uint32 packed, float& r, float& g, float& b)
{
    r = ((packed >> 16) & 0xFF) / 255.0f;
    g = ((packed >> 8) & 0xFF) / 255.0f;
    b = (packed & 0xFF) / 255.0f;
}

static void DecodePackedColor64(csSDK_uint64 packed, float& r, float& g, float& b)
{
    r = ((packed >> 32) & 0xFFFF) / 65535.0f;
    g = ((packed >> 16) & 0xFFFF) / 65535.0f;
    b = (packed & 0xFFFF) / 65535.0f;
}

#if HAS_METAL
static prSuiteError CheckForMetalError(NSError* error)
{
    return error ? suiteError_Fail : suiteError_NoError;
}
#endif


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
enum { kMaxDevices = 12 };
static cl_kernel sKernelCache[kMaxDevices] = {};
#if HAS_METAL
static id<MTLComputePipelineState> sMetalPipelineStateCache[kMaxDevices] = {};
#endif

class GPUFilter : public PrGPUFilterBase
{
public:
    virtual prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData)
    {
        PrGPUFilterBase::Initialize(ioInstanceData);

        if (mDeviceIndex >= kMaxDevices)
        {
            return suiteError_Fail;
        }

        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
        {
#if HAS_CUDA
            return suiteError_NoError;
#else
            return suiteError_Fail;
#endif
        }

        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
        {
            mKernelOpenCL = sKernelCache[mDeviceIndex];
            if (mKernelOpenCL)
            {
                return suiteError_NoError;
            }

            cl_int clResult = CL_SUCCESS;
            const char* define16f = "#define GF_OPENCL_SUPPORTS_16F 1\n";
            const char* sources[2] = { define16f, nullptr };
            size_t sourceSizes[2] = { strlen(define16f), 0 };

#if _WIN32
            sources[1] = k__TPL_MATCH_NAME___OpenCLString;
            sourceSizes[1] = strlen(k__TPL_MATCH_NAME___OpenCLString);
#else
            std::string clSourcePath = __FILE__;
            size_t slashPos = clSourcePath.find_last_of("\\/");
            if (slashPos != std::string::npos)
            {
                clSourcePath.erase(slashPos);
            }
            clSourcePath += "/__TPL_MATCH_NAME__.cl";
            std::ifstream clFile(clSourcePath.c_str());
            if (!clFile.is_open())
            {
                return suiteError_Fail;
            }
            std::string clSource((std::istreambuf_iterator<char>(clFile)), std::istreambuf_iterator<char>());
            sources[1] = clSource.c_str();
            sourceSizes[1] = clSource.size();
#endif

            cl_context context = (cl_context)mDeviceInfo.outContextHandle;
            cl_device_id device = (cl_device_id)mDeviceInfo.outDeviceHandle;
            cl_program program = clCreateProgramWithSource(context, 2, sources, sourceSizes, &clResult);
            if (clResult != CL_SUCCESS || !program)
            {
                return suiteError_Fail;
            }

            clResult = clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0);
            if (clResult != CL_SUCCESS)
            {
                clReleaseProgram(program);
                return suiteError_Fail;
            }

            mKernelOpenCL = clCreateKernel(program, "PluginKernel", &clResult);
            clReleaseProgram(program);
            if (clResult != CL_SUCCESS || !mKernelOpenCL)
            {
                return suiteError_Fail;
            }

            sKernelCache[mDeviceIndex] = mKernelOpenCL;
            return suiteError_NoError;
        }

#if HAS_METAL
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal)
        {
            if (!sMetalPipelineStateCache[mDeviceIndex])
            @autoreleasepool {
                prSuiteError result = suiteError_NoError;
                NSBundle* pluginBundle = [NSBundle bundleWithIdentifier:@"__TPL_BUNDLE_ID__"];
                if (!pluginBundle)
                {
                    Dl_info dlInfo;
                    if (dladdr((const void*)&sMetalPipelineStateCache, &dlInfo) && dlInfo.dli_fname)
                    {
                        NSString* execPath = [NSString stringWithUTF8String:dlInfo.dli_fname];
                        NSString* bundlePath = [[[execPath stringByDeletingLastPathComponent] stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];
                        pluginBundle = [NSBundle bundleWithPath:bundlePath];
                    }
                }

                NSString* pluginBundlePath = [pluginBundle bundlePath];
                NSString* metalLibPath = [pluginBundlePath stringByAppendingPathComponent:@"Contents/Resources/MetalLib/__TPL_MATCH_NAME__.metallib"];
                if (!(metalLibPath && [[NSFileManager defaultManager] fileExistsAtPath:metalLibPath]))
                {
                    return suiteError_Fail;
                }

                NSError* error = nil;
                id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;
                id<MTLLibrary> library = [[device newLibraryWithFile:metalLibPath error:&error] autorelease];
                result = CheckForMetalError(error);
                if (result != suiteError_NoError)
                {
                    return result;
                }

                id<MTLFunction> function = [[library newFunctionWithName:@"PluginKernel"] autorelease];
                if (!function)
                {
                    return suiteError_Fail;
                }

                sMetalPipelineStateCache[mDeviceIndex] = [device newComputePipelineStateWithFunction:function error:&error];
                return CheckForMetalError(error);
            }
            return suiteError_NoError;
        }
#endif

        return suiteError_Fail;
    }

    prSuiteError InitializeDefaultParams(PrGPUFilterInstance* ioInstanceData)
    {
        return suiteError_NoError;
    }

    virtual prSuiteError Shutdown()
    {
        if (mDeviceIndex >= 0 && mDeviceIndex < kMaxDevices)
        {
            if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL && sKernelCache[mDeviceIndex])
            {
                clReleaseKernel(sKernelCache[mDeviceIndex]);
                sKernelCache[mDeviceIndex] = nullptr;
                mKernelOpenCL = nullptr;
            }

#if HAS_METAL
            if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal && sMetalPipelineStateCache[mDeviceIndex])
            {
                [sMetalPipelineStateCache[mDeviceIndex] release];
                sMetalPipelineStateCache[mDeviceIndex] = nil;
            }
#endif
        }

        return PrGPUFilterBase::Shutdown();
    }

    prSuiteError GetFrameDependencies(
        const PrGPUFilterRenderParams* inRenderParams,
        csSDK_int32* ioQueryIndex,
        PrGPUFilterFrameDependency* outFrameRequirements)
    {
        if (*ioQueryIndex == 0)
        {
            outFrameRequirements->outDependencyType = PrGPUDependency_InputFrame;
            outFrameRequirements->outTrackID = 0;
            outFrameRequirements->outSequenceTime = inRenderParams->inSequenceTime;
            return suiteError_NoError;
        }
        return suiteError_NotImplemented;
    }

    prSuiteError Render(
        const PrGPUFilterRenderParams* inRenderParams,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        PPixHand* outFrame)
    {
        if (inFrameCount < 1)
        {
            return suiteError_InvalidParms;
        }

        prSuiteError result = suiteError_NoError;
        const PrTime clipTime = inRenderParams->inClipTime;

        ProcAmpParams params = {};
        void* inFrameData = nullptr;
        void* outFrameData = nullptr;
        mGPUDeviceSuite->GetGPUPPixData(inFrames[0], &inFrameData);
        mGPUDeviceSuite->GetGPUPPixData(*outFrame, &outFrameData);

        PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
        mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);

        prRect bounds = {};
        mPPixSuite->GetBounds(*outFrame, &bounds);
        params.mWidth = bounds.right - bounds.left;
        params.mHeight = bounds.bottom - bounds.top;

        csSDK_int32 rowBytes = 0;
        mPPixSuite->GetRowBytes(*outFrame, &rowBytes);
        params.mPitch = rowBytes / GetGPUBytesPerPixel(pixelFormat);
        params.m16f = (pixelFormat != PrPixelFormat_GPU_BGRA_4444_32f) ? 1 : 0;

        params.mAmount = (float)GetParam(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_AMOUNT, clipTime).mFloat64;

        {
            PrParam modeParam = GetParam(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_POPUP, clipTime);
            int modeRaw = modeParam.mInt32;
            if (modeParam.mType == kPrParamType_Float64)
            {
                modeRaw = (int)(modeParam.mFloat64 + 0.5);
            }
            params.mMode = NormalizePopupValue(modeRaw, 3);
        }

        {
            PrParam colorParam = GetParam(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_COLOR, clipTime);
            if (colorParam.mType == kPrParamType_PrMemoryPtr && colorParam.mMemoryPtr)
            {
                const PF_Pixel* px = reinterpret_cast<const PF_Pixel*>(colorParam.mMemoryPtr);
                params.mColorR = px->red / 255.0f;
                params.mColorG = px->green / 255.0f;
                params.mColorB = px->blue / 255.0f;
            }
            else if (colorParam.mType == kPrParamType_Int64)
            {
                DecodePackedColor64((csSDK_uint64)colorParam.mInt64, params.mColorR, params.mColorG, params.mColorB);
            }
            else
            {
                DecodePackedColor32((csSDK_uint32)colorParam.mInt32, params.mColorR, params.mColorG, params.mColorB);
            }
        }

        const bool isLicensed = IsGpuLicenseAuthenticated();
        (void)isLicensed;

#if HAS_CUDA
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
        {
            PluginKernel_CUDA(
                (const float*)inFrameData,
                (float*)outFrameData,
                (unsigned int)params.mPitch,
                params.m16f,
                &params);
            return suiteError_NoError;
        }
#endif

#if HAS_METAL
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal)
        {
            id<MTLComputePipelineState> pso = sMetalPipelineStateCache[mDeviceIndex];
            if (!pso)
            {
                return suiteError_Fail;
            }

            @autoreleasepool {
                id<MTLCommandQueue> commandQueue = (id<MTLCommandQueue>)mDeviceInfo.outCommandQueueHandle;
                if (!commandQueue)
                {
                    return suiteError_Fail;
                }

                id<MTLBuffer> inBuffer = (id<MTLBuffer>)inFrameData;
                id<MTLBuffer> outBuffer = (id<MTLBuffer>)outFrameData;
                if (!inBuffer || !outBuffer)
                {
                    return suiteError_Fail;
                }

                id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;
                id<MTLBuffer> parameterBuffer = [[device newBufferWithBytes:&params length:sizeof(ProcAmpParams) options:MTLResourceStorageModeManaged] autorelease];
                if (!parameterBuffer)
                {
                    return suiteError_Fail;
                }

                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:pso];
                [encoder setBuffer:inBuffer offset:0 atIndex:0];
                [encoder setBuffer:outBuffer offset:0 atIndex:1];
                [encoder setBuffer:parameterBuffer offset:0 atIndex:2];

                const NSUInteger tgW = 16;
                const NSUInteger tgH = 16;
                MTLSize threadsPerThreadgroup = MTLSizeMake(tgW, tgH, 1);
                MTLSize threadsPerGrid = MTLSizeMake((NSUInteger)params.mWidth, (NSUInteger)params.mHeight, 1);
                [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
            }
            return suiteError_NoError;
        }
#endif

        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
        {
            if (!mKernelOpenCL)
            {
                return suiteError_Fail;
            }

            cl_command_queue queue = (cl_command_queue)mDeviceInfo.outCommandQueueHandle;
            cl_mem inMem = (cl_mem)inFrameData;
            cl_mem outMem = (cl_mem)outFrameData;

            int arg = 0;
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(cl_mem), &inMem);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(cl_mem), &outMem);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(int), &params.mPitch);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(int), &params.mPitch);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(int), &params.m16f);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(int), &params.mWidth);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(int), &params.mHeight);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(float), &params.mAmount);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(float), &params.mColorR);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(float), &params.mColorG);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(float), &params.mColorB);
            clSetKernelArg(mKernelOpenCL, arg++, sizeof(int), &params.mMode);

            const size_t globalWorkSize[2] = {(size_t)params.mWidth, (size_t)params.mHeight};
            cl_int clResult = clEnqueueNDRangeKernel(queue, mKernelOpenCL, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
            if (clResult != CL_SUCCESS)
            {
                return suiteError_Fail;
            }
            clFinish(queue);
            return suiteError_NoError;
        }

        return result;
    }

private:
    cl_kernel mKernelOpenCL = nullptr;
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<GPUFilter>)
