/*******************************************************************/
/*                                                                 */
/*  OST_WindyLines - Particle Line Effect Plugin                   */
/*  for Adobe Premiere Pro                                         */
/*                                                                 */
/*  Copyright (c) 2026 Kiyoto Nakamura. All rights reserved.       */
/*                                                                 */
/*  This plugin was developed using the Adobe Premiere Pro SDK.    */
/*  Portions based on SDK sample code:                             */
/*    Copyright 2012 Adobe Systems Incorporated.                   */
/*    Used in accordance with the Adobe Developer SDK License.     */
/*                                                                 */
/*  This software is not affiliated with or endorsed by Adobe.     */
/*                                                                 */
/*******************************************************************/


#include "OST_WindyLines.h"
#include "OST_WindyLines_Version.h"
#include "OST_WindyLines_WatermarkMask.h"
#include "OST_WindyLines_Common.h"
#include "PrGPUFilterModule.h"
#include "PrSDKVideoSegmentProperties.h"

#if _WIN32
    #include "OST_WindyLines.cl.h"
    #include <CL/cl.h>
#else
    #include <OpenCL/cl.h>
	#include <pwd.h>
	#include <unistd.h>
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
#include <sstream>

// Debug logging function is now in OST_WindyLines.h

#if _WIN32
    #if defined(MAJOR_VERSION)
        #pragma push_macro("MAJOR_VERSION")
        #undef MAJOR_VERSION
        #define OST_WINDYLINES_RESTORE_MAJOR_VERSION 1
    #endif
    #if defined(MINOR_VERSION)
        #pragma push_macro("MINOR_VERSION")
        #undef MINOR_VERSION
        #define OST_WINDYLINES_RESTORE_MINOR_VERSION 1
    #endif
    #if defined(PATCH_LEVEL)
        #pragma push_macro("PATCH_LEVEL")
        #undef PATCH_LEVEL
        #define OST_WINDYLINES_RESTORE_PATCH_LEVEL 1
    #endif
    #include <cuda_runtime.h>
    #if defined(OST_WINDYLINES_RESTORE_PATCH_LEVEL)
        #pragma pop_macro("PATCH_LEVEL")
        #undef OST_WINDYLINES_RESTORE_PATCH_LEVEL
    #endif
    #if defined(OST_WINDYLINES_RESTORE_MINOR_VERSION)
        #pragma pop_macro("MINOR_VERSION")
        #undef OST_WINDYLINES_RESTORE_MINOR_VERSION
    #endif
    #if defined(OST_WINDYLINES_RESTORE_MAJOR_VERSION)
        #pragma pop_macro("MAJOR_VERSION")
        #undef OST_WINDYLINES_RESTORE_MAJOR_VERSION
    #endif
    #include "DirectXUtils.h"
    #include <vector>
    #include <winhttp.h>
    #pragma comment(lib, "winhttp.lib")
    #define HAS_CUDA    1
    #define HAS_DIRECTX 1
    #define HAS_METAL   0

#if HAS_CUDA
static float4* sCudaLineData = nullptr;
static int* sCudaTileOffsets = nullptr;
static int* sCudaTileCounts = nullptr;
static int* sCudaLineIndices = nullptr;
static int* sCudaAlphaBounds = nullptr;
static size_t sCudaLineDataBytes = 0;
static size_t sCudaTileOffsetsBytes = 0;
static size_t sCudaTileCountsBytes = 0;
static size_t sCudaLineIndicesBytes = 0;
static size_t sCudaAlphaBoundsBytes = 0;
static unsigned char* sCudaWatermarkFill = nullptr;
static unsigned char* sCudaWatermarkOutline = nullptr;
static size_t sCudaWatermarkFillBytes = 0;
static size_t sCudaWatermarkOutlineBytes = 0;
static bool sCudaWatermarkUploaded = false;

static void EnsureCudaBuffer(void** buffer, size_t& capacityBytes, size_t requiredBytes)
{
	if (requiredBytes <= capacityBytes)
	{
		return;
	}
	if (*buffer)
	{
		cudaFree(*buffer);
		*buffer = nullptr;
	}
	if (requiredBytes > 0)
	{
		cudaMalloc(buffer, requiredBytes);
		capacityBytes = requiredBytes;
	}
}
#endif
#else
    #define HAS_CUDA    0
    #define HAS_DIRECTX 0
    #define HAS_METAL   1
    #include <Metal/Metal.h>
#endif
#include <math.h>


/*
**The ProcAmp2Params structure MUST match the kernel values sequence exactly.
** Order from OST_WindyLines.cl GF_KERNEL_FUNCTION(ProcAmp2Kernel, ...) values section:
*/
typedef struct
{
	// Values from GF_KERNEL_FUNCTION - MUST match order exactly
	int mPitch;              // inPitch
	int m16f;                // in16f
	unsigned int mWidth;     // inWidth
	unsigned int mHeight;    // inHeight
	float mLineCenterX;      // inLineCenterX
	float mLineCenterY;      // inLineCenterY
	float mOriginOffsetX;    // inOriginOffsetX
	float mOriginOffsetY;    // inOriginOffsetY
	float mLineCos;          // inLineCos
	float mLineSin;          // inLineSin
	float mLineLength;       // inLineLength
	float mLineThickness;    // inLineThickness
	float mLineLifetime;     // inLineLifetime
	float mLineTravel;       // inLineTravel
	float mLineTailFade;     // inLineTailFade
	float mLineDepthStrength;// inLineDepthStrength
	float mLineR;            // inLineR
	float mLineG;            // inLineG
	float mLineB;            // inLineB
	float mLineAA;           // inLineAA
	int mLineCap;            // inLineCap
	float mLineSkew;         // inLineSkew
	int mLineCount;          // inLineCount
	int mLineSeed;           // inLineSeed
	int mLineEasing;         // inLineEasing
	int mLineInterval;       // inLineInterval
	int mLineAllowMidPlay;   // inLineAllowMidPlay
	int mHideElement;        // inHideElement
	int mBlendMode;          // inBlendMode
	float mFrameIndex;       // inFrameIndex
	int mLineDownsample;     // inLineDownsample
	// Note: GF_PTR buffers (inLineData, inTileOffsets, etc.) are passed separately
	int mTileCountX;         // inTileCountX
	int mTileSize;           // inTileSize
	int mFocusEnable;        // inFocusEnable
	float mFocusDepth;       // inFocusDepth
	float mFocusRange;       // inFocusRange
	float mFocusBlurStrength;// inFocusBlurStrength
	int mShadowEnable;       // inShadowEnable
	float mShadowColorR;     // inShadowColorR
	float mShadowColorG;     // inShadowColorG
	float mShadowColorB;     // inShadowColorB
	float mShadowOffsetX;    // inShadowOffsetX
	float mShadowOffsetY;    // inShadowOffsetY
	float mShadowOpacity;    // inShadowOpacity
	float mLineSpawnScaleX;  // inSpawnScaleX
	float mLineSpawnScaleY;  // inSpawnScaleY
	float mSpawnRotationCos; // inSpawnRotationCos
	float mSpawnRotationSin; // inSpawnRotationSin
	int mShowSpawnArea;      // inShowSpawnArea
	float mSpawnAreaColorR;  // inSpawnAreaColorR
	float mSpawnAreaColorG;  // inSpawnAreaColorG
	float mSpawnAreaColorB;  // inSpawnAreaColorB
	int mIsBGRA;             // inIsBGRA
	float mAlphaBoundsMinX;  // inAlphaBoundsMinX
	float mAlphaBoundsMinY;  // inAlphaBoundsMinY
	float mAlphaBoundsWidth; // inAlphaBoundsWidth
	float mAlphaBoundsHeight;// inAlphaBoundsHeight
	int mMotionBlurEnable;   // inMotionBlurEnable
	int mMotionBlurSamples;  // inMotionBlurSamples
	float mMotionBlurStrength;// inMotionBlurStrength
	int mMotionBlurType;     // inMotionBlurType (0=Bidirectional, 1=Trail)
	float mMotionBlurVelocity;// inMotionBlurVelocity (0=fixed, 1=full velocity link)
	// Additional fields for CPU-side use (not passed to kernel)
	int mTileCountY;
	float mSeqTimeHash;
} ProcAmp2Params;

#if HAS_DIRECTX
static std::vector<DXContextPtr> sDXContextCache;
static std::vector<ShaderObjectPtr> sShaderObjectCache;
static std::vector<ShaderObjectPtr> sShaderObjectAlphaCache;
#endif //HAS_DIRECTX

// === License verification: shared with CPU (single source of truth) ===
#include "OST_WindyLines_License.h"

static bool IsGpuLicenseAuthenticated()
{
	RefreshLicenseAuthenticatedState(false);
	return IsLicenseAuthenticated();
}

#if HAS_METAL

prSuiteError CheckForMetalError(NSError *inError)
{
	if (inError)
	{
		//char const * errorDescription = [[inError localizedDescription] cStringUsingEncoding:NSASCIIStringEncoding];
		return suiteError_Fail;  //For debugging, uncomment above line and set breakpoint here
	}
	return suiteError_NoError;
}
#endif //HAS_METAL

#if HAS_CUDA
//  CUDA KERNEL
//  * See OST_WindyLines.cu
extern void ProcAmp2_CUDA(
	float* ioBuffer,
	int pitch,
	int is16f,
	int isBGRA,
	int width,
	int height,
	float lineCenterX,
	float lineCenterY,
	float originOffsetX,
	float originOffsetY,
	float lineCos,
	float lineSin,
	float lineLength,
	float lineThickness,
	float lineLifetime,
	float lineTravel,
	float lineTailFade,
	float lineDepthStrength,
	float lineR,
	float lineG,
	float lineB,
	float lineAA,
	int lineCap,
	float lineSkew,
	int lineCount,
	int lineSeed,
	int lineEasing,
	int lineInterval,
	int lineAllowMidPlay,
	int hideElement,
	int blendMode,
	float frameIndex,
	int lineDownsample,
	float4* lineData,
	int* tileOffsets,
	int* tileCounts,
	int* lineIndices,
	int tileCountX,
	int tileSize,
	int focusEnable,
	float focusDepth,
	float focusRange,
	float focusBlurStrength,
	int shadowEnable,
	float shadowColorR,
	float shadowColorG,
	float shadowColorB,
	float shadowOffsetX,
	float shadowOffsetY,
	float shadowOpacity,
	float spawnScaleX,
	float spawnScaleY,
	float spawnRotationCos,
	float spawnRotationSin,
	int showSpawnArea,
	float spawnAreaColorR,
	float spawnAreaColorG,
	float spawnAreaColorB,
	float alphaBoundsMinX,
	float alphaBoundsMinY,
	float alphaBoundsWidth,
	float alphaBoundsHeight,
	int motionBlurEnable,
	int motionBlurSamples,
	float motionBlurStrength);
extern void ProcAmp2_CUDA_ComputeAlphaBounds(
	float* ioBuffer,
	int pitch,
	int is16f,
	int width,
	int height,
	int* outBounds,
	int stride,
	float alphaThreshold);
extern void WatermarkOverlay_CUDA(
	float* ioBuffer,
	int pitch,
	int is16f,
	int isBGRA,
	int width,
	int height,
	unsigned char* fillMask,
	unsigned char* outlineMask,
	int maskWidth,
	int maskHeight,
	int marginX,
	int marginY,
	float dsScale,
	float fillOpacity,
	float outlineOpacity);
#endif // HAS_CUDA

size_t DivideRoundUp(
    size_t inValue,
    size_t inMultiple)
{
    return inValue ? (inValue + inMultiple - 1) / inMultiple : 0;
}

static void DecodePackedColor(
	csSDK_uint32 packed,
	float& outR,
	float& outG,
	float& outB)
{
	const csSDK_uint32 r = (packed >> 16) & 0xFF;
	const csSDK_uint32 g = (packed >> 8) & 0xFF;
	const csSDK_uint32 b = packed & 0xFF;
	outR = r / 255.0f;
	outG = g / 255.0f;
	outB = b / 255.0f;
}

static void DecodePackedColor64(
	csSDK_uint64 packed,
	float& outR,
	float& outG,
	float& outB)
{
	const csSDK_uint32 r = (csSDK_uint32)((packed >> 32) & 0xFFFF);
	const csSDK_uint32 g = (csSDK_uint32)((packed >> 16) & 0xFFFF);
	const csSDK_uint32 b = (csSDK_uint32)(packed & 0xFFFF);
	outR = r / 65535.0f;
	outG = g / 65535.0f;
	outB = b / 65535.0f;
}

struct Float4
{
	float x;
	float y;
	float z;
	float w;
};

struct LineBinBounds
{
	int minX;
	int maxX;
	int minY;
	int maxY;
};

static float HalfToFloat(csSDK_uint16 h)
{
	const csSDK_uint32 sign = (h & 0x8000u) << 16;
	const csSDK_uint32 exp = (h & 0x7C00u) >> 10;
	const csSDK_uint32 mantissa = (h & 0x03FFu);
	csSDK_uint32 fexp = 0;
	csSDK_uint32 fmant = 0;
	if (exp == 0)
	{
		if (mantissa == 0)
		{
			fexp = 0;
			fmant = 0;
		}
		else
		{
			int e = -1;
			csSDK_uint32 m = mantissa;
			while ((m & 0x0400u) == 0)
			{
				m <<= 1;
				e -= 1;
			}
			m &= 0x03FFu;
			fexp = (csSDK_uint32)(127 - 15 + 1 + e);
			fmant = m << 13;
		}
	}
	else if (exp == 0x1F)
	{
		fexp = 0xFF;
		fmant = mantissa << 13;
	}
	else
	{
		fexp = exp + (127 - 15);
		fmant = mantissa << 13;
	}
	const csSDK_uint32 bits = sign | (fexp << 23) | fmant;
	float out = 0.0f;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static void ComputeAlphaBoundsCPU(
	const char* pixels,
	int width,
	int height,
	int rowBytes,
	bool is16f,
	int stride,
	float threshold,
	int& outMinX,
	int& outMinY,
	int& outMaxX,
	int& outMaxY)
{
	outMinX = width;
	outMinY = height;
	outMaxX = -1;
	outMaxY = -1;
	for (int y = 0; y < height; y += stride)
	{
		const char* row = pixels + y * rowBytes;
		for (int x = 0; x < width; x += stride)
		{
			float alpha = 0.0f;
			if (is16f)
			{
				const csSDK_uint16* p16 = (const csSDK_uint16*)row;
				alpha = HalfToFloat(p16[x * 4 + 3]);
			}
			else
			{
				const float* p32 = (const float*)row;
				alpha = p32[x * 4 + 3];
			}
			if (alpha > threshold)
			{
				if (x < outMinX) outMinX = x;
				if (y < outMinY) outMinY = y;
				if (x > outMaxX) outMaxX = x;
				if (y > outMaxY) outMaxY = y;
			}
		}
	}
	if (outMaxX < outMinX || outMaxY < outMinY)
	{
		outMinX = 0;
		outMinY = 0;
		outMaxX = width > 0 ? (width - 1) : 0;
		outMaxY = height > 0 ? (height - 1) : 0;
	}
}

static int NormalizePopupValue(int value, int maxValue)
{
	// Premiere Pro GPU GetParam() returns 0-based popup values
	// Just clamp to valid range [0, maxValue-1]
	if (value >= 0 && value < maxValue)
	{
		return value;
	}
	return 0;
}

static int NormalizePopupParam(const PrParam& param, int maxValue)
{
	int raw = 0;
	if (param.mType == kPrParamType_Int32)
	{
		raw = param.mInt32;
	}
	else if (param.mType == kPrParamType_Int64)
	{
		raw = (int)param.mInt64;
	}
	else if (param.mType == kPrParamType_Float64)
	{
		raw = (int)(param.mFloat64 + 0.5);
	}
	return NormalizePopupValue(raw, maxValue);
}

static bool BoolParamFromPrParam(const PrParam& param)
{
	switch (param.mType)
	{
		case kPrParamType_Bool:
		case kPrParamType_Int32:
			return param.mInt32 != 0;
		case kPrParamType_Int8:
			return param.mInt8 != 0;
		case kPrParamType_Int16:
			return param.mInt16 != 0;
		case kPrParamType_Int64:
			return param.mInt64 != 0;
		case kPrParamType_Float32:
			return param.mFloat32 >= 0.5f;
		case kPrParamType_Float64:
			return param.mFloat64 >= 0.5;
		default:
			return false;
	}
}


/*
**
*/
enum {kMaxDevices = 12};
static cl_kernel sKernelCache[kMaxDevices] = {};
static cl_kernel sKernelCacheAlpha[kMaxDevices] = {};
static cl_kernel sKernelCacheWatermark[kMaxDevices] = {};
#if HAS_METAL
static id<MTLComputePipelineState> sMetalPipelineStateCache[kMaxDevices] = {};
static id<MTLComputePipelineState> sMetalAlphaPipelineStateCache[kMaxDevices] = {};
static id<MTLComputePipelineState> sMetalWatermarkPipelineStateCache[kMaxDevices] = {};
#endif

/*
**
*/
class ProcAmp2 :
	public PrGPUFilterBase
{
public:
	virtual prSuiteError GetFrameDependencies(
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
	virtual prSuiteError Initialize(
		PrGPUFilterInstance* ioInstanceData)
	{
		// FORCE CPU FALLBACK: Uncomment the line below to test CPU implementation
		// return suiteError_Fail;
		
		PrGPUFilterBase::Initialize(ioInstanceData);

		if (mDeviceIndex > kMaxDevices)
		{
			// Exceeded max device count
			return suiteError_Fail;
		}

		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
		{
#if HAS_CUDA
			// Nothing to do here. CUDA Kernel statically linked
			return suiteError_NoError;
#else
			return suiteError_Fail;
#endif
		}

		// Load and compile the kernel - a real plugin would cache binaries to disk
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
		{
			mKernelOpenCL = sKernelCache[mDeviceIndex];
			mKernelOpenCLAlpha = sKernelCacheAlpha[mDeviceIndex];
			mKernelOpenCLWatermark = sKernelCacheWatermark[mDeviceIndex];
			if (!mKernelOpenCL)
			{
				cl_int result = CL_SUCCESS;
				//A real plugin would check to see if 16f access is actually supported.
#if _WIN32
				char const *k16fString = "#define GF_OPENCL_SUPPORTS_16F 1\n";
				size_t sizes[] = { strlen(k16fString), strlen(kOST_WindyLines_OpenCLString) };
				char const *strings[] = { k16fString, kOST_WindyLines_OpenCLString };
#else
				// Mac: Try to use preprocessed header first, fallback to reading .cl file
				char const *k16fString = "#define GF_OPENCL_SUPPORTS_16F 1\n";
				size_t sizes[2];
				char const *strings[2];
				
				// Try to include preprocessed header (if build step creates it)
				#ifdef OST_WINDYLINES_OPENCL_STRING_AVAILABLE
					// Preprocessed header available
					sizes[0] = strlen(k16fString);
					sizes[1] = strlen(kOST_WindyLines_OpenCLString);
					strings[0] = k16fString;
					strings[1] = kOST_WindyLines_OpenCLString;
				#else
					// Fallback: Read OpenCL source from file
					std::string clSourcePath = __FILE__;
					const size_t slashPos = clSourcePath.find_last_of("\\/");
					if (slashPos != std::string::npos)
					{
						clSourcePath.erase(slashPos);
					}
					clSourcePath += "/OST_WindyLines.cl";
					std::ifstream clFile(clSourcePath.c_str());
					if (!clFile.is_open())
					{
						return suiteError_Fail;
					}
					std::string clSource((std::istreambuf_iterator<char>(clFile)), std::istreambuf_iterator<char>());
					clFile.close();
					
					// Note: This will fail if GF_KERNEL_FUNCTION macros are not preprocessed
					// For production, add a build step to preprocess the .cl file
					sizes[0] = strlen(k16fString);
					sizes[1] = clSource.length();
					strings[0] = k16fString;
					strings[1] = clSource.c_str();
				#endif
#endif
				cl_context context = (cl_context)mDeviceInfo.outContextHandle;
				cl_device_id device = (cl_device_id)mDeviceInfo.outDeviceHandle;
				cl_program program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
				if (result != CL_SUCCESS)
				{
					return suiteError_Fail;
				}

				result = clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0);
				if (result != CL_SUCCESS)
				{
					return suiteError_Fail;
				}

				mKernelOpenCL = clCreateKernel(program, "ProcAmp2Kernel", &result);
				if (result != CL_SUCCESS)
				{
					return suiteError_Fail;
				}
				mKernelOpenCLAlpha = clCreateKernel(program, "AlphaBoundsKernel", &result);
				if (result != CL_SUCCESS)
				{
					return suiteError_Fail;
				}
				mKernelOpenCLWatermark = clCreateKernel(program, "WatermarkOverlayKernel", &result);
				if (result != CL_SUCCESS)
				{
					return suiteError_Fail;
				}

				sKernelCache[mDeviceIndex] = mKernelOpenCL;
				sKernelCacheAlpha[mDeviceIndex] = mKernelOpenCLAlpha;
				sKernelCacheWatermark[mDeviceIndex] = mKernelOpenCLWatermark;
			}
			return suiteError_NoError;
		}
#if HAS_DIRECTX
		// DirectX/HLSL enabled - CUDA disabled
		// NOTE: HLSL/DirectX path is NOT used by Premiere Pro (After Effects only).
		// HLSL files have been moved to legacy/ folder.
		// Premiere Pro uses CUDA or OpenCL only.
		else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_DirectX)
		{
			if (mDeviceIndex >= sDXContextCache.size())
			{
				sDXContextCache.resize(mDeviceIndex + 1);
				sShaderObjectCache.resize(mDeviceIndex + 1);
				sShaderObjectAlphaCache.resize(mDeviceIndex + 1);
			}
			if (!sDXContextCache[mDeviceIndex])
			{
				// Create the DXContext
				DXContextPtr dxContext = std::make_shared<DXContext>();
				if (!dxContext->Initialize(
					(ID3D12Device*)mDeviceInfo.outDeviceHandle,
					(ID3D12CommandQueue*)mDeviceInfo.outCommandQueueHandle))
				{
					return suiteError_Fail;
				}

                std::wstring csoPath, sigPath;
                if (!GetShaderPath(L"OST_WindyLines", csoPath, sigPath))
                {
                    return suiteError_Fail;
                }

				// Load the shader
				sShaderObjectCache[mDeviceIndex] = std::make_shared<ShaderObject>();
				if (!dxContext->LoadShader(
					csoPath.c_str(),
					sigPath.c_str(),
					sShaderObjectCache[mDeviceIndex]))
				{
					return suiteError_Fail;
				}

				// Load the alpha shader
				std::wstring alphaCsoPath, alphaSigPath;
				if (!GetShaderPath(L"OST_WindyLines_Alpha", alphaCsoPath, alphaSigPath))
				{
					return suiteError_Fail;
				}
				sShaderObjectAlphaCache[mDeviceIndex] = std::make_shared<ShaderObject>();
				if (!dxContext->LoadShader(
					alphaCsoPath.c_str(),
					alphaSigPath.c_str(),
					sShaderObjectAlphaCache[mDeviceIndex]))
				{
					return suiteError_Fail;
				}

				// Cache the DXContext
				sDXContextCache[mDeviceIndex] = dxContext;
			}
			return suiteError_NoError;
		}
#endif //HAS_DIRECTX
#if HAS_METAL
		else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal)
		{
			//A real plugin would check version here - the PremierePro 10.3.0 release has Metal, but does not support Metal plugins.
            if (mDeviceIndex > sizeof(sMetalPipelineStateCache) / sizeof(id<MTLComputePipelineState>))
            {
                return suiteError_Fail;        // Exceeded max device count
            }
            
			if (!sMetalPipelineStateCache[mDeviceIndex])
			@autoreleasepool{
				prSuiteError result = suiteError_NoError;

                // Use correct bundle identifier for this plugin
                NSString *pluginBundlePath = [[NSBundle bundleWithIdentifier:@"MyCompany.OST-WindyLines"] bundlePath];
                NSString *metalLibPath = [pluginBundlePath stringByAppendingPathComponent:@"Contents/Resources/MetalLib/OST_WindyLines.metallib"];
                if(!(metalLibPath && [[NSFileManager defaultManager] fileExistsAtPath:metalLibPath]))
                {
                    // Metallib file missing in path
                    return suiteError_Fail;
                }
                
				//Create a library from source
				NSError *error = nil;
				id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;
                id<MTLLibrary> library = [[device newLibraryWithFile:metalLibPath error:&error] autorelease];
				result = CheckForMetalError(error);

				//Create pipeline state from function extracted from library
				if (result == suiteError_NoError)
				{
					id<MTLFunction> function = nil;
					NSString *name = [NSString stringWithCString:"ProcAmp2Kernel" encoding:NSUTF8StringEncoding];
					function = [[library newFunctionWithName:name] autorelease];
					
                    sMetalPipelineStateCache[mDeviceIndex] = [device newComputePipelineStateWithFunction:function error:&error];
					result = CheckForMetalError(error);
				}
				if (result == suiteError_NoError)
				{
					id<MTLFunction> function = nil;
					NSString *name = [NSString stringWithCString:"AlphaBoundsKernel" encoding:NSUTF8StringEncoding];
					function = [[library newFunctionWithName:name] autorelease];

					sMetalAlphaPipelineStateCache[mDeviceIndex] = [device newComputePipelineStateWithFunction:function error:&error];
					result = CheckForMetalError(error);
				}
				if (result == suiteError_NoError)
				{
					id<MTLFunction> function = nil;
					NSString *name = [NSString stringWithCString:"WatermarkOverlayKernel" encoding:NSUTF8StringEncoding];
					function = [[library newFunctionWithName:name] autorelease];

					sMetalWatermarkPipelineStateCache[mDeviceIndex] = [device newComputePipelineStateWithFunction:function error:&error];
					result = CheckForMetalError(error);
				}
				
				return result;
			}
			return suiteError_NoError;
		}
#endif //HAS_METAL

		// Sample code is only accelerated with OpenCL & Metal
		return suiteError_Fail;
	}

	prSuiteError Render(
		const PrGPUFilterRenderParams* inRenderParams,
		const PPixHand* inFrames,
		csSDK_size_t inFrameCount,
		PPixHand* outFrame)
	{
		// NOTE: GPU fallback disabled on Windows.
		// Returning suiteError_Fail here causes Premiere Pro on Windows to render
		// a black frame instead of falling back to CPU.  Watermark is drawn by
		// the CPU SmartRender path when the sequence is exported or when GPU is
		// unavailable, so this guard is not needed for copy-protection.
		// if (!IsGpuLicenseAuthenticated())
		// {
		// 	return suiteError_Fail;
		// }

		auto normalizePopup = [](int value, int maxValue) {
			if (value >= 1 && value <= maxValue)
			{
				return value - 1;
			}
			if (value >= 0 && value < maxValue)
			{
				return value;
			}
			return 0;
		};

		// read the parameters
		ProcAmp2Params params;

		void* frameData = 0;
		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &frameData);

		void* srcFrameData = 0;
		if (inFrameCount > 0)
		{
			mGPUDeviceSuite->GetGPUPPixData(inFrames[0], &srcFrameData);
		}

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
		PrPixelFormat srcPixelFormat = pixelFormat;
		if (inFrameCount > 0)
		{
			mPPixSuite->GetPixelFormat(inFrames[0], &srcPixelFormat);
		}

		prRect bounds = {};
		mPPixSuite->GetBounds(*outFrame, &bounds);
		params.mWidth = bounds.right - bounds.left;
		params.mHeight = bounds.bottom - bounds.top;

		csSDK_int32 rowBytes = 0;
		mPPixSuite->GetRowBytes(*outFrame, &rowBytes);
		params.mPitch = rowBytes / GetGPUBytesPerPixel(pixelFormat);
		params.m16f = pixelFormat != PrPixelFormat_GPU_BGRA_4444_32f;
		csSDK_int32 srcRowBytes = 0;
		if (inFrameCount > 0)
		{
			mPPixSuite->GetRowBytes(inFrames[0], &srcRowBytes);
		}
		char* srcPixels = nullptr;
		if (inFrameCount > 0)
		{
			mPPixSuite->GetPixels(inFrames[0], PrPPixBufferAccess_ReadOnly, &srcPixels);
		}

		// Cache-consistent approach using media in point to calculate clip-relative frame
		const PrTime clipTime = inRenderParams->inClipTime;
		const bool allowMidPlay = BoolParamFromPrParam(GetParam(OST_WINDYLINES_LINE_ALLOW_MIDPLAY, inRenderParams->inClipTime));
		const bool hideElement = BoolParamFromPrParam(GetParam(OST_WINDYLINES_HIDE_ELEMENT, inRenderParams->inClipTime));
		const int blendMode = NormalizePopupParam(GetParam(OST_WINDYLINES_BLEND_MODE, inRenderParams->inClipTime), 4);
		const bool shadowEnable = BoolParamFromPrParam(GetParam(OST_WINDYLINES_SHADOW_ENABLE, inRenderParams->inClipTime));
		float shadowColorR = 0.0f, shadowColorG = 0.0f, shadowColorB = 0.0f;
		{
			PrParam shadowColorParam = GetParam(OST_WINDYLINES_SHADOW_COLOR, inRenderParams->inClipTime);
			if (shadowColorParam.mType == kPrParamType_PrMemoryPtr && shadowColorParam.mMemoryPtr)
			{
				const PF_Pixel* color = reinterpret_cast<const PF_Pixel*>(shadowColorParam.mMemoryPtr);
				shadowColorR = color->red / 255.0f;
				shadowColorG = color->green / 255.0f;
				shadowColorB = color->blue / 255.0f;
			}
			else if (shadowColorParam.mType == kPrParamType_Int64)
			{
				DecodePackedColor64(static_cast<csSDK_uint64>(shadowColorParam.mInt64), shadowColorR, shadowColorG, shadowColorB);
			}
		}
		const float shadowOffsetX = static_cast<float>(GetParam(OST_WINDYLINES_SHADOW_OFFSET_X, inRenderParams->inClipTime).mFloat64);
		const float shadowOffsetY = static_cast<float>(GetParam(OST_WINDYLINES_SHADOW_OFFSET_Y, inRenderParams->inClipTime).mFloat64);
		const float shadowOpacity = static_cast<float>(GetParam(OST_WINDYLINES_SHADOW_OPACITY, inRenderParams->inClipTime).mFloat64);
		const float spawnScaleX = static_cast<float>(GetParam(OST_WINDYLINES_LINE_SPAWN_SCALE_X, inRenderParams->inClipTime).mFloat64) / 100.0f;
		const float spawnScaleY = static_cast<float>(GetParam(OST_WINDYLINES_LINE_SPAWN_SCALE_Y, inRenderParams->inClipTime).mFloat64) / 100.0f;
		const float spawnRotationDeg = static_cast<float>(GetParam(OST_WINDYLINES_LINE_SPAWN_ROTATION, inRenderParams->inClipTime).mFloat64);
		const float spawnRotationRad = spawnRotationDeg * 3.14159265f / 180.0f;
		const float spawnCos = cosf(spawnRotationRad);
		const float spawnSin = sinf(spawnRotationRad);
		const bool showSpawnArea = BoolParamFromPrParam(GetParam(OST_WINDYLINES_LINE_SHOW_SPAWN_AREA, inRenderParams->inClipTime));
		// Read spawn area color
		float spawnAreaColorR = 0.5f, spawnAreaColorG = 0.5f, spawnAreaColorB = 1.0f;  // Default light blue
		{
			const PrParam spawnAreaColorParam = GetParam(OST_WINDYLINES_LINE_SPAWN_AREA_COLOR, inRenderParams->inClipTime);
			if (spawnAreaColorParam.mType == kPrParamType_PrMemoryPtr && spawnAreaColorParam.mMemoryPtr)
			{
				const PF_Pixel* colorPtr = reinterpret_cast<const PF_Pixel*>(spawnAreaColorParam.mMemoryPtr);
				spawnAreaColorR = colorPtr->red / 255.0f;
				spawnAreaColorG = colorPtr->green / 255.0f;
				spawnAreaColorB = colorPtr->blue / 255.0f;
			}
			else if (spawnAreaColorParam.mType == kPrParamType_Int64)
			{
				DecodePackedColor64(spawnAreaColorParam.mInt64, spawnAreaColorR, spawnAreaColorG, spawnAreaColorB);
			}
		}
		const float originOffsetX = static_cast<float>(GetParam(OST_WINDYLINES_ORIGIN_OFFSET_X, inRenderParams->inClipTime).mFloat64);
		const float originOffsetY = static_cast<float>(GetParam(OST_WINDYLINES_ORIGIN_OFFSET_Y, inRenderParams->inClipTime).mFloat64);
		const int animPattern = NormalizePopupParam(GetParam(OST_WINDYLINES_ANIM_PATTERN, inRenderParams->inClipTime), 3);
		const float centerGap = static_cast<float>(GetParam(OST_WINDYLINES_CENTER_GAP, inRenderParams->inClipTime).mFloat64);
		// Focus parameters removed
	
	// GPU's inClipTime is absolute media time, NOT clip-relative.
	// We need to find the clip start in sequence time to compute clip-relative frame.
	// Strategy: Walk the VideoSegment tree from timeline root, find the ClipNode that
	// has our mNodeID as an operator, then read TrackItemStartAsTicks from the ClipNode.
	const PrTime ticksPerFrame = inRenderParams->inRenderTicksPerFrame;
	const PrTime seqTime = inRenderParams->inSequenceTime;
	
	PrTime trackItemStart = 0;
	bool trackItemStartFound = false;
	
	// Get hash of our own mNodeID for matching against operators in the tree
	prPluginID myNodeHash = {};
	{
		char myNodeType[256] = {};
		csSDK_int32 myNodeFlags = 0;
		if (mVideoSegmentSuite)
			mVideoSegmentSuite->GetNodeInfo(mNodeID, myNodeType, &myNodeHash, &myNodeFlags);
	}
	
	// If not cached, walk the segment tree to find our ClipNode
	if (!trackItemStartFound && mVideoSegmentSuite)
	{
		// One-time tree dump for diagnostics
		static bool sTreeDumped = false;
		const bool doDump = !sTreeDumped;
		if (doDump) sTreeDumped = true;
		
		if (doDump)
			DebugLog("GPU TREE: mNodeID=%d myHash=%08x%08x%08x%08x",
				mNodeID,
				((unsigned int*)&myNodeHash)[0], ((unsigned int*)&myNodeHash)[1],
				((unsigned int*)&myNodeHash)[2], ((unsigned int*)&myNodeHash)[3]);
		
		csSDK_int32 segmentsID = -1;
		if (PrSuiteErrorSucceeded(mVideoSegmentSuite->AcquireVideoSegmentsID(mTimelineID, &segmentsID)) && segmentsID >= 0)
		{
			csSDK_int32 numSegments = 0;
			mVideoSegmentSuite->GetSegmentCount(segmentsID, &numSegments);
			if (doDump)
				DebugLog("GPU TREE: numSegments=%d seqTime=%lld", numSegments, (long long)seqTime);
			
			// Search segments for one that contains our current time
			for (csSDK_int32 segIdx = 0; segIdx < numSegments && !trackItemStartFound; ++segIdx)
			{
				PrTime segStart = 0, segEnd = 0, segOffset = 0;
				prPluginID segHash = {};
				if (!PrSuiteErrorSucceeded(mVideoSegmentSuite->GetSegmentInfo(segmentsID, segIdx, &segStart, &segEnd, &segOffset, &segHash)))
					continue;
				
				// Only process segments that contain our current sequence time
				if (seqTime < segStart || seqTime >= segEnd)
					continue;
				
				if (doDump)
					DebugLog("GPU TREE: seg[%d] start=%lld end=%lld offset=%lld", segIdx, (long long)segStart, (long long)segEnd, (long long)segOffset);
				
				csSDK_int32 rootNodeID = -1;
				if (!PrSuiteErrorSucceeded(mVideoSegmentSuite->AcquireNodeID(segmentsID, &segHash, &rootNodeID)) || rootNodeID < 0)
					continue;
				
				// BFS: walk tree looking for ClipNodes with our effect as operator
				std::vector<csSDK_int32> nodeQueue;
				nodeQueue.push_back(rootNodeID);
				std::vector<csSDK_int32> nodesToRelease;
				
				while (!nodeQueue.empty() && !trackItemStartFound)
				{
					csSDK_int32 currentNode = nodeQueue.back();
					nodeQueue.pop_back();
					
					char nodeType[256] = {};
					prPluginID nodeHash = {};
					csSDK_int32 nodeFlags = 0;
					mVideoSegmentSuite->GetNodeInfo(currentNode, nodeType, &nodeHash, &nodeFlags);
					
					// Check operators on this node
					csSDK_int32 opCount = 0;
					mVideoSegmentSuite->GetNodeOperatorCount(currentNode, &opCount);
					
					if (doDump && (opCount > 0 || strstr(nodeType, "Clip")))
						DebugLog("GPU TREE:   node=%d type='%s' ops=%d flags=%d", currentNode, nodeType, opCount, nodeFlags);
					
					for (csSDK_int32 opIdx = 0; opIdx < opCount && !trackItemStartFound; ++opIdx)
					{
						csSDK_int32 operatorID = -1;
						if (PrSuiteErrorSucceeded(mVideoSegmentSuite->AcquireOperatorNodeID(currentNode, opIdx, &operatorID)) && operatorID >= 0)
						{
							char opType[256] = {};
							prPluginID opHash = {};
							csSDK_int32 opFlags = 0;
							mVideoSegmentSuite->GetNodeInfo(operatorID, opType, &opHash, &opFlags);
							
							if (doDump)
								DebugLog("GPU TREE:     op[%d]=%d type='%s' hash=%08x%08x%08x%08x",
									opIdx, operatorID, opType,
									((unsigned int*)&opHash)[0], ((unsigned int*)&opHash)[1],
									((unsigned int*)&opHash)[2], ((unsigned int*)&opHash)[3]);
							
							// Match by hash comparison (acquired IDs differ from mNodeID)
							bool isMatch = (operatorID == mNodeID) ||
								(memcmp(&opHash, &myNodeHash, sizeof(prPluginID)) == 0);
							
							// Also try matching by FilterMatchName + RuntimeInstanceID
							if (!isMatch)
							{
								// Check if this operator is our WindyLines effect
								PrMemoryPtr nameBuf = nullptr;
								if (PrSuiteErrorSucceeded(mVideoSegmentSuite->GetNodeProperty(operatorID, kVideoSegmentProperty_Effect_FilterMatchName, &nameBuf)) && nameBuf)
								{
									if (strstr((const char*)nameBuf, "WindyLines") || strstr((const char*)nameBuf, "windy"))
									{
										isMatch = true;
										if (doDump)
											DebugLog("GPU TREE:     MATCH by FilterMatchName='%s'", (const char*)nameBuf);
									}
									mMemoryManagerSuite->PrDisposePtr(nameBuf);
								}
							}
							
							if (isMatch)
							{
								// Found our effect! Read TrackItemStartAsTicks from the parent
								PrMemoryPtr buffer = nullptr;
								if (PrSuiteErrorSucceeded(mVideoSegmentSuite->GetNodeProperty(currentNode, kVideoSegmentProperty_Clip_TrackItemStartAsTicks, &buffer)) && buffer)
								{
									std::istringstream stream((const char*)buffer);
									stream >> trackItemStart;
									trackItemStartFound = true;
									DebugLog("GPU CLIP FOUND: mNodeID=%d parentNode=%d parentType='%s' TrackItemStart=%lld (frame %lld)",
										mNodeID, currentNode, nodeType, (long long)trackItemStart,
										ticksPerFrame > 0 ? (long long)(trackItemStart / ticksPerFrame) : 0LL);
									mMemoryManagerSuite->PrDisposePtr(buffer);
								}
								else
								{
									DebugLog("GPU CLIP FOUND but no TrackItemStart: parentNode=%d type='%s'", currentNode, nodeType);
								}
							}
							mVideoSegmentSuite->ReleaseVideoNodeID(operatorID);
						}
					}
					
					// Traverse inputs
					csSDK_int32 inputCount = 0;
					mVideoSegmentSuite->GetNodeInputCount(currentNode, &inputCount);
					for (csSDK_int32 inIdx = 0; inIdx < inputCount && !trackItemStartFound; ++inIdx)
					{
						PrTime offset = 0;
						csSDK_int32 inputID = -1;
						if (PrSuiteErrorSucceeded(mVideoSegmentSuite->AcquireInputNodeID(currentNode, inIdx, &offset, &inputID)) && inputID >= 0)
						{
							nodeQueue.push_back(inputID);
							nodesToRelease.push_back(inputID);
						}
					}
				}
				
				for (csSDK_int32 nodeToRelease : nodesToRelease)
					mVideoSegmentSuite->ReleaseVideoNodeID(nodeToRelease);
				mVideoSegmentSuite->ReleaseVideoNodeID(rootNodeID);
			}
			
			mVideoSegmentSuite->ReleaseVideoSegmentsID(segmentsID);
		}
		
		if (!trackItemStartFound)
		{
			trackItemStart = 0;
			DebugLog("GPU CLIP WARN: Could not find ClipNode for mNodeID=%d, falling back to trackItemStart=0", mNodeID);
		}
	}
	
	// Clip-relative frame index: (seqTime - trackItemStart) / ticksPerFrame
	const csSDK_int64 frameIndex = (ticksPerFrame > 0) ? ((seqTime - trackItemStart) / ticksPerFrame) : 0;
	
	// DEBUG: Log time values (first 5 frames only)
	{
		static int sDebugFrameCount = 0;
		if (sDebugFrameCount < 5)
		{
			sDebugFrameCount++;
			DebugLog("GPU TIME DEBUG: clipTime=%lld seqTime=%lld trackItemStart=%lld ticksPerFrame=%lld frameIndex=%lld found=%d",
				(long long)clipTime, (long long)seqTime, (long long)trackItemStart,
				(long long)ticksPerFrame, (long long)frameIndex, trackItemStartFound ? 1 : 0);
		}
	}
	
	// Color Mode and Palette Setup (unified index)
	// Get unified preset parameter
	// IMPORTANT: Premiere Pro's GetParam() returns 0-based values for popup params
	// so we do NOT normalize (no -1 conversion needed)
	PrParam rawPresetParam = GetParam(OST_WINDYLINES_COLOR_PRESET, inRenderParams->inClipTime);
	int rawPresetIndex = 0;
	if (rawPresetParam.mType == kPrParamType_Int32)
		rawPresetIndex = rawPresetParam.mInt32;
	else if (rawPresetParam.mType == kPrParamType_Int64)
		rawPresetIndex = (int)rawPresetParam.mInt64;
	else if (rawPresetParam.mType == kPrParamType_Float64)
		rawPresetIndex = (int)(rawPresetParam.mFloat64 + 0.5);
	
	// GetParam returns 0-based: 0=単色, 1=Sep, 2=カスタム, 3=Sep, 4=Rainbow, ...
	// Clamp to valid range
	const int unifiedPresetIndex = (rawPresetIndex >= 0 && rawPresetIndex < kUnifiedPresetCount) ? rawPresetIndex : 0;
	
	// Convert unified index to colorMode and presetIndex
	int colorMode, presetIndex;
	UnifiedIndexToColorModeAndPreset(unifiedPresetIndex, colorMode, presetIndex);
	
	// Debug logging
	// 0-based: 0=単色, 1=Sep, 2=カスタム, 3=Sep, 4=Rainbow, 5=Pastel, 6=Forest, ...
	DebugLog("[GPU ColorPreset] Raw(0-based): %d → colorMode: %d, presetIndex: %d", 
		rawPresetIndex, colorMode, presetIndex);
	
	// Build color palette (8 colors)
	float colorPalette[8][3];  // RGB normalized
	
	if (colorMode == 0)  // Single (0-based)
	{
		// Single color mode: all 8 slots have the same color
		float singleR = 1.0f, singleG = 1.0f, singleB = 1.0f;
		PrParam lineColorParam = GetParam(OST_WINDYLINES_LINE_COLOR, inRenderParams->inClipTime);
		if (lineColorParam.mType == kPrParamType_PrMemoryPtr && lineColorParam.mMemoryPtr)
		{
			const PF_Pixel* color = reinterpret_cast<const PF_Pixel*>(lineColorParam.mMemoryPtr);
			singleR = color->red / 255.0f;
			singleG = color->green / 255.0f;
			singleB = color->blue / 255.0f;
		}
		else if (lineColorParam.mType == kPrParamType_Int32)
		{
			DecodePackedColor(static_cast<csSDK_uint32>(lineColorParam.mInt32), singleR, singleG, singleB);
		}
		else if (lineColorParam.mType == kPrParamType_Int64)
		{
			DecodePackedColor64(static_cast<csSDK_uint64>(lineColorParam.mInt64), singleR, singleG, singleB);
		}
		for (int i = 0; i < 8; ++i)
		{
			colorPalette[i][0] = singleR;
			colorPalette[i][1] = singleG;
			colorPalette[i][2] = singleB;
		}
		DebugLog("[GPU ColorPreset] Single color mode: R=%.2f G=%.2f B=%.2f", singleR, singleG, singleB);
	}
	else if (colorMode == 1)  // Custom (0-based)
	{
		// Custom mode: load from custom color parameters
		const int customColorParams[8] = {
			OST_WINDYLINES_CUSTOM_COLOR_1, OST_WINDYLINES_CUSTOM_COLOR_2,
			OST_WINDYLINES_CUSTOM_COLOR_3, OST_WINDYLINES_CUSTOM_COLOR_4,
			OST_WINDYLINES_CUSTOM_COLOR_5, OST_WINDYLINES_CUSTOM_COLOR_6,
			OST_WINDYLINES_CUSTOM_COLOR_7, OST_WINDYLINES_CUSTOM_COLOR_8
		};
		for (int i = 0; i < 8; ++i)
		{
			float r = 1.0f, g = 1.0f, b = 1.0f;
			PrParam colorParam = GetParam(customColorParams[i], inRenderParams->inClipTime);
			if (colorParam.mType == kPrParamType_PrMemoryPtr && colorParam.mMemoryPtr)
			{
				const PF_Pixel* color = reinterpret_cast<const PF_Pixel*>(colorParam.mMemoryPtr);
				r = color->red / 255.0f;
				g = color->green / 255.0f;
				b = color->blue / 255.0f;
			}
			else if (colorParam.mType == kPrParamType_Int32)
			{
				DecodePackedColor(static_cast<csSDK_uint32>(colorParam.mInt32), r, g, b);
			}
			else if (colorParam.mType == kPrParamType_Int64)
			{
				DecodePackedColor64(static_cast<csSDK_uint64>(colorParam.mInt64), r, g, b);
			}
			colorPalette[i][0] = r;
			colorPalette[i][1] = g;
			colorPalette[i][2] = b;
		}
		DebugLog("[GPU ColorPreset] Custom mode: Loaded 8 custom colors, Color[0]: R=%.2f G=%.2f B=%.2f", 
			colorPalette[0][0], colorPalette[0][1], colorPalette[0][2]);
	}
	else  // Preset (colorMode == 2, 0-based)
	{
		BuildPresetPalette(colorPalette, presetIndex);
		DebugLog("[GPU ColorPreset] Loaded 8 colors, Color[0]: R=%.2f G=%.2f B=%.2f", 
			colorPalette[0][0], colorPalette[0][1], colorPalette[0][2]);
	}
	
	// Default line color (for compatibility, use first palette color)
	float lineR = colorPalette[0][0];
	float lineG = colorPalette[0][1];
	float lineB = colorPalette[0][2];

	bool isBGRA = true;
#if defined(PrPixelFormat_GPU_VUYA_4444_32f)
		if (pixelFormat == PrPixelFormat_GPU_VUYA_4444_32f)
		{
			isBGRA = false;
		}
#endif
#if defined(PrPixelFormat_GPU_VUYA_4444_16f)
		if (pixelFormat == PrPixelFormat_GPU_VUYA_4444_16f)
		{
			isBGRA = false;
		}
#endif
#if defined(PrPixelFormat_GPU_BGRA_4444_32f)
		if (pixelFormat == PrPixelFormat_GPU_BGRA_4444_32f)
		{
			isBGRA = true;
		}
#endif
#if defined(PrPixelFormat_GPU_BGRA_4444_16f)
		if (pixelFormat == PrPixelFormat_GPU_BGRA_4444_16f)
		{
			isBGRA = true;
		}
#endif
		// Fallback for observed pixelFormat codes in logs.
		if (pixelFormat == (PrPixelFormat)1094992704 || pixelFormat == (PrPixelFormat)1631863616)
		{
			isBGRA = true;
		}
		params.mIsBGRA = isBGRA ? 1 : 0;
		float outC0 = 0.0f;
		float outC1 = 0.0f;
		float outC2 = 0.0f;
		if (isBGRA)
		{
			outC0 = lineB;
			outC1 = lineG;
			outC2 = lineR;
		}
		else
		{
			// Assume VUYA for GPU paths when not BGRA.
			const float lineY = lineR * 0.299f + lineG * 0.587f + lineB * 0.114f;
			const float lineU = lineR * -0.168736f + lineG * -0.331264f + lineB * 0.5f;
			const float lineV = lineR * 0.5f + lineG * -0.418688f + lineB * -0.081312f;
			outC0 = lineV;
			outC1 = lineU;
			outC2 = lineY;
		}

		const float lineThickness = static_cast<float>(GetParam(OST_WINDYLINES_LINE_THICKNESS, inRenderParams->inClipTime).mFloat64);
	const float lineLength = static_cast<float>(GetParam(OST_WINDYLINES_LINE_LENGTH, inRenderParams->inClipTime).mFloat64);
	const int lineCap = NormalizePopupParam(GetParam(OST_WINDYLINES_LINE_CAP, inRenderParams->inClipTime), 2);
		const float lineAngle = static_cast<float>(GetParam(OST_WINDYLINES_LINE_ANGLE, inRenderParams->inClipTime).mFloat32);
		const float lineSkewRaw = static_cast<float>(GetParam(OST_WINDYLINES_LINE_SKEW, inRenderParams->inClipTime).mFloat64);
		const float lineSkew = std::isfinite(lineSkewRaw)
			? (lineSkewRaw < LINE_SKEW_MIN_VALUE ? LINE_SKEW_MIN_VALUE : (lineSkewRaw > LINE_SKEW_MAX_VALUE ? LINE_SKEW_MAX_VALUE : lineSkewRaw))
			: LINE_SKEW_DFLT;
		const float lineAA = static_cast<float>(GetParam(OST_WINDYLINES_LINE_AA, inRenderParams->inClipTime).mFloat64);
		
		// Spawn Source: if "Full Frame" selected, ignore alpha threshold
		const int spawnSource = NormalizePopupParam(GetParam(OST_WINDYLINES_SPAWN_SOURCE, inRenderParams->inClipTime), 2);
		float lineAlphaThreshold = static_cast<float>(GetParam(OST_WINDYLINES_LINE_ALPHA_THRESH, inRenderParams->inClipTime).mFloat64);
		if (spawnSource == SPAWN_SOURCE_FULL_FRAME) {
			lineAlphaThreshold = 1.0f;  // Full frame: ignore alpha, spawn everywhere
		}
		
		const int lineOriginMode = NormalizePopupParam(GetParam(OST_WINDYLINES_LINE_ORIGIN_MODE, inRenderParams->inClipTime), 3);
		const float dsx = inRenderParams->inDownsampleFactorX;
		const float dsy = inRenderParams->inDownsampleFactorY;
		const float dsMax = dsx > dsy ? dsx : dsy;
		const float dsScale = dsMax >= 1.0f ? (1.0f / dsMax) : (dsMax > 0.0f ? dsMax : 1.0f);
		const int lineDownsample = dsMax > 1.0f ? (int)(dsMax + 0.5f) : 1;
		const float lineCountF = static_cast<float>(GetParam(OST_WINDYLINES_LINE_COUNT, inRenderParams->inClipTime).mFloat64);
		const float lineLifetime = static_cast<float>(GetParam(OST_WINDYLINES_LINE_LIFETIME, inRenderParams->inClipTime).mFloat64);
		const float lineInterval = static_cast<float>(GetParam(OST_WINDYLINES_LINE_INTERVAL, inRenderParams->inClipTime).mFloat64);
		const float lineStartTime = static_cast<float>(GetParam(OST_WINDYLINES_LINE_START_TIME, inRenderParams->inClipTime).mFloat64);
		const float lineDuration = static_cast<float>(GetParam(OST_WINDYLINES_LINE_DURATION, inRenderParams->inClipTime).mFloat64);
		const float lineSeedF = static_cast<float>(GetParam(OST_WINDYLINES_LINE_SEED, inRenderParams->inClipTime).mFloat64);
		const int lineEasing = NormalizePopupParam(GetParam(OST_WINDYLINES_LINE_EASING, inRenderParams->inClipTime), 28);
		const float lineTravel = static_cast<float>(GetParam(OST_WINDYLINES_LINE_TRAVEL, inRenderParams->inClipTime).mFloat64);
		const float lineTailFade = static_cast<float>(GetParam(OST_WINDYLINES_LINE_TAIL_FADE, inRenderParams->inClipTime).mFloat64);
		const float lineDepthStrength = static_cast<float>(GetParam(OST_WINDYLINES_LINE_DEPTH_STRENGTH, inRenderParams->inClipTime).mFloat64) / 10.0f; // Normalize 0-10 to 0-1
		
		// Linkage parameters - will be applied after alphaBounds calculation
		const int lengthLinkage = NormalizePopupParam(GetParam(OST_WINDYLINES_LENGTH_LINKAGE, inRenderParams->inClipTime), 3);
		const float lengthLinkageRate = static_cast<float>(GetParam(OST_WINDYLINES_LENGTH_LINKAGE_RATE, inRenderParams->inClipTime).mFloat64) / 100.0f;
		const int thicknessLinkage = NormalizePopupParam(GetParam(OST_WINDYLINES_THICKNESS_LINKAGE, inRenderParams->inClipTime), 3);
		const float thicknessLinkageRate = static_cast<float>(GetParam(OST_WINDYLINES_THICKNESS_LINKAGE_RATE, inRenderParams->inClipTime).mFloat64) / 100.0f;
		const int travelLinkage = NormalizePopupParam(GetParam(OST_WINDYLINES_TRAVEL_LINKAGE, inRenderParams->inClipTime), 3);
		const float travelLinkageRate = static_cast<float>(GetParam(OST_WINDYLINES_TRAVEL_LINKAGE_RATE, inRenderParams->inClipTime).mFloat64) / 100.0f;
		
		const int allowMidPlayInt = allowMidPlay ? 1 : 0;
		// Center is now controlled by Origin Offset X/Y only
		const float lineCenterX = params.mWidth * 0.5f;
		const float lineCenterY = params.mHeight * 0.5f;

		const float lineRadians = lineAngle * static_cast<float>(M_PI / 180.0);
		const float lineCos = cos(lineRadians);
		const float lineSin = sin(lineRadians);

		params.mLineCenterX = lineCenterX;
		params.mLineCenterY = lineCenterY;
		params.mLineCos = lineCos;
		params.mLineSin = lineSin;
		// Note: Line length, thickness, travel will be set after alphaBounds calculation and linkage
		const float lineAAScaled = lineAA * dsScale;
		const float shadowOffsetXScaled = shadowOffsetX * dsScale;
		const float shadowOffsetYScaled = shadowOffsetY * dsScale;
		const float originOffsetXScaled = originOffsetX * dsScale;
		const float originOffsetYScaled = originOffsetY * dsScale;
		params.mLineTailFade = lineTailFade;
		params.mLineDepthStrength = lineDepthStrength;
		if (isBGRA)
		{
			params.mLineR = outC0 < 0.0f ? 0.0f : (outC0 > 1.0f ? 1.0f : outC0);
			params.mLineG = outC1 < 0.0f ? 0.0f : (outC1 > 1.0f ? 1.0f : outC1);
			params.mLineB = outC2 < 0.0f ? 0.0f : (outC2 > 1.0f ? 1.0f : outC2);
		}
		else
		{
			// VUYA expects signed chroma; clamp Y to [0,1], U/V to [-0.5,0.5].
			params.mLineR = outC0 < -0.5f ? -0.5f : (outC0 > 0.5f ? 0.5f : outC0); // V
			params.mLineG = outC1 < -0.5f ? -0.5f : (outC1 > 0.5f ? 0.5f : outC1); // U
			params.mLineB = outC2 < 0.0f ? 0.0f : (outC2 > 1.0f ? 1.0f : outC2);     // Y
		}
		params.mLineAA = lineAAScaled;
		params.mLineSkew = lineSkew;
		const int lineCountDefault = lineCountF < 1.0f ? 1 : (lineCountF > 5000.0f ? 5000 : (int)lineCountF);
		const bool skipFirstFrame = false;
		const int lineCount = skipFirstFrame ? 0 : lineCountDefault;
		params.mLineCap = lineCap;
		params.mLineCount = lineCount;
		params.mLineSeed = (int)lineSeedF;
		params.mLineEasing = lineEasing;
		params.mLineInterval = lineInterval < 0.5f ? 0 : (int)(lineInterval + 0.5f);
		params.mLineAllowMidPlay = allowMidPlayInt;
		params.mHideElement = hideElement ? 1 : 0;
		params.mBlendMode = blendMode;
		params.mShadowEnable = shadowEnable ? 1 : 0;
		// Pre-convert shadow color to output format
		if (isBGRA)
		{
			// BGRA format: store as BGR (reversed RGB)
			params.mShadowColorR = shadowColorB;  // B
			params.mShadowColorG = shadowColorG;  // G
			params.mShadowColorB = shadowColorR;  // R
		}
		else
		{
			// VUYA format: convert RGB to VUY
			params.mShadowColorR = shadowColorR * 0.5f + shadowColorG * -0.418688f + shadowColorB * -0.081312f; // V
			params.mShadowColorG = shadowColorR * -0.168736f + shadowColorG * -0.331264f + shadowColorB * 0.5f; // U
			params.mShadowColorB = shadowColorR * 0.299f + shadowColorG * 0.587f + shadowColorB * 0.114f;       // Y
		}
		params.mShadowOffsetX = shadowOffsetXScaled;
		params.mShadowOffsetY = shadowOffsetYScaled;
		params.mShadowOpacity = shadowOpacity;
		params.mLineSpawnScaleX = spawnScaleX;
		params.mLineSpawnScaleY = spawnScaleY;
		params.mSpawnRotationCos = spawnCos;
		params.mSpawnRotationSin = spawnSin;
		params.mShowSpawnArea = showSpawnArea ? 1 : 0;
		// Pre-convert spawn area color to output format (like shadow color)
		if (isBGRA)
		{
			params.mSpawnAreaColorR = spawnAreaColorB;  // B
			params.mSpawnAreaColorG = spawnAreaColorG;  // G
			params.mSpawnAreaColorB = spawnAreaColorR;  // R
		}
		else
		{
			// VUYA format: convert RGB to VUY
			params.mSpawnAreaColorR = spawnAreaColorR * 0.5f + spawnAreaColorG * -0.418688f + spawnAreaColorB * -0.081312f; // V
			params.mSpawnAreaColorG = spawnAreaColorR * -0.168736f + spawnAreaColorG * -0.331264f + spawnAreaColorB * 0.5f; // U
			params.mSpawnAreaColorB = spawnAreaColorR * 0.299f + spawnAreaColorG * 0.587f + spawnAreaColorB * 0.114f;       // Y
		}
		// Note: mAlphaBounds* fields are set later after alpha bounds calculation
		params.mOriginOffsetX = originOffsetXScaled;
		params.mOriginOffsetY = originOffsetYScaled;
		params.mLineDownsample = lineDownsample;
		params.mFrameIndex = (float)frameIndex;
		// Use clipTime for cache hash - fully cache-consistent
		params.mSeqTimeHash = (float)((clipTime / 1000000) % 10000000);
		// Focus parameters removed (keep defaults)
		params.mFocusEnable = 0;
		params.mFocusDepth = 0.0f;
		params.mFocusRange = 0.0f;
		params.mFocusBlurStrength = 0.0f;
		// Motion Blur parameters
		const bool motionBlurEnable = GetParam(OST_WINDYLINES_MOTION_BLUR_ENABLE, inRenderParams->inClipTime).mBool;
		const int motionBlurSamples = static_cast<int>(GetParam(OST_WINDYLINES_MOTION_BLUR_SAMPLES, inRenderParams->inClipTime).mFloat64 + 0.5f);
		const float motionBlurStrength = static_cast<float>(GetParam(OST_WINDYLINES_MOTION_BLUR_STRENGTH, inRenderParams->inClipTime).mFloat64);
		const int motionBlurType = 0;  // Always use Trail mode (unified behavior)
		const float motionBlurVelocity = 0.0f;  // Removed parameter, use default
		params.mMotionBlurEnable = motionBlurEnable ? 1 : 0;
		params.mMotionBlurSamples = motionBlurSamples < 1 ? 1 : (motionBlurSamples > 32 ? 32 : motionBlurSamples);
		params.mMotionBlurStrength = motionBlurStrength;
		params.mMotionBlurType = motionBlurType;
		params.mMotionBlurVelocity = motionBlurVelocity;
		// Note: Color is now stored per-line in lineData, not in params

		const int tileSize = 32;
		const int tileCountX = (params.mWidth + tileSize - 1) / tileSize;
		const int tileCountY = (params.mHeight + tileSize - 1) / tileSize;
		const int tileCount = tileCountX * tileCountY;
		params.mTileCountX = tileCountX;
		params.mTileCountY = tileCountY;
		params.mTileSize = tileSize;
		
		// Optimization: Pre-compute inverse of tileSize for faster division
		const float invTileSize = 1.0f / (float)tileSize;

		const float lineCenterOffsetX = lineCenterX - (float)params.mWidth * 0.5f;
		const float lineCenterOffsetY = lineCenterY - (float)params.mHeight * 0.5f;
		const float lifeFrames = lineLifetime < 1.0f ? 1.0f : lineLifetime;
		// Note: travelRange, baseLength, baseThickness will be set after alphaBounds calculation
		const float aa = lineAAScaled;
		const int intervalFrames = params.mLineInterval;
		const float period = lifeFrames + (float)intervalFrames;
		const int allowMid = allowMidPlayInt;
		const int easingType = lineEasing;
		const int seed = params.mLineSeed;
		
		// Start Time + Duration: control when lines spawn
		const float startTimeFrames = lineStartTime;
		const float endTimeFrames = (lineDuration > 0.0f) ? (lineStartTime + lineDuration) : 0.0f;
		const int totalLines = lineCount;
		const float timeFrames = (float)frameIndex;
		const float angle = lineAngle;

		int alphaMinX = 0;
		int alphaMinY = 0;
		int alphaMaxX = params.mWidth > 0 ? (params.mWidth - 1) : 0;
		int alphaMaxY = params.mHeight > 0 ? (params.mHeight - 1) : 0;
		const int alphaStride = 4;
		const float alphaThreshold = lineAlphaThreshold;
		
		// Dynamic processing mode based on Random Seed value
		// Seed 100 -> Alternative mode, others -> Standard mode
		const bool useAlternativeMode = (params.mLineSeed == 100);
		
#if HAS_CUDA
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
		{
			float* ioBuffer = (float*)frameData;
			int boundsInit[4] = { params.mWidth, params.mHeight, -1, -1 };
			EnsureCudaBuffer((void**)&sCudaAlphaBounds, sCudaAlphaBoundsBytes, sizeof(boundsInit));
			cudaMemcpy(sCudaAlphaBounds, boundsInit, sizeof(boundsInit), cudaMemcpyHostToDevice);
			ProcAmp2_CUDA_ComputeAlphaBounds(
				ioBuffer,
				params.mPitch,
				params.m16f,
				params.mWidth,
				params.mHeight,
				sCudaAlphaBounds,
				alphaStride,
				alphaThreshold);
			cudaMemcpy(boundsInit, sCudaAlphaBounds, sizeof(boundsInit), cudaMemcpyDeviceToHost);
			if (boundsInit[2] >= boundsInit[0] && boundsInit[3] >= boundsInit[1])
			{
				alphaMinX = boundsInit[0];
				alphaMinY = boundsInit[1];
				alphaMaxX = boundsInit[2];
				alphaMaxY = boundsInit[3];
			}
		}
#endif
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL && mKernelOpenCLAlpha)
		{
			cl_mem srcBuffer = (cl_mem)(srcFrameData ? srcFrameData : frameData);
			const int srcPitch = srcRowBytes > 0 ? (srcRowBytes / GetGPUBytesPerPixel(srcPixelFormat)) : params.mPitch;
			const int src16f = srcPixelFormat != PrPixelFormat_GPU_BGRA_4444_32f;
			cl_context context = (cl_context)mDeviceInfo.outContextHandle;
			cl_command_queue queue = (cl_command_queue)mDeviceInfo.outCommandQueueHandle;
			cl_int clResult = CL_SUCCESS;
			cl_mem boundsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 4, nullptr, &clResult);
			if (clResult == CL_SUCCESS && boundsBuffer)
			{
				int boundsInit[4] = { (int)params.mWidth, (int)params.mHeight, -1, -1 };
				clEnqueueWriteBuffer(queue, boundsBuffer, CL_TRUE, 0, sizeof(boundsInit), boundsInit, 0, nullptr, nullptr);
			clSetKernelArg(mKernelOpenCLAlpha, 0, sizeof(cl_mem), &srcBuffer);
			clSetKernelArg(mKernelOpenCLAlpha, 1, sizeof(cl_mem), &boundsBuffer);
			clSetKernelArg(mKernelOpenCLAlpha, 2, sizeof(int), &srcPitch);
			clSetKernelArg(mKernelOpenCLAlpha, 3, sizeof(int), &src16f);
			clSetKernelArg(mKernelOpenCLAlpha, 4, sizeof(unsigned int), &params.mWidth);
			clSetKernelArg(mKernelOpenCLAlpha, 5, sizeof(unsigned int), &params.mHeight);
			clSetKernelArg(mKernelOpenCLAlpha, 6, sizeof(int), &alphaStride);
			clSetKernelArg(mKernelOpenCLAlpha, 7, sizeof(float), &alphaThreshold);
				size_t threadBlock[2] = { 16, 16 };
				size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };
				clEnqueueNDRangeKernel(queue, mKernelOpenCLAlpha, 2, 0, grid, threadBlock, 0, 0, 0);
				clEnqueueReadBuffer(queue, boundsBuffer, CL_TRUE, 0, sizeof(boundsInit), boundsInit, 0, nullptr, nullptr);
				if (boundsInit[2] >= boundsInit[0] && boundsInit[3] >= boundsInit[1])
				{
					alphaMinX = boundsInit[0];
					alphaMinY = boundsInit[1];
					alphaMaxX = boundsInit[2];
					alphaMaxY = boundsInit[3];
				}
				clReleaseMemObject(boundsBuffer);
			}
		}
		// DirectX/HLSL for alpha calculation
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_DirectX)
		{
#if HAS_DIRECTX
			DXContextPtr dxContext = sDXContextCache[mDeviceIndex];
			if (dxContext && sShaderObjectAlphaCache[mDeviceIndex])
			{
				struct AlphaParams
				{
					int pitch;
					int is16f;
					int width;
					int height;
					int stride;
					float threshold;
				} alphaParams = {};
				alphaParams.pitch = params.mPitch;
				alphaParams.is16f = params.m16f;
				alphaParams.width = params.mWidth;
				alphaParams.height = params.mHeight;
				alphaParams.stride = alphaStride;
				alphaParams.threshold = alphaThreshold;

				Microsoft::WRL::ComPtr<ID3D12Resource> boundsBuffer;
				const UINT boundsBytes = 16;
				D3D12_RESOURCE_DESC desc = {};
				desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
				desc.Width = boundsBytes;
				desc.Height = 1;
				desc.DepthOrArraySize = 1;
				desc.MipLevels = 1;
				desc.Format = DXGI_FORMAT_UNKNOWN;
				desc.SampleDesc.Count = 1;
				desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
				desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
				D3D12_HEAP_PROPERTIES heapProps = {};
				heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
				ID3D12Device* device = (ID3D12Device*)mDeviceInfo.outDeviceHandle;
				if (SUCCEEDED(device->CreateCommittedResource(
					&heapProps,
					D3D12_HEAP_FLAG_NONE,
					&desc,
					D3D12_RESOURCE_STATE_GENERIC_READ,
					nullptr,
					IID_PPV_ARGS(boundsBuffer.GetAddressOf()))))
				{
					int boundsInit[4] = { params.mWidth, params.mHeight, -1, -1 };
					void* mapped = nullptr;
					if (SUCCEEDED(boundsBuffer->Map(0, nullptr, &mapped)) && mapped)
					{
						memcpy(mapped, boundsInit, sizeof(boundsInit));
						boundsBuffer->Unmap(0, nullptr);
					}

					DXShaderExecution alphaExec(
						dxContext,
						sShaderObjectAlphaCache[mDeviceIndex],
						2);
					alphaExec.SetParamBuffer(&alphaParams, sizeof(alphaParams));
					alphaExec.SetUnorderedAccessView(boundsBuffer.Get(), boundsBytes);
					alphaExec.SetShaderResourceView(
						(ID3D12Resource*)(srcFrameData ? srcFrameData : frameData),
						params.mHeight * rowBytes);
					if (alphaExec.Execute((UINT)DivideRoundUp(params.mWidth, 16), (UINT)DivideRoundUp(params.mHeight, 16)))
					{
						void* outMapped = nullptr;
						if (SUCCEEDED(boundsBuffer->Map(0, nullptr, &outMapped)) && outMapped)
						{
							const int* outBounds = (const int*)outMapped;
							if (outBounds[2] >= outBounds[0] && outBounds[3] >= outBounds[1])
							{
								alphaMinX = outBounds[0];
								alphaMinY = outBounds[1];
								alphaMaxX = outBounds[2];
								alphaMaxY = outBounds[3];
							}
							boundsBuffer->Unmap(0, nullptr);
						}
					}
				}
			}
#endif
		}
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal)
		{
#if HAS_METAL
			if (sMetalAlphaPipelineStateCache[mDeviceIndex])
			{
				id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;
				id<MTLCommandQueue> queue = (id<MTLCommandQueue>)mDeviceInfo.outCommandQueueHandle;
				id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
				id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
				id<MTLBuffer> srcBuffer = (id<MTLBuffer>)(srcFrameData ? srcFrameData : frameData);

				struct AlphaParams
				{
					int pitch;
					int is16f;
					int width;
					int height;
					int stride;
					float threshold;
				} alphaParams = {};
				const int srcPitch = srcRowBytes > 0 ? (srcRowBytes / GetGPUBytesPerPixel(srcPixelFormat)) : params.mPitch;
				alphaParams.pitch = srcPitch;
				alphaParams.is16f = srcPixelFormat != PrPixelFormat_GPU_BGRA_4444_32f;
				alphaParams.width = params.mWidth;
				alphaParams.height = params.mHeight;
				alphaParams.stride = alphaStride;
				alphaParams.threshold = alphaThreshold;

				int boundsInit[4] = { (int)params.mWidth, (int)params.mHeight, -1, -1 };
				id<MTLBuffer> boundsBuffer = [[device newBufferWithBytes:boundsInit
					length:sizeof(boundsInit)
					options:MTLResourceStorageModeShared] autorelease];
				id<MTLBuffer> paramBuffer = [[device newBufferWithBytes:&alphaParams
					length:sizeof(alphaParams)
					options:MTLResourceStorageModeShared] autorelease];

				[computeEncoder setComputePipelineState:sMetalAlphaPipelineStateCache[mDeviceIndex]];
				[computeEncoder setBuffer:srcBuffer offset:0 atIndex:0];
				[computeEncoder setBuffer:boundsBuffer offset:0 atIndex:1];
				[computeEncoder setBuffer:paramBuffer offset:0 atIndex:2];
				MTLSize threadsPerGroup = {[sMetalAlphaPipelineStateCache[mDeviceIndex] threadExecutionWidth], 16, 1};
				MTLSize numThreadgroups = {DivideRoundUp(params.mWidth, threadsPerGroup.width), DivideRoundUp(params.mHeight, threadsPerGroup.height), 1};
				[computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
				[computeEncoder endEncoding];
				[commandBuffer commit];
				[commandBuffer waitUntilCompleted];

				const int* outBounds = (const int*)[boundsBuffer contents];
				if (outBounds && outBounds[2] >= outBounds[0] && outBounds[3] >= outBounds[1])
				{
					alphaMinX = outBounds[0];
					alphaMinY = outBounds[1];
					alphaMaxX = outBounds[2];
					alphaMaxY = outBounds[3];
				}
			}
#endif
		}
		if (mDeviceInfo.outDeviceFramework != PrGPUDeviceFramework_CUDA &&
			mDeviceInfo.outDeviceFramework != PrGPUDeviceFramework_OpenCL &&
			mDeviceInfo.outDeviceFramework != PrGPUDeviceFramework_DirectX &&
			mDeviceInfo.outDeviceFramework != PrGPUDeviceFramework_Metal &&
			srcPixels && srcRowBytes > 0)
		{
			const bool src16f = srcPixelFormat != PrPixelFormat_GPU_BGRA_4444_32f;
			ComputeAlphaBoundsCPU(srcPixels, params.mWidth, params.mHeight, srcRowBytes, src16f,
				alphaStride, alphaThreshold, alphaMinX, alphaMinY, alphaMaxX, alphaMaxY);
		}
		const float alphaBoundsMinX = (float)alphaMinX + lineCenterOffsetX;
		const float alphaBoundsMinY = (float)alphaMinY + lineCenterOffsetY;
		const float alphaBoundsWidth = (float)(alphaMaxX - alphaMinX + 1);
		const float alphaBoundsHeight = (float)(alphaMaxY - alphaMinY + 1);
		const float alphaBoundsWidthSafe = alphaBoundsWidth > 0.0f ? alphaBoundsWidth : (float)params.mWidth;
		const float alphaBoundsHeightSafe = alphaBoundsHeight > 0.0f ? alphaBoundsHeight : (float)params.mHeight;
		
		const float lineLengthScaled = ApplyLinkageValue(
			lineLength,
			lengthLinkage,
			lengthLinkageRate,
			alphaBoundsWidthSafe,
			alphaBoundsHeightSafe,
			dsScale);
		const float lineThicknessScaled_temp = ApplyLinkageValue(
			lineThickness,
			thicknessLinkage,
			thicknessLinkageRate,
			alphaBoundsWidthSafe,
			alphaBoundsHeightSafe,
			dsScale);
		const float lineThicknessScaled = lineThicknessScaled_temp < 1.0f ? 1.0f : lineThicknessScaled_temp;
		const float lineTravelScaled = ApplyLinkageValue(
			lineTravel,
			travelLinkage,
			travelLinkageRate,
			alphaBoundsWidthSafe,
			alphaBoundsHeightSafe,
			dsScale);
		
		params.mLineLength = lineLengthScaled;
		params.mLineThickness = lineThicknessScaled;
		params.mLineLifetime = lineLifetime;
		params.mLineTravel = lineTravelScaled;
		
		// Define variables for line generation
		const float travelRange = lineTravelScaled;
		const float baseLength = lineLengthScaled;
		const float baseThickness = lineThicknessScaled;
		
		// Set alpha bounds in params for spawn area preview
		params.mAlphaBoundsMinX = alphaBoundsMinX;
		params.mAlphaBoundsMinY = alphaBoundsMinY;
		params.mAlphaBoundsWidth = alphaBoundsWidth > 0.0f ? alphaBoundsWidth : (float)params.mWidth;
		params.mAlphaBoundsHeight = alphaBoundsHeight > 0.0f ? alphaBoundsHeight : (float)params.mHeight;

		std::vector<Float4> lineData;
		std::vector<LineBinBounds> lineBounds;
		std::vector<int> tileCounts(tileCount, 0);
		std::vector<int> tileOffsets(tileCount + 1, 0);
		std::vector<int> lineIndices;
		
		// Pre-allocate to maximum size, will resize down after loop
		lineData.resize(lineCount * 4);  // 4 Float4s per line: position, size, color, extra(appearAlpha)
		lineBounds.resize(lineCount);

		int skipThick = 0, skipStartTime = 0, skipEndTime = 0, skipAge = 0;
		int outputIndex = 0;  // Track actual number of lines added
		for (int i = 0; i < totalLines; ++i)
		{
			const csSDK_uint32 base = (csSDK_uint32)(seed * 1315423911u) + (csSDK_uint32)i * 2654435761u;
			const float depth = Rand01(base + 6);
			const float depthScale = DepthScale(depth, lineDepthStrength);

			// Apply depth scaling but ensure minimum 1px for thickness and length
			const float baseLenScaled = baseLength * depthScale;
			const float baseThickScaled = baseThickness * depthScale;
			const float baseLen = baseLenScaled < 1.0f ? 1.0f : baseLenScaled;
			const float baseThick = baseThickScaled < 1.0f ? 1.0f : baseThickScaled;

			const float rx = Rand01(base + 1);
			const float ry = Rand01(base + 2);
			const float rstart = Rand01(base + 5);
			const float startFrame = rstart * period;

			// Matching CPU algorithm: age = fmodf(timeFrames - startFrame, period)
			// Then skip if cycleStartFrame < lineStartTime
			float age = fmodf(timeFrames - startFrame, period);
			if (age < 0.0f)
			{
				age += period;
			}
			
			// Start Time + End Time support: skip cycles outside the active time range
			{
				// Calculate when this cycle started
				const float cycleStartFrame = timeFrames - age;
				// Skip if this cycle started before startTime
				if (cycleStartFrame < startTimeFrames)
				{
					skipStartTime++;
					continue;
				}
				// Skip if endTime is set and this cycle started after endTime
				if (lineDuration > 0.0f && cycleStartFrame >= endTimeFrames)
				{
					skipEndTime++;
					continue;
				}
			}

			// Skip lines that have finished their animation cycle
			if (age > lifeFrames)
			{
				skipAge++;
				continue;
			}

			const float t = lifeFrames > 0.0f ? (age / lifeFrames) : 0.0f;
			const float tMove = ApplyEasing(t, easingType);
			const float maxLen = baseLen;
			const float travelScaled = travelRange * depthScale;
			
			// Calculate instantaneous velocity (derivative of eased progress)
			// Use small delta to approximate derivative
			const float tDelta = 1.0f / lifeFrames;  // One frame's worth of t
			const float tPrev = t > tDelta ? (t - tDelta) : 0.0f;
			const float easedPrev = ApplyEasing(tPrev, easingType);
			const float easedT = ApplyEasing(t, easingType);  // Calculate once, reuse below
			// Normalized velocity: how much the eased value changes per frame
			// Range: 0 to ~2 (depends on easing, highest at steep parts)
			const float instantVelocity = (easedT - easedPrev) / tDelta;
			
			// "Head extends from tail, then tail retracts" animation
			// Total travel distance includes line length for proper appearance/disappearance
			const float totalTravelDist = travelScaled + maxLen;  // Total distance for full animation
			const float tailStartPos = -0.5f * travelScaled - maxLen;  // Start hidden on left
			
			const float travelT = easedT;  // Reuse pre-computed eased value
			const float currentTravelPos = tailStartPos + totalTravelDist * travelT;
			
			float headPosX, tailPosX, currentLength;
			
			// Length animation: use sin(π * easedT) for smooth 0→max→0 curve
			// This links length to travel easing - when travel is slow, length changes slow too
			// sin(π * t) gives: 0 at t=0, 1 at t=0.5, 0 at t=1
			const float lengthFactor = sinf((float)M_PI * easedT);  // Reuse pre-computed eased value
			currentLength = maxLen * lengthFactor;
			
			// Position: head extends from tail, then tail retracts toward head
			// Use easedT to determine position within the length animation
			if (easedT <= 0.5f)
			{
				// First half: tail at travel position, head extends forward
				tailPosX = currentTravelPos;
				headPosX = tailPosX + currentLength;
			}
			else
			{
				// Second half: head at max position, tail catches up
				headPosX = currentTravelPos + maxLen;
				tailPosX = headPosX - currentLength;
			}
			
			// For shader: center = midpoint between head and tail
			const float segCenterX = (headPosX + tailPosX) * 0.5f;
			
			// Round cap adjustment: subtract the round cap radius from line length
			// so the visual length matches the intended length
			// Round cap adds (thickness/2) on each end = thickness total
			float effectiveLength = currentLength;
			float effectiveThickness = baseThick;
			const int lineCap = params.mLineCap;  // 0=Flat, 1=Round
			if (lineCap == 1)
			{
				// Subtract cap radius from each end
				effectiveLength = currentLength - baseThick;
				if (effectiveLength < 0.0f)
				{
					// Line too short - shrink thickness to match length
					effectiveThickness = currentLength;
					effectiveLength = 0.0f;
				}
			}
			
			const float halfLen = effectiveLength * 0.5f;
			const float halfThick = effectiveThickness * 0.5f;
			const float alphaCenterX = alphaBoundsMinX + alphaBoundsWidthSafe * 0.5f;
			const float alphaCenterY = alphaBoundsMinY + alphaBoundsHeightSafe * 0.5f;
			
			// Wind Origin: adjust spawn area position (overall atmosphere, not per-line animation)
			// Apply offset in the direction of line angle (both X and Y components)
			// Use maxLen*0.5 (max possible halfLen) for conservative compensation
			// Note: some minor protrusion is inherent in head/tail animation
			// since each line is at a different phase with different extent
			const float maxHalfLen = maxLen * 0.5f;
			float originOffset = 0.0f;
			if (lineOriginMode == 1)  // Forward
			{
				originOffset = 0.5f * travelScaled + maxHalfLen;
			}
			else if (lineOriginMode == 2)  // Backward
			{
				originOffset = -(0.5f * travelScaled + maxHalfLen);
			}
			
			// Animation Pattern adjustments
			// Pattern 0: Simple - all same direction
			// Pattern 1: Half Reverse - every other line reversed
			// Pattern 2: Split - sides go opposite directions (angle-linked)
			// Center Gap applies to all patterns when > 0
			
			float adjustedPosX = rx;
			float adjustedPosY = ry;
			float adjustedAngle = angle;
			
			// Pre-compute original angle cos/sin for perpendicular calculations
			const float origCos = cosf(angle * 3.14159265f / 180.0f);
			const float origSin = sinf(angle * 3.14159265f / 180.0f);
			
			// Calculate perpendicular axis for center gap and Split pattern (aspect-corrected)
			const float invW = alphaBoundsWidthSafe > 0.0f ? (1.0f / alphaBoundsWidthSafe) : 1.0f;
			const float invH = alphaBoundsHeightSafe > 0.0f ? (1.0f / alphaBoundsHeightSafe) : 1.0f;
			const float dirX = origCos * invW;
			const float dirY = origSin * invH;
			float perpX = -dirY;
			float perpY = dirX;
			const float perpLen = sqrtf(perpX * perpX + perpY * perpY);
			if (perpLen > 0.00001f)
			{
				const float invPerpLen = 1.0f / perpLen;
				perpX *= invPerpLen;
				perpY *= invPerpLen;
			}
			const float sideValue = (rx - 0.5f) * perpX + (ry - 0.5f) * perpY;
			
			// Apply center gap (hide lines in center zone)
			if (centerGap > 0.0f && sideValue > -centerGap && sideValue < centerGap)
			{
				// Center zone - hide line
				adjustedPosX = -10.0f;
				adjustedPosY = -10.0f;
			}
			else
			{
				// Pattern-specific direction adjustments
				if (animPattern == 1)  // Half Reverse: 50% of lines go opposite direction
				{
					if (i % 2 == 1)
					{
						adjustedAngle = angle + 180.0f;
					}
				}
				else if (animPattern == 2)  // Split: sides go opposite directions
				{
					if (sideValue < 0.0f)
					{
						adjustedAngle = angle + 180.0f;  // Negative side flows opposite
					}
					// Positive side keeps original angle
				}
				// animPattern == 0 (Simple): no direction adjustment
			}
			
			const float lineCos = cosf(adjustedAngle * 3.14159265f / 180.0f);
			const float lineSin = sinf(adjustedAngle * 3.14159265f / 180.0f);
			// Apply spawn rotation to the spawn position offset
			const float spawnOffsetX = (adjustedPosX - 0.5f) * alphaBoundsWidthSafe * spawnScaleX;
			const float spawnOffsetY = (adjustedPosY - 0.5f) * alphaBoundsHeightSafe * spawnScaleY;
			const float rotatedSpawnX = spawnOffsetX * spawnCos - spawnOffsetY * spawnSin;
			const float rotatedSpawnY = spawnOffsetX * spawnSin + spawnOffsetY * spawnCos;
			const float centerX = alphaCenterX + rotatedSpawnX + originOffset * lineCos + originOffsetXScaled;
			const float centerY = alphaCenterY + rotatedSpawnY + originOffset * lineSin + originOffsetYScaled;

		// Select color from palette: Simple mode uses 0, Preset/Custom uses random based on seed
		int colorIndex = 0;
		if (colorMode != 0)  // Preset or Custom mode
		{
			// Use existing seed + line index for random color selection
			const csSDK_uint32 colorBase = (csSDK_uint32)(seed * 1315423911u) + (csSDK_uint32)i * 2654435761u + 12345u;
			colorIndex = (int)(Rand01(colorBase) * 8.0f);
			if (colorIndex > 7) colorIndex = 7;
		}
		
		// Debug: log first few lines' color selection
		if (i < 3) {
			DebugLog("[GPU ColorSelect] Line %d: colorMode=%d, colorIndex=%d, RGB=(%.2f,%.2f,%.2f)", 
				i, colorMode, colorIndex, 
				colorPalette[colorIndex][0], colorPalette[colorIndex][1], colorPalette[colorIndex][2]);
		}
		
		// Get line color from palette and convert to output color space
		float outColor0, outColor1, outColor2;
		if (isBGRA)
		{
			// BGRA format: pixel[0]=B, pixel[1]=G, pixel[2]=R
			outColor0 = colorPalette[colorIndex][2]; // B
			outColor1 = colorPalette[colorIndex][1]; // G
			outColor2 = colorPalette[colorIndex][0]; // R
		}
		else
		{
			// VUYA conversion: pixel[0]=V, pixel[1]=U, pixel[2]=Y
			const float r = colorPalette[colorIndex][0];
			const float g = colorPalette[colorIndex][1];
			const float b = colorPalette[colorIndex][2];
			outColor0 = r * 0.5f + g * -0.418688f + b * -0.081312f; // V
			outColor1 = r * -0.168736f + g * -0.331264f + b * 0.5f; // U
			outColor2 = r * 0.299f + g * 0.587f + b * 0.114f;       // Y
		}
		
		Float4 d0 = { centerX, centerY, lineCos, lineSin };
		Float4 d1 = { halfLen, halfThick, segCenterX, depth };  // Store depth value for blend mode
		Float4 d2 = { outColor0, outColor1, outColor2, instantVelocity };  // Line color + velocity
		Float4 d3 = { 1.0f, 0.0f, 0.0f, 0.0f };  // Reserved for future use
		
		// Use direct indexing instead of push_back (faster)
		lineData[outputIndex * 4 + 0] = d0;
		lineData[outputIndex * 4 + 1] = d1;
		lineData[outputIndex * 4 + 2] = d2;
		lineData[outputIndex * 4 + 3] = d3;

		const float radius = fabsf(segCenterX) + halfLen + halfThick + aa + fabsf(params.mLineSkew) * (halfThick + aa);
		const float minXf = centerX + segCenterX * lineCos - radius;
		const float maxXf = centerX + segCenterX * lineCos + radius;
		const float minYf = centerY + segCenterX * lineSin - radius;
		const float maxYf = centerY + segCenterX * lineSin + radius;

		LineBinBounds bounds;
		const int maxXClamp = params.mWidth > 0 ? (params.mWidth - 1) : 0;
		const int maxYClamp = params.mHeight > 0 ? (params.mHeight - 1) : 0;
		bounds.minX = minXf < 0.0f ? 0 : (minXf > (float)maxXClamp ? maxXClamp : (int)minXf);
		bounds.maxX = maxXf < 0.0f ? 0 : (maxXf > (float)maxXClamp ? maxXClamp : (int)maxXf);
		bounds.minY = minYf < 0.0f ? 0 : (minYf > (float)maxYClamp ? maxYClamp : (int)minYf);
		bounds.maxY = maxYf < 0.0f ? 0 : (maxYf > (float)maxYClamp ? maxYClamp : (int)maxYf);
		
		// Use direct indexing instead of push_back (faster)
		lineBounds[outputIndex] = bounds;

		const int minTileX = (int)((float)bounds.minX * invTileSize);
		const int maxTileX = (int)((float)bounds.maxX * invTileSize);
		const int minTileY = (int)((float)bounds.minY * invTileSize);
		const int maxTileY = (int)((float)bounds.maxY * invTileSize);

		for (int ty = minTileY; ty <= maxTileY; ++ty)
		{
			for (int tx = minTileX; tx <= maxTileX; ++tx)
			{
				const int tileIndex = ty * tileCountX + tx;
				if (tileIndex >= 0 && tileIndex < tileCount)
				{
					tileCounts[tileIndex] += 1;
				}
			}
		}
		
		// Increment output index after successfully adding this line
		outputIndex++;
		}
		
		// Resize down to actual number of lines added (removing skipped lines)
		lineData.resize(outputIndex * 4);
		lineBounds.resize(outputIndex);
		
		int running = 0;
		for (int i = 0; i < tileCount; ++i)
		{
			tileOffsets[i] = running;
			running += tileCounts[i];
		}
		tileOffsets[tileCount] = running;
		lineIndices.resize(running);

		std::vector<int> tileCursor(tileCounts);
		for (int i = 0; i < (int)lineBounds.size(); ++i)
		{
			const LineBinBounds& bounds = lineBounds[i];
			const int minTileX = (int)((float)bounds.minX * invTileSize);
			const int maxTileX = (int)((float)bounds.maxX * invTileSize);
			const int minTileY = (int)((float)bounds.minY * invTileSize);
			const int maxTileY = (int)((float)bounds.maxY * invTileSize);
			for (int ty = minTileY; ty <= maxTileY; ++ty)
			{
				for (int tx = minTileX; tx <= maxTileX; ++tx)
				{
					const int tileIndex = ty * tileCountX + tx;
					if (tileIndex >= 0 && tileIndex < tileCount)
					{
						const int insertIndex = tileOffsets[tileIndex] + (tileCounts[tileIndex] - tileCursor[tileIndex]);
						tileCursor[tileIndex] -= 1;
						lineIndices[insertIndex] = i;
					}
				}
			}
		}

		const size_t lineDataBytes = lineData.size() * sizeof(Float4);
		const size_t tileOffsetsBytes = tileOffsets.size() * sizeof(int);
		const size_t tileCountsBytes = tileCounts.size() * sizeof(int);
		const size_t lineIndicesBytes = lineIndices.size() * sizeof(int);
		
		// Buffer optimization: over-allocate by 50% to reduce re-allocation frequency
		const size_t lineDataBytesWithOverhead = lineDataBytes * 3 / 2;
		const size_t tileOffsetsBytesWithOverhead = tileOffsetsBytes * 3 / 2;
		const size_t tileCountsBytesWithOverhead = tileCountsBytes * 3 / 2;
		const size_t lineIndicesBytesWithOverhead = lineIndicesBytes * 3 / 2;

		// CUDA rendering
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
		{
#if HAS_CUDA
			float* ioBuffer = (float*)frameData;

			EnsureCudaBuffer((void**)&sCudaLineData, sCudaLineDataBytes, lineDataBytesWithOverhead);
			EnsureCudaBuffer((void**)&sCudaTileOffsets, sCudaTileOffsetsBytes, tileOffsetsBytesWithOverhead);
			EnsureCudaBuffer((void**)&sCudaTileCounts, sCudaTileCountsBytes, tileCountsBytesWithOverhead);
			EnsureCudaBuffer((void**)&sCudaLineIndices, sCudaLineIndicesBytes, lineIndicesBytesWithOverhead);
			if (lineDataBytes > 0)
			{
				cudaMemcpy(sCudaLineData, lineData.data(), lineDataBytes, cudaMemcpyHostToDevice);
			}
			if (tileOffsetsBytes > 0)
			{
				cudaMemcpy(sCudaTileOffsets, tileOffsets.data(), tileOffsetsBytes, cudaMemcpyHostToDevice);
			}
			if (tileCountsBytes > 0)
			{
				cudaMemcpy(sCudaTileCounts, tileCounts.data(), tileCountsBytes, cudaMemcpyHostToDevice);
			}
			if (lineIndicesBytes > 0)
			{
				cudaMemcpy(sCudaLineIndices, lineIndices.data(), lineIndicesBytes, cudaMemcpyHostToDevice);
			}

			ProcAmp2_CUDA(
			ioBuffer,
			params.mPitch,
			params.m16f,
			params.mIsBGRA,
			params.mWidth,
			params.mHeight,
			params.mLineCenterX,
			params.mLineCenterY,
			params.mOriginOffsetX,
			params.mOriginOffsetY,
			params.mLineCos,
			params.mLineSin,
			params.mLineLength,
			params.mLineThickness,
			params.mLineLifetime,
			params.mLineTravel,
			params.mLineTailFade,
			params.mLineDepthStrength,
			params.mLineR,
			params.mLineG,
			params.mLineB,
			params.mLineAA,
			params.mLineCap,
			params.mLineSkew,
			params.mLineCount,
			params.mLineSeed,
			params.mLineEasing,
			params.mLineInterval,
			params.mLineAllowMidPlay,
			params.mHideElement,
			params.mBlendMode,
			params.mFrameIndex,
			params.mLineDownsample,
			sCudaLineData,
			sCudaTileOffsets,
			sCudaTileCounts,
			sCudaLineIndices,
			params.mTileCountX,
			params.mTileSize,
			params.mFocusEnable,
			params.mFocusDepth,
			params.mFocusRange,
			params.mFocusBlurStrength,
			params.mShadowEnable,
			params.mShadowColorR,
			params.mShadowColorG,
			params.mShadowColorB,
			params.mShadowOffsetX,
			params.mShadowOffsetY,
			params.mShadowOpacity,
			params.mLineSpawnScaleX,
			params.mLineSpawnScaleY,
			params.mSpawnRotationCos,
			params.mSpawnRotationSin,
			params.mShowSpawnArea,
			params.mSpawnAreaColorR,
			params.mSpawnAreaColorG,
			params.mSpawnAreaColorB,
			params.mAlphaBoundsMinX,
			params.mAlphaBoundsMinY,
			params.mAlphaBoundsWidth,
			params.mAlphaBoundsHeight,
			params.mMotionBlurEnable,
			params.mMotionBlurSamples,
			params.mMotionBlurStrength);

			// --- Watermark overlay (free mode) ---
			if (!IsGpuLicenseAuthenticated())
			{
				const size_t wmMaskBytes = (size_t)FreeModeWatermark::kMaskWidth * FreeModeWatermark::kMaskHeight;
				EnsureCudaBuffer((void**)&sCudaWatermarkFill, sCudaWatermarkFillBytes, wmMaskBytes);
				EnsureCudaBuffer((void**)&sCudaWatermarkOutline, sCudaWatermarkOutlineBytes, wmMaskBytes);
				// Only upload static mask data once (it never changes)
				if (!sCudaWatermarkUploaded) {
					cudaMemcpy(sCudaWatermarkFill, FreeModeWatermark::kFillMask, wmMaskBytes, cudaMemcpyHostToDevice);
					cudaMemcpy(sCudaWatermarkOutline, FreeModeWatermark::kOutlineMask, wmMaskBytes, cudaMemcpyHostToDevice);
					sCudaWatermarkUploaded = true;
				}
				WatermarkOverlay_CUDA(
					ioBuffer,
					params.mPitch,
					params.m16f,
					params.mIsBGRA,
					params.mWidth,
					params.mHeight,
					sCudaWatermarkFill,
					sCudaWatermarkOutline,
					FreeModeWatermark::kMaskWidth,
					FreeModeWatermark::kMaskHeight,
					FreeModeWatermark::kMarginX,
					FreeModeWatermark::kMarginY,
					dsScale,
					FreeModeWatermark::kFillOpacity,
					FreeModeWatermark::kOutlineOpacity);
			}

			return cudaPeekAtLastError() == cudaSuccess ? suiteError_NoError : suiteError_Fail;
#else
			return suiteError_NotImplemented;
#endif
		}
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
		{
			cl_mem buffer = (cl_mem)frameData;

			cl_context context = (cl_context)mDeviceInfo.outContextHandle;
			cl_int clResult = CL_SUCCESS;
			auto createBuffer = [&](size_t sizeBytes, const void* src) -> cl_mem {
				const size_t allocSize = sizeBytes > 0 ? sizeBytes : sizeof(int);
				cl_mem_flags flags = CL_MEM_READ_ONLY;
				if (sizeBytes > 0 && src)
				{
					flags |= CL_MEM_COPY_HOST_PTR;
				}
				return clCreateBuffer(context, flags, allocSize, (void*)src, &clResult);
			};

			cl_mem lineDataBuffer = createBuffer(lineDataBytes, lineDataBytes > 0 ? lineData.data() : nullptr);
			cl_mem tileOffsetsBuffer = createBuffer(tileOffsetsBytes, tileOffsetsBytes > 0 ? tileOffsets.data() : nullptr);
			cl_mem tileCountsBuffer = createBuffer(tileCountsBytes, tileCountsBytes > 0 ? tileCounts.data() : nullptr);
			cl_mem lineIndicesBuffer = createBuffer(lineIndicesBytes, lineIndicesBytes > 0 ? lineIndices.data() : nullptr);
			
			// Set the arguments - buffers first
			clSetKernelArg(mKernelOpenCL, 0, sizeof(cl_mem), &buffer);
			clSetKernelArg(mKernelOpenCL, 1, sizeof(cl_mem), &lineDataBuffer);
			clSetKernelArg(mKernelOpenCL, 2, sizeof(cl_mem), &tileOffsetsBuffer);
			clSetKernelArg(mKernelOpenCL, 3, sizeof(cl_mem), &tileCountsBuffer);
			clSetKernelArg(mKernelOpenCL, 4, sizeof(cl_mem), &lineIndicesBuffer);
			// Scalar parameters
			clSetKernelArg(mKernelOpenCL, 5, sizeof(int), &params.mPitch);
			clSetKernelArg(mKernelOpenCL, 6, sizeof(int), &params.m16f);
			clSetKernelArg(mKernelOpenCL, 7, sizeof(int), &params.mWidth);
			clSetKernelArg(mKernelOpenCL, 8, sizeof(int), &params.mHeight);
			clSetKernelArg(mKernelOpenCL, 9, sizeof(float), &params.mLineCenterX);
			clSetKernelArg(mKernelOpenCL, 10, sizeof(float), &params.mLineCenterY);
			clSetKernelArg(mKernelOpenCL, 11, sizeof(float), &params.mOriginOffsetX);
			clSetKernelArg(mKernelOpenCL, 12, sizeof(float), &params.mOriginOffsetY);
			clSetKernelArg(mKernelOpenCL, 13, sizeof(float), &params.mLineCos);
			clSetKernelArg(mKernelOpenCL, 14, sizeof(float), &params.mLineSin);
		clSetKernelArg(mKernelOpenCL, 15, sizeof(float), &params.mLineLength);
		clSetKernelArg(mKernelOpenCL, 16, sizeof(float), &params.mLineThickness);
		clSetKernelArg(mKernelOpenCL, 17, sizeof(float), &params.mLineLifetime);
		clSetKernelArg(mKernelOpenCL, 18, sizeof(float), &params.mLineTravel);
		clSetKernelArg(mKernelOpenCL, 19, sizeof(float), &params.mLineTailFade);
		clSetKernelArg(mKernelOpenCL, 20, sizeof(float), &params.mLineDepthStrength);
		clSetKernelArg(mKernelOpenCL, 21, sizeof(float), &params.mLineR);
		clSetKernelArg(mKernelOpenCL, 22, sizeof(float), &params.mLineG);
		clSetKernelArg(mKernelOpenCL, 23, sizeof(float), &params.mLineB);
		clSetKernelArg(mKernelOpenCL, 24, sizeof(float), &params.mLineAA);
		clSetKernelArg(mKernelOpenCL, 25, sizeof(int), &params.mLineCap);
		clSetKernelArg(mKernelOpenCL, 26, sizeof(int), &params.mLineCount);
		clSetKernelArg(mKernelOpenCL, 27, sizeof(int), &params.mLineSeed);
		clSetKernelArg(mKernelOpenCL, 28, sizeof(int), &params.mLineEasing);
		clSetKernelArg(mKernelOpenCL, 29, sizeof(int), &params.mLineInterval);
		clSetKernelArg(mKernelOpenCL, 30, sizeof(int), &params.mLineAllowMidPlay);
		clSetKernelArg(mKernelOpenCL, 31, sizeof(int), &params.mHideElement);
		clSetKernelArg(mKernelOpenCL, 32, sizeof(int), &params.mBlendMode);
		clSetKernelArg(mKernelOpenCL, 33, sizeof(float), &params.mFrameIndex);
		clSetKernelArg(mKernelOpenCL, 34, sizeof(int), &params.mLineDownsample);
		clSetKernelArg(mKernelOpenCL, 35, sizeof(int), &params.mTileCountX);
		clSetKernelArg(mKernelOpenCL, 36, sizeof(int), &params.mTileSize);
		clSetKernelArg(mKernelOpenCL, 37, sizeof(int), &params.mFocusEnable);
		clSetKernelArg(mKernelOpenCL, 38, sizeof(float), &params.mFocusDepth);
		clSetKernelArg(mKernelOpenCL, 39, sizeof(float), &params.mFocusRange);
		clSetKernelArg(mKernelOpenCL, 40, sizeof(float), &params.mFocusBlurStrength);
		clSetKernelArg(mKernelOpenCL, 41, sizeof(int), &params.mShadowEnable);
		clSetKernelArg(mKernelOpenCL, 42, sizeof(float), &params.mShadowColorR);
		clSetKernelArg(mKernelOpenCL, 43, sizeof(float), &params.mShadowColorG);
		clSetKernelArg(mKernelOpenCL, 44, sizeof(float), &params.mShadowColorB);
		clSetKernelArg(mKernelOpenCL, 45, sizeof(float), &params.mShadowOffsetX);
		clSetKernelArg(mKernelOpenCL, 46, sizeof(float), &params.mShadowOffsetY);
		clSetKernelArg(mKernelOpenCL, 47, sizeof(float), &params.mShadowOpacity);
		clSetKernelArg(mKernelOpenCL, 48, sizeof(float), &params.mLineSpawnScaleX);
		clSetKernelArg(mKernelOpenCL, 49, sizeof(float), &params.mLineSpawnScaleY);
		clSetKernelArg(mKernelOpenCL, 50, sizeof(float), &params.mSpawnRotationCos);
		clSetKernelArg(mKernelOpenCL, 51, sizeof(float), &params.mSpawnRotationSin);
		clSetKernelArg(mKernelOpenCL, 52, sizeof(int), &params.mShowSpawnArea);
		clSetKernelArg(mKernelOpenCL, 53, sizeof(float), &params.mSpawnAreaColorR);
		clSetKernelArg(mKernelOpenCL, 54, sizeof(float), &params.mSpawnAreaColorG);
		clSetKernelArg(mKernelOpenCL, 55, sizeof(float), &params.mSpawnAreaColorB);
		clSetKernelArg(mKernelOpenCL, 56, sizeof(int), &params.mIsBGRA);
		clSetKernelArg(mKernelOpenCL, 57, sizeof(float), &params.mAlphaBoundsMinX);
		clSetKernelArg(mKernelOpenCL, 58, sizeof(float), &params.mAlphaBoundsMinY);
		clSetKernelArg(mKernelOpenCL, 59, sizeof(float), &params.mAlphaBoundsWidth);
		clSetKernelArg(mKernelOpenCL, 60, sizeof(float), &params.mAlphaBoundsHeight);
		clSetKernelArg(mKernelOpenCL, 61, sizeof(int), &params.mMotionBlurEnable);
		clSetKernelArg(mKernelOpenCL, 62, sizeof(int), &params.mMotionBlurSamples);
		clSetKernelArg(mKernelOpenCL, 63, sizeof(float), &params.mMotionBlurStrength);
		clSetKernelArg(mKernelOpenCL, 64, sizeof(float), &params.mMotionBlurVelocity);
		clSetKernelArg(mKernelOpenCL, 65, sizeof(float), &params.mLineSkew);

			// Launch the kernel
			size_t threadBlock[2] = { 16, 16 };
			size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1])};

			cl_int result = clEnqueueNDRangeKernel(
				(cl_command_queue)mDeviceInfo.outCommandQueueHandle,
				mKernelOpenCL,
				2,
				0,
				grid,
				threadBlock,
				0,
				0,
				0);
			if (lineDataBuffer) { clReleaseMemObject(lineDataBuffer); }
			if (tileOffsetsBuffer) { clReleaseMemObject(tileOffsetsBuffer); }
			if (tileCountsBuffer) { clReleaseMemObject(tileCountsBuffer); }
			if (lineIndicesBuffer) { clReleaseMemObject(lineIndicesBuffer); }

			// --- Watermark overlay (free mode) ---
			if (result == CL_SUCCESS && !IsGpuLicenseAuthenticated())
			{
				const size_t wmMaskBytes = (size_t)FreeModeWatermark::kMaskWidth * FreeModeWatermark::kMaskHeight;
				cl_mem wmFillBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wmMaskBytes, (void*)FreeModeWatermark::kFillMask, &clResult);
				cl_mem wmOutlineBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wmMaskBytes, (void*)FreeModeWatermark::kOutlineMask, &clResult);

				int wmMaskW = FreeModeWatermark::kMaskWidth;
				int wmMaskH = FreeModeWatermark::kMaskHeight;
				int wmMarginX = FreeModeWatermark::kMarginX;
				int wmMarginY = FreeModeWatermark::kMarginY;
				int wmWidth = (int)params.mWidth;
				int wmHeight = (int)params.mHeight;
				int wmPitch = params.mPitch;
				int wm16f = params.m16f;
				int wmIsBGRA = params.mIsBGRA;

				cl_int argIdx = 0;
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(cl_mem), &buffer);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(cl_mem), &wmFillBuf);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(cl_mem), &wmOutlineBuf);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wmPitch);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wm16f);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wmIsBGRA);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wmWidth);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wmHeight);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wmMaskW);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wmMaskH);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wmMarginX);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(int), &wmMarginY);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(float), &dsScale);

				float wmFillOpacity = FreeModeWatermark::kFillOpacity;
				float wmOutlineOpacity = FreeModeWatermark::kOutlineOpacity;
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(float), &wmFillOpacity);
				clSetKernelArg(mKernelOpenCLWatermark, argIdx++, sizeof(float), &wmOutlineOpacity);

				int scaledW = (int)(FreeModeWatermark::kMaskWidth * dsScale);
				int scaledH = (int)(FreeModeWatermark::kMaskHeight * dsScale);
				if (scaledW < 1) scaledW = 1;
				if (scaledH < 1) scaledH = 1;
				size_t wmThreadBlock[2] = { 16, 16 };
				size_t wmGrid[2] = { RoundUp((unsigned int)scaledW, (unsigned int)wmThreadBlock[0]),
				                     RoundUp((unsigned int)scaledH, (unsigned int)wmThreadBlock[1]) };

				clEnqueueNDRangeKernel(
					(cl_command_queue)mDeviceInfo.outCommandQueueHandle,
					mKernelOpenCLWatermark,
					2, 0, wmGrid, wmThreadBlock, 0, 0, 0);

				if (wmFillBuf) { clReleaseMemObject(wmFillBuf); }
				if (wmOutlineBuf) { clReleaseMemObject(wmOutlineBuf); }
			}

			return result == CL_SUCCESS ? suiteError_NoError : suiteError_Fail;
		}
#if HAS_DIRECTX
		// DirectX/HLSL rendering
		else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_DirectX)
		{


			// Setup the shader execution
			DXShaderExecution shaderExecution(
				sDXContextCache[mDeviceIndex],
				sShaderObjectCache[mDeviceIndex],
				6);

			auto createUploadBuffer = [&](size_t dataSizeBytes, const void* data) -> Microsoft::WRL::ComPtr<ID3D12Resource> {
				Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
				if (dataSizeBytes == 0)
				{
					dataSizeBytes = sizeof(int);
				}
				const size_t allocSize = (dataSizeBytes + 3) & ~3ULL;
				D3D12_RESOURCE_DESC desc = {};
				desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
				desc.Width = allocSize;
				desc.Height = 1;
				desc.DepthOrArraySize = 1;
				desc.MipLevels = 1;
				desc.Format = DXGI_FORMAT_UNKNOWN;
				desc.SampleDesc.Count = 1;
				desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
				D3D12_HEAP_PROPERTIES heapProps = {};
				heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
				ID3D12Device* device = (ID3D12Device*)mDeviceInfo.outDeviceHandle;
				if (FAILED(device->CreateCommittedResource(
					&heapProps,
					D3D12_HEAP_FLAG_NONE,
					&desc,
					D3D12_RESOURCE_STATE_GENERIC_READ,
					nullptr,
					IID_PPV_ARGS(buffer.GetAddressOf()))))
				{
					return nullptr;
				}
				if (data && dataSizeBytes > 0)
				{
					void* mapped = nullptr;
					if (SUCCEEDED(buffer->Map(0, nullptr, &mapped)) && mapped)
					{
						memcpy(mapped, data, dataSizeBytes);
						buffer->Unmap(0, nullptr);
					}
				}
				return buffer;
			};

			Microsoft::WRL::ComPtr<ID3D12Resource> lineDataBuffer = createUploadBuffer(lineDataBytes, lineDataBytes > 0 ? lineData.data() : nullptr);
			Microsoft::WRL::ComPtr<ID3D12Resource> tileOffsetsBuffer = createUploadBuffer(tileOffsetsBytes, tileOffsetsBytes > 0 ? tileOffsets.data() : nullptr);
			Microsoft::WRL::ComPtr<ID3D12Resource> tileCountsBuffer = createUploadBuffer(tileCountsBytes, tileCountsBytes > 0 ? tileCounts.data() : nullptr);
			Microsoft::WRL::ComPtr<ID3D12Resource> lineIndicesBuffer = createUploadBuffer(lineIndicesBytes, lineIndicesBytes > 0 ? lineIndices.data() : nullptr);
			const UINT lineDataSrvBytes = (UINT)(lineDataBytes > 0 ? lineDataBytes : sizeof(int));
			const UINT tileOffsetsSrvBytes = (UINT)(tileOffsetsBytes > 0 ? tileOffsetsBytes : sizeof(int));
			const UINT tileCountsSrvBytes = (UINT)(tileCountsBytes > 0 ? tileCountsBytes : sizeof(int));
			const UINT lineIndicesSrvBytes = (UINT)(lineIndicesBytes > 0 ? lineIndicesBytes : sizeof(int));

			// Note: Ordering should match the root signature
			shaderExecution.SetParamBuffer(&params, sizeof(ProcAmp2Params));
			shaderExecution.SetUnorderedAccessView(
				(ID3D12Resource*)frameData,
				params.mHeight * rowBytes);
			shaderExecution.SetShaderResourceView(lineDataBuffer.Get(), lineDataSrvBytes);
			shaderExecution.SetShaderResourceView(tileOffsetsBuffer.Get(), tileOffsetsSrvBytes);
			shaderExecution.SetShaderResourceView(tileCountsBuffer.Get(), tileCountsSrvBytes);
			shaderExecution.SetShaderResourceView(lineIndicesBuffer.Get(), lineIndicesSrvBytes);
			// Note: 16, 16 used to divide below matches the threadblock in the hlsl file
			if (!shaderExecution.Execute((UINT)DivideRoundUp(params.mWidth, 16), (UINT)DivideRoundUp(params.mHeight, 16))) { return suiteError_Fail; }
			return suiteError_NoError;
		}
#endif
#if HAS_METAL
		else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal)
		{
            @autoreleasepool {
                prSuiteError result = suiteError_NoError;
                
                // v43: If no lines to draw and hideElement is off, skip kernel and return original image unchanged
                // When hideElement is on, we must run the kernel to clear the original element to transparent
                if (lineData.empty() && !hideElement)
                {
                    return suiteError_NoError;
                }
                
                //Set the arguments
                id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;
                id<MTLBuffer> parameterBuffer = [[device newBufferWithBytes:&params
                    length:sizeof(ProcAmp2Params)
                    options:MTLResourceStorageModeManaged] autorelease];
                    
                // v44: Create properly initialized buffers even when empty
                // newBufferWithBytes:nullptr creates uninitialized memory, causing issues
                static const int dummyInt = 0;
                static const Float4 dummyFloat4 = {0.0f, 0.0f, 0.0f, 0.0f};
                
				const NSUInteger lineDataLen = lineDataBytes > 0 ? (NSUInteger)lineDataBytes : sizeof(dummyFloat4);
				const NSUInteger tileOffsetsLen = tileOffsetsBytes > 0 ? (NSUInteger)tileOffsetsBytes : sizeof(dummyInt);
				const NSUInteger tileCountsLen = tileCountsBytes > 0 ? (NSUInteger)tileCountsBytes : sizeof(dummyInt);
				const NSUInteger lineIndicesLen = lineIndicesBytes > 0 ? (NSUInteger)lineIndicesBytes : sizeof(dummyInt);
				
				id<MTLBuffer> lineDataBuffer = [[device newBufferWithBytes:(lineDataBytes > 0 ? lineData.data() : &dummyFloat4)
					length:lineDataLen
					options:MTLResourceStorageModeManaged] autorelease];
				id<MTLBuffer> tileOffsetsBuffer = [[device newBufferWithBytes:(tileOffsetsBytes > 0 ? tileOffsets.data() : &dummyInt)
					length:tileOffsetsLen
					options:MTLResourceStorageModeManaged] autorelease];
				id<MTLBuffer> tileCountsBuffer = [[device newBufferWithBytes:(tileCountsBytes > 0 ? tileCounts.data() : &dummyInt)
					length:tileCountsLen
					options:MTLResourceStorageModeManaged] autorelease];
				id<MTLBuffer> lineIndicesBuffer = [[device newBufferWithBytes:(lineIndicesBytes > 0 ? lineIndices.data() : &dummyInt)
					length:lineIndicesLen
                    options:MTLResourceStorageModeManaged] autorelease];
                
                //Launch the command
                id<MTLCommandQueue> queue = (id<MTLCommandQueue>)mDeviceInfo.outCommandQueueHandle;
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
                id<MTLBuffer> ioBuffer = (id<MTLBuffer>)frameData;
                
                MTLSize threadsPerGroup = {[sMetalPipelineStateCache[mDeviceIndex] threadExecutionWidth], 16, 1};
                MTLSize numThreadgroups = {DivideRoundUp(params.mWidth, threadsPerGroup.width), DivideRoundUp(params.mHeight, threadsPerGroup.height), 1};
                
                // Buffer indices must match GF_KERNEL_FUNCTION buffers section order:
                // 0: ioImage, 1: inLineData, 2: inTileOffsets, 3: inTileCounts, 4: inLineIndices
                // Then values struct at index 5
                [computeEncoder setComputePipelineState:sMetalPipelineStateCache[mDeviceIndex]];
                [computeEncoder setBuffer:ioBuffer offset:0 atIndex:0];
				[computeEncoder setBuffer:lineDataBuffer offset:0 atIndex:1];
				[computeEncoder setBuffer:tileOffsetsBuffer offset:0 atIndex:2];
				[computeEncoder setBuffer:tileCountsBuffer offset:0 atIndex:3];
				[computeEncoder setBuffer:lineIndicesBuffer offset:0 atIndex:4];
                [computeEncoder setBuffer:parameterBuffer offset:0 atIndex:5];
                [computeEncoder	dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
                [computeEncoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];  // Wait for GPU to finish
                // result = CheckForMetalError([commandBuffer error]);

				// --- Watermark overlay (free mode) ---
				if (result == suiteError_NoError && !IsGpuLicenseAuthenticated())
				{
					const size_t wmMaskBytes = (size_t)FreeModeWatermark::kMaskWidth * FreeModeWatermark::kMaskHeight;
					id<MTLBuffer> wmFillBuf = [[device newBufferWithBytes:FreeModeWatermark::kFillMask
						length:(NSUInteger)wmMaskBytes
						options:MTLResourceStorageModeManaged] autorelease];
					id<MTLBuffer> wmOutlineBuf = [[device newBufferWithBytes:FreeModeWatermark::kOutlineMask
						length:(NSUInteger)wmMaskBytes
						options:MTLResourceStorageModeManaged] autorelease];

					// Scalar parameters packed into a struct
					struct WatermarkParams {
						int pitch;
						int is16f;
						int isBGRA;
						unsigned int width;
						unsigned int height;
						int maskWidth;
						int maskHeight;
						int marginX;
						int marginY;
						float dsScale;
						float fillOpacity;
						float outlineOpacity;
					};
					WatermarkParams wmParams;
					wmParams.pitch = params.mPitch;
					wmParams.is16f = params.m16f;
					wmParams.isBGRA = params.mIsBGRA;
					wmParams.width = params.mWidth;
					wmParams.height = params.mHeight;
					wmParams.maskWidth = FreeModeWatermark::kMaskWidth;
					wmParams.maskHeight = FreeModeWatermark::kMaskHeight;
					wmParams.marginX = FreeModeWatermark::kMarginX;
					wmParams.marginY = FreeModeWatermark::kMarginY;
					wmParams.dsScale = dsScale;
					wmParams.fillOpacity = FreeModeWatermark::kFillOpacity;
					wmParams.outlineOpacity = FreeModeWatermark::kOutlineOpacity;

					id<MTLBuffer> wmParamBuffer = [[device newBufferWithBytes:&wmParams
						length:sizeof(wmParams)
						options:MTLResourceStorageModeManaged] autorelease];

					int scaledW = (int)(FreeModeWatermark::kMaskWidth * dsScale);
					int scaledH = (int)(FreeModeWatermark::kMaskHeight * dsScale);
					if (scaledW < 1) scaledW = 1;
					if (scaledH < 1) scaledH = 1;

					id<MTLCommandBuffer> wmCmdBuf = [queue commandBuffer];
					id<MTLComputeCommandEncoder> wmEncoder = [wmCmdBuf computeCommandEncoder];
					[wmEncoder setComputePipelineState:sMetalWatermarkPipelineStateCache[mDeviceIndex]];
					[wmEncoder setBuffer:ioBuffer offset:0 atIndex:0];
					[wmEncoder setBuffer:wmFillBuf offset:0 atIndex:1];
					[wmEncoder setBuffer:wmOutlineBuf offset:0 atIndex:2];
					[wmEncoder setBuffer:wmParamBuffer offset:0 atIndex:3];

					MTLSize wmThreadsPerGroup = {[sMetalWatermarkPipelineStateCache[mDeviceIndex] threadExecutionWidth], 16, 1};
					MTLSize wmNumThreadgroups = {DivideRoundUp((size_t)scaledW, wmThreadsPerGroup.width), DivideRoundUp((size_t)scaledH, wmThreadsPerGroup.height), 1};
					[wmEncoder dispatchThreadgroups:wmNumThreadgroups threadsPerThreadgroup:wmThreadsPerGroup];
					[wmEncoder endEncoding];
					[wmCmdBuf commit];
					[wmCmdBuf waitUntilCompleted];
				}
                
                return result;
            }
		}
#endif //HAS_METAL
		return suiteError_Fail;
	}
	
	static prSuiteError Shutdown(
		piSuitesPtr piSuites,
		csSDK_int32 inIndex)
	{
#if HAS_DIRECTX
        // Note: DirectX: If deferred execution is implemented, a GPU sync is
        // necessary before the plugin shutdown.
        if (inIndex < sDXContextCache.size() && sDXContextCache[inIndex])
        {
            sDXContextCache[inIndex].reset();
        }
        if (inIndex < sShaderObjectCache.size() && sShaderObjectCache[inIndex])
        {
            sShaderObjectCache[inIndex].reset();
        }
		if (inIndex < sShaderObjectAlphaCache.size() && sShaderObjectAlphaCache[inIndex])
		{
			sShaderObjectAlphaCache[inIndex].reset();
        }
#endif
#if HAS_METAL
        @autoreleasepool
        {
            if(sMetalPipelineStateCache[inIndex])
            {
                [sMetalPipelineStateCache[inIndex] release];
                sMetalPipelineStateCache[inIndex] = nil;
            }
			if (sMetalAlphaPipelineStateCache[inIndex])
			{
				[sMetalAlphaPipelineStateCache[inIndex] release];
				sMetalAlphaPipelineStateCache[inIndex] = nil;
            }
			if (sMetalWatermarkPipelineStateCache[inIndex])
			{
				[sMetalWatermarkPipelineStateCache[inIndex] release];
				sMetalWatermarkPipelineStateCache[inIndex] = nil;
			}
        }
#endif
#if HAS_CUDA
		if (sCudaLineData) { cudaFree(sCudaLineData); sCudaLineData = nullptr; sCudaLineDataBytes = 0; }
		if (sCudaTileOffsets) { cudaFree(sCudaTileOffsets); sCudaTileOffsets = nullptr; sCudaTileOffsetsBytes = 0; }
		if (sCudaTileCounts) { cudaFree(sCudaTileCounts); sCudaTileCounts = nullptr; sCudaTileCountsBytes = 0; }
		if (sCudaLineIndices) { cudaFree(sCudaLineIndices); sCudaLineIndices = nullptr; sCudaLineIndicesBytes = 0; }
		if (sCudaAlphaBounds) { cudaFree(sCudaAlphaBounds); sCudaAlphaBounds = nullptr; sCudaAlphaBoundsBytes = 0; }
		if (sCudaWatermarkFill) { cudaFree(sCudaWatermarkFill); sCudaWatermarkFill = nullptr; sCudaWatermarkFillBytes = 0; }
		if (sCudaWatermarkOutline) { cudaFree(sCudaWatermarkOutline); sCudaWatermarkOutline = nullptr; sCudaWatermarkOutlineBytes = 0; }
#endif
		return suiteError_NoError;
	}

private:
	cl_kernel mKernelOpenCL;
	cl_kernel mKernelOpenCLAlpha = nullptr;
	cl_kernel mKernelOpenCLWatermark = nullptr;
};

#if ENABLE_GPU_RENDERING
DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ProcAmp2>)
#endif
