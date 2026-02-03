	/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2012 Adobe Systems Incorporated                       */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Systems Incorporated and its suppliers, if    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Systems Incorporated and its    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Systems         */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/


#include "SDK_ProcAmp.h"
#include "SDK_ProcAmp_Version.h"
#include "PrGPUFilterModule.h"
#include "PrSDKVideoSegmentProperties.h"

#if _WIN32
    #include "SDK_ProcAmp.cl.h"
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
#if _WIN32
    #if defined(MAJOR_VERSION)
        #pragma push_macro("MAJOR_VERSION")
        #undef MAJOR_VERSION
        #define SDK_PROCAMP_RESTORE_MAJOR_VERSION 1
    #endif
    #if defined(MINOR_VERSION)
        #pragma push_macro("MINOR_VERSION")
        #undef MINOR_VERSION
        #define SDK_PROCAMP_RESTORE_MINOR_VERSION 1
    #endif
    #if defined(PATCH_LEVEL)
        #pragma push_macro("PATCH_LEVEL")
        #undef PATCH_LEVEL
        #define SDK_PROCAMP_RESTORE_PATCH_LEVEL 1
    #endif
    #include <cuda_runtime.h>
    #if defined(SDK_PROCAMP_RESTORE_PATCH_LEVEL)
        #pragma pop_macro("PATCH_LEVEL")
        #undef SDK_PROCAMP_RESTORE_PATCH_LEVEL
    #endif
    #if defined(SDK_PROCAMP_RESTORE_MINOR_VERSION)
        #pragma pop_macro("MINOR_VERSION")
        #undef SDK_PROCAMP_RESTORE_MINOR_VERSION
    #endif
    #if defined(SDK_PROCAMP_RESTORE_MAJOR_VERSION)
        #pragma pop_macro("MAJOR_VERSION")
        #undef SDK_PROCAMP_RESTORE_MAJOR_VERSION
    #endif
    #include "DirectXUtils.h"
    #include <vector>
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
** Order from SDK_ProcAmp.cl GF_KERNEL_FUNCTION(ProcAmp2Kernel, ...) values section:
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

// No state tracking needed - using period-based approach for cache consistency

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
//  * See SDK_ProcAmp.cu
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

static csSDK_uint32 HashUInt(csSDK_uint32 x)
{
	x ^= x >> 16;
	x *= 0x7feb352d;
	x ^= x >> 15;
	x *= 0x846ca68b;
	x ^= x >> 16;
	return x;
}

static float Rand01(csSDK_uint32 x)
{
	return (HashUInt(x) & 0x00FFFFFF) / 16777215.0f;
}

static float DepthScale(float depth, float strength)
{
	// Shrink lines based on depth: depth=1 (front) keeps scale=1.0, depth=0 (back) shrinks
	const float v = 1.0f - (1.0f - depth) * strength;
	return v < 0.05f ? 0.05f : v;
}

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

static float ApplyEasing(float t, int easingType)
{
	switch (easingType)
	{
		case 0: return t; // Linear
		// SmoothStep (moved to 1-2)
		case 1: return t * t * (3.0f - 2.0f * t); // SmoothStep (3rd order Hermite)
		case 2: return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // SmootherStep (5th order, Ken Perlin)
		// Sine (3-6)
		case 3: return 1.0f - cosf((float)M_PI * 0.5f * t); // InSine (slow→fast)
		case 4: return sinf((float)M_PI * 0.5f * t); // OutSine (fast→slow)
		case 5: return 0.5f * (1.0f - cosf((float)M_PI * t)); // InOutSine
		case 6: { // OutInSine
			if (t < 0.5f) {
				return 0.5f * ApplyEasing(t * 2.0f, 4);  // OutSine
			} else {
				return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 3);  // InSine
			}
		}
		// Quad (7-10)
		case 7: return t * t; // InQuad
		case 8: return 1.0f - (1.0f - t) * (1.0f - t); // OutQuad
		case 9: {
			const float u = t * 2.0f;
			if (u < 1.0f) { return 0.5f * u * u; }
			const float v = u - 1.0f;
			return 0.5f + 0.5f * (1.0f - (1.0f - v) * (1.0f - v));
		}
		case 10: { // OutInQuad
			if (t < 0.5f) {
				return 0.5f * ApplyEasing(t * 2.0f, 8);  // OutQuad
			} else {
				return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 7);  // InQuad
			}
		}
		// Cubic (11-14)
		case 11: return t * t * t; // InCubic
		case 12: {
			const float u = 1.0f - t;
			return 1.0f - u * u * u; // OutCubic
		}
		case 13: {
			const float u = t * 2.0f;
			if (u < 1.0f) { return 0.5f * u * u * u; }
			const float v = u - 1.0f;
			return 0.5f + 0.5f * (1.0f - (1.0f - v) * (1.0f - v) * (1.0f - v));
		}
		case 14: { // OutInCubic
			if (t < 0.5f) {
				return 0.5f * ApplyEasing(t * 2.0f, 12);  // OutCubic
			} else {
				return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 11);  // InCubic
			}
		}
		// Circular (15-18)
		case 15: return 1.0f - sqrtf(1.0f - t * t); // InCirc
		case 16: { // OutCirc
			const float u = t - 1.0f;
			return sqrtf(1.0f - u * u);
		}
		case 17: { // InOutCirc
			const float u = t * 2.0f;
			if (u < 1.0f) {
				return 0.5f * (1.0f - sqrtf(1.0f - u * u));
			}
			const float v = u - 2.0f;
			return 0.5f * (sqrtf(1.0f - v * v) + 1.0f);
		}
		case 18: { // OutInCirc
			if (t < 0.5f) {
				return 0.5f * ApplyEasing(t * 2.0f, 16);  // OutCirc
			} else {
				return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 15);  // InCirc
			}
		}
		// Back easing (overshoots) (19-21)
		case 19: { // InBack
			const float s = 1.70158f;
			return t * t * ((s + 1.0f) * t - s);
		}
		case 20: { // OutBack
			const float s = 1.70158f;
			const float u = t - 1.0f;
			return u * u * ((s + 1.0f) * u + s) + 1.0f;
		}
		case 21: { // InOutBack
			const float s = 1.70158f * 1.525f;
			const float u = t * 2.0f;
			if (u < 1.0f) {
				return 0.5f * u * u * ((s + 1.0f) * u - s);
			}
			const float v = u - 2.0f;
			return 0.5f * (v * v * ((s + 1.0f) * v + s) + 2.0f);
		}
		// Elastic easing (22-24)
		case 22: { // InElastic
			if (t == 0.0f) return 0.0f;
			if (t == 1.0f) return 1.0f;
			const float p = 0.3f;
			return -powf(2.0f, 10.0f * (t - 1.0f)) * sinf((t - 1.0f - p / 4.0f) * (2.0f * (float)M_PI) / p);
		}
		case 23: { // OutElastic
			if (t == 0.0f) return 0.0f;
			if (t == 1.0f) return 1.0f;
			const float p = 0.3f;
			return powf(2.0f, -10.0f * t) * sinf((t - p / 4.0f) * (2.0f * (float)M_PI) / p) + 1.0f;
		}
		case 24: { // InOutElastic
			if (t == 0.0f) return 0.0f;
			if (t == 1.0f) return 1.0f;
			const float p = 0.45f;
			const float s = p / 4.0f;
			const float u = t * 2.0f;
			if (u < 1.0f) {
				return -0.5f * powf(2.0f, 10.0f * (u - 1.0f)) * sinf((u - 1.0f - s) * (2.0f * (float)M_PI) / p);
			}
			return powf(2.0f, -10.0f * (u - 1.0f)) * sinf((u - 1.0f - s) * (2.0f * (float)M_PI) / p) * 0.5f + 1.0f;
		}
		// Bounce easing (25-27)
		case 25: { // InBounce
			const float u = 1.0f - t;
			float b;
			if (u < 1.0f / 2.75f) {
				b = 7.5625f * u * u;
			} else if (u < 2.0f / 2.75f) {
				const float v = u - 1.5f / 2.75f;
				b = 7.5625f * v * v + 0.75f;
			} else if (u < 2.5f / 2.75f) {
				const float v = u - 2.25f / 2.75f;
				b = 7.5625f * v * v + 0.9375f;
			} else {
				const float v = u - 2.625f / 2.75f;
				b = 7.5625f * v * v + 0.984375f;
			}
			return 1.0f - b;
		}
		case 26: { // OutBounce
			if (t < 1.0f / 2.75f) {
				return 7.5625f * t * t;
			} else if (t < 2.0f / 2.75f) {
				const float u = t - 1.5f / 2.75f;
				return 7.5625f * u * u + 0.75f;
			} else if (t < 2.5f / 2.75f) {
				const float u = t - 2.25f / 2.75f;
				return 7.5625f * u * u + 0.9375f;
			} else {
				const float u = t - 2.625f / 2.75f;
				return 7.5625f * u * u + 0.984375f;
			}
		}
		case 27: { // InOutBounce
			if (t < 0.5f) {
				const float u = 1.0f - t * 2.0f;
				float b;
				if (u < 1.0f / 2.75f) {
					b = 7.5625f * u * u;
				} else if (u < 2.0f / 2.75f) {
					const float v = u - 1.5f / 2.75f;
					b = 7.5625f * v * v + 0.75f;
				} else if (u < 2.5f / 2.75f) {
					const float v = u - 2.25f / 2.75f;
					b = 7.5625f * v * v + 0.9375f;
				} else {
					const float v = u - 2.625f / 2.75f;
					b = 7.5625f * v * v + 0.984375f;
				}
				return (1.0f - b) * 0.5f;
			} else {
				const float u = t * 2.0f - 1.0f;
				float b;
				if (u < 1.0f / 2.75f) {
					b = 7.5625f * u * u;
				} else if (u < 2.0f / 2.75f) {
					const float v = u - 1.5f / 2.75f;
					b = 7.5625f * v * v + 0.75f;
				} else if (u < 2.5f / 2.75f) {
					const float v = u - 2.25f / 2.75f;
					b = 7.5625f * v * v + 0.9375f;
				} else {
					const float v = u - 2.625f / 2.75f;
					b = 7.5625f * v * v + 0.984375f;
				}
				return b * 0.5f + 0.5f;
			}
		}
		default:
			return t;
	}
}

// Derivative of easing function (instantaneous velocity factor)
// Returns normalized velocity: 1.0 = linear speed, >1.0 = faster, <1.0 = slower
static float ApplyEasingDerivative(float t, int easingType)
{
	// For complex easing types, use numerical approximation
	const float epsilon = 0.001f;
	switch (easingType)
	{
		case 0: return 1.0f; // Linear: constant velocity
		// For all other types - use numerical differentiation
		case 1: case 2: // SmoothStep, SmootherStep
		case 3: case 4: case 5: case 6: // Sine (In, Out, InOut, OutIn)
		case 7: case 8: case 9: case 10: // Quad (In, Out, InOut, OutIn)
		case 11: case 12: case 13: case 14: // Cubic (In, Out, InOut, OutIn)
		case 15: case 16: case 17: case 18: // Circ (In, Out, InOut, OutIn)
		case 19: case 20: case 21: // Back (In, Out, InOut)
		case 22: case 23: case 24: // Elastic (In, Out, InOut)
		case 25: case 26: case 27: // Bounce (In, Out, InOut)
		{
			const float t1 = t > epsilon ? t - epsilon : 0.0f;
			const float t2 = t < 1.0f - epsilon ? t + epsilon : 1.0f;
			const float dt = t2 - t1;
			if (dt > 0.0f) {
				return (ApplyEasing(t2, easingType) - ApplyEasing(t1, easingType)) / dt;
			}
			return 1.0f;
		}
		default:
			return 1.0f;
	}
}

static int NormalizePopupValue(int value, int maxValue)
{
	if (value >= 0 && value < maxValue)
	{
		return value;
	}
	if (value >= 1 && value <= maxValue)
	{
		return value - 1;
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
	if (param.mType == kPrParamType_Bool)
	{
		return param.mInt32 != 0;
	}
	if (param.mType == kPrParamType_Int8)
	{
		return param.mInt8 != 0;
	}
	if (param.mType == kPrParamType_Int16)
	{
		return param.mInt16 != 0;
	}
	if (param.mType == kPrParamType_Int32)
	{
		return param.mInt32 != 0;
	}
	if (param.mType == kPrParamType_Int64)
	{
		return param.mInt64 != 0;
	}
	if (param.mType == kPrParamType_Float64)
	{
		return param.mFloat64 >= 0.5;
	}
	if (param.mType == kPrParamType_Float32)
	{
		return param.mFloat32 >= 0.5f;
	}
	return false;
}


/*
**
*/
enum {kMaxDevices = 12};
static cl_kernel sKernelCache[kMaxDevices] = {};
static cl_kernel sKernelCacheAlpha[kMaxDevices] = {};
#if HAS_METAL
static id<MTLComputePipelineState> sMetalPipelineStateCache[kMaxDevices] = {};
static id<MTLComputePipelineState> sMetalAlphaPipelineStateCache[kMaxDevices] = {};
#endif

/*
**
*/
class ProcAmp2 :
	public PrGPUFilterBase
{
public:
	// Cached clip start ticks (from VideoSegmentSuite, per nodeID)
	csSDK_int64 mCachedClipStartTicks = -1;
	bool mClipStartQueried = false;
	
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
		PrGPUFilterBase::Initialize(ioInstanceData);

		if (mDeviceIndex > kMaxDevices)
		{
			// Exceeded max device count
			return suiteError_Fail;
		}

		// NOTE: GPU Override parameter is checked in Render() function
		// because we don't have access to parameters during Initialize()

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
			if (!mKernelOpenCL)
			{
				cl_int result = CL_SUCCESS;
				//A real plugin would check to see if 16f access is actually supported.
#if _WIN32
				char const *k16fString = "#define GF_OPENCL_SUPPORTS_16F 1\n";
				size_t sizes[] = { strlen(k16fString), strlen(kSDK_ProcAmp_OpenCLString) };
				char const *strings[] = { k16fString, kSDK_ProcAmp_OpenCLString };
#else
				// Mac: Try to use preprocessed header first, fallback to reading .cl file
				char const *k16fString = "#define GF_OPENCL_SUPPORTS_16F 1\n";
				size_t sizes[2];
				char const *strings[2];
				
				// Try to include preprocessed header (if build step creates it)
				#ifdef SDK_PROCAMP_OPENCL_STRING_AVAILABLE
					// Preprocessed header available
					sizes[0] = strlen(k16fString);
					sizes[1] = strlen(kSDK_ProcAmp_OpenCLString);
					strings[0] = k16fString;
					strings[1] = kSDK_ProcAmp_OpenCLString;
				#else
					// Fallback: Read OpenCL source from file
					std::string clSourcePath = __FILE__;
					const size_t slashPos = clSourcePath.find_last_of("\\/");
					if (slashPos != std::string::npos)
					{
						clSourcePath.erase(slashPos);
					}
					clSourcePath += "/SDK_ProcAmp.cl";
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

				sKernelCache[mDeviceIndex] = mKernelOpenCL;
				sKernelCacheAlpha[mDeviceIndex] = mKernelOpenCLAlpha;
			}
			return suiteError_NoError;
		}
#if HAS_DIRECTX
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
                if (!GetShaderPath(L"SDK_ProcAmp", csoPath, sigPath))
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
				if (!GetShaderPath(L"SDK_ProcAmp_Alpha", alphaCsoPath, alphaSigPath))
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

                NSString *pluginBundlePath = [[NSBundle bundleWithIdentifier:@"MyCompany.SDK-ProcAmp"] bundlePath];
                NSString *metalLibPath = [pluginBundlePath stringByAppendingPathComponent:@"Contents/Resources/MetalLib/SDK_ProcAmp.metallib"];
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
		const bool allowMidPlay = BoolParamFromPrParam(GetParam(SDK_PROCAMP_LINE_ALLOW_MIDPLAY, inRenderParams->inClipTime));
		const bool hideElement = BoolParamFromPrParam(GetParam(SDK_PROCAMP_HIDE_ELEMENT, inRenderParams->inClipTime));
		const int blendMode = NormalizePopupParam(GetParam(SDK_PROCAMP_BLEND_MODE, inRenderParams->inClipTime), 4);
		const bool shadowEnable = BoolParamFromPrParam(GetParam(SDK_PROCAMP_SHADOW_ENABLE, inRenderParams->inClipTime));
		float shadowColorR = 0.0f, shadowColorG = 0.0f, shadowColorB = 0.0f;
		{
			PrParam shadowColorParam = GetParam(SDK_PROCAMP_SHADOW_COLOR, inRenderParams->inClipTime);
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
		const float shadowOffsetX = static_cast<float>(GetParam(SDK_PROCAMP_SHADOW_OFFSET_X, inRenderParams->inClipTime).mFloat64);
		const float shadowOffsetY = static_cast<float>(GetParam(SDK_PROCAMP_SHADOW_OFFSET_Y, inRenderParams->inClipTime).mFloat64);
		const float shadowOpacity = static_cast<float>(GetParam(SDK_PROCAMP_SHADOW_OPACITY, inRenderParams->inClipTime).mFloat64);
		const float spawnScaleX = static_cast<float>(GetParam(SDK_PROCAMP_LINE_SPAWN_SCALE_X, inRenderParams->inClipTime).mFloat64) / 100.0f;
		const float spawnScaleY = static_cast<float>(GetParam(SDK_PROCAMP_LINE_SPAWN_SCALE_Y, inRenderParams->inClipTime).mFloat64) / 100.0f;
		const float spawnRotationDeg = static_cast<float>(GetParam(SDK_PROCAMP_LINE_SPAWN_ROTATION, inRenderParams->inClipTime).mFloat64);
		const float spawnRotationRad = spawnRotationDeg * 3.14159265f / 180.0f;
		const float spawnCos = cosf(spawnRotationRad);
		const float spawnSin = sinf(spawnRotationRad);
		const bool showSpawnArea = BoolParamFromPrParam(GetParam(SDK_PROCAMP_LINE_SHOW_SPAWN_AREA, inRenderParams->inClipTime));
		// Read spawn area color
		float spawnAreaColorR = 0.5f, spawnAreaColorG = 0.5f, spawnAreaColorB = 1.0f;  // Default light blue
		{
			const PrParam spawnAreaColorParam = GetParam(SDK_PROCAMP_LINE_SPAWN_AREA_COLOR, inRenderParams->inClipTime);
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
		const float originOffsetX = static_cast<float>(GetParam(SDK_PROCAMP_ORIGIN_OFFSET_X, inRenderParams->inClipTime).mFloat64);
		const float originOffsetY = static_cast<float>(GetParam(SDK_PROCAMP_ORIGIN_OFFSET_Y, inRenderParams->inClipTime).mFloat64);
		const int animPattern = NormalizePopupParam(GetParam(SDK_PROCAMP_ANIM_PATTERN, inRenderParams->inClipTime), 3);
		const float centerGap = static_cast<float>(GetParam(SDK_PROCAMP_CENTER_GAP, inRenderParams->inClipTime).mFloat64);
		// Focus parameters removed
	
	// v51: DevGuide approach - use clipTime with SharedClipData
	// clipTime = absolute media time in ticks
	// clipStartFrame = first frame of clip in media time (from CPU or estimated)
	// frameIndex = mediaFrameIndex - clipStartFrame (0-based from clip start)
	const PrTime ticksPerFrame = inRenderParams->inRenderTicksPerFrame;
	const csSDK_int64 mediaFrameIndex = (ticksPerFrame > 0) ? (clipTime / ticksPerFrame) : 0;
	
	// Try to get clipStartFrame from SharedClipData (set by CPU renderer)
	// Use clipTime as key since it's consistent for the same media
	csSDK_int64 clipStartFrame = SharedClipData::GetClipStart(mediaFrameIndex);
	
	// If not found from CPU, try to find the smallest clipStartFrame <= mediaFrameIndex
	if (clipStartFrame < 0)
	{
		// Fallback: GPU-only estimation using clipTime - seqTime relationship
		// For the same clip, (clipTime - seqTime) is constant
		const PrTime seqTime = inRenderParams->inSequenceTime;
		const PrTime clipOffset = clipTime - seqTime;
		
		// Use clipOffset as key to track this clip instance
		clipStartFrame = SharedClipData::GetClipStart(clipOffset);
		if (clipStartFrame < 0)
		{
			// First encounter: estimate clipStartFrame from seqTime
			// If seqTime=0, clipStartFrame = mediaFrameIndex (clip starts at media time corresponding to timeline start)
			const csSDK_int64 seqFrameIndex = (ticksPerFrame > 0) ? (seqTime / ticksPerFrame) : 0;
			clipStartFrame = mediaFrameIndex - seqFrameIndex;
			SharedClipData::SetClipStart(clipOffset, clipStartFrame);
		}
	}
	
	// Calculate clip-relative frame index
	csSDK_int64 frameIndex = mediaFrameIndex - clipStartFrame;
	if (frameIndex < 0) frameIndex = 0;
	
		// Color Mode and Palette Setup
		// normalizePopup returns 0-based: 0=Single, 1=Preset, 2=Custom
		const int colorMode = NormalizePopupParam(GetParam(SDK_PROCAMP_COLOR_MODE, inRenderParams->inClipTime), 3);
		const int presetIndex = NormalizePopupParam(GetParam(SDK_PROCAMP_COLOR_PRESET, inRenderParams->inClipTime), 33);
		
		// Build color palette (8 colors)
		float colorPalette[8][3];  // RGB normalized
		
		if (colorMode == 0)  // Single (0-based)
		{
			// Single color mode: all 8 slots have the same color
			float singleR = 1.0f, singleG = 1.0f, singleB = 1.0f;
			PrParam lineColorParam = GetParam(SDK_PROCAMP_LINE_COLOR, inRenderParams->inClipTime);
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
		}
		else if (colorMode == 1)  // Preset (0-based)
		{
			// Preset mode: load from preset palette (presetIndex is 0-based)
			const PresetColor* preset = GetPresetPalette(presetIndex + 1);  // GetPresetPalette expects 1-based
			for (int i = 0; i < 8; ++i)
			{
				colorPalette[i][0] = preset[i].r / 255.0f;
				colorPalette[i][1] = preset[i].g / 255.0f;
				colorPalette[i][2] = preset[i].b / 255.0f;
			}
		}
		else  // Custom (colorMode == 2, 0-based)
		{
			// Custom mode: load from custom color parameters
			const int customColorParams[8] = {
				SDK_PROCAMP_CUSTOM_COLOR_1, SDK_PROCAMP_CUSTOM_COLOR_2,
				SDK_PROCAMP_CUSTOM_COLOR_3, SDK_PROCAMP_CUSTOM_COLOR_4,
				SDK_PROCAMP_CUSTOM_COLOR_5, SDK_PROCAMP_CUSTOM_COLOR_6,
				SDK_PROCAMP_CUSTOM_COLOR_7, SDK_PROCAMP_CUSTOM_COLOR_8
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

		const float lineThickness = static_cast<float>(GetParam(SDK_PROCAMP_LINE_THICKNESS, inRenderParams->inClipTime).mFloat64);
	const float lineLength = static_cast<float>(GetParam(SDK_PROCAMP_LINE_LENGTH, inRenderParams->inClipTime).mFloat64);
	const int lineCap = NormalizePopupParam(GetParam(SDK_PROCAMP_LINE_CAP, inRenderParams->inClipTime), 2);
		const float lineAngle = static_cast<float>(GetParam(SDK_PROCAMP_LINE_ANGLE, inRenderParams->inClipTime).mFloat32);
		const float lineAA = static_cast<float>(GetParam(SDK_PROCAMP_LINE_AA, inRenderParams->inClipTime).mFloat64);
		
		// Spawn Source: if "Full Frame" selected, ignore alpha threshold
		const int spawnSource = NormalizePopupParam(GetParam(SDK_PROCAMP_SPAWN_SOURCE, inRenderParams->inClipTime), 2);
		float lineAlphaThreshold = static_cast<float>(GetParam(SDK_PROCAMP_LINE_ALPHA_THRESH, inRenderParams->inClipTime).mFloat64);
		if (spawnSource == SPAWN_SOURCE_FULL_FRAME) {
			lineAlphaThreshold = 1.0f;  // Full frame: ignore alpha, spawn everywhere
		}
		
		const int lineOriginMode = NormalizePopupParam(GetParam(SDK_PROCAMP_LINE_ORIGIN_MODE, inRenderParams->inClipTime), 3);
		const float dsx = inRenderParams->inDownsampleFactorX;
		const float dsy = inRenderParams->inDownsampleFactorY;
		const float dsMax = dsx > dsy ? dsx : dsy;
		const float dsScale = dsMax >= 1.0f ? (1.0f / dsMax) : (dsMax > 0.0f ? dsMax : 1.0f);
		const int lineDownsample = dsMax > 1.0f ? (int)(dsMax + 0.5f) : 1;
		const float lineCountF = static_cast<float>(GetParam(SDK_PROCAMP_LINE_COUNT, inRenderParams->inClipTime).mFloat64);
		const float lineLifetime = static_cast<float>(GetParam(SDK_PROCAMP_LINE_LIFETIME, inRenderParams->inClipTime).mFloat64);
		const float lineInterval = static_cast<float>(GetParam(SDK_PROCAMP_LINE_INTERVAL, inRenderParams->inClipTime).mFloat64);
		const float lineStartTime = static_cast<float>(GetParam(SDK_PROCAMP_LINE_START_TIME, inRenderParams->inClipTime).mFloat64);
		const float lineDuration = static_cast<float>(GetParam(SDK_PROCAMP_LINE_DURATION, inRenderParams->inClipTime).mFloat64);
		const float lineSeedF = static_cast<float>(GetParam(SDK_PROCAMP_LINE_SEED, inRenderParams->inClipTime).mFloat64);
		const int lineEasing = NormalizePopupParam(GetParam(SDK_PROCAMP_LINE_EASING, inRenderParams->inClipTime), 28);
		const float lineTravel = static_cast<float>(GetParam(SDK_PROCAMP_LINE_TRAVEL, inRenderParams->inClipTime).mFloat64);
		const float lineTailFade = static_cast<float>(GetParam(SDK_PROCAMP_LINE_TAIL_FADE, inRenderParams->inClipTime).mFloat64);
		const float lineDepthStrength = static_cast<float>(GetParam(SDK_PROCAMP_LINE_DEPTH_STRENGTH, inRenderParams->inClipTime).mFloat64) / 10.0f; // Normalize 0-10 to 0-1
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
		const float lineLengthScaled = lineLength * dsScale;
		// Ensure minimum thickness of 1.0px even at low preview resolutions
		const float lineThicknessScaled = (lineThickness * dsScale) < 1.0f ? 1.0f : (lineThickness * dsScale);
		const float lineTravelScaled = lineTravel * dsScale;
		const float lineAAScaled = lineAA * dsScale;
		const float shadowOffsetXScaled = shadowOffsetX * dsScale;
		const float shadowOffsetYScaled = shadowOffsetY * dsScale;
		const float originOffsetXScaled = originOffsetX * dsScale;
		const float originOffsetYScaled = originOffsetY * dsScale;
	params.mLineLength = lineLengthScaled;
	params.mLineThickness = lineThicknessScaled;
	params.mLineLifetime = lineLifetime;
	params.mLineTravel = lineTravelScaled;
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
		const bool motionBlurEnable = GetParam(SDK_PROCAMP_MOTION_BLUR_ENABLE, inRenderParams->inClipTime).mBool;
		const int motionBlurSamples = static_cast<int>(GetParam(SDK_PROCAMP_MOTION_BLUR_SAMPLES, inRenderParams->inClipTime).mFloat64 + 0.5f);
		const float motionBlurStrength = static_cast<float>(GetParam(SDK_PROCAMP_MOTION_BLUR_STRENGTH, inRenderParams->inClipTime).mFloat64);
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

		const float lineCenterOffsetX = lineCenterX - (float)params.mWidth * 0.5f;
		const float lineCenterOffsetY = lineCenterY - (float)params.mHeight * 0.5f;
		const float lifeFrames = lineLifetime < 1.0f ? 1.0f : lineLifetime;
		const float travelRange = lineTravelScaled;
		const float baseLength = lineLengthScaled;
		const float baseThickness = lineThicknessScaled;
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
		lineData.reserve(lineCount * 4);  // 4 Float4s per line: position, size, color, extra(appearAlpha)
		lineBounds.reserve(lineCount);

		int skipThick = 0, skipStartTime = 0, skipEndTime = 0, skipAge = 0;
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

			// Note: allowMidPlay functionality is now handled by negative Start Time
			// Start Time < 0 allows lines to appear mid-animation at clip start

			// v43: Calculate effective time relative to Start Time parameter
			// If startTimeFrames > 0, offset time so animation "starts" at that frame
			const float effectiveTime = timeFrames - startTimeFrames;
			
			// Skip if current frame is before start time
			if (effectiveTime < 0.0f)
			{
				skipStartTime++;
				continue;
			}

			// v43: Calculate age - lines don't start until their startFrame has elapsed
			// This ensures animation starts "fresh" at Start Time
			float age = 0.0f;
			if (period > 0.0f)
			{
				// Each line has a random startFrame delay within the period
				// Line should not appear until effectiveTime >= startFrame for its first cycle
				const float timeSinceLineStart = effectiveTime - startFrame;
				
				if (timeSinceLineStart < 0.0f)
				{
					// This line hasn't started its first cycle yet
					skipStartTime++;
					continue;
				}
				
				// Calculate age within current cycle
				age = fmodf(timeSinceLineStart, period);
			}

			// v49: End Time support - lines that started before endTime continue to draw
			// Only skip lines whose current cycle started AFTER endTime
			if (endTimeFrames > 0.0f && period > 0.0f)
			{
				// Calculate when this line's current cycle started (in absolute frame time)
				const float timeSinceLineStart = effectiveTime - startFrame;
				const float cycleNumber = floorf(timeSinceLineStart / period);
				const float cycleStartTime = startTimeFrames + startFrame + cycleNumber * period;
				
				// If this cycle started after endTime, skip this line
				if (cycleStartTime >= endTimeFrames)
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
			const float easedCurr = ApplyEasing(t, easingType);
			// Normalized velocity: how much the eased value changes per frame
			// Range: 0 to ~2 (depends on easing, highest at steep parts)
			const float instantVelocity = (easedCurr - easedPrev) / tDelta;
			
			// "Head extends from tail, then tail retracts" animation
			// Total travel distance includes line length for proper appearance/disappearance
			const float totalTravelDist = travelScaled + maxLen;  // Total distance for full animation
			const float tailStartPos = -0.5f * travelScaled - maxLen;  // Start hidden on left
			
			const float travelT = ApplyEasing(t, easingType);
			const float currentTravelPos = tailStartPos + totalTravelDist * travelT;
			
			float headPosX, tailPosX, currentLength;
			
			// Length animation: use sin(π * easedT) for smooth 0→max→0 curve
			// This links length to travel easing - when travel is slow, length changes slow too
			// sin(π * t) gives: 0 at t=0, 1 at t=0.5, 0 at t=1
			const float easedT = ApplyEasing(t, easingType);  // Same easing as travel
			const float lengthFactor = sinf((float)M_PI * easedT);
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
			float originOffset = 0.0f;
			if (lineOriginMode == 1)  // Forward
			{
				originOffset = 0.5f * travelScaled;
			}
			else if (lineOriginMode == 2)  // Backward
			{
				originOffset = -0.5f * travelScaled;
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
				perpX /= perpLen;
				perpY /= perpLen;
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
			
			lineData.push_back(d0);
			lineData.push_back(d1);
			lineData.push_back(d2);
			lineData.push_back(d3);

			const float radius = fabsf(segCenterX) + halfLen + halfThick + aa;
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
			lineBounds.push_back(bounds);

			const int minTileX = bounds.minX / tileSize;
			const int maxTileX = bounds.maxX / tileSize;
			const int minTileY = bounds.minY / tileSize;
			const int maxTileY = bounds.maxY / tileSize;

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
		}
		
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
			const int minTileX = bounds.minX / tileSize;
			const int maxTileX = bounds.maxX / tileSize;
			const int minTileY = bounds.minY / tileSize;
			const int maxTileY = bounds.maxY / tileSize;
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

		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
		{
#if HAS_CUDA
			float* ioBuffer = (float*)frameData;

			EnsureCudaBuffer((void**)&sCudaLineData, sCudaLineDataBytes, lineDataBytes);
			EnsureCudaBuffer((void**)&sCudaTileOffsets, sCudaTileOffsetsBytes, tileOffsetsBytes);
			EnsureCudaBuffer((void**)&sCudaTileCounts, sCudaTileCountsBytes, tileCountsBytes);
			EnsureCudaBuffer((void**)&sCudaLineIndices, sCudaLineIndicesBytes, lineIndicesBytes);
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
			return result == CL_SUCCESS ? suiteError_NoError : suiteError_Fail;
		}
#if HAS_DIRECTX
		else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_DirectX)
		{
			// The kernel expects pitch in bytes
			params.mPitch = rowBytes;

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
                
                // v43: If no lines to draw, skip kernel and return original image unchanged
                if (lineData.empty())
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
        }
#endif
#if HAS_CUDA
		if (sCudaLineData) { cudaFree(sCudaLineData); sCudaLineData = nullptr; sCudaLineDataBytes = 0; }
		if (sCudaTileOffsets) { cudaFree(sCudaTileOffsets); sCudaTileOffsets = nullptr; sCudaTileOffsetsBytes = 0; }
		if (sCudaTileCounts) { cudaFree(sCudaTileCounts); sCudaTileCounts = nullptr; sCudaTileCountsBytes = 0; }
		if (sCudaLineIndices) { cudaFree(sCudaLineIndices); sCudaLineIndices = nullptr; sCudaLineIndicesBytes = 0; }
		if (sCudaAlphaBounds) { cudaFree(sCudaAlphaBounds); sCudaAlphaBounds = nullptr; sCudaAlphaBoundsBytes = 0; }
#endif
		return suiteError_NoError;
	}

private:
	cl_kernel mKernelOpenCL;
	cl_kernel mKernelOpenCLAlpha = nullptr;
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ProcAmp2>)
