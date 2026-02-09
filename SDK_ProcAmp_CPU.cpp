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
#include "SDK_ProcAmp_ParamNames.h"
#include "SDK_ProcAmp_Version.h"
#include "AE_EffectSuites.h"
#include "PrSDKAESupport.h"
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdarg>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Debug logging function is now in SDK_ProcAmp.h

// Debounce for preset button double-fire issue
static std::atomic<uint32_t> sLastPresetClickTime{ 0 };
static const uint32_t kPresetDebounceMs = 200;

static uint32_t GetCurrentTimeMs()
{
#ifdef _WIN32
	return GetTickCount();
#else
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return static_cast<uint32_t>(tv.tv_sec * 1000 + tv.tv_usec / 1000);
#endif
}

static int NormalizePopupValue(int value, int maxValue)
{
	// Premiere Pro popup values are 1-based, convert to 0-based
	if (value >= 1 && value <= maxValue)
	{
		return value - 1;
	}
	// Already 0-based or out of range
	if (value >= 0 && value < maxValue)
	{
		return value;
	}
	return 0;
}

// Define shared static variables for CPU-GPU clip start sharing
std::unordered_map<csSDK_int64, csSDK_int64> SharedClipData::clipStartMap;
std::mutex SharedClipData::mapMutex;

// ========================================================================
// Phase 3-1: Easing Look-Up Table (LUT)
// ========================================================================

#define EASING_LUT_SIZE 256
#define EASING_COUNT 28

// LUT storage: [easing_type][sample_index]
static float sEasingLUT[EASING_COUNT][EASING_LUT_SIZE];
static bool sEasingLUTInitialized = false;

// Forward declaration
static float ApplyEasing(float t, int easing);

/**
 * Initialize easing LUT by pre-computing all 28 easing functions
 * at 256 sample points (0.0 to 1.0)
 */
static void InitializeEasingLUT()
{
	if (sEasingLUTInitialized) return;
	
	for (int easingType = 0; easingType < EASING_COUNT; ++easingType)
	{
		for (int i = 0; i < EASING_LUT_SIZE; ++i)
		{
			const float t = static_cast<float>(i) / static_cast<float>(EASING_LUT_SIZE - 1);
			sEasingLUT[easingType][i] = ApplyEasing(t, easingType);
		}
	}
	
	sEasingLUTInitialized = true;
}

/**
 * Fast easing lookup using pre-computed LUT
 * @param t Input value [0.0, 1.0]
 * @param easingType Easing type [0-27]
 * @return Eased value
 */
static inline float ApplyEasingLUT(float t, int easingType)
{
	// Clamp input
	if (t <= 0.0f) return 0.0f;
	if (t >= 1.0f) return 1.0f;
	
	// Bounds check
	if (easingType < 0 || easingType >= EASING_COUNT) {
		return t; // Fallback to linear
	}
	
	// Map t to LUT index with linear interpolation
	const float fidx = t * static_cast<float>(EASING_LUT_SIZE - 1);
	const int idx = static_cast<int>(fidx);
	const float frac = fidx - static_cast<float>(idx);
	
	if (idx >= EASING_LUT_SIZE - 1) {
		return sEasingLUT[easingType][EASING_LUT_SIZE - 1];
	}
	
	// Linear interpolation between samples
	const float v0 = sEasingLUT[easingType][idx];
	const float v1 = sEasingLUT[easingType][idx + 1];
	return v0 + (v1 - v0) * frac;
}

// ========================================================================
// Phase 3-2: Trigonometric Function Look-Up Table (LUT)
// ========================================================================

#define TRIG_LUT_SIZE 256

// LUT for sine: covers [0, 2π]
static float sSinLUT[TRIG_LUT_SIZE];
static bool sTrigLUTInitialized = false;

/**
 * Initialize trigonometric LUT
 * Pre-computes sin values for [0, 2π] range
 */
static void InitializeTrigLUT()
{
	if (sTrigLUTInitialized) return;
	
	for (int i = 0; i < TRIG_LUT_SIZE; ++i)
	{
		const float angle = (2.0f * static_cast<float>(M_PI) * static_cast<float>(i)) / static_cast<float>(TRIG_LUT_SIZE);
		sSinLUT[i] = sinf(angle);
	}
	
	sTrigLUTInitialized = true;
}

/**
 * Fast sine lookup using pre-computed LUT
 * @param angle Angle in radians
 * @return sin(angle)
 */
static inline float FastSin(float angle)
{
	// Normalize angle to [0, 2π]
	const float twoPi = 2.0f * static_cast<float>(M_PI);
	float normalized = fmodf(angle, twoPi);
	if (normalized < 0.0f) normalized += twoPi;
	
	// Map to LUT index with linear interpolation
	const float fidx = (normalized / twoPi) * static_cast<float>(TRIG_LUT_SIZE - 1);
	const int idx = static_cast<int>(fidx);
	const float frac = fidx - static_cast<float>(idx);
	
	if (idx >= TRIG_LUT_SIZE - 1) {
		return sSinLUT[0];  // Wrap around
	}
	
	// Linear interpolation
	const float v0 = sSinLUT[idx];
	const float v1 = sSinLUT[idx + 1];
	return v0 + (v1 - v0) * frac;
}

/**
 * Fast cosine lookup using pre-computed LUT
 * Uses identity: cos(x) = sin(x + π/2)
 * @param angle Angle in radians
 * @return cos(angle)
 */
static inline float FastCos(float angle)
{
	return FastSin(angle + static_cast<float>(M_PI) * 0.5f);
}

// ========================================================================
// Phase 2-1: Shared SDF (Signed Distance Field) Functions
// ========================================================================

/**
 * Box SDF: Distance from point to rounded rectangle
 * Optimized for compiler auto-vectorization (branchless)
 * @param px Local X coordinate (along line axis)
 * @param py Local Y coordinate (perpendicular to line)
 * @param halfLen Half of line length
 * @param halfThick Half of line thickness
 * @return Signed distance (negative = inside, positive = outside)
 */
static inline float SDFBox(float px, float py, float halfLen, float halfThick)
{
	const float dxBox = fabsf(px) - halfLen;
	const float dyBox = fabsf(py) - halfThick;
	// Branchless max(0, x) using fmaxf for SIMD-friendly code
	const float ox = fmaxf(dxBox, 0.0f);
	const float oy = fmaxf(dyBox, 0.0f);
	const float outside = sqrtf(ox * ox + oy * oy);
	const float inside = fminf(fmaxf(dxBox, dyBox), 0.0f);
	return outside + inside;
}

/**
 * Capsule SDF: Distance from point to rounded line (capsule)
 * Optimized for compiler auto-vectorization (branchless)
 * @param px Local X coordinate (along line axis)
 * @param py Local Y coordinate (perpendicular to line)
 * @param halfLen Half of line length
 * @param halfThick Half of line thickness (radius)
 * @return Signed distance (negative = inside, positive = outside)
 */
static inline float SDFCapsule(float px, float py, float halfLen, float halfThick)
{
	const float ax = fabsf(px) - halfLen;
	// Branchless max(0, x) using fmaxf for SIMD-friendly code
	const float qx = fmaxf(ax, 0.0f);
	return sqrtf(qx * qx + py * py) - halfThick;
}


// ========================================================================
// Phase 2-2: Shared Blending Functions
// ========================================================================

/**
 * Premultiplied alpha compositing (over operation)
 * Optimized: branchless division handling
 * @param srcR Source color R
 * @param srcG Source color G
 * @param srcB Source color B
 * @param srcA Source alpha
 * @param dstR Destination color R (in/out)
 * @param dstG Destination color G (in/out)
 * @param dstB Destination color B (in/out)
 * @param dstA Destination alpha (in/out)
 */
static inline void BlendPremultiplied(
	float srcR, float srcG, float srcB, float srcA,
	float& dstR, float& dstG, float& dstB, float& dstA)
{
	const float invSrcA = 1.0f - srcA;
	const float outA = srcA + dstA * invSrcA;
	// Branchless: use fmaxf to avoid division by zero
	const float invOutA = 1.0f / fmaxf(outA, 1e-6f);
	dstR = (srcR * srcA + dstR * dstA * invSrcA) * invOutA;
	dstG = (srcG * srcA + dstG * dstA * invSrcA) * invOutA;
	dstB = (srcB * srcA + dstB * dstA * invSrcA) * invOutA;
	dstA = outA;
}

/**
 * Un-premultiplied alpha accumulation (for front line accumulation)
 * Optimized: branchless division handling
 * @param srcR Source color R
 * @param srcG Source color G
 * @param srcB Source color B
 * @param srcA Source alpha
 * @param dstR Destination color R (in/out)
 * @param dstG Destination color G (in/out)
 * @param dstB Destination color B (in/out)
 * @param dstA Destination alpha (in/out)
 */
static inline void BlendUnpremultiplied(
	float srcR, float srcG, float srcB, float srcA,
	float& dstR, float& dstG, float& dstB, float& dstA)
{
	const float invSrcA = 1.0f - srcA;
	const float outA = srcA + dstA * invSrcA;
	// Branchless: use fmaxf to avoid division by zero
	const float invOutA = 1.0f / fmaxf(outA, 1e-6f);
	dstR = (srcR * srcA + dstR * dstA * invSrcA) * invOutA;
	dstG = (srcG * srcA + dstG * dstA * invSrcA) * invOutA;
	dstB = (srcB * srcA + dstB * dstA * invSrcA) * invOutA;
	dstA = outA;
}


/*
**
*/
static PF_Err GlobalSetup(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	out_data->my_version	= PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);

	if (in_data->appl_id == 'PrMr')
	{
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
	}

	out_data->out_flags |= PF_OutFlag_USE_OUTPUT_EXTENT;
	out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
	out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
	out_data->out_flags |= PF_OutFlag_SEND_UPDATE_PARAMS_UI;
	out_data->out_flags2 |= PF_OutFlag2_PRESERVES_FULLY_OPAQUE_PIXELS;
	// Tell Premiere this effect uses timecode/sequence position to help with cache invalidation.
	out_data->out_flags2 |= PF_OutFlag2_I_USE_TIMECODE;

	// Initialize LUTs
	InitializeEasingLUT();
	InitializeTrigLUT();

	return PF_Err_NONE;
}
/*
**
*/
static PF_Err GlobalSetdown(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	return PF_Err_NONE;
}

struct LineParams
{
	float startFrame;
	float posX;
	float posY;
	float baseLen;
	float baseThick;
	float angle;
	float depthScale;
	float depthValue;  // Depth value (0-1) for blend mode
};

// LineDerived: Pre-computed values for each line
// Layout optimized for cache efficiency - frequently accessed fields first
// Total size: 60 bytes (fits in one 64-byte cache line)
struct alignas(64) LineDerived
{
	// Hot path: Early skip check (offset 0-7)
	float halfThick;   // Used for tiny line skip and bounding box
	float halfLen;     // Used for bounding box check
	
	// Hot path: Coordinate transform (offset 8-27)
	float centerX;
	float centerY;
	float cosA;
	float sinA;
	float segCenterX;
	
	// Hot path: Rendering (offset 28-35)
	float depthAlpha;      // Pre-computed depth fade alpha
	float invDenom;        // Pre-computed 1 / (2.0f * halfLen) for tail fade
	
	// Medium frequency: Color and effects (offset 36-47)
	float depth;           // Depth value for blend mode
	float focusAlpha;      // Alpha multiplier for focus blur
	float appearAlpha;     // Alpha multiplier for appear/disappear fade
	
	// Lower frequency (offset 48-55)
	float lineVelocity;    // Instantaneous velocity for motion blur
	int colorIndex;        // Palette color index (0-7)
	
	// Padding to align to cache line boundary (offset 56-63)
	int _padding;
};

struct LineInstanceState
{
	std::vector<LineParams> lineParams;
	std::vector<LineDerived> lineDerived;
	std::vector<int> tileOffsets;
	std::vector<int> tileCounts;
	std::vector<int> tileIndices;
	std::vector<char> lineActive;
	int lineCount = 0;
	int lineSeed = 0;
	float lineDepthStrength = 0.0f;
	int lineInterval = 0;
};

struct ClipTimeState
{
	A_long startTime = 0;
	A_long lastTime = 0;
	bool valid = false;
};

struct InstanceState
{
	std::mutex mutex;
	LineInstanceState lineState;
	ClipTimeState clipTime;
	bool allowMidPlayCached = false;
};

static std::unordered_map<const void*, std::shared_ptr<InstanceState>> sInstanceStates;
static std::mutex sInstanceStatesMutex;

static const void* GetInstanceKey(const PF_InData* in_data)
{
	if (in_data && in_data->effect_ref)
	{
		return in_data->effect_ref;
	}
	if (in_data && in_data->sequence_data)
	{
		return in_data->sequence_data;
	}
	return in_data;
}

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

static float EaseInOutSine(float t)
{
	return 0.5f * (1.0f - cosf((float)M_PI * t));
}

static float DepthScale(float depth, float strength)
{
	// Shrink lines based on depth: depth=1 (front) keeps scale=1.0, depth=0 (back) shrinks
	const float v = 1.0f - (1.0f - depth) * strength;
	return v < 0.05f ? 0.05f : v;
}

static void SyncLineColorParams(PF_ParamDef* params[])
{
	if (!params || !params[SDK_PROCAMP_LINE_COLOR] ||
		!params[SDK_PROCAMP_LINE_COLOR_R] ||
		!params[SDK_PROCAMP_LINE_COLOR_G] ||
		!params[SDK_PROCAMP_LINE_COLOR_B])
	{
		return;
	}
	const PF_Pixel color = params[SDK_PROCAMP_LINE_COLOR]->u.cd.value;
	params[SDK_PROCAMP_LINE_COLOR_R]->u.fs_d.value = color.red / 255.0f;
	params[SDK_PROCAMP_LINE_COLOR_G]->u.fs_d.value = color.green / 255.0f;
	params[SDK_PROCAMP_LINE_COLOR_B]->u.fs_d.value = color.blue / 255.0f;
}

static void HideLineColorParams(PF_InData* in_data)
{
	if (!in_data)
	{
		return;
	}
	PF_ParamDef def;
	AEFX_CLR_STRUCT(def);
	def.ui_flags = PF_PUI_INVISIBLE;
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
	paramUtils->PF_UpdateParamUI(in_data->effect_ref, SDK_PROCAMP_LINE_COLOR_R, &def);
	paramUtils->PF_UpdateParamUI(in_data->effect_ref, SDK_PROCAMP_LINE_COLOR_G, &def);
	paramUtils->PF_UpdateParamUI(in_data->effect_ref, SDK_PROCAMP_LINE_COLOR_B, &def);
}

// Hide/Show Alpha Threshold based on Spawn Source selection
// UI visibility update is disabled because PF_UpdateParamUI breaks slider range
// The alpha threshold value is forced to 1.0 in Render when spawnSource == FULL_FRAME
static void UpdateAlphaThresholdVisibility(PF_InData* in_data, PF_ParamDef* params[])
{
	if (!in_data || !params)
	{
		return;
	}
}

static float ApplyEasing(float t, int easing)
{
	if (t < 0.0f) t = 0.0f;
	if (t > 1.0f) t = 1.0f;
	switch (easing)
	{
		case 0: // Linear
			return t;
		// SmoothStep (1-2)
		case 1: // SmoothStep (3rd order Hermite)
			return t * t * (3.0f - 2.0f * t);
		case 2: // SmootherStep (5th order, Ken Perlin)
			return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
		// Sine (3-6)
		case 3: // InSine (slow→fast)
			return 1.0f - cosf((float)M_PI * t * 0.5f);
		case 4: // OutSine (fast→slow)
			return sinf((float)M_PI * t * 0.5f);
		case 5: // InOutSine
			return EaseInOutSine(t);
		case 6: // OutInSine
		{
			if (t < 0.5f) {
				return 0.5f * ApplyEasing(t * 2.0f, 4);  // OutSine
			} else {
				return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 3);  // InSine
			}
		}
		// Quad (7-10)
		case 7: // InQuad
			return t * t;
		case 8: // OutQuad
			return 1.0f - (1.0f - t) * (1.0f - t);
		case 9: // InOutQuad
		{
			const float u = t * 2.0f;
			if (u < 1.0f) { return 0.5f * u * u; }
			const float v = u - 1.0f;
			return 0.5f + 0.5f * (1.0f - (1.0f - v) * (1.0f - v));
		}
		case 10: // OutInQuad
		{
			if (t < 0.5f) {
				return 0.5f * ApplyEasing(t * 2.0f, 8);  // OutQuad
			} else {
				return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 7);  // InQuad
			}
		}
		// Cubic (11-14)
		case 11: // InCubic
			return t * t * t;
		case 12: // OutCubic
		{
			const float u = 1.0f - t;
			return 1.0f - u * u * u;
		}
		case 13: // InOutCubic
		{
			const float u = t * 2.0f;
			if (u < 1.0f) { return 0.5f * u * u * u; }
			const float v = u - 1.0f;
			return 0.5f + 0.5f * (1.0f - (1.0f - v) * (1.0f - v) * (1.0f - v));
		}
		case 14: // OutInCubic
		{
			if (t < 0.5f) {
				return 0.5f * ApplyEasing(t * 2.0f, 12);  // OutCubic
			} else {
				return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 11);  // InCubic
			}
		}
		// Circular (15-18)
		case 15: // InCirc
			return 1.0f - sqrtf(1.0f - t * t);
		case 16: // OutCirc
		{
			const float u = t - 1.0f;
			return sqrtf(1.0f - u * u);
		}
		case 17: // InOutCirc
		{
			const float u = t * 2.0f;
			if (u < 1.0f) {
				return 0.5f * (1.0f - sqrtf(1.0f - u * u));
			}
			const float v = u - 2.0f;
			return 0.5f * (sqrtf(1.0f - v * v) + 1.0f);
		}
		case 18: // OutInCirc
		{
			if (t < 0.5f) {
				return 0.5f * ApplyEasing(t * 2.0f, 16);  // OutCirc
			} else {
				return 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 15);  // InCirc
			}
		}
		// Back easing (overshoots) (19-21)
		case 19: // InBack
		{
			const float s = 1.70158f;
			return t * t * ((s + 1.0f) * t - s);
		}
		case 20: // OutBack
		{
			const float s = 1.70158f;
			const float u = t - 1.0f;
			return u * u * ((s + 1.0f) * u + s) + 1.0f;
		}
		case 21: // InOutBack
		{
			const float s = 1.70158f * 1.525f;
			const float u = t * 2.0f;
			if (u < 1.0f) {
				return 0.5f * u * u * ((s + 1.0f) * u - s);
			}
			const float v = u - 2.0f;
			return 0.5f * (v * v * ((s + 1.0f) * v + s) + 2.0f);
		}
		// Elastic easing (22-24)
		case 22: // InElastic
		{
			if (t == 0.0f) return 0.0f;
			if (t == 1.0f) return 1.0f;
			const float p = 0.3f;
			return -powf(2.0f, 10.0f * (t - 1.0f)) * sinf((t - 1.0f - p / 4.0f) * (2.0f * (float)M_PI) / p);
		}
		case 23: // OutElastic
		{
			if (t == 0.0f) return 0.0f;
			if (t == 1.0f) return 1.0f;
			const float p = 0.3f;
			return powf(2.0f, -10.0f * t) * sinf((t - p / 4.0f) * (2.0f * (float)M_PI) / p) + 1.0f;
		}
		case 24: // InOutElastic
		{
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
		case 25: // InBounce
		{
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
		case 26: // OutBounce
		{
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
		case 27: // InOutBounce
		{
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

static void ApplyRectColorUi(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[])
{
	(void)in_data;
	(void)out_data;
	(void)params;
}

// Update visibility of mode-dependent parameters (no checkbox groups)
static void UpdatePseudoGroupVisibility(
	PF_InData* in_data,
	PF_ParamDef* params[])
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
	if (!paramUtils.get())
	{
		return;
	}

	// Helper lambda to set visibility
	auto setVisible = [&](int paramId, bool visible)
	{
		PF_ParamDef paramCopy = *params[paramId];
		if (visible)
		{
			paramCopy.ui_flags &= ~PF_PUI_INVISIBLE;
		}
		else
		{
			paramCopy.ui_flags |= PF_PUI_INVISIBLE;
		}
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, &paramCopy);
	};

	// ========================================
	// Flat parameters (always visible):
	// - Basic: LINE_COUNT, LINE_LIFETIME, LINE_TRAVEL, LINE_SEED
	// - Color: COLOR_MODE, LINE_COLOR, COLOR_PRESET
	// - Appearance: LINE_THICKNESS, LINE_LENGTH, LINE_CAP, LINE_ANGLE, LINE_AA, LINE_TAIL_FADE
	// - Position: LINE_ORIGIN_MODE, LINE_INTERVAL, SPAWN_SCALE_X/Y, SPAWN_ROTATION, 
	//             SHOW_SPAWN_AREA, SPAWN_AREA_COLOR, ORIGIN_OFFSET_X/Y
	// - Animation: ANIM_PATTERN, CENTER_GAP, LINE_EASING, LINE_START_TIME, LINE_DURATION
	// ========================================

	// Custom Colors 1-8: visible only when Color Mode = Custom (value 3)
	const int colorMode = params[SDK_PROCAMP_COLOR_MODE]->u.pd.value;
	const bool showCustomColors = (colorMode == 3); // 3 = Custom
	setVisible(SDK_PROCAMP_CUSTOM_COLOR_1, showCustomColors);
	setVisible(SDK_PROCAMP_CUSTOM_COLOR_2, showCustomColors);
	setVisible(SDK_PROCAMP_CUSTOM_COLOR_3, showCustomColors);
	setVisible(SDK_PROCAMP_CUSTOM_COLOR_4, showCustomColors);
	setVisible(SDK_PROCAMP_CUSTOM_COLOR_5, showCustomColors);
	setVisible(SDK_PROCAMP_CUSTOM_COLOR_6, showCustomColors);
	setVisible(SDK_PROCAMP_CUSTOM_COLOR_7, showCustomColors);
	setVisible(SDK_PROCAMP_CUSTOM_COLOR_8, showCustomColors);

	// ========================================
	// Linkage Parameters Visibility Control
	// ========================================
	
	// Travel Distance Linkage: show/hide rate and value based on mode
	const int travelLinkage = params[SDK_PROCAMP_TRAVEL_LINKAGE]->u.pd.value;
	const bool travelLinkageOff = (travelLinkage == 1); // 1 = Off (1-based popup)
	setVisible(SDK_PROCAMP_TRAVEL_LINKAGE_RATE, !travelLinkageOff); // Show rate when linkage is ON
	setVisible(SDK_PROCAMP_LINE_TRAVEL, travelLinkageOff);          // Show value when linkage is OFF
	
	// Thickness Linkage: show/hide rate and value based on mode
	const int thicknessLinkage = params[SDK_PROCAMP_THICKNESS_LINKAGE]->u.pd.value;
	const bool thicknessLinkageOff = (thicknessLinkage == 1); // 1 = Off (1-based popup)
	setVisible(SDK_PROCAMP_THICKNESS_LINKAGE_RATE, !thicknessLinkageOff); // Show rate when linkage is ON
	setVisible(SDK_PROCAMP_LINE_THICKNESS, thicknessLinkageOff);           // Show value when linkage is OFF
	
	// Length Linkage: show/hide rate and value based on mode
	const int lengthLinkage = params[SDK_PROCAMP_LENGTH_LINKAGE]->u.pd.value;
	const bool lengthLinkageOff = (lengthLinkage == 1); // 1 = Off (1-based popup)
	setVisible(SDK_PROCAMP_LENGTH_LINKAGE_RATE, !lengthLinkageOff); // Show rate when linkage is ON
	setVisible(SDK_PROCAMP_LINE_LENGTH, lengthLinkageOff);           // Show value when linkage is OFF

	// Single Color and Color Preset: always visible regardless of Color Mode
	// (User requested these to never be disabled)
	setVisible(SDK_PROCAMP_LINE_COLOR, true);
	setVisible(SDK_PROCAMP_COLOR_PRESET, true);

	// Shadow / Advanced / Focus params are always visible (no checkbox groups)
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
		default:
		{
			const float t1 = t > epsilon ? t - epsilon : 0.0f;
			const float t2 = t < 1.0f - epsilon ? t + epsilon : 1.0f;
			const float dt = t2 - t1;
			if (dt > 0.0f) {
				return (ApplyEasing(t2, easingType) - ApplyEasing(t1, easingType)) / dt;
			}
			return 1.0f;
		}
	}
}

static void ApplyEffectPreset(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	int presetIndex)
{
	if (presetIndex < 0 || presetIndex >= kEffectPresetCount)
	{
		return;
	}

	const EffectPreset& preset = kEffectPresets[presetIndex];
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
	if (!paramUtils.get())
	{
		return;
	}

	auto updateFloat = [&](int paramId, float value)
	{
		params[paramId]->u.fs_d.value = value;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	auto updatePopup = [&](int paramId, int value)
	{
		params[paramId]->u.pd.value = value;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	auto updateAngle = [&](int paramId, float degrees)
	{
		const A_long fixedAngle = FLOAT2FIX(degrees);
		params[paramId]->u.ad.value = fixedAngle;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	// Basic settings
	updateFloat(SDK_PROCAMP_LINE_COUNT, static_cast<float>(preset.count));
	updateFloat(SDK_PROCAMP_LINE_LIFETIME, preset.lifetime);
	updateFloat(SDK_PROCAMP_LINE_TRAVEL, preset.travel);
	
	// Appearance
	updateFloat(SDK_PROCAMP_LINE_THICKNESS, preset.thickness);
	updateFloat(SDK_PROCAMP_LINE_LENGTH, preset.length);
	updateAngle(SDK_PROCAMP_LINE_ANGLE, preset.angle);
	updateFloat(SDK_PROCAMP_LINE_TAIL_FADE, preset.tailFade);
	updateFloat(SDK_PROCAMP_LINE_AA, preset.aa);
	
	// Position & Spawn
	updatePopup(SDK_PROCAMP_LINE_ORIGIN_MODE, preset.originMode);
	updateFloat(SDK_PROCAMP_LINE_SPAWN_SCALE_X, preset.spawnScaleX);
	updateFloat(SDK_PROCAMP_LINE_SPAWN_SCALE_Y, preset.spawnScaleY);
	updateFloat(SDK_PROCAMP_ORIGIN_OFFSET_X, preset.originOffsetX);
	updateFloat(SDK_PROCAMP_ORIGIN_OFFSET_Y, preset.originOffsetY);
	updateFloat(SDK_PROCAMP_LINE_INTERVAL, preset.interval);
	
	// Animation
	updatePopup(SDK_PROCAMP_ANIM_PATTERN, preset.animPattern);
	updateFloat(SDK_PROCAMP_CENTER_GAP, preset.centerGap);
	updatePopup(SDK_PROCAMP_LINE_EASING, preset.easing + 1);
	updateFloat(SDK_PROCAMP_LINE_START_TIME, preset.startTime);
	updateFloat(SDK_PROCAMP_LINE_DURATION, preset.duration);
	
	// Advanced
	updatePopup(SDK_PROCAMP_BLEND_MODE, preset.blendMode);
	updateFloat(SDK_PROCAMP_LINE_DEPTH_STRENGTH, preset.depthStrength);
	
	// New parameters
	updatePopup(SDK_PROCAMP_LINE_CAP, preset.lineCap);
	updatePopup(SDK_PROCAMP_COLOR_MODE, preset.colorMode);
	updatePopup(SDK_PROCAMP_COLOR_PRESET, preset.colorPreset);
	updatePopup(SDK_PROCAMP_SPAWN_SOURCE, preset.spawnSource);
	
	auto updateCheckbox = [&](int paramId, bool value)
	{
		params[paramId]->u.bd.value = value ? 1 : 0;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};
	updateCheckbox(SDK_PROCAMP_HIDE_ELEMENT, preset.hideElement);

	// Force UI refresh and re-render
	out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
	out_data->out_flags |= PF_OutFlag_REFRESH_UI;
}

static void ApplyDefaultEffectParams(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[])
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
	if (!paramUtils.get())
	{
		return;
	}

	auto updateFloat = [&](int paramId, float value)
	{
		params[paramId]->u.fs_d.value = value;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	auto updatePopup = [&](int paramId, int value)
	{
		params[paramId]->u.pd.value = value;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	auto updateAngle = [&](int paramId, float degrees)
	{
		const A_long fixedAngle = FLOAT2FIX(degrees);
		params[paramId]->u.ad.value = fixedAngle;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	// Basic settings
	updateFloat(SDK_PROCAMP_LINE_COUNT, static_cast<float>(LINE_COUNT_DFLT));
	updateFloat(SDK_PROCAMP_LINE_LIFETIME, static_cast<float>(LINE_LIFETIME_DFLT));
	updateFloat(SDK_PROCAMP_LINE_TRAVEL, static_cast<float>(LINE_TRAVEL_DFLT));
	
	// Appearance
	updateFloat(SDK_PROCAMP_LINE_THICKNESS, static_cast<float>(LINE_THICKNESS_DFLT));
	updateFloat(SDK_PROCAMP_LINE_LENGTH, static_cast<float>(LINE_LENGTH_DFLT));
	updateAngle(SDK_PROCAMP_LINE_ANGLE, 0.0f);
	updateFloat(SDK_PROCAMP_LINE_TAIL_FADE, static_cast<float>(LINE_TAIL_FADE_DFLT));
	updateFloat(SDK_PROCAMP_LINE_AA, static_cast<float>(LINE_AA_DFLT));
	
	// Position & Spawn
	updatePopup(SDK_PROCAMP_LINE_ORIGIN_MODE, LINE_ORIGIN_MODE_DFLT);
	updateFloat(SDK_PROCAMP_LINE_SPAWN_SCALE_X, static_cast<float>(LINE_SPAWN_SCALE_X_DFLT));
	updateFloat(SDK_PROCAMP_LINE_SPAWN_SCALE_Y, static_cast<float>(LINE_SPAWN_SCALE_Y_DFLT));
	updateFloat(SDK_PROCAMP_ORIGIN_OFFSET_X, static_cast<float>(ORIGIN_OFFSET_X_DFLT));
	updateFloat(SDK_PROCAMP_ORIGIN_OFFSET_Y, static_cast<float>(ORIGIN_OFFSET_Y_DFLT));
	updateFloat(SDK_PROCAMP_LINE_INTERVAL, static_cast<float>(LINE_INTERVAL_DFLT));
	
	// Animation
	updatePopup(SDK_PROCAMP_ANIM_PATTERN, ANIM_PATTERN_DFLT);
	updateFloat(SDK_PROCAMP_CENTER_GAP, static_cast<float>(CENTER_GAP_DFLT));
	updatePopup(SDK_PROCAMP_LINE_EASING, LINE_EASING_DFLT);
	updateFloat(SDK_PROCAMP_LINE_START_TIME, static_cast<float>(LINE_START_TIME_DFLT));
	updateFloat(SDK_PROCAMP_LINE_DURATION, static_cast<float>(LINE_DURATION_DFLT));
	
	// Advanced
	updatePopup(SDK_PROCAMP_BLEND_MODE, BLEND_MODE_DFLT);
	updateFloat(SDK_PROCAMP_LINE_DEPTH_STRENGTH, static_cast<float>(LINE_DEPTH_DFLT));
	
	// New parameters (use defaults from .h file)
	updatePopup(SDK_PROCAMP_LINE_CAP, LINE_CAP_DFLT);
	updatePopup(SDK_PROCAMP_COLOR_MODE, COLOR_MODE_DFLT);
	updatePopup(SDK_PROCAMP_COLOR_PRESET, COLOR_PRESET_DFLT);
	updatePopup(SDK_PROCAMP_SPAWN_SOURCE, SPAWN_SOURCE_DFLT);
	
	auto updateCheckbox = [&](int paramId, bool value)
	{
		params[paramId]->u.bd.value = value ? 1 : 0;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};
	updateCheckbox(SDK_PROCAMP_HIDE_ELEMENT, HIDE_ELEMENT_DFLT);

	out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
	out_data->out_flags |= PF_OutFlag_REFRESH_UI;
}

/*
**
*/
// UI labels are the first string in each PF_ADD_* call below.
// Change those strings to rename parameters in the effect UI.
static PF_Err ParamsSetup(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_ParamDef	def;

	// ============================================================
	// Effect Preset (top-level, no group)
	// ============================================================
	AEFX_CLR_STRUCT(def);
	std::string presetLabels = "デフォルト|";
	for (int i = 0; i < kEffectPresetCount; ++i)
	{
		presetLabels += kEffectPresets[i].name;
		if (i < kEffectPresetCount - 1)
		{
			presetLabels += "|";
		}
	}
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_EFFECT_PRESET,
		1 + kEffectPresetCount,
		1,
		presetLabels.c_str(),
		SDK_PROCAMP_EFFECT_PRESET);

	// Random Seed (moved to top, after preset)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SEED,
		LINE_SEED_MIN_VALUE,
		LINE_SEED_MAX_VALUE,
		LINE_SEED_MIN_SLIDER,
		LINE_SEED_MAX_SLIDER,
		LINE_SEED_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_LINE_SEED);

	// ============================================================
	// Basic Settings
	// ============================================================

	// Line Count
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_LINE_COUNT,
		LINE_COUNT_MIN_VALUE,
		LINE_COUNT_MAX_VALUE,
		LINE_COUNT_MIN_SLIDER,
		LINE_COUNT_MAX_SLIDER,
		LINE_COUNT_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_LINE_COUNT);

	// Line Lifetime (frames)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_LIFETIME,
		LINE_LIFETIME_MIN_VALUE,
		LINE_LIFETIME_MAX_VALUE,
		LINE_LIFETIME_MIN_SLIDER,
		LINE_LIFETIME_MAX_SLIDER,
		LINE_LIFETIME_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_LINE_LIFETIME);

	// Spawn Interval (moved from Position group)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_INTERVAL,
		LINE_INTERVAL_MIN_VALUE,
		LINE_INTERVAL_MAX_VALUE,
		LINE_INTERVAL_MIN_SLIDER,
		LINE_INTERVAL_MAX_SLIDER,
		LINE_INTERVAL_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_LINE_INTERVAL);

	// Easing (moved here, after Interval)
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_EASING,
		28,
		LINE_EASING_DFLT,
		PM_EASING,
		SDK_PROCAMP_LINE_EASING);

	// ============================================================
	// Color Settings
	// ============================================================

	// Color Mode
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_COLOR_MODE,
		3,
		COLOR_MODE_DFLT,
		PM_COLOR_MODE,
		SDK_PROCAMP_COLOR_MODE);

	// Single Color
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_COLOR(
		P_COLOR,
		LINE_COLOR_DFLT_R8,
		LINE_COLOR_DFLT_G8,
		LINE_COLOR_DFLT_B8,
		SDK_PROCAMP_LINE_COLOR);

	// Color Preset
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_COLOR_PRESET,
		33,
		COLOR_PRESET_DFLT,
		PM_COLOR_PRESET,
		SDK_PROCAMP_COLOR_PRESET);

	// Custom Colors 1-8
	AEFX_CLR_STRUCT(def);
	def.ui_flags = PF_PUI_ECW_SEPARATOR;
	PF_ADD_COLOR(P_CUSTOM_1, 255, 0, 0, SDK_PROCAMP_CUSTOM_COLOR_1);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_2, 255, 128, 0, SDK_PROCAMP_CUSTOM_COLOR_2);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_3, 255, 255, 0, SDK_PROCAMP_CUSTOM_COLOR_3);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_4, 0, 255, 0, SDK_PROCAMP_CUSTOM_COLOR_4);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_5, 0, 255, 255, SDK_PROCAMP_CUSTOM_COLOR_5);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_6, 0, 0, 255, SDK_PROCAMP_CUSTOM_COLOR_6);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_7, 128, 0, 255, SDK_PROCAMP_CUSTOM_COLOR_7);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_8, 255, 0, 255, SDK_PROCAMP_CUSTOM_COLOR_8);

	// Travel Distance Linkage (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_TRAVEL_LINKAGE,
		3,
		LINKAGE_MODE_DFLT,
		PM_LINKAGE_MODE,
		SDK_PROCAMP_TRAVEL_LINKAGE);

	// Travel Distance Linkage Rate (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_TRAVEL_LINKAGE_RATE,
		LINKAGE_RATE_MIN_VALUE,
		LINKAGE_RATE_MAX_VALUE,
		LINKAGE_RATE_MIN_SLIDER,
		LINKAGE_RATE_MAX_SLIDER,
		LINKAGE_RATE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_TRAVEL_LINKAGE_RATE);

	// Travel Distance (moved from Basic Settings)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_TRAVEL,
		LINE_TRAVEL_MIN_VALUE,
		LINE_TRAVEL_MAX_VALUE,
		LINE_TRAVEL_MIN_SLIDER,
		LINE_TRAVEL_MAX_SLIDER,
		LINE_TRAVEL_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_TRAVEL);

	// ============================================================
	// Appearance
	// ============================================================

	// Thickness Linkage (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_THICKNESS_LINKAGE,
		3,
		LINKAGE_MODE_DFLT,
		PM_LINKAGE_MODE,
		SDK_PROCAMP_THICKNESS_LINKAGE);

	// Thickness Linkage Rate (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_THICKNESS_LINKAGE_RATE,
		LINKAGE_RATE_MIN_VALUE,
		LINKAGE_RATE_MAX_VALUE,
		LINKAGE_RATE_MIN_SLIDER,
		LINKAGE_RATE_MAX_SLIDER,
		LINKAGE_RATE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_THICKNESS_LINKAGE_RATE);

	// Line Thickness
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_THICKNESS,
		LINE_THICKNESS_MIN_VALUE,
		LINE_THICKNESS_MAX_VALUE,
		LINE_THICKNESS_MIN_SLIDER,
		LINE_THICKNESS_MAX_SLIDER,
		LINE_THICKNESS_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_THICKNESS);

	// Length Linkage (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_LENGTH_LINKAGE,
		3,
		LINKAGE_MODE_DFLT,
		PM_LINKAGE_MODE,
		SDK_PROCAMP_LENGTH_LINKAGE);

	// Length Linkage Rate (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_LENGTH_LINKAGE_RATE,
		LINKAGE_RATE_MIN_VALUE,
		LINKAGE_RATE_MAX_VALUE,
		LINKAGE_RATE_MIN_SLIDER,
		LINKAGE_RATE_MAX_SLIDER,
		LINKAGE_RATE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_LENGTH_LINKAGE_RATE);

	// Line Length
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_LENGTH,
		LINE_LENGTH_MIN_VALUE,
		LINE_LENGTH_MAX_VALUE,
		LINE_LENGTH_MIN_SLIDER,
		LINE_LENGTH_MAX_SLIDER,
		LINE_LENGTH_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_LENGTH);

	// Line Angle
	AEFX_CLR_STRUCT(def);
	PF_ADD_ANGLE(
		P_ANGLE,
		0,
		SDK_PROCAMP_LINE_ANGLE);

	// Line Cap
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_LINE_CAP,
		2,
		LINE_CAP_DFLT,
		PM_LINE_CAP,
		SDK_PROCAMP_LINE_CAP);

	// Tail Fade
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_TAIL_FADE,
		LINE_TAIL_FADE_MIN_VALUE,
		LINE_TAIL_FADE_MAX_VALUE,
		LINE_TAIL_FADE_MIN_SLIDER,
		LINE_TAIL_FADE_MAX_SLIDER,
		LINE_TAIL_FADE_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_TAIL_FADE);

	// ============================================================
	// Line Origin (線�E起点)
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_POSITION_HEADER, SDK_PROCAMP_POSITION_HEADER);

	// 1. Spawn Source
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_SPAWN_SOURCE,
		2,
		SPAWN_SOURCE_DFLT,
		P_SPAWN_SOURCE_CHOICES,
		SDK_PROCAMP_SPAWN_SOURCE);

	// 2. Alpha Threshold
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_ALPHA_THRESH,
		LINE_ALPHA_THRESH_MIN_VALUE,
		LINE_ALPHA_THRESH_MAX_VALUE,
		LINE_ALPHA_THRESH_MIN_SLIDER,
		LINE_ALPHA_THRESH_MAX_SLIDER,
		LINE_ALPHA_THRESH_DFLT,
		PF_Precision_THOUSANDTHS,
		0,
		0,
		SDK_PROCAMP_LINE_ALPHA_THRESH);

	// 3. Wind Origin Mode (moved before Animation Pattern)
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_ORIGIN_MODE,
		3,
		LINE_ORIGIN_MODE_DFLT,
		PM_ORIGIN_MODE,
		SDK_PROCAMP_LINE_ORIGIN_MODE);

	// 4. Animation Pattern (Direction)
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_ANIM_PATTERN,
		3,
		ANIM_PATTERN_DFLT,
		PM_ANIM_PATTERN,
		SDK_PROCAMP_ANIM_PATTERN);

	// 5. Start Time
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_START_TIME,
		LINE_START_TIME_MIN_VALUE,
		LINE_START_TIME_MAX_VALUE,
		LINE_START_TIME_MIN_SLIDER,
		LINE_START_TIME_MAX_SLIDER,
		LINE_START_TIME_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_LINE_START_TIME);

	// 6. Duration
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_DURATION,
		LINE_DURATION_MIN_VALUE,
		LINE_DURATION_MAX_VALUE,
		LINE_DURATION_MIN_SLIDER,
		LINE_DURATION_MAX_SLIDER,
		LINE_DURATION_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_LINE_DURATION);

	// 7. Depth Strength (moved from Advanced)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_DEPTH_STRENGTH,
		LINE_DEPTH_MIN_VALUE,
		LINE_DEPTH_MAX_VALUE,
		LINE_DEPTH_MIN_SLIDER,
		LINE_DEPTH_MAX_SLIDER,
		LINE_DEPTH_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_DEPTH_STRENGTH);

	// 8. Center Gap (moved before Offset X)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_CENTER_GAP,
		CENTER_GAP_MIN_VALUE,
		CENTER_GAP_MAX_VALUE,
		CENTER_GAP_MIN_SLIDER,
		CENTER_GAP_MAX_SLIDER,
		CENTER_GAP_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		SDK_PROCAMP_CENTER_GAP);

	// 9. Origin Offset X
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_OFFSET_X,
		ORIGIN_OFFSET_X_MIN_VALUE,
		ORIGIN_OFFSET_X_MAX_VALUE,
		ORIGIN_OFFSET_X_MIN_SLIDER,
		ORIGIN_OFFSET_X_MAX_SLIDER,
		ORIGIN_OFFSET_X_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_ORIGIN_OFFSET_X);

	// 10. Origin Offset Y
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_OFFSET_Y,
		ORIGIN_OFFSET_Y_MIN_VALUE,
		ORIGIN_OFFSET_Y_MAX_VALUE,
		ORIGIN_OFFSET_Y_MIN_SLIDER,
		ORIGIN_OFFSET_Y_MAX_SLIDER,
		ORIGIN_OFFSET_Y_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_ORIGIN_OFFSET_Y);

	// 11. Spawn Scale X
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SPAWN_SCALE_X,
		LINE_SPAWN_SCALE_X_MIN_VALUE,
		LINE_SPAWN_SCALE_X_MAX_VALUE,
		LINE_SPAWN_SCALE_X_MIN_SLIDER,
		LINE_SPAWN_SCALE_X_MAX_SLIDER,
		LINE_SPAWN_SCALE_X_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_SPAWN_SCALE_X);

	// 12. Spawn Scale Y
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SPAWN_SCALE_Y,
		LINE_SPAWN_SCALE_Y_MIN_VALUE,
		LINE_SPAWN_SCALE_Y_MAX_VALUE,
		LINE_SPAWN_SCALE_Y_MIN_SLIDER,
		LINE_SPAWN_SCALE_Y_MAX_SLIDER,
		LINE_SPAWN_SCALE_Y_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_SPAWN_SCALE_Y);

	// 13. Spawn Rotation
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SPAWN_ROTATION,
		LINE_SPAWN_ROTATION_MIN_VALUE,
		LINE_SPAWN_ROTATION_MAX_VALUE,
		LINE_SPAWN_ROTATION_MIN_SLIDER,
		LINE_SPAWN_ROTATION_MAX_SLIDER,
		LINE_SPAWN_ROTATION_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_SPAWN_ROTATION);

	// 14. Show Spawn Area
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(P_SHOW_SPAWN, "", SHOW_SPAWN_AREA_DFLT, 0, SDK_PROCAMP_LINE_SHOW_SPAWN_AREA);

	// 16. Spawn Area Color
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_SPAWN_COLOR, 128, 128, 255, SDK_PROCAMP_LINE_SPAWN_AREA_COLOR);  // Light blue default

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(SDK_PROCAMP_POSITION_TOPIC_END);

	// ============================================================
	// Shadow
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_SHADOW, SDK_PROCAMP_SHADOW_HEADER);

	// Shadow Enable
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(P_SHADOW_ENABLE, "", SHADOW_ENABLE_DFLT, 0, SDK_PROCAMP_SHADOW_ENABLE);

	// Shadow Color
	AEFX_CLR_STRUCT(def);
	def.u.cd.value.red = static_cast<A_u_short>(SHADOW_COLOR_R_DFLT * 65535);
	def.u.cd.value.green = static_cast<A_u_short>(SHADOW_COLOR_G_DFLT * 65535);
	def.u.cd.value.blue = static_cast<A_u_short>(SHADOW_COLOR_B_DFLT * 65535);
	def.u.cd.value.alpha = 65535;
	def.u.cd.dephault = def.u.cd.value;
	PF_ADD_COLOR(P_SHADOW_COLOR, def.u.cd.value.red, def.u.cd.value.green, def.u.cd.value.blue, SDK_PROCAMP_SHADOW_COLOR);

	// Shadow Offset X
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SHADOW_OFFSET_X,
		SHADOW_OFFSET_X_MIN,
		SHADOW_OFFSET_X_MAX,
		SHADOW_OFFSET_X_MIN,
		SHADOW_OFFSET_X_MAX,
		SHADOW_OFFSET_X_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_SHADOW_OFFSET_X);

	// Shadow Offset Y
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SHADOW_OFFSET_Y,
		SHADOW_OFFSET_Y_MIN,
		SHADOW_OFFSET_Y_MAX,
		SHADOW_OFFSET_Y_MIN,
		SHADOW_OFFSET_Y_MAX,
		SHADOW_OFFSET_Y_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_SHADOW_OFFSET_Y);

	// Shadow Opacity
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SHADOW_OPACITY,
		SHADOW_OPACITY_MIN,
		SHADOW_OPACITY_MAX,
		SHADOW_OPACITY_MIN,
		SHADOW_OPACITY_MAX,
		SHADOW_OPACITY_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		SDK_PROCAMP_SHADOW_OPACITY);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(SDK_PROCAMP_SHADOW_TOPIC_END);

	// ============================================================
	// Motion Blur
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_MOTION_BLUR, SDK_PROCAMP_MOTION_BLUR_HEADER);

	// Motion Blur Enable
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(P_MOTION_BLUR, "", MOTION_BLUR_ENABLE_DFLT, 0, SDK_PROCAMP_MOTION_BLUR_ENABLE);

	// Motion Blur Samples (Quality)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_BLUR_SAMPLES,
		MOTION_BLUR_SAMPLES_MIN_VALUE,
		MOTION_BLUR_SAMPLES_MAX_VALUE,
		MOTION_BLUR_SAMPLES_MIN_SLIDER,
		MOTION_BLUR_SAMPLES_MAX_SLIDER,
		MOTION_BLUR_SAMPLES_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_MOTION_BLUR_SAMPLES);

	// Motion Blur Shutter Angle (0-360°)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_BLUR_ANGLE,
		MOTION_BLUR_ANGLE_MIN_VALUE,
		MOTION_BLUR_ANGLE_MAX_VALUE,
		MOTION_BLUR_ANGLE_MIN_SLIDER,
		MOTION_BLUR_ANGLE_MAX_SLIDER,
		MOTION_BLUR_ANGLE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		SDK_PROCAMP_MOTION_BLUR_STRENGTH);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(SDK_PROCAMP_MOTION_BLUR_TOPIC_END);

	// ============================================================
	// Advanced
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_ADVANCED_HEADER, SDK_PROCAMP_ADVANCED_HEADER);

	// Anti-Aliasing (moved to Advanced)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_AA,
		LINE_AA_MIN_VALUE,
		LINE_AA_MAX_VALUE,
		LINE_AA_MIN_SLIDER,
		LINE_AA_MAX_SLIDER,
		LINE_AA_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		SDK_PROCAMP_LINE_AA);

	// Hide Element (lines only)
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(P_HIDE_ELEMENT, "", HIDE_ELEMENT_DFLT, 0, SDK_PROCAMP_HIDE_ELEMENT);

	// Blend Mode
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_BLEND_MODE,
		4,
		BLEND_MODE_DFLT,
		PM_BLEND_MODE,
		SDK_PROCAMP_BLEND_MODE);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(SDK_PROCAMP_ADVANCED_TOPIC_END);

	// ============================================================
	// Hidden Parameters (for backwards compatibility)
	// ============================================================
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;
	PF_ADD_CHECKBOX("", "", LINE_ALLOW_MIDPLAY_DFLT, 0, SDK_PROCAMP_LINE_ALLOW_MIDPLAY);

	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;
	PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
		LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
		PF_Precision_TENTHS, 0, 0, SDK_PROCAMP_LINE_COLOR_R);

	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;
	PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
		LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
		PF_Precision_TENTHS, 0, 0, SDK_PROCAMP_LINE_COLOR_G);

	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;
	PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
		LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
		PF_Precision_TENTHS, 0, 0, SDK_PROCAMP_LINE_COLOR_B);

	out_data->num_params = SDK_PROCAMP_NUM_PARAMS;
	return PF_Err_NONE;
}

/*
**
*/
static PF_Err Render(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
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

	if (in_data->appl_id == 'PrMr')
	{
		PF_LayerDef* src = &params[0]->u.ld;
		PF_LayerDef* dest = output;

		const char* srcData = (const char*)src->data;
		char* destData = (char*)dest->data;

		// Color Mode and Palette Setup
		// normalizePopup returns 0-based: 0=Single, 1=Preset, 2=Custom
		const int colorMode = normalizePopup(params[SDK_PROCAMP_COLOR_MODE]->u.pd.value, 3);
		const int presetIndex = normalizePopup(params[SDK_PROCAMP_COLOR_PRESET]->u.pd.value, 33);
		
		// Build color palette (8 colors, RGB normalized)
		float colorPalette[8][3];
		
		if (colorMode == 0)  // Single (0-based)
		{
			// Single color mode: all 8 slots have the same color
			const PF_Pixel singleColor = params[SDK_PROCAMP_LINE_COLOR]->u.cd.value;
			const float singleR = singleColor.red / 255.0f;
			const float singleG = singleColor.green / 255.0f;
			const float singleB = singleColor.blue / 255.0f;
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
				const PF_Pixel customColor = params[customColorParams[i]]->u.cd.value;
				colorPalette[i][0] = customColor.red / 255.0f;
				colorPalette[i][1] = customColor.green / 255.0f;
				colorPalette[i][2] = customColor.blue / 255.0f;
			}
		}
		
		// Convert color palette to VUYA format for rendering
		float paletteV[8], paletteU[8], paletteY[8];
		for (int i = 0; i < 8; ++i)
		{
			const float r = colorPalette[i][0];
			const float g = colorPalette[i][1];
			const float b = colorPalette[i][2];
			paletteY[i] = r * 0.299f + g * 0.587f + b * 0.114f;
			paletteU[i] = r * -0.168736f + g * -0.331264f + b * 0.5f;
			paletteV[i] = r * 0.5f + g * -0.418688f + b * -0.081312f;
		}
		
		// Default line color (for compatibility, use first palette color)
		const float lineR = colorPalette[0][0];
		const float lineG = colorPalette[0][1];
		const float lineB = colorPalette[0][2];
		const float lineYVal = paletteY[0];
		const float lineUVal = paletteU[0];
		const float lineVVal = paletteV[0];
	(void)lineR; (void)lineG; (void)lineB;
	(void)lineYVal; (void)lineUVal; (void)lineVVal;

	const float lineThickness = (float)params[SDK_PROCAMP_LINE_THICKNESS]->u.fs_d.value;
	const float lineLength = (float)params[SDK_PROCAMP_LINE_LENGTH]->u.fs_d.value;
	const int lineCap = normalizePopup(params[SDK_PROCAMP_LINE_CAP]->u.pd.value, 2);
	const float lineAngle = (float)FIX_2_FLOAT(params[SDK_PROCAMP_LINE_ANGLE]->u.ad.value);
	const float lineAA = (float)params[SDK_PROCAMP_LINE_AA]->u.fs_d.value;
	
	// Spawn Source: if "Full Frame" selected, ignore alpha threshold
	const int spawnSource = normalizePopup(params[SDK_PROCAMP_SPAWN_SOURCE]->u.pd.value, 2);
		float lineAlphaThreshold = (float)params[SDK_PROCAMP_LINE_ALPHA_THRESH]->u.fs_d.value;
		if (spawnSource == SPAWN_SOURCE_FULL_FRAME) {
			lineAlphaThreshold = 1.0f;  // Full frame: ignore alpha, spawn everywhere
		}
		const int lineOriginMode = normalizePopup(params[SDK_PROCAMP_LINE_ORIGIN_MODE]->u.pd.value, 3);
		const float dsx = (in_data->downsample_x.den != 0) ? ((float)in_data->downsample_x.num / (float)in_data->downsample_x.den) : 1.0f;
		const float dsy = (in_data->downsample_y.den != 0) ? ((float)in_data->downsample_y.num / (float)in_data->downsample_y.den) : 1.0f;
		const float dsMax = dsx > dsy ? dsx : dsy;
		const float dsScale = dsMax >= 1.0f ? (1.0f / dsMax) : (dsMax > 0.0f ? dsMax : 1.0f);
		const float lineThicknessScaled = lineThickness * dsScale;
		const float lineLengthScaled = lineLength * dsScale;
		const float lineAAScaled = lineAA * dsScale;
		const float effectiveAA = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;
		// Center is now controlled by Origin Offset X/Y only
		const int lineCount = (int)params[SDK_PROCAMP_LINE_COUNT]->u.fs_d.value;
		const float lineLifetime = (float)params[SDK_PROCAMP_LINE_LIFETIME]->u.fs_d.value;
		const float lineInterval = (float)params[SDK_PROCAMP_LINE_INTERVAL]->u.fs_d.value;
		const int lineSeed = (int)params[SDK_PROCAMP_LINE_SEED]->u.fs_d.value;
		const int lineEasing = normalizePopup(params[SDK_PROCAMP_LINE_EASING]->u.pd.value, 28);
		const float lineTravel = (float)params[SDK_PROCAMP_LINE_TRAVEL]->u.fs_d.value;
		const float lineTravelScaled = lineTravel * dsScale;
		const float lineTailFade = (float)params[SDK_PROCAMP_LINE_TAIL_FADE]->u.fs_d.value;
		const float lineDepthStrength = (float)params[SDK_PROCAMP_LINE_DEPTH_STRENGTH]->u.fs_d.value / 10.0f; // Normalize 0-10 to 0-1
	// allowMidPlay is now replaced by negative Start Time - kept for backward compatibility but ignored
	// const bool allowMidPlay = params[SDK_PROCAMP_LINE_ALLOW_MIDPLAY]->u.bd.value != 0;
	const bool hideElement = params[SDK_PROCAMP_HIDE_ELEMENT]->u.bd.value != 0;
	const int blendMode = NormalizePopupValue((int)params[SDK_PROCAMP_BLEND_MODE]->u.pd.value, 4);
	
	// Shadow parameters
	const bool shadowEnable = params[SDK_PROCAMP_SHADOW_ENABLE]->u.bd.value != 0;
	const float shadowColorR = (float)params[SDK_PROCAMP_SHADOW_COLOR]->u.cd.value.red / 255.0f;
	const float shadowColorG = (float)params[SDK_PROCAMP_SHADOW_COLOR]->u.cd.value.green / 255.0f;
	const float shadowColorB = (float)params[SDK_PROCAMP_SHADOW_COLOR]->u.cd.value.blue / 255.0f;
	// Convert shadow color to YUV
	const float shadowY = shadowColorR * 0.299f + shadowColorG * 0.587f + shadowColorB * 0.114f;
	const float shadowU = shadowColorR * -0.168736f + shadowColorG * -0.331264f + shadowColorB * 0.5f;
	const float shadowV = shadowColorR * 0.5f + shadowColorG * -0.418688f + shadowColorB * -0.081312f;
	const float shadowOffsetX = (float)params[SDK_PROCAMP_SHADOW_OFFSET_X]->u.fs_d.value * dsScale;
	const float shadowOffsetY = (float)params[SDK_PROCAMP_SHADOW_OFFSET_Y]->u.fs_d.value * dsScale;
	const float shadowOpacity = (float)params[SDK_PROCAMP_SHADOW_OPACITY]->u.fs_d.value;
	
	// Motion Blur parameters
	const bool motionBlurEnable = params[SDK_PROCAMP_MOTION_BLUR_ENABLE]->u.bd.value != 0;
	const int motionBlurSamples = (int)params[SDK_PROCAMP_MOTION_BLUR_SAMPLES]->u.fs_d.value;
	const float motionBlurStrength = (float)params[SDK_PROCAMP_MOTION_BLUR_STRENGTH]->u.fs_d.value;
	
	// Focus (Depth of Field) parameters
	// Focus parameters removed
	const float spawnScaleX = (float)params[SDK_PROCAMP_LINE_SPAWN_SCALE_X]->u.fs_d.value / 100.0f;
	const float spawnScaleY = (float)params[SDK_PROCAMP_LINE_SPAWN_SCALE_Y]->u.fs_d.value / 100.0f;
	const float spawnRotationDeg = (float)params[SDK_PROCAMP_LINE_SPAWN_ROTATION]->u.fs_d.value;
	const float spawnRotationRad = spawnRotationDeg * 3.14159265f / 180.0f;
	const float spawnCos = FastCos(spawnRotationRad);
	const float spawnSin = FastSin(spawnRotationRad);
	const bool showSpawnArea = params[SDK_PROCAMP_LINE_SHOW_SPAWN_AREA]->u.bd.value != 0;
	const PF_Pixel spawnAreaColorPx = params[SDK_PROCAMP_LINE_SPAWN_AREA_COLOR]->u.cd.value;
	const float spawnAreaColorR = spawnAreaColorPx.red / 255.0f;
	const float spawnAreaColorG = spawnAreaColorPx.green / 255.0f;
	const float spawnAreaColorB = spawnAreaColorPx.blue / 255.0f;
	// Convert spawn area color to YUV
	const float spawnAreaY = spawnAreaColorR * 0.299f + spawnAreaColorG * 0.587f + spawnAreaColorB * 0.114f;
	const float spawnAreaU = spawnAreaColorR * -0.168736f + spawnAreaColorG * -0.331264f + spawnAreaColorB * 0.5f;
	const float spawnAreaV = spawnAreaColorR * 0.5f + spawnAreaColorG * -0.418688f + spawnAreaColorB * -0.081312f;
	// CPU rendering uses current_time which is clip-relative time.
		// This ensures cache consistency - same clip frame = same result.
		const A_long clipTime = in_data->current_time; // Clip-relative time
		const A_long frameIndex = (in_data->time_step != 0) ? (clipTime / in_data->time_step) : 0;
		
		// Try to get clip start using PF_UtilitySuite
		A_long clipStartFrame = 0;
		A_long trackItemStart = 0;
		{
			AEFX_SuiteScoper<PF_UtilitySuite> utilitySuite(in_data, kPFUtilitySuite, kPFUtilitySuiteVersion, out_data);
			if (utilitySuite.get())
			{
				utilitySuite->GetClipStart(in_data->effect_ref, &clipStartFrame);
				utilitySuite->GetTrackItemStart(in_data->effect_ref, &trackItemStart);
			}
		}
		
		// Share clipStartFrame with GPU renderer
		// Key: clipStartFrame itself, Value: clipStartFrame
		// GPU can find the correct clipStart by looking for keys <= mediaFrameIndex
		if (clipStartFrame > 0)
		{
			SharedClipData::SetClipStart(clipStartFrame, clipStartFrame);
		}

		const float angleRadians = (float)(M_PI / 180) * lineAngle;
		const float lineCos = cos(angleRadians);
		const float lineSin = sin(angleRadians);
		// Optimized branchless saturate using fminf/fmaxf
		auto saturate = [](float v) { return fminf(fmaxf(v, 0.0f), 1.0f); };

		// Use local state for stateless rendering (no global caching)
		LineInstanceState localLineState;
		LineInstanceState* lineState = &localLineState;

		// Branchless clamp for line count
		const int clampedLineCount = (int)fminf(fmaxf((float)lineCount, 1.0f), 5000.0f);
		const int intervalFrames = lineInterval < 0.5f ? 0 : (int)(lineInterval + 0.5f);
		
		// Generate line params locally each frame for stateless rendering
		lineState->lineCount = clampedLineCount;
		lineState->lineSeed = lineSeed;
		lineState->lineDepthStrength = lineDepthStrength;
		lineState->lineInterval = intervalFrames;
		lineState->lineParams.assign(clampedLineCount, {});
		for (int i = 0; i < clampedLineCount; ++i)
		{
			const csSDK_uint32 base = (csSDK_uint32)(lineSeed * 1315423911u) + (csSDK_uint32)i * 2654435761u;
			const float rx = Rand01(base + 1);
			const float ry = Rand01(base + 2);
			const float rstart = Rand01(base + 5);
			const float rdepth = Rand01(base + 6);
			const float depthScale = DepthScale(rdepth, lineDepthStrength);

			LineParams lp;
			lp.posX = rx;
			lp.posY = ry;
			lp.baseLen = lineLengthScaled * depthScale;
			lp.baseThick = lineThicknessScaled * depthScale;
			lp.angle = lineAngle;
			const float life = lineLifetime > 1.0f ? lineLifetime : 1.0f;
			const float interval = intervalFrames > 0 ? (float)intervalFrames : 0.0f;
			const float period = life + interval;
			float startFrame = rstart * period; // Sequence-time based, no offset
			lp.startFrame = startFrame;
			lp.depthScale = depthScale;
			lp.depthValue = rdepth;
			lineState->lineParams[i] = lp;
		}

		lineState->lineDerived.assign(lineState->lineCount, {});
		lineState->lineActive.assign(lineState->lineCount, 0);

		const float life = lineLifetime > 1.0f ? lineLifetime : 1.0f;
		const float interval = intervalFrames > 0 ? (float)intervalFrames : 0.0f;
		const float period = life + interval;
		// No center offset - use Origin Offset X/Y instead
		const float centerOffsetX = 0.0f;
		const float centerOffsetY = 0.0f;
		const float alphaThreshold = lineAlphaThreshold;
		const int alphaStride = 4;
		int alphaMinX = output->width;
		int alphaMinY = output->height;
		int alphaMaxX = -1;
		int alphaMaxY = -1;
		for (int y = 0; y < output->height; y += alphaStride)
		{
			const float* row = (const float*)(srcData + y * src->rowbytes);
			for (int x = 0; x < output->width; x += alphaStride)
			{
				const float aSample = row[x * 4 + 3];
				if (aSample > alphaThreshold)
				{
					if (x < alphaMinX) alphaMinX = x;
					if (y < alphaMinY) alphaMinY = y;
					if (x > alphaMaxX) alphaMaxX = x;
					if (y > alphaMaxY) alphaMaxY = y;
				}

			}
		}
		if (alphaMaxX < alphaMinX || alphaMaxY < alphaMinY)
		{
			alphaMinX = 0;
			alphaMinY = 0;
			alphaMaxX = output->width > 0 ? (output->width - 1) : 0;
			alphaMaxY = output->height > 0 ? (output->height - 1) : 0;
		}
		const float alphaBoundsMinX = (float)alphaMinX + centerOffsetX;
		const float alphaBoundsMinY = (float)alphaMinY + centerOffsetY;
		const float alphaBoundsWidth = (float)(alphaMaxX - alphaMinX + 1);
		const float alphaBoundsHeight = (float)(alphaMaxY - alphaMinY + 1);
	
	// Start Time + Duration: control when lines spawn
	const float lineStartTime = (float)params[SDK_PROCAMP_LINE_START_TIME]->u.fs_d.value;
	const float lineDuration = (float)params[SDK_PROCAMP_LINE_DURATION]->u.fs_d.value;
	// Calculate effective end time (0 duration = infinite)
	const float lineEndTime = (lineDuration > 0.0f) ? (lineStartTime + lineDuration) : 0.0f;
	
	// Use frameIndex directly for sequence-time based rendering.
	const float timeFramesBase = (float)frameIndex;
	
	// Origin Offset X/Y (px) - 線の起点のオフセット
	// Apply downsample scale to origin offsets (user inputs in full-resolution pixels)
	const float userOriginOffsetX = (float)params[SDK_PROCAMP_ORIGIN_OFFSET_X]->u.fs_d.value * dsScale;
	const float userOriginOffsetY = (float)params[SDK_PROCAMP_ORIGIN_OFFSET_Y]->u.fs_d.value * dsScale;
	
	// Animation Pattern (1=Simple, 2=Half Reverse, 3=Split)
	const int animPattern = params[SDK_PROCAMP_ANIM_PATTERN]->u.pd.value;
	const float centerGap = (float)params[SDK_PROCAMP_CENTER_GAP]->u.fs_d.value;
	
	for (int i = 0; lineState && i < lineState->lineCount; ++i)
		{
			const LineParams& lp = lineState->lineParams[i];
			const float timeFrames = timeFramesBase;
			// Note: allowMidPlay functionality is now handled by negative Start Time
			// Start Time < 0 allows lines to appear mid-animation at clip start
			float age = fmodf(timeFrames - lp.startFrame, period);
			if (age < 0.0f)
			{
				age += period;
			}
			
			// Start Time + End Time support: skip cycles outside the active time range
			{
				// Calculate when this cycle started
				const float cycleStartFrame = timeFrames - age;
				// Skip if this cycle started before startTime
				if (cycleStartFrame < lineStartTime)
				{
					continue;
				}
				// Skip if endTime is set and this cycle started after endTime
				if (lineEndTime > 0.0f && cycleStartFrame >= lineEndTime)
				{
					continue;
				}
			}
			
			if (age > life)
			{
				continue;
			}
		const float t = age / life;
		const float tMove = ApplyEasingLUT(t, lineEasing);
		(void)tMove;  // Unused, kept for compatibility
		const float maxLen = lp.baseLen;
		const float travelRange = lineTravelScaled * lp.depthScale;
		
		// "Head extends from tail, then tail retracts" animation (matches GPU logic)????
		// Total travel distance includes line length for proper appearance/disappearance
		const float totalTravelDist = travelRange + maxLen;  // Total distance for full animation
		const float tailStartPos = -0.5f * travelRange - maxLen;  // Start hidden on left
		
		const float travelT = ApplyEasingLUT(t, lineEasing);
		const float currentTravelPos = tailStartPos + totalTravelDist * travelT;
		
		float headPosX, tailPosX, currentLength;
		
		if (t <= 0.5f)
		{
			// First half: tail at current travel position, head extends from it
			const float extendT = ApplyEasingLUT(t * 2.0f, lineEasing);
			tailPosX = currentTravelPos;
			headPosX = tailPosX + maxLen * extendT;
			currentLength = maxLen * extendT;
		}
		else
		{
			// Second half: head at current travel position + maxLen, tail retracts toward it
			const float retractT = ApplyEasingLUT((t - 0.5f) * 2.0f, lineEasing);
			headPosX = currentTravelPos + maxLen;
			tailPosX = headPosX - maxLen * (1.0f - retractT);
			currentLength = maxLen * (1.0f - retractT);
		}
		
		// For rendering: center = midpoint between head and tail
		const float segCenterX = (headPosX + tailPosX) * 0.5f;
		
		// Appear/Disappear scale + fade: smooth fade-in at start, fade-out at end
		// Uses easeOutCubic for appear (fast start, very slow end - more natural)
		// Uses easeInCubic for disappear (very slow start, fast end - more natural)
		const float appearDuration = 0.10f;   // 10% of lifetime for appear
		const float disappearDuration = 0.10f; // 10% of lifetime for disappear
		float appearScale = 1.0f;
		float appearAlpha = 1.0f;
		if (t < appearDuration)
		{
			// Appear: easeOutCubic (1 - (1-t)^3) - starts fast, ends very smoothly
			const float at = t / appearDuration;
			const float inv = 1.0f - at;
			const float eased = 1.0f - inv * inv * inv;
			appearScale = eased;
			appearAlpha = eased;  // Fade in with scale
		}
		else if (t > (1.0f - disappearDuration))
		{
			// Disappear: easeInCubic (t^3) - starts very smoothly, ends fast
			const float dt = (1.0f - t) / disappearDuration;
			const float eased = dt * dt * dt;
			appearScale = eased;
			appearAlpha = eased;  // Fade out with scale
		}
		
		const float halfLen = currentLength * 0.5f * appearScale;

		// Skip if thickness is less than 1px (effectively invisible)
		const bool isTiny = (lp.baseThick * appearScale < 1.0f);
		lineState->lineActive[i] = isTiny ? 0 : 1;
		if (isTiny)
		{
			continue;
		}

		// Wind Origin: adjust spawn area position (overall atmosphere, not per-line animation)
		// Apply offset in the direction of line angle (both X and Y components)
		float originOffset = 0.0f;
		if (lineOriginMode == 1)  // Forward
		{
			originOffset = 0.5f * travelRange;
		}
		else if (lineOriginMode == 2)  // Backward
		{
			originOffset = -0.5f * travelRange;
		}

		// Animation Pattern adjustments
		// Pattern 1: Simple - all same direction
		// Pattern 2: Half Reverse - every other line reversed
		// Pattern 3: Split - sides go opposite directions (angle-linked)
		// Center Gap applies to all patterns when > 0
		
		float adjustedPosX = lp.posX;
		float adjustedPosY = lp.posY;
		float adjustedAngle = lineAngle;
		
		// Calculate perpendicular axis for center gap and Split pattern (aspect-corrected)
		const float invW = alphaBoundsWidth > 0.0f ? (1.0f / alphaBoundsWidth) : 1.0f;
		const float invH = alphaBoundsHeight > 0.0f ? (1.0f / alphaBoundsHeight) : 1.0f;
		const float dirX = lineCos * invW;
		const float dirY = lineSin * invH;
		float perpX = -dirY;  // Perpendicular to movement direction
		float perpY = dirX;
		const float perpLen = sqrtf(perpX * perpX + perpY * perpY);
		if (perpLen > 0.00001f)
		{
			perpX /= perpLen;
			perpY /= perpLen;
		}
		const float sideValue = (lp.posX - 0.5f) * perpX + (lp.posY - 0.5f) * perpY;
		
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
			if (animPattern == 2)  // Half Reverse: 50% of lines go opposite direction
			{
				if (i % 2 == 1)
				{
					adjustedAngle = lineAngle + 180.0f;
				}
			}
			else if (animPattern == 3)  // Split: sides go opposite directions
			{
				if (sideValue < 0.0f)
				{
					adjustedAngle = lineAngle + 180.0f;  // Negative side flows opposite
				}
				// Positive side keeps original angle
			}
			// animPattern == 1 (Simple): no direction adjustment
		}
		
		const float adjustedCos = FastCos(adjustedAngle * 3.14159265f / 180.0f);
		const float adjustedSin = FastSin(adjustedAngle * 3.14159265f / 180.0f);

		LineDerived ld;
		const float alphaCenterX = alphaBoundsMinX + alphaBoundsWidth * 0.5f;
		const float alphaCenterY = alphaBoundsMinY + alphaBoundsHeight * 0.5f;
		// Apply spawn rotation to the spawn position offset
		const float spawnOffsetX = (adjustedPosX - 0.5f) * alphaBoundsWidth * spawnScaleX;
		const float spawnOffsetY = (adjustedPosY - 0.5f) * alphaBoundsHeight * spawnScaleY;
		const float rotatedSpawnX = spawnOffsetX * spawnCos - spawnOffsetY * spawnSin;
		const float rotatedSpawnY = spawnOffsetX * spawnSin + spawnOffsetY * spawnCos;
		ld.centerX = alphaCenterX + rotatedSpawnX + originOffset * adjustedCos + userOriginOffsetX;
		ld.centerY = alphaCenterY + rotatedSpawnY + originOffset * adjustedSin + userOriginOffsetY;
		ld.cosA = adjustedCos;
		ld.sinA = adjustedSin;
		ld.halfLen = halfLen;
		ld.segCenterX = segCenterX;
		ld.depth = lp.depthValue;  // Store depth value for consistent blend mode

		// Focus (Depth of Field) disabled
		ld.halfThick = lp.baseThick * 0.5f * appearScale;
		ld.focusAlpha = 1.0f;
		ld.appearAlpha = appearAlpha;  // Appear/disappear fade alpha
		ld.lineVelocity = ApplyEasingDerivative(t, lineEasing);  // Motion blur velocity
		
		// Select color from palette: Single mode uses 0, Preset/Custom uses random based on seed
		if (colorMode == 0)  // Single mode
		{
			ld.colorIndex = 0;
		}
		else  // Preset or Custom mode
		{
			// Use existing seed + line index for random color selection
			const csSDK_uint32 colorBase = (csSDK_uint32)(lineSeed * 1315423911u) + (csSDK_uint32)i * 2654435761u + 12345u;
			ld.colorIndex = (int)(Rand01(colorBase) * 8.0f);
			if (ld.colorIndex > 7) ld.colorIndex = 7;
		}
		
		// Pre-compute expensive per-pixel calculations once per line
		const float depthScale = DepthScale(ld.depth, lineDepthStrength);
		const float depthFadeT = saturate((depthScale - 0.2f) / (0.6f - 0.2f));
		ld.depthAlpha = 0.05f + 0.95f * depthFadeT;
		const float denom = (2.0f * ld.halfLen) > 0.0001f ? (2.0f * ld.halfLen) : 0.0001f;
		ld.invDenom = 1.0f / denom;
		ld._padding = 0;  // Initialize padding for clean memory
		
		lineState->lineDerived[i] = ld;
	}

		const int tileSize = 32;
		const int tileCountX = (output->width + tileSize - 1) / tileSize;
		const int tileCountY = (output->height + tileSize - 1) / tileSize;
		const int tileCount = tileCountX * tileCountY;

		if (lineState)
		{
			lineState->tileCounts.assign(tileCount, 0);
		}
		for (int i = 0; lineState && i < lineState->lineCount; ++i)
		{
			if (!lineState->lineActive[i])
			{
				continue;
			}
			const LineDerived& ld = lineState->lineDerived[i];
			const float radius = fabsf(ld.segCenterX) + ld.halfLen + ld.halfThick + lineAAScaled;
			int minX = (int)((ld.centerX - radius) / tileSize);
			int maxX = (int)((ld.centerX + radius) / tileSize);
			int minY = (int)((ld.centerY - radius) / tileSize);
			int maxY = (int)((ld.centerY + radius) / tileSize);
			if (minX < 0) minX = 0;
			if (minY < 0) minY = 0;
			if (maxX >= tileCountX) maxX = tileCountX - 1;
			if (maxY >= tileCountY) maxY = tileCountY - 1;
			for (int ty = minY; ty <= maxY; ++ty)
			{
				for (int tx = minX; tx <= maxX; ++tx)
				{
					lineState->tileCounts[ty * tileCountX + tx] += 1;
				}
			}
		}

		if (lineState)
		{
			lineState->tileOffsets.assign(tileCount + 1, 0);
			for (int i = 0; i < tileCount; ++i)
			{
				lineState->tileOffsets[i + 1] = lineState->tileOffsets[i] + lineState->tileCounts[i];
			}
		}
		if (lineState)
		{
			lineState->tileIndices.assign(lineState->tileOffsets[tileCount], 0);
			std::vector<int> tileCursor = lineState->tileOffsets;
			for (int i = 0; i < lineState->lineCount; ++i)
			{
				if (!lineState->lineActive[i])
				{
					continue;
				}
				const LineDerived& ld = lineState->lineDerived[i];
				const float radius = fabsf(ld.segCenterX) + ld.halfLen + ld.halfThick + lineAAScaled;
				int minX = (int)((ld.centerX - radius) / tileSize);
				int maxX = (int)((ld.centerX + radius) / tileSize);
				int minY = (int)((ld.centerY - radius) / tileSize);
				int maxY = (int)((ld.centerY + radius) / tileSize);
				if (minX < 0) minX = 0;
				if (minY < 0) minY = 0;
				if (maxX >= tileCountX) maxX = tileCountX - 1;
				if (maxY >= tileCountY) maxY = tileCountY - 1;
				for (int ty = minY; ty <= maxY; ++ty)
				{
					for (int tx = minX; tx <= maxX; ++tx)
					{
						const int idx = ty * tileCountX + tx;
						lineState->tileIndices[tileCursor[idx]++] = i;
					}
				}
			}
		}

		// Main render loop - compiler optimization hints
#if defined(__clang__)
#pragma clang loop unroll_count(4)
#endif
		for (int y = 0; y < output->height; ++y, srcData += src->rowbytes, destData += dest->rowbytes)
		{
			for (int x = 0; x < output->width; ++x)
			{
				float v, u, luma, a;
				if (hideElement)
				{
					// Hide original element - start with transparent black
					v = 0.0f;
					u = 0.0f;
					luma = 0.0f;
					a = 0.0f;
				}
				else
				{
					v = ((const float*)srcData)[x * 4 + 0];
					u = ((const float*)srcData)[x * 4 + 1];
					luma = ((const float*)srcData)[x * 4 + 2];
					a = ((const float*)srcData)[x * 4 + 3];
				}

				float outV = v;
				float outU = u;
				float outY = luma;
				const float originalAlpha = a;  // Save original alpha for blend modes
				const float originalV = v;  // Save original color for Alpha XOR mode
				const float originalU = u;
				const float originalY = luma;
				// Accumulate front lines separately for blend mode 2
				float frontV = 0.0f;
				float frontU = 0.0f;
				float frontY = 0.0f;
				float frontA = 0.0f;
				float frontAppearAlpha = 1.0f;
				
				// Track line-only alpha for Alpha XOR mode (blend mode 3)
				float lineOnlyAlpha = 0.0f;

				const int tileX = x / tileSize;
				const int tileY = y / tileSize;
				const int tileIndex = tileY * tileCountX + tileX;
				const int start = lineState ? lineState->tileOffsets[tileIndex] : 0;
				const int count = lineState ? lineState->tileCounts[tileIndex] : 0;
				// Main line pass (with shadow drawn first for each line)
				for (int i = 0; i < count; ++i)
				{
					const LineDerived& ld = lineState->lineDerived[lineState->tileIndices[start + i]];
					const float depthAlpha = ld.depthAlpha;  // Use pre-computed

					// === STEP 1-4: Skip extremely tiny lines (< 1 pixel thick) ===
					if (ld.halfThick < 0.5f)
					{
						continue;  // Invisible line - skip all processing
					}

					// === STEP 2: Early skip optimization ===
					// Check if pixel is outside line bounding box (with shadow offset margin)
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
					
				// Draw shadow first (before the line) with motion blur
				if (shadowEnable)
				{
					const float sdx = (x + 0.5f) - (ld.centerX + shadowOffsetX);
					const float sdy = (y + 0.5f) - (ld.centerY + shadowOffsetY);
					float scoverage = 0.0f;

					// Motion blur for shadow (matching OpenCL/Metal implementation)
					if (motionBlurEnable && motionBlurSamples > 1)
					{
						const int samples = motionBlurSamples;
						const float shutterFraction = motionBlurStrength / 360.0f;
						const float pixelsPerFrame = lineTravel / lineLifetime;
						const float effectiveVelocity = pixelsPerFrame * ld.lineVelocity;
						const float blurRange = effectiveVelocity * shutterFraction;
						if (blurRange > 0.5f)
						{
							float saccumA = 0.0f;

							for (int s = 0; s < samples; ++s)
							{
								const float t = (float)s / fmaxf((float)(samples - 1), 1.0f);
								const float sampleOffset = blurRange * t;

								float spxSample = sdx * ld.cosA + sdy * ld.sinA;
								const float spySample = -sdx * ld.sinA + sdy * ld.cosA;
								spxSample -= (ld.segCenterX + sampleOffset);

								float sdistSample = (lineCap == 0)
									? SDFBox(spxSample, spySample, ld.halfLen, ld.halfThick)
									: SDFCapsule(spxSample, spySample, ld.halfLen, ld.halfThick);

								const float stailT = saturate((spxSample + ld.halfLen) * ld.invDenom);  // Use pre-computed
								const float stailFade = 1.0f + (stailT - 1.0f) * lineTailFade;
								const float saa = effectiveAA;
								float sampleCoverage = 0.0f;
								if (saa > 0.0f)
								{
									const float tt = saturate((sdistSample - saa) / (0.0f - saa));
									sampleCoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade;
								}

								saccumA += sampleCoverage;
							}

							scoverage = (saccumA / (float)samples) * shadowOpacity * depthAlpha;
						}
						else
						{
							// Blur too small, use single sample
							float spx = sdx * ld.cosA + sdy * ld.sinA;
							const float spy = -sdx * ld.sinA + sdy * ld.cosA;
							spx -= ld.segCenterX;

							float sdist = (lineCap == 0)
								? SDFBox(spx, spy, ld.halfLen, ld.halfThick)
								: SDFCapsule(spx, spy, ld.halfLen, ld.halfThick);

							const float stailT = saturate((spx + ld.halfLen) * ld.invDenom);  // Use pre-computed
							const float stailFade = 1.0f + (stailT - 1.0f) * lineTailFade;
							const float saa = effectiveAA;
							if (saa > 0.0f)
							{
								const float tt = saturate((sdist - saa) / (0.0f - saa));
								scoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * shadowOpacity * depthAlpha;
							}
						}
					}
					else
					{
						// No motion blur - single sample
						float spx = sdx * ld.cosA + sdy * ld.sinA;
						const float spy = -sdx * ld.sinA + sdy * ld.cosA;
						spx -= ld.segCenterX;

						float sdist = (lineCap == 0)
							? SDFBox(spx, spy, ld.halfLen, ld.halfThick)
							: SDFCapsule(spx, spy, ld.halfLen, ld.halfThick);

						const float stailT = saturate((spx + ld.halfLen) * ld.invDenom);  // Use pre-computed
						const float stailFade = 1.0f + (stailT - 1.0f) * lineTailFade;
						const float saa = effectiveAA;
						if (saa > 0.0f)
						{
							const float tt = saturate((sdist - saa) / (0.0f - saa));
							scoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * shadowOpacity * depthAlpha;
						}
					}
						if (scoverage > 0.001f)
						{
							float shadowBlend = scoverage;
							if (blendMode == 0)
							{
								// Back: keep shadow behind the original element
								shadowBlend = scoverage * (1.0f - originalAlpha);
							}
							else if (blendMode == 2 && ld.depth < 0.5f)
							{
								// Back portion of "Back and Front": keep shadow behind the original element
								shadowBlend = scoverage * (1.0f - originalAlpha);
							}

							// Shadow: premultiplied alpha compositing (v62 fix)
							float invShadow = 1.0f - shadowBlend;
							float outAlpha = shadowBlend + a * invShadow;
							if (outAlpha > 0.0f) {
								outY = (shadowY * shadowBlend + outY * a * invShadow) / outAlpha;
								outU = (shadowU * shadowBlend + outU * a * invShadow) / outAlpha;
								outV = (shadowV * shadowBlend + outV * a * invShadow) / outAlpha;
							}
							a = outAlpha;
						}
					}
					
					// Draw the main line with motion blur
					const float dx = (x + 0.5f) - ld.centerX;
					const float dy = (y + 0.5f) - ld.centerY;
					float coverage = 0.0f;

					// Motion blur sampling
					if (motionBlurEnable && motionBlurSamples > 1)
					{
						const int samples = motionBlurSamples;
						const float shutterFraction = motionBlurStrength / 360.0f;
						const float pixelsPerFrame = lineTravel / lineLifetime;
						const float effectiveVelocity = pixelsPerFrame * ld.lineVelocity;
						const float blurRange = effectiveVelocity * shutterFraction;
	
						if (blurRange > 0.5f)
						{
							// Multi-sample motion blur with uniform averaging
							float accumA = 0.0f;

							for (int s = 0; s < samples; ++s)
							{
								const float t = (float)s / fmaxf((float)(samples - 1), 1.0f);
								const float sampleOffset = blurRange * t;

								float pxSample = dx * ld.cosA + dy * ld.sinA;
								const float pySample = -dx * ld.sinA + dy * ld.cosA;
								pxSample -= (ld.segCenterX + sampleOffset);

								float distSample = (lineCap == 0)
									? SDFBox(pxSample, pySample, ld.halfLen, ld.halfThick)
									: SDFCapsule(pxSample, pySample, ld.halfLen, ld.halfThick);

								const float tailT = saturate((pxSample + ld.halfLen) * ld.invDenom);  // Use pre-computed
								const float tailFade = 1.0f + (tailT - 1.0f) * lineTailFade;
								const float aa = effectiveAA;
								float sampleCoverage = 0.0f;
								if (aa > 0.0f)
								{
									const float tt = saturate((distSample - aa) / (0.0f - aa));
									sampleCoverage = tt * tt * (3.0f - 2.0f * tt) * tailFade;
								}

								accumA += sampleCoverage;
							}

							coverage = (accumA / (float)samples) * ld.focusAlpha * depthAlpha;
						}
						else
						{
							// Blur too small, use single sample
							float px = dx * ld.cosA + dy * ld.sinA;
							const float py = -dx * ld.sinA + dy * ld.cosA;
							px -= ld.segCenterX;

							float dist = (lineCap == 0)
								? SDFBox(px, py, ld.halfLen, ld.halfThick)
								: SDFCapsule(px, py, ld.halfLen, ld.halfThick);

							const float aa = effectiveAA;
									const float tailT = saturate((px + ld.halfLen) * ld.invDenom);  // Use pre-computed
							const float tailFade = 1.0f + (tailT - 1.0f) * lineTailFade;
							if (aa > 0.0f)
							{
								const float tt = saturate((dist - aa) / (0.0f - aa));
								coverage = tt * tt * (3.0f - 2.0f * tt) * tailFade * ld.focusAlpha * depthAlpha;
							}
						}
					}
					else
					{
						// No motion blur (single sample)
						float px = dx * ld.cosA + dy * ld.sinA;
						const float py = -dx * ld.sinA + dy * ld.cosA;
						px -= ld.segCenterX;

						float dist = (lineCap == 0)
							? SDFBox(px, py, ld.halfLen, ld.halfThick)
							: SDFCapsule(px, py, ld.halfLen, ld.halfThick);

						const float aa = effectiveAA;
							const float tailT = saturate((px + ld.halfLen) * ld.invDenom);  // Use pre-computed
						const float tailFade = 1.0f + (tailT - 1.0f) * lineTailFade;
						if (aa > 0.0f)
						{
							const float tt = saturate((dist - aa) / (0.0f - aa));
							coverage = tt * tt * (3.0f - 2.0f * tt) * tailFade * ld.focusAlpha * depthAlpha;
						}
					}
					if (coverage > 0.001f)
					{
						const int ci = ld.colorIndex >= 0 && ld.colorIndex < 8 ? ld.colorIndex : 0;
						
						// Save alpha before this line's contribution
						const float prevAlpha = a;
						(void)prevAlpha;  // Unused, kept for future use
						
						// Apply blend mode
						if (blendMode == 0)  // Back (behind element)
						{
							float srcAlpha = coverage * (1.0f - originalAlpha);
							BlendPremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
							                   outY, outU, outV, a);
						}
						else if (blendMode == 1)  // Front (in front of element)
						{
							float srcAlpha = coverage;
							BlendPremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
							                   outY, outU, outV, a);
						}
						else if (blendMode == 2)  // Back and Front (split by per-line depth)
						{
							// Use stored depth value from line data (consistent across frames)
							if (ld.depth < 0.5f)
							{
								// Back mode (full)
								float srcAlpha = coverage * (1.0f - originalAlpha);
								BlendPremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
								                   outY, outU, outV, a);
							}
							else
							{
								// Front mode (full) -> accumulate separately, apply after loop
								// Use un-premultiplied accumulation (same as CUDA/OpenCL)
								float srcAlpha = coverage;
								BlendUnpremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
								                     frontY, frontU, frontV, frontA);
								frontAppearAlpha = std::min(frontAppearAlpha, ld.appearAlpha);
							}
						}
						else if (blendMode == 3)  // Alpha (XOR with original element only)
						{
							// Line-to-line blending: premultiplied compositing
							float srcAlpha = coverage;
							BlendPremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
							                   outY, outU, outV, a);
							// Track line-only alpha
							lineOnlyAlpha = std::max(lineOnlyAlpha, coverage * ld.appearAlpha);
						}
					}
				}

				// Apply XOR with original element AFTER all lines are drawn (blend mode 3)
				if (blendMode == 3 && originalAlpha > 0.0f)
				{
					// Alpha XOR mode: 
					// - Where lines exist: XOR alpha with original, show original color
					// - Where only original exists (no lines): show original element as-is
					
					if (lineOnlyAlpha > 0.0f)
					{
						// XOR alpha calculation: where overlap exists, alpha is reduced
						const float xorAlpha = saturate(originalAlpha + lineOnlyAlpha - (originalAlpha * lineOnlyAlpha * 2.0f));
						
						// For color: where original element exists, show original color
						outV = outV * (1.0f - originalAlpha) + originalV * originalAlpha;
						outU = outU * (1.0f - originalAlpha) + originalU * originalAlpha;
						outY = outY * (1.0f - originalAlpha) + originalY * originalAlpha;
						a = xorAlpha;
					}
					else
					{
						// No lines here, but original element exists - show original element
						outV = originalV;
						outU = originalU;
						outY = originalY;
						a = originalAlpha;
					}
				}

				// Apply front lines after back lines (blend mode 2)
				if (blendMode == 2 && frontA > 0.0f)
				{
				float prevAlpha = a;
				float invFrontA = 1.0f - frontA;
				float newAlpha = frontA + a * invFrontA;
				if (newAlpha > 0.0f) {
					outV = (frontV * frontA + outV * a * invFrontA) / newAlpha;
					outU = (frontU * frontA + outU * a * invFrontA) / newAlpha;
					outY = (frontY * frontA + outY * a * invFrontA) / newAlpha;
				}
				a = prevAlpha + (newAlpha - prevAlpha) * frontAppearAlpha;
			}

			// Draw spawn area preview (filled with inverted colors)
			if (showSpawnArea)
				{
					const float alphaCenterX = alphaBoundsMinX + alphaBoundsWidth * 0.5f;
					const float alphaCenterY = alphaBoundsMinY + alphaBoundsHeight * 0.5f;
					const float halfW = alphaBoundsWidth * spawnScaleX * 0.5f;
					const float halfH = alphaBoundsHeight * spawnScaleY * 0.5f;
					
					// Transform pixel position to rotated spawn space
					const float relX = (x + 0.5f) - alphaCenterX - userOriginOffsetX;
					const float relY = (y + 0.5f) - alphaCenterY - userOriginOffsetY;
					// Inverse rotate to check bounds
					const float localX = relX * spawnCos + relY * spawnSin;
					const float localY = -relX * spawnSin + relY * spawnCos;
					
					// Check if inside the spawn area (filled)
					if (fabsf(localX) <= halfW && fabsf(localY) <= halfH)
					{
						// Blend with spawn area color at 50%
						const float blendAlpha = 0.5f;
						const float baseV = (a <= 0.0f) ? spawnAreaV : outV;
						const float baseU = (a <= 0.0f) ? spawnAreaU : outU;
						const float baseY = (a <= 0.0f) ? spawnAreaY : outY;
						float blendedV = baseV + (spawnAreaV - baseV) * blendAlpha;
						float blendedU = baseU + (spawnAreaU - baseU) * blendAlpha;
						float blendedY = baseY + (spawnAreaY - baseY) * blendAlpha;
						outV = blendedV;
						outU = blendedU;
						outY = blendedY;
						a = std::max(a, blendAlpha);
					}
				}

				// DEBUG: Draw RED CIRCLE in top-left corner to indicate CPU is being used
				// Shape: Circle (CPU = fallback)
				{
					const float cx = 20.0f;  // Circle center X
					const float cy = 20.0f;  // Circle center Y
					const float radius = 15.0f;
					const float dx = (x + 0.5f) - cx;
					const float dy = (y + 0.5f) - cy;
					if (dx * dx + dy * dy <= radius * radius)
					{
					outV = 0.5f;   // V for red in YUV
					outU = 0.0f;   // U for red in YUV
					outY = 0.3f;   // Y for red in YUV (low luma)
					a = 1.0f;      // A = 1
				}
			}
			
			// Note: Premultiplied alpha compositing already produces correctly weighted colors
			// No additional premultiplication needed - output is straight alpha format
			// Premiere Pro handles the compositing correctly with straight alpha

			((float*)destData)[x * 4 + 0] = outV;
			((float*)destData)[x * 4 + 1] = outU;
			((float*)destData)[x * 4 + 2] = outY;
			((float*)destData)[x * 4 + 3] = a;
		}
	}
}	return PF_Err_NONE;
}

/*
**
*/
#if _WIN32 || defined(MSWindows)
#define DllExport   __declspec( dllexport )
#else
#define DllExport	__attribute__((visibility("default")))
#endif
extern "C" DllExport PF_Err EffectMain(
	PF_Cmd inCmd,
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* inOutput,
	void* extra)
{
	PF_Err err = PF_Err_NONE;
	switch (inCmd)
	{
	case PF_Cmd_GLOBAL_SETUP:
		err = GlobalSetup(in_data, out_data, params, inOutput);
		break;
	case PF_Cmd_GLOBAL_SETDOWN:
		err = GlobalSetdown(in_data, out_data, params, inOutput);
		break;
	case PF_Cmd_PARAMS_SETUP:
		err = ParamsSetup(in_data, out_data, params, inOutput);
		break;
	case PF_Cmd_USER_CHANGED_PARAM:
	{
		PF_UserChangedParamExtra* changedExtra = reinterpret_cast<PF_UserChangedParamExtra*>(extra);
		
		WriteLog("USER_CHANGED_PARAM: changedExtra=%p, param_index=%d", 
			changedExtra, changedExtra ? changedExtra->param_index : -1);
		
		// Effect Preset: apply preset parameters or defaults
		if (changedExtra && changedExtra->param_index == SDK_PROCAMP_EFFECT_PRESET)
		{
			const int presetValue = params[SDK_PROCAMP_EFFECT_PRESET]->u.pd.value;
			WriteLog("Effect Preset changed: presetValue=%d (1=Default, 2+=Preset[n-2])", presetValue);
			
			if (presetValue == 1)
			{
				WriteLog("Applying default effect params...");
				ApplyDefaultEffectParams(in_data, out_data, params);
				WriteLog("Default effect params applied.");
			}
			else if (presetValue > 1)
			{
				// Debounce: ignore double-fire within 200ms
				const uint32_t currentTime = GetCurrentTimeMs();
				const uint32_t lastTime = sLastPresetClickTime.load();
				WriteLog("Debounce check: currentTime=%u, lastTime=%u, diff=%u, threshold=%u",
					currentTime, lastTime, currentTime - lastTime, kPresetDebounceMs);
				if (currentTime - lastTime < kPresetDebounceMs)
				{
					WriteLog("DEBOUNCE: Ignoring duplicate event");
					break;  // Ignore duplicate event
				}
				sLastPresetClickTime.store(currentTime);
				WriteLog("Applying effect preset index=%d...", presetValue - 2);
				ApplyEffectPreset(in_data, out_data, params, presetValue - 2);
				WriteLog("Effect preset applied.");
			}
			else
			{
				WriteLog("presetValue <= 0, no action taken");
			}
		}
		
		// Color Preset: auto-switch Color Mode to "Preset"
		if (changedExtra && changedExtra->param_index == SDK_PROCAMP_COLOR_PRESET)
		{
			const int currentMode = params[SDK_PROCAMP_COLOR_MODE]->u.pd.value;
			if (currentMode != COLOR_MODE_PRESET)
			{
				params[SDK_PROCAMP_COLOR_MODE]->u.pd.value = COLOR_MODE_PRESET;
				params[SDK_PROCAMP_COLOR_MODE]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
				AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
				if (paramUtils.get())
				{
					paramUtils->PF_UpdateParamUI(in_data->effect_ref, SDK_PROCAMP_COLOR_MODE, params[SDK_PROCAMP_COLOR_MODE]);
				}
				out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
			}
		}
		
		// Single Color: auto-switch Color Mode to "Single"
		if (changedExtra && changedExtra->param_index == SDK_PROCAMP_LINE_COLOR)
		{
			const int currentMode = params[SDK_PROCAMP_COLOR_MODE]->u.pd.value;
			if (currentMode != COLOR_MODE_SINGLE)
			{
				params[SDK_PROCAMP_COLOR_MODE]->u.pd.value = COLOR_MODE_SINGLE;
				params[SDK_PROCAMP_COLOR_MODE]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
				AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
				if (paramUtils.get())
				{
					paramUtils->PF_UpdateParamUI(in_data->effect_ref, SDK_PROCAMP_COLOR_MODE, params[SDK_PROCAMP_COLOR_MODE]);
				}
				out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
			}
		}
		
		// Spawn Source: enable/disable Alpha Threshold based on selection
		if (changedExtra && changedExtra->param_index == SDK_PROCAMP_SPAWN_SOURCE)
		{
			// Update Alpha Threshold visibility
			UpdateAlphaThresholdVisibility(in_data, params);
			out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
		}
		
		// Linkage parameters: update visibility when linkage mode changes
		if (changedExtra && (
			changedExtra->param_index == SDK_PROCAMP_TRAVEL_LINKAGE ||
			changedExtra->param_index == SDK_PROCAMP_THICKNESS_LINKAGE ||
			changedExtra->param_index == SDK_PROCAMP_LENGTH_LINKAGE))
		{
			WriteLog("Linkage parameter changed: param_index=%d", changedExtra->param_index);
			UpdatePseudoGroupVisibility(in_data, params);
			out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
		}
		
		// Color Mode: update custom colors visibility
		if (changedExtra && changedExtra->param_index == SDK_PROCAMP_COLOR_MODE)
		{
			WriteLog("Color Mode parameter changed");
			UpdatePseudoGroupVisibility(in_data, params);
			out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
		}
		
		ApplyRectColorUi(in_data, out_data, params);
		SyncLineColorParams(params);
		HideLineColorParams(in_data);
		UpdateAlphaThresholdVisibility(in_data, params);
		UpdatePseudoGroupVisibility(in_data, params);
	}
		break;
	case PF_Cmd_UPDATE_PARAMS_UI:
	{
		// Update UI state for all parameters
		ApplyRectColorUi(in_data, out_data, params);
		SyncLineColorParams(params);
		HideLineColorParams(in_data);
		UpdateAlphaThresholdVisibility(in_data, params);
		UpdatePseudoGroupVisibility(in_data, params);
	}
		break;
	case PF_Cmd_RENDER:
		SyncLineColorParams(params);
		HideLineColorParams(in_data);
		err = Render(in_data, out_data, params, inOutput);
		break;
	}
	return err;
}
