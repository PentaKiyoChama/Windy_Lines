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
#include "SDK_ProcAmp_ParamOrder.h"
#include "SDK_ProcAmp_Version.h"
#include "AE_EffectSuites.h"
#include "PrSDKAESupport.h"
#include <algorithm>
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
std::unordered_map<csSDK_int64, ElementBounds> SharedClipData::elementBoundsMap;
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
	
	// Pre-computed tile boundaries (optimization: avoid redundant calculation)
	int tileMinX;
	int tileMinY;
	int tileMaxX;
	int tileMaxY;
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
	
	// Memory pool optimization: track reserved capacity
	int maxLineCapacity = 0;
	int maxTileCapacity = 0;
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
	
	// Get raw popup value (1-based), convert to 0-based: 1->0, 2->1
	const int rawValue = params[SDK_PROCAMP_SPAWN_SOURCE]->u.pd.value;
	const int spawnSource = (rawValue > 0) ? rawValue - 1 : 0;  // 0=Full, 1=Element
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
	
	// Linkage parameters
	updatePopup(SDK_PROCAMP_LENGTH_LINKAGE, preset.lengthLinkage);
	updateFloat(SDK_PROCAMP_LENGTH_LINKAGE_RATE, preset.lengthLinkageRate);
	updatePopup(SDK_PROCAMP_THICKNESS_LINKAGE, preset.thicknessLinkage);
	updateFloat(SDK_PROCAMP_THICKNESS_LINKAGE_RATE, preset.thicknessLinkageRate);
	updatePopup(SDK_PROCAMP_TRAVEL_LINKAGE, preset.travelLinkage);
	updateFloat(SDK_PROCAMP_TRAVEL_LINKAGE_RATE, preset.travelLinkageRate);
	
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
	
	// Iterate through parameters in display order defined by PARAM_DISPLAY_ORDER
	for (int i = 0; i < SDK_PROCAMP_NUM_PARAMS; ++i)
	{
		int paramId = PARAM_DISPLAY_ORDER[i];
		
		switch (paramId)
		{
			case SDK_PROCAMP_INPUT:
				// Input layer is handled automatically by After Effects
				break;
			
			case SDK_PROCAMP_EFFECT_PRESET:
				// Effect Preset (top-level, no group)
				AEFX_CLR_STRUCT(def);
				{
					std::string presetLabels = "デフォルト|";
					for (int j = 0; j < kEffectPresetCount; ++j)
					{
						presetLabels += kEffectPresets[j].name;
						if (j < kEffectPresetCount - 1)
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
				}
				break;
			
			case SDK_PROCAMP_LINE_SEED:
				// Random Seed
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
				break;
			
			case SDK_PROCAMP_LINE_COUNT:
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
				break;
			
			case SDK_PROCAMP_LINE_LIFETIME:
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
				break;
			
			case SDK_PROCAMP_LINE_INTERVAL:
				// Spawn Interval
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
				break;
			
			case SDK_PROCAMP_LINE_TRAVEL:
				// Travel Distance
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
				break;
			
			case SDK_PROCAMP_LINE_EASING:
				// Easing
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_EASING,
					28,
					LINE_EASING_DFLT,
					PM_EASING,
					SDK_PROCAMP_LINE_EASING);
				break;
			
			case SDK_PROCAMP_COLOR_MODE:
				// Color Mode
				AEFX_CLR_STRUCT(def);
				def.flags = PF_ParamFlag_SUPERVISE;
				PF_ADD_POPUP(
					P_COLOR_MODE,
					3,
					COLOR_MODE_DFLT,
					PM_COLOR_MODE,
					SDK_PROCAMP_COLOR_MODE);
				break;
			
			case SDK_PROCAMP_LINE_COLOR:
				// Single Color
				AEFX_CLR_STRUCT(def);
				def.flags = PF_ParamFlag_SUPERVISE;
				PF_ADD_COLOR(
					P_COLOR,
					LINE_COLOR_DFLT_R8,
					LINE_COLOR_DFLT_G8,
					LINE_COLOR_DFLT_B8,
					SDK_PROCAMP_LINE_COLOR);
				break;
			
			case SDK_PROCAMP_COLOR_PRESET:
				// Color Preset
				AEFX_CLR_STRUCT(def);
				def.flags = PF_ParamFlag_SUPERVISE;
				PF_ADD_POPUP(
					P_COLOR_PRESET,
					33,
					COLOR_PRESET_DFLT,
					PM_COLOR_PRESET,
					SDK_PROCAMP_COLOR_PRESET);
				break;
			
			case SDK_PROCAMP_CUSTOM_COLOR_1:
				// Custom Color 1
				AEFX_CLR_STRUCT(def);
				def.ui_flags = PF_PUI_ECW_SEPARATOR;
				PF_ADD_COLOR(P_CUSTOM_1, 255, 0, 0, SDK_PROCAMP_CUSTOM_COLOR_1);
				break;
			
			case SDK_PROCAMP_CUSTOM_COLOR_2:
				// Custom Color 2
				AEFX_CLR_STRUCT(def);
				PF_ADD_COLOR(P_CUSTOM_2, 255, 128, 0, SDK_PROCAMP_CUSTOM_COLOR_2);
				break;
			
			case SDK_PROCAMP_CUSTOM_COLOR_3:
				// Custom Color 3
				AEFX_CLR_STRUCT(def);
				PF_ADD_COLOR(P_CUSTOM_3, 255, 255, 0, SDK_PROCAMP_CUSTOM_COLOR_3);
				break;
			
			case SDK_PROCAMP_CUSTOM_COLOR_4:
				// Custom Color 4
				AEFX_CLR_STRUCT(def);
				PF_ADD_COLOR(P_CUSTOM_4, 0, 255, 0, SDK_PROCAMP_CUSTOM_COLOR_4);
				break;
			
			case SDK_PROCAMP_CUSTOM_COLOR_5:
				// Custom Color 5
				AEFX_CLR_STRUCT(def);
				PF_ADD_COLOR(P_CUSTOM_5, 0, 255, 255, SDK_PROCAMP_CUSTOM_COLOR_5);
				break;
			
			case SDK_PROCAMP_CUSTOM_COLOR_6:
				// Custom Color 6
				AEFX_CLR_STRUCT(def);
				PF_ADD_COLOR(P_CUSTOM_6, 0, 0, 255, SDK_PROCAMP_CUSTOM_COLOR_6);
				break;
			
			case SDK_PROCAMP_CUSTOM_COLOR_7:
				// Custom Color 7
				AEFX_CLR_STRUCT(def);
				PF_ADD_COLOR(P_CUSTOM_7, 128, 0, 255, SDK_PROCAMP_CUSTOM_COLOR_7);
				break;
			
			case SDK_PROCAMP_CUSTOM_COLOR_8:
				// Custom Color 8
				AEFX_CLR_STRUCT(def);
				PF_ADD_COLOR(P_CUSTOM_8, 255, 0, 255, SDK_PROCAMP_CUSTOM_COLOR_8);
				break;
			
			case SDK_PROCAMP_LINE_THICKNESS:
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
				break;
			
			case SDK_PROCAMP_LINE_LENGTH:
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
				break;
			
			case SDK_PROCAMP_LINE_ANGLE:
				// Line Angle
				AEFX_CLR_STRUCT(def);
				PF_ADD_ANGLE(
					P_ANGLE,
					0,
					SDK_PROCAMP_LINE_ANGLE);
				break;
			
			case SDK_PROCAMP_LINE_CAP:
				// Line Cap
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_LINE_CAP,
					2,
					LINE_CAP_DFLT,
					PM_LINE_CAP,
					SDK_PROCAMP_LINE_CAP);
				break;
			
			case SDK_PROCAMP_LINE_TAIL_FADE:
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
				break;
			
			case SDK_PROCAMP_POSITION_HEADER:
				// Line Origin Topic Start
				AEFX_CLR_STRUCT(def);
				PF_ADD_TOPIC(P_POSITION_HEADER, SDK_PROCAMP_POSITION_HEADER);
				break;
			
			case SDK_PROCAMP_SPAWN_SOURCE:
				// Spawn Source
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_SPAWN_SOURCE,
					2,
					SPAWN_SOURCE_DFLT,
					P_SPAWN_SOURCE_CHOICES,
					SDK_PROCAMP_SPAWN_SOURCE);
				break;
			
			case SDK_PROCAMP_LINE_ALPHA_THRESH:
				// Alpha Threshold
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
				break;
			
			case SDK_PROCAMP_LINE_ORIGIN_MODE:
				// Wind Origin Mode
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_ORIGIN_MODE,
					3,
					LINE_ORIGIN_MODE_DFLT,
					PM_ORIGIN_MODE,
					SDK_PROCAMP_LINE_ORIGIN_MODE);
				break;
			
			case SDK_PROCAMP_ANIM_PATTERN:
				// Animation Pattern (Direction)
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_ANIM_PATTERN,
					3,
					ANIM_PATTERN_DFLT,
					PM_ANIM_PATTERN,
					SDK_PROCAMP_ANIM_PATTERN);
				break;
			
			case SDK_PROCAMP_LINE_START_TIME:
				// Start Time
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
				break;
			
			case SDK_PROCAMP_LINE_DURATION:
				// Duration
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
				break;
			
			case SDK_PROCAMP_LINE_DEPTH_STRENGTH:
				// Depth Strength
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
				break;
			
			case SDK_PROCAMP_CENTER_GAP:
				// Center Gap
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
				break;
			
			case SDK_PROCAMP_ORIGIN_OFFSET_X:
				// Origin Offset X
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
				break;
			
			case SDK_PROCAMP_ORIGIN_OFFSET_Y:
				// Origin Offset Y
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
				break;
			
			case SDK_PROCAMP_LINE_SPAWN_SCALE_X:
				// Spawn Scale X
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
				break;
			
			case SDK_PROCAMP_LINE_SPAWN_SCALE_Y:
				// Spawn Scale Y
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
				break;
			
			case SDK_PROCAMP_LINE_SPAWN_ROTATION:
				// Spawn Rotation
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
				break;
			
			case SDK_PROCAMP_LINE_SHOW_SPAWN_AREA:
				// Show Spawn Area
				AEFX_CLR_STRUCT(def);
				PF_ADD_CHECKBOX(P_SHOW_SPAWN, "", SHOW_SPAWN_AREA_DFLT, 0, SDK_PROCAMP_LINE_SHOW_SPAWN_AREA);
				break;
			
			case SDK_PROCAMP_LINE_SPAWN_AREA_COLOR:
				// Spawn Area Color
				AEFX_CLR_STRUCT(def);
				PF_ADD_COLOR(P_SPAWN_COLOR, 128, 128, 255, SDK_PROCAMP_LINE_SPAWN_AREA_COLOR);
				break;
			
			case SDK_PROCAMP_POSITION_TOPIC_END:
				// Line Origin Topic End
				AEFX_CLR_STRUCT(def);
				PF_END_TOPIC(SDK_PROCAMP_POSITION_TOPIC_END);
				break;
			
			case SDK_PROCAMP_SHADOW_HEADER:
				// Shadow Topic Start
				AEFX_CLR_STRUCT(def);
				PF_ADD_TOPIC(P_SHADOW, SDK_PROCAMP_SHADOW_HEADER);
				break;
			
			case SDK_PROCAMP_SHADOW_ENABLE:
				// Shadow Enable
				AEFX_CLR_STRUCT(def);
				PF_ADD_CHECKBOX(P_SHADOW_ENABLE, "", SHADOW_ENABLE_DFLT, 0, SDK_PROCAMP_SHADOW_ENABLE);
				break;
			
			case SDK_PROCAMP_SHADOW_COLOR:
				// Shadow Color
				AEFX_CLR_STRUCT(def);
				def.u.cd.value.red = static_cast<A_u_short>(SHADOW_COLOR_R_DFLT * 65535);
				def.u.cd.value.green = static_cast<A_u_short>(SHADOW_COLOR_G_DFLT * 65535);
				def.u.cd.value.blue = static_cast<A_u_short>(SHADOW_COLOR_B_DFLT * 65535);
				def.u.cd.value.alpha = 65535;
				def.u.cd.dephault = def.u.cd.value;
				PF_ADD_COLOR(P_SHADOW_COLOR, def.u.cd.value.red, def.u.cd.value.green, def.u.cd.value.blue, SDK_PROCAMP_SHADOW_COLOR);
				break;
			
			case SDK_PROCAMP_SHADOW_OFFSET_X:
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
				break;
			
			case SDK_PROCAMP_SHADOW_OFFSET_Y:
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
				break;
			
			case SDK_PROCAMP_SHADOW_OPACITY:
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
				break;
			
			case SDK_PROCAMP_SHADOW_TOPIC_END:
				// Shadow Topic End
				AEFX_CLR_STRUCT(def);
				PF_END_TOPIC(SDK_PROCAMP_SHADOW_TOPIC_END);
				break;
			
			case SDK_PROCAMP_MOTION_BLUR_HEADER:
				// Motion Blur Topic Start
				AEFX_CLR_STRUCT(def);
				PF_ADD_TOPIC(P_MOTION_BLUR, SDK_PROCAMP_MOTION_BLUR_HEADER);
				break;
			
			case SDK_PROCAMP_MOTION_BLUR_ENABLE:
				// Motion Blur Enable
				AEFX_CLR_STRUCT(def);
				PF_ADD_CHECKBOX(P_MOTION_BLUR, "", MOTION_BLUR_ENABLE_DFLT, 0, SDK_PROCAMP_MOTION_BLUR_ENABLE);
				break;
			
			case SDK_PROCAMP_MOTION_BLUR_SAMPLES:
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
				break;
			
			case SDK_PROCAMP_MOTION_BLUR_STRENGTH:
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
				break;
			
			case SDK_PROCAMP_MOTION_BLUR_TOPIC_END:
				// Motion Blur Topic End
				AEFX_CLR_STRUCT(def);
				PF_END_TOPIC(SDK_PROCAMP_MOTION_BLUR_TOPIC_END);
				break;
			
			case SDK_PROCAMP_ADVANCED_HEADER:
				// Advanced Topic Start
				AEFX_CLR_STRUCT(def);
				PF_ADD_TOPIC(P_ADVANCED_HEADER, SDK_PROCAMP_ADVANCED_HEADER);
				break;
			
			case SDK_PROCAMP_LINE_AA:
				// Anti-Aliasing
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
				break;
			
			case SDK_PROCAMP_HIDE_ELEMENT:
				// Hide Element (lines only)
				AEFX_CLR_STRUCT(def);
				PF_ADD_CHECKBOX(P_HIDE_ELEMENT, "", HIDE_ELEMENT_DFLT, 0, SDK_PROCAMP_HIDE_ELEMENT);
				break;
			
			case SDK_PROCAMP_BLEND_MODE:
				// Blend Mode
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_BLEND_MODE,
					4,
					BLEND_MODE_DFLT,
					PM_BLEND_MODE,
					SDK_PROCAMP_BLEND_MODE);
				break;
			
			case SDK_PROCAMP_ADVANCED_TOPIC_END:
				// Advanced Topic End
				AEFX_CLR_STRUCT(def);
				PF_END_TOPIC(SDK_PROCAMP_ADVANCED_TOPIC_END);
				break;
			
			case SDK_PROCAMP_LINKAGE_HEADER:
				// Linkage Topic Start
				AEFX_CLR_STRUCT(def);
				PF_ADD_TOPIC(P_LINKAGE_HEADER, SDK_PROCAMP_LINKAGE_HEADER);
				break;
			
			case SDK_PROCAMP_LENGTH_LINKAGE:
				// Length Linkage
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_LENGTH_LINKAGE,
					3,
					LINKAGE_MODE_DFLT,
					PM_LINKAGE_MODE,
					SDK_PROCAMP_LENGTH_LINKAGE);
				break;
			
			case SDK_PROCAMP_LENGTH_LINKAGE_RATE:
				// Length Linkage Rate
				AEFX_CLR_STRUCT(def);
				PF_ADD_FLOAT_SLIDERX(
					P_LENGTH_LINKAGE_RATE,
					LINKAGE_RATE_MIN_VALUE,
					LINKAGE_RATE_MAX_VALUE,
					LINKAGE_RATE_MIN_SLIDER,
					LINKAGE_RATE_MAX_SLIDER,
					LINKAGE_RATE_DFLT,
					PF_Precision_TENTHS,
					0,
					0,
					SDK_PROCAMP_LENGTH_LINKAGE_RATE);
				break;
			
			case SDK_PROCAMP_THICKNESS_LINKAGE:
				// Thickness Linkage
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_THICKNESS_LINKAGE,
					3,
					LINKAGE_MODE_DFLT,
					PM_LINKAGE_MODE,
					SDK_PROCAMP_THICKNESS_LINKAGE);
				break;
			
			case SDK_PROCAMP_THICKNESS_LINKAGE_RATE:
				// Thickness Linkage Rate
				AEFX_CLR_STRUCT(def);
				PF_ADD_FLOAT_SLIDERX(
					P_THICKNESS_LINKAGE_RATE,
					LINKAGE_RATE_MIN_VALUE,
					LINKAGE_RATE_MAX_VALUE,
					LINKAGE_RATE_MIN_SLIDER,
					LINKAGE_RATE_MAX_SLIDER,
					LINKAGE_RATE_DFLT,
					PF_Precision_TENTHS,
					0,
					0,
					SDK_PROCAMP_THICKNESS_LINKAGE_RATE);
				break;
			
			case SDK_PROCAMP_TRAVEL_LINKAGE:
				// Travel Distance Linkage
				AEFX_CLR_STRUCT(def);
				PF_ADD_POPUP(
					P_TRAVEL_LINKAGE,
					3,
					LINKAGE_MODE_DFLT,
					PM_LINKAGE_MODE,
					SDK_PROCAMP_TRAVEL_LINKAGE);
				break;
			
			case SDK_PROCAMP_TRAVEL_LINKAGE_RATE:
				// Travel Distance Linkage Rate
				AEFX_CLR_STRUCT(def);
				PF_ADD_FLOAT_SLIDERX(
					P_TRAVEL_LINKAGE_RATE,
					LINKAGE_RATE_MIN_VALUE,
					LINKAGE_RATE_MAX_VALUE,
					LINKAGE_RATE_MIN_SLIDER,
					LINKAGE_RATE_MAX_SLIDER,
					LINKAGE_RATE_DFLT,
					PF_Precision_TENTHS,
					0,
					0,
					SDK_PROCAMP_TRAVEL_LINKAGE_RATE);
				break;
			
			case SDK_PROCAMP_LINKAGE_TOPIC_END:
				// Linkage Topic End
				AEFX_CLR_STRUCT(def);
				PF_END_TOPIC(SDK_PROCAMP_LINKAGE_TOPIC_END);
				break;
			
			case SDK_PROCAMP_LINE_ALLOW_MIDPLAY:
				// Hidden Parameter: Allow Midplay
				AEFX_CLR_STRUCT(def);
				def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
				def.ui_flags = PF_PUI_INVISIBLE;
				PF_ADD_CHECKBOX("", "", LINE_ALLOW_MIDPLAY_DFLT, 0, SDK_PROCAMP_LINE_ALLOW_MIDPLAY);
				break;
			
			case SDK_PROCAMP_LINE_COLOR_R:
				// Hidden Parameter: Line Color R
				AEFX_CLR_STRUCT(def);
				def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
				def.ui_flags = PF_PUI_INVISIBLE;
				PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
					LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
					PF_Precision_TENTHS, 0, 0, SDK_PROCAMP_LINE_COLOR_R);
				break;
			
			case SDK_PROCAMP_LINE_COLOR_G:
				// Hidden Parameter: Line Color G
				AEFX_CLR_STRUCT(def);
				def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
				def.ui_flags = PF_PUI_INVISIBLE;
				PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
					LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
					PF_Precision_TENTHS, 0, 0, SDK_PROCAMP_LINE_COLOR_G);
				break;
			
			case SDK_PROCAMP_LINE_COLOR_B:
				// Hidden Parameter: Line Color B
				AEFX_CLR_STRUCT(def);
				def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
				def.ui_flags = PF_PUI_INVISIBLE;
				PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
					LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
					PF_Precision_TENTHS, 0, 0, SDK_PROCAMP_LINE_COLOR_B);
				break;
		}
	}
	
	out_data->num_params = SDK_PROCAMP_NUM_PARAMS;
	return PF_Err_NONE;
}