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


#ifndef OST_WINDYLINES_H
#define OST_WINDYLINES_H

#include "AEConfig.h"

#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "A.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "Param_Utils.h"

#include "PrSDKAESupport.h"

#include <math.h>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <ctime>

// ========== GPU RENDERING CONTROL ==========
// Set to 1 to enable GPU rendering (CUDA/OpenCL/DirectX)
// Set to 0 to force CPU-only rendering (for testing CPU path)
#define ENABLE_GPU_RENDERING 1

// ========== DEBUG RENDER MARKERS ==========
// Set to 1 to enable visual markers in top-left corner (GPU/CPU indicator)
// Set to 0 to disable completely (zero performance impact)
#define ENABLE_DEBUG_RENDER_MARKERS 0
// ========== DEBUG LOGGING (Common) ==========
#ifdef _DEBUG
static std::mutex sLogMutex;
static void WriteLog(const char* format, ...)
{
	std::lock_guard<std::mutex> lock(sLogMutex);
	
	// Platform-specific log paths
#ifdef _WIN32
	const char* paths[] = {
		"C:\\Temp\\OST_WindyLines_Log.txt",
		"C:\\Users\\Owner\\Desktop\\OST_WindyLines_Log.txt"
	};
#else
	// Mac/Unix paths
	const char* pathTemplates[] = {
		"/tmp/OST_WindyLines_Log.txt",
		"~/Desktop/OST_WindyLines_Log.txt"
	};
	// Expand ~ for home directory on Unix
	char expandedPath[512] = "";
	const char* paths[2];
	paths[0] = pathTemplates[0];
	
	if (pathTemplates[1][0] == '~') {
		const char* home = getenv("HOME");
		if (home) {
			snprintf(expandedPath, sizeof(expandedPath), "%s%s", home, pathTemplates[1] + 1);
			paths[1] = expandedPath;
		} else {
			paths[1] = pathTemplates[1];  // Fallback to unexpanded path
		}
	} else {
		paths[1] = pathTemplates[1];
	}
#endif
	
	FILE* fp = nullptr;
	for (int i = 0; i < 2; ++i)
	{
#ifdef _WIN32
		errno_t err = fopen_s(&fp, paths[i], "a");
		if (err == 0 && fp)
#else
		fp = fopen(paths[i], "a");
		if (fp)
#endif
		{
			// Time stamp
			time_t now = time(nullptr);
			char timeStr[64];
			struct tm timeInfo;
#ifdef _WIN32
			localtime_s(&timeInfo, &now);
#else
			localtime_r(&now, &timeInfo);
#endif
			strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &timeInfo);
			fprintf(fp, "[%s] ", timeStr);
			
			// Actual log message
			va_list args;
			va_start(args, format);
			vfprintf(fp, format, args);
			va_end(args);
			fprintf(fp, "\n");
			fflush(fp);
			fclose(fp);
			break;
		}
	}
}
#endif // _DEBUG

// Release builds disable logging for performance and security
#ifdef _DEBUG
    #define DebugLog WriteLog
#else
    #define DebugLog(...)  // No-op in release builds
#endif
// ===========================================


/*
** Defaults & ranges (edit here to change default values)
*/
enum
{
	OST_WINDYLINES_INPUT = 0,
	
	// Effect Preset (top-level)
	OST_WINDYLINES_EFFECT_PRESET,                // 1. Effect preset selector
	OST_WINDYLINES_LINE_SEED,                    // 2. Random seed
	
	// ▼ Basic Settings
	OST_WINDYLINES_LINE_COUNT,                   // 3. Number of lines
	OST_WINDYLINES_LINE_LIFETIME,                // 4. Line lifetime (frames)
	OST_WINDYLINES_LINE_INTERVAL,                // 5. Spawn interval (frames)
	
	// ▼ Color Settings
	OST_WINDYLINES_COLOR_MODE,                   // 6. Single/Preset/Custom
	OST_WINDYLINES_COLOR_PRESET,                 // 8. Preset selection popup
	OST_WINDYLINES_LINE_COLOR,                   // 9. Single color picker
	OST_WINDYLINES_CUSTOM_COLOR_1,               // 10-17. Custom colors 1-8
	OST_WINDYLINES_CUSTOM_COLOR_2,
	OST_WINDYLINES_CUSTOM_COLOR_3,
	OST_WINDYLINES_CUSTOM_COLOR_4,
	OST_WINDYLINES_CUSTOM_COLOR_5,
	OST_WINDYLINES_CUSTOM_COLOR_6,
	OST_WINDYLINES_CUSTOM_COLOR_7,
	OST_WINDYLINES_CUSTOM_COLOR_8,
	OST_WINDYLINES_LINE_EASING,                  // Easing function
	OST_WINDYLINES_TRAVEL_LINKAGE,               // Travel distance linkage (Off/Width/Height)
	OST_WINDYLINES_TRAVEL_LINKAGE_RATE,          // 19. Travel distance linkage rate (%)
	OST_WINDYLINES_LINE_TRAVEL,                  // 20. Travel distance (px)
	
	// ▼ Appearance
	OST_WINDYLINES_THICKNESS_LINKAGE,            // 21. Thickness linkage (Off/Width/Height)
	OST_WINDYLINES_THICKNESS_LINKAGE_RATE,       // 22. Thickness linkage rate (%)
	OST_WINDYLINES_LINE_THICKNESS,               // 23. Line thickness (px)
	OST_WINDYLINES_LENGTH_LINKAGE,               // 24. Length linkage (Off/Width/Height)
	OST_WINDYLINES_LENGTH_LINKAGE_RATE,          // 25. Length linkage rate (%)
	OST_WINDYLINES_LINE_LENGTH,                  // 26. Line length (px)
	OST_WINDYLINES_LINE_ANGLE,                   // 27. Line angle (degrees)
	OST_WINDYLINES_LINE_CAP,                     // 28. Line cap style
	OST_WINDYLINES_LINE_TAIL_FADE,               // 29. Tail fade amount
	
	// ▼ Position & Spawn - Line Origin
	OST_WINDYLINES_POSITION_HEADER,              // 30. Line origin topic start
	OST_WINDYLINES_SPAWN_SOURCE,                 // 31. Spawn source (Full Frame / Element Bounds)
	OST_WINDYLINES_LINE_ALPHA_THRESH,            // 32. Alpha threshold
	OST_WINDYLINES_LINE_ORIGIN_MODE,             // 33. Wind origin mode
	OST_WINDYLINES_ANIM_PATTERN,                 // 34. Animation pattern (direction)
	OST_WINDYLINES_LINE_START_TIME,              // 35. Start time (frames)
	OST_WINDYLINES_LINE_DURATION,                // 36. Duration (frames)
	OST_WINDYLINES_LINE_DEPTH_STRENGTH,          // 37. Depth strength
	OST_WINDYLINES_CENTER_GAP,                   // 38. Center gap
	OST_WINDYLINES_ORIGIN_OFFSET_X,              // 39. Origin Offset X (px)
	OST_WINDYLINES_ORIGIN_OFFSET_Y,              // 40. Origin Offset Y (px)
	OST_WINDYLINES_LINE_SPAWN_SCALE_X,           // 41. Spawn area scale X (%)
	OST_WINDYLINES_LINE_SPAWN_SCALE_Y,           // 42. Spawn area scale Y (%)
	OST_WINDYLINES_LINE_SPAWN_ROTATION,          // 43. Spawn area rotation (degrees)
	OST_WINDYLINES_LINE_SHOW_SPAWN_AREA,         // 44. Show spawn area preview
	OST_WINDYLINES_LINE_SPAWN_AREA_COLOR,        // 45. Spawn area color
	OST_WINDYLINES_POSITION_TOPIC_END,           // 46. Line origin topic end
	
	// ▼ Shadow - Topic group
	OST_WINDYLINES_SHADOW_HEADER,                // 47. Shadow topic start
	OST_WINDYLINES_SHADOW_ENABLE,                // 48. Shadow on/off
	OST_WINDYLINES_SHADOW_COLOR,                 // 49. Shadow color
	OST_WINDYLINES_SHADOW_OFFSET_X,              // 50. Shadow offset X (px)
	OST_WINDYLINES_SHADOW_OFFSET_Y,              // 51. Shadow offset Y (px)
	OST_WINDYLINES_SHADOW_OPACITY,               // 52. Shadow opacity (0-1)
	OST_WINDYLINES_SHADOW_TOPIC_END,             // 53. Shadow topic end
	
	// ▼ Motion Blur - Topic group
	OST_WINDYLINES_MOTION_BLUR_HEADER,           // 54. Motion blur topic start
	OST_WINDYLINES_MOTION_BLUR_ENABLE,           // 55. Motion blur on/off
	OST_WINDYLINES_MOTION_BLUR_SAMPLES,          // 56. Motion blur quality (samples)
	OST_WINDYLINES_MOTION_BLUR_STRENGTH,         // 57. Motion blur strength
	OST_WINDYLINES_MOTION_BLUR_TOPIC_END,        // 58. Motion blur topic end
	
	// ▼ Advanced - Topic group
	OST_WINDYLINES_ADVANCED_HEADER,              // 59. Advanced topic start
	OST_WINDYLINES_LINE_AA,                      // 60. Anti-aliasing
	OST_WINDYLINES_HIDE_ELEMENT,                 // 61. Hide original element (lines only)
	OST_WINDYLINES_BLEND_MODE,                   // 62. Blend mode with element
	OST_WINDYLINES_ADVANCED_TOPIC_END,           // 63. Advanced topic end
	
	// Hidden params (for backwards compatibility)
	OST_WINDYLINES_LINE_ALLOW_MIDPLAY,           // 64. (hidden)
	OST_WINDYLINES_LINE_COLOR_R,                 // 65. (hidden)
	OST_WINDYLINES_LINE_COLOR_G,                 // 66. (hidden)
	OST_WINDYLINES_LINE_COLOR_B,                 // 67. (hidden)
	
	// ▼ License
	OST_WINDYLINES_LICENSE_HEADER,               // 68. License topic start
	OST_WINDYLINES_LICENSE_STATUS,               // 69. License status / activate popup
	OST_WINDYLINES_LICENSE_TOPIC_END,            // 70. License topic end
	
	OST_WINDYLINES_NUM_PARAMS
};

// Color Mode enum
enum ColorMode
{
	COLOR_MODE_SINGLE = 1,   // Single color for all lines
	COLOR_MODE_PRESET = 2,   // Use preset palette
	COLOR_MODE_CUSTOM = 3    // User-defined 8 colors
};

// Blend Mode enum
enum BlendMode
{
	BLEND_MODE_BACK = 1,         // Draw behind element
	BLEND_MODE_FRONT = 2,        // Draw in front of element
	BLEND_MODE_BACK_FRONT = 3,   // Back and Front based on depth strength
	BLEND_MODE_ALPHA = 4         // Alpha transparency where overlapping
};

// Linkage Mode enum
enum LinkageMode
{
	LINKAGE_MODE_OFF = 0,        // No linkage
	LINKAGE_MODE_WIDTH = 1,      // Link to element width
	LINKAGE_MODE_HEIGHT = 2      // Link to element height
};

// Spawn Source enum (0-based, matches NormalizePopupParam output)
// Menu: "画面全体|要素範囲"
enum SpawnSource
{
	SPAWN_SOURCE_FULL_FRAME = 0,      // Entire frame (menu index 0)
	SPAWN_SOURCE_ELEMENT_BOUNDS = 1   // Element alpha bounds (menu index 1)
};

// Effect preset structure
struct EffectPreset
{
	const char* name;
	// Basic
	int count;
	float lifetime;
	float travel;
	// Appearance
	float thickness;
	float length;
	float angle;        // degrees
	float tailFade;
	float aa;
	// Position & Spawn
	int originMode;     // 1=Center, 2=Forward, 3=Backward
	float spawnScaleX;  // %
	float spawnScaleY;  // %
	float originOffsetX;
	float originOffsetY;
	float interval;
	// Animation
	int animPattern;    // 1=Simple, 2=HalfReverse, 3=Split
	float centerGap;    // 0.0-0.5 (center empty zone)
	int easing;         // 0-based index
	float startTime;
	float duration;     // 0=infinite
	// Advanced
	int blendMode;      // 1=Back, 2=Front, 3=BackAndFront, 4=Alpha
	float depthStrength;
	// New fields
	int lineCap;          // 0=Flat, 1=Round
	int unifiedPresetIndex;  // 0=単色, 1=カスタム, 2=separator, 3+=color presets (0-based)
	int spawnSource;      // 1=Full Frame, 2=Element
	bool hideElement;     // Hide original element
	// Linkage fields
	int lengthLinkage;    // 0=Off, 1=Width, 2=Height
	float lengthLinkageRate;    // %
	int thicknessLinkage;  // 0=Off, 1=Width, 2=Height
	float thicknessLinkageRate; // %
	int travelLinkage;     // 0=Off, 1=Width, 2=Height
	float travelLinkageRate;    // %
};

// Preset data array - auto-generated from presets.tsv
// Run preset_converter.py to regenerate OST_WindyLines_Presets.h
#include "OST_WindyLines_Presets.h"

/*
**
*/
#define	LINE_COLOR_DFLT_R8			255
#define	LINE_COLOR_DFLT_G8			255
#define	LINE_COLOR_DFLT_B8			255

#define	LINE_THICKNESS_MIN_VALUE	1
#define	LINE_THICKNESS_MAX_VALUE	200
#define	LINE_THICKNESS_MIN_SLIDER	1
#define	LINE_THICKNESS_MAX_SLIDER	200
#define	LINE_THICKNESS_DFLT			8

#define	LINE_LENGTH_MIN_VALUE		0
#define	LINE_LENGTH_MAX_VALUE		2000
#define	LINE_LENGTH_MIN_SLIDER		0
#define	LINE_LENGTH_MAX_SLIDER		2000
#define	LINE_LENGTH_DFLT			200

#define	LINE_AA_MIN_VALUE			0
#define	LINE_AA_MAX_VALUE			5
#define	LINE_AA_MIN_SLIDER			0
#define	LINE_AA_MAX_SLIDER			5
#define	LINE_AA_DFLT				0

#define	LINE_ALPHA_THRESH_MIN_VALUE	0
#define	LINE_ALPHA_THRESH_MAX_VALUE	1
#define	LINE_ALPHA_THRESH_MIN_SLIDER	0
#define	LINE_ALPHA_THRESH_MAX_SLIDER	1
#define	LINE_ALPHA_THRESH_DFLT			0.02f

#define	LINE_ORIGIN_MODE_DFLT			1


#define	LINE_COUNT_MIN_VALUE		1
#define	LINE_COUNT_MAX_VALUE		5000
#define	LINE_COUNT_MIN_SLIDER		1
#define	LINE_COUNT_MAX_SLIDER		5000
#define	LINE_COUNT_DFLT				100

#define	LINE_LIFETIME_MIN_VALUE		1
#define	LINE_LIFETIME_MAX_VALUE		600
#define	LINE_LIFETIME_MIN_SLIDER	1
#define	LINE_LIFETIME_MAX_SLIDER	600
#define	LINE_LIFETIME_DFLT			20

#define	LINE_INTERVAL_MIN_VALUE		0
#define	LINE_INTERVAL_MAX_VALUE		600
#define	LINE_INTERVAL_MIN_SLIDER	0
#define	LINE_INTERVAL_MAX_SLIDER	600
#define	LINE_INTERVAL_DFLT			180

#define	LINE_SEED_MIN_VALUE			0
#define	LINE_SEED_MAX_VALUE			1000000
#define	LINE_SEED_MIN_SLIDER		0
#define	LINE_SEED_MAX_SLIDER		1000000
#define	LINE_SEED_DFLT				1

#define	LINE_EASING_DFLT			2

#define	LINE_TRAVEL_MIN_VALUE		0
#define	LINE_TRAVEL_MAX_VALUE		5000
#define	LINE_TRAVEL_MIN_SLIDER		0
#define	LINE_TRAVEL_MAX_SLIDER		2000
#define	LINE_TRAVEL_DFLT			300

#define	LINE_TAIL_FADE_MIN_VALUE	0
#define	LINE_TAIL_FADE_MAX_VALUE	1
#define	LINE_TAIL_FADE_MIN_SLIDER	0
#define	LINE_TAIL_FADE_MAX_SLIDER	1
#define	LINE_TAIL_FADE_DFLT			0

#define	LINE_DEPTH_MIN_VALUE		0
#define	LINE_DEPTH_MAX_VALUE		10
#define	LINE_DEPTH_MIN_SLIDER		0
#define	LINE_DEPTH_MAX_SLIDER		10
#define	LINE_DEPTH_DFLT				3

#define	LINE_ALLOW_MIDPLAY_DFLT		1

#define	LINE_COLOR_CH_MIN_VALUE		0
#define	LINE_COLOR_CH_MAX_VALUE		1
#define	LINE_COLOR_CH_MIN_SLIDER	0
#define	LINE_COLOR_CH_MAX_SLIDER	1
#define	LINE_COLOR_CH_DFLT			1

// Default values for color parameters
#define COLOR_MODE_DFLT         1   // Deprecated: now unified with COLOR_PRESET
#define COLOR_PRESET_DFLT       5   // Rainbow (UI values: 1=単色, 2=Sep, 3=カスタム, 4=Sep, 5=Rainbow, ...)

// Line Cap default
#define LINE_CAP_DFLT           1   // Round (1=Flat, 2=Round)

// Spawn Source default
#define SPAWN_SOURCE_DFLT       2   // Full Frame (1=Full Frame, 2=Element)

// Shadow parameters
#define SHADOW_ENABLE_DFLT      0       // Off by default
#define SHADOW_COLOR_R_DFLT     0.0     // Black shadow by default
#define SHADOW_COLOR_G_DFLT     0.0
#define SHADOW_COLOR_B_DFLT     0.0
#define SHADOW_OFFSET_X_MIN     -100
#define SHADOW_OFFSET_X_MAX     100
#define SHADOW_OFFSET_X_DFLT    5
#define SHADOW_OFFSET_Y_MIN     -100
#define SHADOW_OFFSET_Y_MAX     100
#define SHADOW_OFFSET_Y_DFLT    5
#define SHADOW_OPACITY_MIN      0.0
#define SHADOW_OPACITY_MAX      1.0
#define SHADOW_OPACITY_DFLT     0.5

// Hide Element (show lines only)
#define HIDE_ELEMENT_DFLT       0   // Off by default (show element + lines)

// Default value for blend mode
#define BLEND_MODE_DFLT         4   // Front&back

// Line spawn area scale (%)
#define	LINE_SPAWN_SCALE_X_MIN_VALUE	0
#define	LINE_SPAWN_SCALE_X_MAX_VALUE	200
#define	LINE_SPAWN_SCALE_X_MIN_SLIDER	0
#define	LINE_SPAWN_SCALE_X_MAX_SLIDER	200
#define	LINE_SPAWN_SCALE_X_DFLT			75  // 100% = current behavior

#define	LINE_SPAWN_SCALE_Y_MIN_VALUE	0
#define	LINE_SPAWN_SCALE_Y_MAX_VALUE	200
#define	LINE_SPAWN_SCALE_Y_MIN_SLIDER	0
#define	LINE_SPAWN_SCALE_Y_MAX_SLIDER	200
#define	LINE_SPAWN_SCALE_Y_DFLT			100  // 100% = current behavior

// Spawn Rotation (degrees)
#define	LINE_SPAWN_ROTATION_MIN_VALUE	-180
#define	LINE_SPAWN_ROTATION_MAX_VALUE	180
#define	LINE_SPAWN_ROTATION_MIN_SLIDER	-180
#define	LINE_SPAWN_ROTATION_MAX_SLIDER	180
#define	LINE_SPAWN_ROTATION_DFLT		0

// Show Spawn Area
#define	SHOW_SPAWN_AREA_DFLT			0  // Off by default

// Origin Offset (px)
#define	ORIGIN_OFFSET_X_MIN_VALUE	-2000
#define	ORIGIN_OFFSET_X_MAX_VALUE	2000
#define	ORIGIN_OFFSET_X_MIN_SLIDER	-2000
#define	ORIGIN_OFFSET_X_MAX_SLIDER	2000
#define	ORIGIN_OFFSET_X_DFLT		0

#define	ORIGIN_OFFSET_Y_MIN_VALUE	-2000
#define	ORIGIN_OFFSET_Y_MAX_VALUE	2000
#define	ORIGIN_OFFSET_Y_MIN_SLIDER	-2000
#define	ORIGIN_OFFSET_Y_MAX_SLIDER	2000
#define	ORIGIN_OFFSET_Y_DFLT		0

// Animation Pattern
#define	ANIM_PATTERN_DFLT			1  // 1=Simple (default)

// Start Time (frames) - negative values allow "mid-play" effect
#define	LINE_START_TIME_MIN_VALUE	-36000
#define	LINE_START_TIME_MAX_VALUE	36000
#define	LINE_START_TIME_MIN_SLIDER	-300
#define	LINE_START_TIME_MAX_SLIDER	300
#define	LINE_START_TIME_DFLT		-300  // Start immediately (0 = no mid-play)

// Duration (frames) - 0 = infinite
#define	LINE_DURATION_MIN_VALUE		0
#define	LINE_DURATION_MAX_VALUE		36000
#define	LINE_DURATION_MIN_SLIDER	0
#define	LINE_DURATION_MAX_SLIDER	1800
#define	LINE_DURATION_DFLT			0  // 0=infinite (no end)

// Center Gap (applies to all patterns)
#define	CENTER_GAP_MIN_VALUE		0.0
#define	CENTER_GAP_MAX_VALUE		0.5
#define	CENTER_GAP_MIN_SLIDER		0.0
#define	CENTER_GAP_MAX_SLIDER		0.5
#define	CENTER_GAP_DFLT				0

// Motion Blur
#define	MOTION_BLUR_ENABLE_DFLT			0		// Off by default
#define	MOTION_BLUR_SAMPLES_MIN_VALUE	1
#define	MOTION_BLUR_SAMPLES_MAX_VALUE	32
#define	MOTION_BLUR_SAMPLES_MIN_SLIDER	1
#define	MOTION_BLUR_SAMPLES_MAX_SLIDER	32
#define	MOTION_BLUR_SAMPLES_DFLT		8		// 8 samples default
#define	MOTION_BLUR_ANGLE_MIN_VALUE		0.0
#define	MOTION_BLUR_ANGLE_MAX_VALUE		360.0
#define	MOTION_BLUR_ANGLE_MIN_SLIDER	0.0
#define	MOTION_BLUR_ANGLE_MAX_SLIDER	360.0
#define	MOTION_BLUR_ANGLE_DFLT			180.0	// 180° = film standard shutter angle

// Linkage Parameters
#define	LINKAGE_MODE_DFLT				0		// 0=Off
#define	LINKAGE_RATE_MIN_VALUE			0.0
#define	LINKAGE_RATE_MAX_VALUE			500.0
#define	LINKAGE_RATE_MIN_SLIDER			0.0
#define	LINKAGE_RATE_MAX_SLIDER			200.0
#define	LINKAGE_RATE_DFLT				2.0	// 100% = 1:1 ratio

// Preset colors count (8 colors per preset)
#define PRESET_COLORS_COUNT     8

/*
**
*/
#define	MAJOR_VERSION		1
#define	MINOR_VERSION		0
#define	BUG_VERSION			0
#define	STAGE_VERSION		PF_Stage_DEVELOP
#define	BUILD_VERSION		0

/*
** Preset Color Definitions
** Format: {A, R, G, B} - each value 0-255
*/
#ifdef __cplusplus

// Color preset definitions - auto-generated from color_presets.tsv
// Run color_preset_converter.py to regenerate OST_WindyLines_ColorPresets.h
#include "OST_WindyLines_ColorPresets.h"

#endif // __cplusplus

/*
** Shared data between CPU and GPU renderers for clip start detection
** Key: clipTime offset (clipTime - seqTime, constant per clip)
** Value: clipStart in frames (from CPU's PF_UtilitySuite::GetClipStart)
*/
#ifdef __cplusplus
#include <unordered_map>
#include <mutex>

// Element bounds structure for linkage feature
struct ElementBounds
{
	int minX;
	int minY;
	int maxX;
	int maxY;
	bool isValid;
	
	ElementBounds() : minX(-1), minY(-1), maxX(-1), maxY(-1), isValid(false) {}
	ElementBounds(int x1, int y1, int x2, int y2) : minX(x1), minY(y1), maxX(x2), maxY(y2), isValid(x2 >= x1 && y2 >= y1) {}
};

struct SharedClipData
{
	static std::unordered_map<csSDK_int64, csSDK_int64> clipStartMap;
	static std::unordered_map<csSDK_int64, ElementBounds> elementBoundsMap;
	static std::mutex mapMutex;
	
	static void SetClipStart(csSDK_int64 clipOffset, csSDK_int64 clipStartFrame)
	{
		std::lock_guard<std::mutex> lock(mapMutex);
		clipStartMap[clipOffset] = clipStartFrame;
	}
	
	static csSDK_int64 GetClipStart(csSDK_int64 clipOffset)
	{
		std::lock_guard<std::mutex> lock(mapMutex);
		auto it = clipStartMap.find(clipOffset);
		return (it != clipStartMap.end()) ? it->second : -1;
	}
	
	static void SetElementBounds(csSDK_int64 clipKey, const ElementBounds& bounds)
	{
		std::lock_guard<std::mutex> lock(mapMutex);
		elementBoundsMap[clipKey] = bounds;
	}
	
	static ElementBounds GetElementBounds(csSDK_int64 clipKey)
	{
		std::lock_guard<std::mutex> lock(mapMutex);
		auto it = elementBoundsMap.find(clipKey);
		return (it != elementBoundsMap.end()) ? it->second : ElementBounds();
	}
};
#endif

#endif