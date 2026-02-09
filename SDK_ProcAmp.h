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


#ifndef SDK_PROCAMP_H
#define SDK_PROCAMP_H

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

// ========== DEBUG RENDER MARKERS ==========
// Set to 1 to enable visual markers in top-left corner (GPU/CPU indicator)
// Set to 0 to disable completely (zero performance impact)
#define ENABLE_DEBUG_RENDER_MARKERS 0
// ========== DEBUG LOGGING (Common) ==========
static std::mutex sLogMutex;
static void WriteLog(const char* format, ...)
{
	std::lock_guard<std::mutex> lock(sLogMutex);
	
	// Try C:\Temp first, then Desktop
	const char* paths[] = {
		"C:\\Temp\\SDK_ProcAmp_Log.txt",
		"C:\\Users\\Owner\\Desktop\\SDK_ProcAmp_Log.txt"
	};
	
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

#define DebugLog WriteLog
// ===========================================


/*
** Defaults & ranges (edit here to change default values)
*/
enum
{
	SDK_PROCAMP_INPUT = 0,
	
	// Effect Preset (top-level)
	SDK_PROCAMP_EFFECT_PRESET,                // 1. Effect preset selector
	SDK_PROCAMP_LINE_SEED,                    // 2. Random seed
	
	// ▼ Basic Settings
	SDK_PROCAMP_LINE_COUNT,                   // 3. Number of lines
	SDK_PROCAMP_LINE_LIFETIME,                // 4. Line lifetime (frames)
	SDK_PROCAMP_LINE_INTERVAL,                // 5. Spawn interval (frames)
	SDK_PROCAMP_LINE_EASING,                  // 6. Easing function
	
	// ▼ Color Settings
	SDK_PROCAMP_COLOR_MODE,                   // 7. Single/Preset/Custom
	SDK_PROCAMP_LINE_COLOR,                   // 8. Single color picker
	SDK_PROCAMP_COLOR_PRESET,                 // 9. Preset selection popup
	SDK_PROCAMP_CUSTOM_COLOR_1,               // 10-17. Custom colors 1-8
	SDK_PROCAMP_CUSTOM_COLOR_2,
	SDK_PROCAMP_CUSTOM_COLOR_3,
	SDK_PROCAMP_CUSTOM_COLOR_4,
	SDK_PROCAMP_CUSTOM_COLOR_5,
	SDK_PROCAMP_CUSTOM_COLOR_6,
	SDK_PROCAMP_CUSTOM_COLOR_7,
	SDK_PROCAMP_CUSTOM_COLOR_8,
	SDK_PROCAMP_TRAVEL_LINKAGE,               // 18. Travel distance linkage (Off/Width/Height)
	SDK_PROCAMP_TRAVEL_LINKAGE_RATE,          // 19. Travel distance linkage rate (%)
	SDK_PROCAMP_LINE_TRAVEL,                  // 20. Travel distance (px)
	
	// ▼ Appearance
	SDK_PROCAMP_THICKNESS_LINKAGE,            // 21. Thickness linkage (Off/Width/Height)
	SDK_PROCAMP_THICKNESS_LINKAGE_RATE,       // 22. Thickness linkage rate (%)
	SDK_PROCAMP_LINE_THICKNESS,               // 23. Line thickness (px)
	SDK_PROCAMP_LENGTH_LINKAGE,               // 24. Length linkage (Off/Width/Height)
	SDK_PROCAMP_LENGTH_LINKAGE_RATE,          // 25. Length linkage rate (%)
	SDK_PROCAMP_LINE_LENGTH,                  // 26. Line length (px)
	SDK_PROCAMP_LINE_ANGLE,                   // 27. Line angle (degrees)
	SDK_PROCAMP_LINE_CAP,                     // 28. Line cap style
	SDK_PROCAMP_LINE_TAIL_FADE,               // 29. Tail fade amount
	
	// ▼ Position & Spawn - Line Origin
	SDK_PROCAMP_POSITION_HEADER,              // 30. Line origin topic start
	SDK_PROCAMP_SPAWN_SOURCE,                 // 31. Spawn source (Full Frame / Element Bounds)
	SDK_PROCAMP_LINE_ALPHA_THRESH,            // 32. Alpha threshold
	SDK_PROCAMP_LINE_ORIGIN_MODE,             // 33. Wind origin mode
	SDK_PROCAMP_ANIM_PATTERN,                 // 34. Animation pattern (direction)
	SDK_PROCAMP_LINE_START_TIME,              // 35. Start time (frames)
	SDK_PROCAMP_LINE_DURATION,                // 36. Duration (frames)
	SDK_PROCAMP_LINE_DEPTH_STRENGTH,          // 37. Depth strength
	SDK_PROCAMP_CENTER_GAP,                   // 38. Center gap
	SDK_PROCAMP_ORIGIN_OFFSET_X,              // 39. Origin Offset X (px)
	SDK_PROCAMP_ORIGIN_OFFSET_Y,              // 40. Origin Offset Y (px)
	SDK_PROCAMP_LINE_SPAWN_SCALE_X,           // 41. Spawn area scale X (%)
	SDK_PROCAMP_LINE_SPAWN_SCALE_Y,           // 42. Spawn area scale Y (%)
	SDK_PROCAMP_LINE_SPAWN_ROTATION,          // 43. Spawn area rotation (degrees)
	SDK_PROCAMP_LINE_SHOW_SPAWN_AREA,         // 44. Show spawn area preview
	SDK_PROCAMP_LINE_SPAWN_AREA_COLOR,        // 45. Spawn area color
	SDK_PROCAMP_POSITION_TOPIC_END,           // 46. Line origin topic end
	
	// ▼ Shadow - Topic group
	SDK_PROCAMP_SHADOW_HEADER,                // 47. Shadow topic start
	SDK_PROCAMP_SHADOW_ENABLE,                // 48. Shadow on/off
	SDK_PROCAMP_SHADOW_COLOR,                 // 49. Shadow color
	SDK_PROCAMP_SHADOW_OFFSET_X,              // 50. Shadow offset X (px)
	SDK_PROCAMP_SHADOW_OFFSET_Y,              // 51. Shadow offset Y (px)
	SDK_PROCAMP_SHADOW_OPACITY,               // 52. Shadow opacity (0-1)
	SDK_PROCAMP_SHADOW_TOPIC_END,             // 53. Shadow topic end
	
	// ▼ Motion Blur - Topic group
	SDK_PROCAMP_MOTION_BLUR_HEADER,           // 54. Motion blur topic start
	SDK_PROCAMP_MOTION_BLUR_ENABLE,           // 55. Motion blur on/off
	SDK_PROCAMP_MOTION_BLUR_SAMPLES,          // 56. Motion blur quality (samples)
	SDK_PROCAMP_MOTION_BLUR_STRENGTH,         // 57. Motion blur strength
	SDK_PROCAMP_MOTION_BLUR_TOPIC_END,        // 58. Motion blur topic end
	
	// ▼ Advanced - Topic group
	SDK_PROCAMP_ADVANCED_HEADER,              // 59. Advanced topic start
	SDK_PROCAMP_LINE_AA,                      // 60. Anti-aliasing
	SDK_PROCAMP_HIDE_ELEMENT,                 // 61. Hide original element (lines only)
	SDK_PROCAMP_BLEND_MODE,                   // 62. Blend mode with element
	SDK_PROCAMP_ADVANCED_TOPIC_END,           // 63. Advanced topic end
	
	// Hidden params (for backwards compatibility)
	SDK_PROCAMP_LINE_ALLOW_MIDPLAY,           // 64. (hidden)
	SDK_PROCAMP_LINE_COLOR_R,                 // 65. (hidden)
	SDK_PROCAMP_LINE_COLOR_G,                 // 66. (hidden)
	SDK_PROCAMP_LINE_COLOR_B,                 // 67. (hidden)
	
	SDK_PROCAMP_NUM_PARAMS
};

// Color Mode enum
enum ColorMode
{
	COLOR_MODE_SINGLE = 1,   // Single color for all lines
	COLOR_MODE_PRESET = 2,   // Use preset palette
	COLOR_MODE_CUSTOM = 3    // User-defined 8 colors
};

// Color Preset enum (matches OSTBoxWind presets)
enum ColorPreset
{
	COLOR_PRESET_RAINBOW = 1,
	COLOR_PRESET_RAINBOW_PASTEL,
	COLOR_PRESET_FOREST,
	COLOR_PRESET_CYBER,
	COLOR_PRESET_HAZARD,
	COLOR_PRESET_SAKURA,
	COLOR_PRESET_DESERT,
	COLOR_PRESET_STAR_DUST,
	COLOR_PRESET_WAKABA,
	COLOR_PRESET_DANGER_ZONE,
	COLOR_PRESET_YOEN,
	COLOR_PRESET_SOKAI,
	COLOR_PRESET_DREAMY_WIND,
	COLOR_PRESET_SUNSET,
	COLOR_PRESET_OCEAN,
	COLOR_PRESET_AUTUMN,
	COLOR_PRESET_SNOW,
	COLOR_PRESET_DEEP_SEA,
	COLOR_PRESET_MORNING_DEW,
	COLOR_PRESET_NIGHT_SKY,
	COLOR_PRESET_FLAME,
	COLOR_PRESET_EARTH,
	COLOR_PRESET_JEWEL,
	COLOR_PRESET_PASTEL2,
	COLOR_PRESET_CITY_NIGHT,
	COLOR_PRESET_MOONLIGHT,
	COLOR_PRESET_DAZZLING_LIGHT,
	COLOR_PRESET_NEON_BLAST,
	COLOR_PRESET_TOXIC_SWAMP,
	COLOR_PRESET_COSMIC_STORM,
	COLOR_PRESET_LAVA_FLOW,
	COLOR_PRESET_GOLD,
	COLOR_PRESET_MONOCHROME,
	COLOR_PRESET_COUNT
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
	int colorMode;        // 1=Single, 2=Preset, 3=Custom
	int colorPreset;      // 1-33
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
// Run preset_converter.py to regenerate SDK_ProcAmp_Presets.h
#include "SDK_ProcAmp_Presets.h"

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
#define COLOR_MODE_DFLT         2   // 
#define COLOR_PRESET_DFLT       1   // Rainbow

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

// Color structure for presets (ARGB format)
struct PresetColor {
	unsigned char a, r, g, b;
};

// Preset color palettes (8 colors each)
namespace ColorPresets {
	// Rainbow
	const PresetColor kRainbow[8] = {
		{255, 255, 0, 0}, {255, 255, 128, 0}, {255, 255, 255, 0}, {255, 0, 255, 0},
		{255, 0, 0, 255}, {255, 74, 0, 130}, {255, 140, 0, 255}, {255, 255, 0, 255}
	};
	// Pastel Rainbow
	const PresetColor kPastel[8] = {
		{255, 255, 178, 178}, {255, 255, 217, 178}, {255, 255, 255, 178}, {255, 178, 255, 178},
		{255, 178, 178, 255}, {255, 217, 178, 255}, {255, 255, 178, 255}, {255, 255, 204, 255}
	};
	// Forest
	const PresetColor kForest[8] = {
		{255, 102, 229, 128}, {255, 153, 242, 204}, {255, 51, 178, 229}, {255, 178, 255, 178},
		{255, 102, 204, 255}, {255, 76, 204, 102}, {255, 153, 204, 255}, {255, 204, 255, 229}
	};
	// Cyber
	const PresetColor kCyber[8] = {
		{255, 0, 204, 255}, {255, 128, 0, 255}, {255, 0, 255, 128}, {255, 51, 153, 255},
		{255, 255, 0, 204}, {255, 0, 102, 153}, {255, 204, 255, 0}, {255, 178, 178, 178}
	};
	// Hazard
	const PresetColor kHazard[8] = {
		{255, 255, 51, 0}, {255, 255, 128, 0}, {255, 0, 0, 0}, {255, 204, 0, 0},
		{255, 153, 76, 0}, {255, 51, 51, 51}, {255, 255, 178, 0}, {255, 128, 0, 0}
	};
	// Sakura (Spring)
	const PresetColor kSakura[8] = {
		{255, 255, 178, 204}, {255, 229, 229, 242}, {255, 178, 255, 178}, {255, 255, 217, 229},
		{255, 153, 204, 102}, {255, 255, 153, 191}, {255, 242, 242, 255}, {255, 153, 204, 229}
	};
	// Desert
	const PresetColor kDesert[8] = {
		{255, 204, 153, 51}, {255, 178, 102, 0}, {255, 102, 51, 25}, {255, 229, 178, 76},
		{255, 51, 25, 12}, {255, 153, 76, 0}, {255, 128, 102, 76}, {255, 25, 12, 0}
	};
	// Star Dust
	const PresetColor kStarDust[8] = {
		{255, 12, 0, 51}, {255, 76, 0, 128}, {255, 229, 229, 255}, {255, 25, 25, 102},
		{255, 153, 128, 204}, {255, 51, 0, 89}, {255, 178, 178, 178}, {255, 0, 0, 25}
	};
	// Wakaba (Young Leaf)
	const PresetColor kWakaba[8] = {
		{255, 204, 255, 51}, {255, 153, 229, 102}, {255, 255, 229, 0}, {255, 102, 204, 76},
		{255, 229, 255, 128}, {255, 178, 242, 153}, {255, 255, 255, 153}, {255, 76, 178, 51}
	};
	// Danger Zone (Flame)
	const PresetColor kDangerZone[8] = {
		{255, 0, 0, 0}, {255, 255, 204, 0}, {255, 255, 102, 0}, {255, 51, 51, 51},
		{255, 229, 178, 0}, {255, 204, 76, 0}, {255, 25, 25, 25}, {255, 255, 153, 0}
	};
	// Yoen (Bewitching)
	const PresetColor kYoen[8] = {
		{255, 102, 0, 128}, {255, 153, 0, 204}, {255, 51, 0, 76}, {255, 204, 25, 76},
		{255, 76, 25, 102}, {255, 229, 0, 153}, {255, 25, 0, 38}, {255, 178, 51, 229}
	};
	// Sokai (Exhilarating)
	const PresetColor kSokai[8] = {
		{255, 0, 178, 255}, {255, 76, 255, 178}, {255, 204, 242, 255}, {255, 0, 128, 204},
		{255, 128, 255, 204}, {255, 25, 229, 255}, {255, 229, 255, 242}, {255, 0, 153, 229}
	};
	// Dreamy
	const PresetColor kDreamy[8] = {
		{255, 153, 229, 255}, {255, 255, 191, 229}, {255, 204, 255, 255}, {255, 255, 204, 242},
		{255, 102, 204, 255}, {255, 255, 166, 204}, {255, 229, 242, 255}, {255, 255, 204, 217}
	};
	// Sunset
	const PresetColor kSunset[8] = {
		{255, 255, 102, 0}, {255, 255, 153, 51}, {255, 255, 204, 102}, {255, 255, 69, 0},
		{255, 255, 140, 0}, {255, 204, 85, 0}, {255, 178, 34, 34}, {255, 255, 165, 0}
	};
	// Ocean
	const PresetColor kOcean[8] = {
		{255, 0, 119, 190}, {255, 0, 191, 255}, {255, 135, 206, 235}, {255, 0, 150, 255},
		{255, 30, 144, 255}, {255, 100, 149, 237}, {255, 173, 216, 230}, {255, 0, 206, 209}
	};
	// Autumn
	const PresetColor kAutumn[8] = {
		{255, 34, 139, 34}, {255, 107, 142, 35}, {255, 154, 205, 50}, {255, 255, 215, 0},
		{255, 255, 165, 0}, {255, 255, 69, 0}, {255, 220, 20, 60}, {255, 178, 34, 34}
	};
	// Snow
	const PresetColor kSnow[8] = {
		{255, 255, 255, 255}, {255, 240, 248, 255}, {255, 230, 230, 250}, {255, 248, 248, 255},
		{255, 220, 220, 220}, {255, 245, 245, 245}, {255, 211, 211, 211}, {255, 192, 192, 192}
	};
	// Deep Sea
	const PresetColor kDeepSea[8] = {
		{255, 0, 0, 139}, {255, 25, 25, 112}, {255, 0, 0, 205}, {255, 72, 61, 139},
		{255, 106, 90, 205}, {255, 123, 104, 238}, {255, 147, 112, 219}, {255, 138, 43, 226}
	};
	// Morning Dew (Party)
	const PresetColor kMorningDew[8] = {
		{255, 255, 255, 0}, {255, 255, 165, 0}, {255, 255, 20, 147}, {255, 0, 191, 255},
		{255, 30, 144, 255}, {255, 50, 205, 50}, {255, 138, 43, 226}, {255, 255, 69, 0}
	};
	// Night Sky
	const PresetColor kNightSky[8] = {
		{255, 25, 25, 112}, {255, 0, 0, 139}, {255, 72, 61, 139}, {255, 106, 90, 205},
		{255, 123, 104, 238}, {255, 147, 112, 219}, {255, 138, 43, 226}, {255, 148, 0, 211}
	};
	// Flame (Amethyst Garden)
	const PresetColor kFlame[8] = {
		{255, 186, 85, 211}, {255, 221, 160, 221}, {255, 238, 130, 238}, {255, 218, 112, 214},
		{255, 255, 182, 193}, {255, 255, 192, 203}, {255, 230, 230, 250}, {255, 216, 191, 216}
	};
	// Earth (Flower Field)
	const PresetColor kEarth[8] = {
		{255, 34, 139, 34}, {255, 50, 205, 50}, {255, 144, 238, 144}, {255, 255, 182, 193},
		{255, 255, 20, 147}, {255, 219, 112, 147}, {255, 186, 85, 211}, {255, 138, 43, 226}
	};
	// Jewel
	const PresetColor kJewel[8] = {
		{255, 255, 0, 255}, {255, 0, 255, 255}, {255, 255, 215, 0}, {255, 50, 205, 50},
		{255, 255, 20, 147}, {255, 30, 144, 255}, {255, 138, 43, 226}, {255, 255, 105, 180}
	};
	// Pastel 2
	const PresetColor kPastel2[8] = {
		{255, 255, 182, 193}, {255, 255, 218, 185}, {255, 255, 228, 196}, {255, 255, 240, 245},
		{255, 230, 230, 250}, {255, 221, 160, 221}, {255, 255, 228, 225}, {255, 255, 218, 185}
	};
	// City Night (Earth Color)
	const PresetColor kCityNight[8] = {
		{255, 101, 67, 33}, {255, 139, 69, 19}, {255, 160, 82, 45}, {255, 205, 133, 63},
		{255, 222, 184, 135}, {255, 238, 203, 173}, {255, 245, 222, 179}, {255, 255, 228, 196}
	};
	// Moonlight
	const PresetColor kMoonlight[8] = {
		{255, 248, 248, 255}, {255, 230, 230, 250}, {255, 176, 196, 222}, {255, 119, 136, 153},
		{255, 112, 128, 144}, {255, 135, 206, 235}, {255, 173, 216, 230}, {255, 240, 248, 255}
	};
	// Dazzling Light
	const PresetColor kDazzlingLight[8] = {
		{255, 255, 255, 255}, {255, 255, 255, 224}, {255, 255, 250, 205}, {255, 255, 255, 0},
		{255, 255, 215, 0}, {255, 255, 228, 181}, {255, 255, 248, 220}, {255, 255, 255, 240}
	};
	// Neon Blast
	const PresetColor kNeonBlast[8] = {
		{255, 255, 0, 255}, {255, 255, 20, 147}, {255, 255, 105, 180}, {255, 219, 112, 147},
		{255, 186, 85, 211}, {255, 138, 43, 226}, {255, 148, 0, 211}, {255, 199, 21, 133}
	};
	// Toxic Swamp (Warning)
	const PresetColor kToxicSwamp[8] = {
		{255, 255, 0, 0}, {255, 255, 69, 0}, {255, 255, 140, 0}, {255, 255, 165, 0},
		{255, 255, 255, 0}, {255, 255, 215, 0}, {255, 139, 0, 0}, {255, 128, 0, 0}
	};
	// Cosmic Storm (Aurora)
	const PresetColor kCosmicStorm[8] = {
		{255, 0, 255, 127}, {255, 46, 139, 87}, {255, 0, 250, 154}, {255, 127, 255, 212},
		{255, 173, 216, 230}, {255, 138, 43, 226}, {255, 147, 112, 219}, {255, 255, 20, 147}
	};
	// Lava Flow
	const PresetColor kLavaFlow[8] = {
		{255, 255, 0, 0}, {255, 255, 69, 0}, {255, 255, 20, 147}, {255, 139, 0, 0},
		{255, 255, 140, 0}, {255, 220, 20, 60}, {255, 178, 34, 34}, {255, 128, 0, 0}
	};
	// Gold
	const PresetColor kGold[8] = {
		{255, 255, 215, 0}, {255, 255, 223, 0}, {255, 255, 255, 0}, {255, 255, 218, 185},
		{255, 238, 203, 173}, {255, 205, 133, 63}, {255, 184, 134, 11}, {255, 139, 69, 19}
	};
	// Monochrome
	const PresetColor kMonochrome[8] = {
		{255, 255, 255, 255}, {255, 240, 240, 240}, {255, 211, 211, 211}, {255, 169, 169, 169},
		{255, 128, 128, 128}, {255, 105, 105, 105}, {255, 64, 64, 64}, {255, 0, 0, 0}
	};
}

// Preset color lookup table
inline const PresetColor* GetPresetPalette(int presetIndex) {
	switch (presetIndex) {
		case COLOR_PRESET_RAINBOW:        return ColorPresets::kRainbow;
		case COLOR_PRESET_RAINBOW_PASTEL: return ColorPresets::kPastel;
		case COLOR_PRESET_FOREST:         return ColorPresets::kForest;
		case COLOR_PRESET_CYBER:          return ColorPresets::kCyber;
		case COLOR_PRESET_HAZARD:         return ColorPresets::kHazard;
		case COLOR_PRESET_SAKURA:         return ColorPresets::kSakura;
		case COLOR_PRESET_DESERT:         return ColorPresets::kDesert;
		case COLOR_PRESET_STAR_DUST:      return ColorPresets::kStarDust;
		case COLOR_PRESET_WAKABA:         return ColorPresets::kWakaba;
		case COLOR_PRESET_DANGER_ZONE:    return ColorPresets::kDangerZone;
		case COLOR_PRESET_YOEN:           return ColorPresets::kYoen;
		case COLOR_PRESET_SOKAI:          return ColorPresets::kSokai;
		case COLOR_PRESET_DREAMY_WIND:    return ColorPresets::kDreamy;
		case COLOR_PRESET_SUNSET:         return ColorPresets::kSunset;
		case COLOR_PRESET_OCEAN:          return ColorPresets::kOcean;
		case COLOR_PRESET_AUTUMN:         return ColorPresets::kAutumn;
		case COLOR_PRESET_SNOW:           return ColorPresets::kSnow;
		case COLOR_PRESET_DEEP_SEA:       return ColorPresets::kDeepSea;
		case COLOR_PRESET_MORNING_DEW:    return ColorPresets::kMorningDew;
		case COLOR_PRESET_NIGHT_SKY:      return ColorPresets::kNightSky;
		case COLOR_PRESET_FLAME:          return ColorPresets::kFlame;
		case COLOR_PRESET_EARTH:          return ColorPresets::kEarth;
		case COLOR_PRESET_JEWEL:          return ColorPresets::kJewel;
		case COLOR_PRESET_PASTEL2:        return ColorPresets::kPastel2;
		case COLOR_PRESET_CITY_NIGHT:     return ColorPresets::kCityNight;
		case COLOR_PRESET_MOONLIGHT:      return ColorPresets::kMoonlight;
		case COLOR_PRESET_DAZZLING_LIGHT: return ColorPresets::kDazzlingLight;
		case COLOR_PRESET_NEON_BLAST:     return ColorPresets::kNeonBlast;
		case COLOR_PRESET_TOXIC_SWAMP:    return ColorPresets::kToxicSwamp;
		case COLOR_PRESET_COSMIC_STORM:   return ColorPresets::kCosmicStorm;
		case COLOR_PRESET_LAVA_FLOW:      return ColorPresets::kLavaFlow;
		case COLOR_PRESET_GOLD:           return ColorPresets::kGold;
		case COLOR_PRESET_MONOCHROME:     return ColorPresets::kMonochrome;
		default:                          return ColorPresets::kRainbow;
	}
}

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