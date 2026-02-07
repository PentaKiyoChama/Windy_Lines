/**
 * SDK_ProcAmp_ParamOrder.h
 * 
 * Parameter Display Order Definition
 * 
 * This file defines the display order of parameters independently from
 * their enum ID values in SDK_ProcAmp.h. This allows easy parameter
 * reordering by simply changing the order in this file.
 * 
 * HOW TO USE:
 * 1. To reorder parameters, simply rearrange the items in PARAM_DISPLAY_ORDER array
 * 2. Group markers (topics) must stay together with their content:
 *    - SDK_PROCAMP_*_HEADER starts a group
 *    - Content parameters follow
 *    - SDK_PROCAMP_*_TOPIC_END ends the group
 * 3. The enum IDs in SDK_ProcAmp.h remain stable for backwards compatibility
 * 4. ParamsSetup will automatically register parameters in the order specified here
 * 
 * BENEFITS:
 * - Enum order can be kept stable (backwards compatibility)
 * - Display order is centrally managed in one place
 * - Easy to visualize and modify parameter grouping
 * - No need to modify ParamsSetup function when reordering
 */

#ifndef SDK_PROCAMP_PARAM_ORDER_H
#define SDK_PROCAMP_PARAM_ORDER_H

#include "SDK_ProcAmp.h"

/**
 * Parameter Display Order Array
 * 
 * This array defines the order in which parameters appear in the UI.
 * The order here determines the sequence of PF_ADD_* calls in ParamsSetup.
 * 
 * Rules:
 * - SDK_PROCAMP_INPUT must always be first
 * - Topic headers and endings must enclose their parameters
 * - Hidden parameters are typically at the end
 */
static const int PARAM_DISPLAY_ORDER[] = {
	SDK_PROCAMP_INPUT,                     // 0. Input layer (required first)
	
	// Top-level parameters (no group)
	SDK_PROCAMP_EFFECT_PRESET,             // 1. Effect preset selector
	SDK_PROCAMP_LINE_SEED,                 // 2. Random seed
	
	// ▼ Basic Settings (ungrouped)
	SDK_PROCAMP_LINE_COUNT,                // 3. Number of lines
	SDK_PROCAMP_LINE_LIFETIME,             // 4. Line lifetime (frames)
	SDK_PROCAMP_LINE_INTERVAL,             // 5. Spawn interval (frames)
	SDK_PROCAMP_LINE_TRAVEL,               // 6. Travel distance (px)
	SDK_PROCAMP_LINE_EASING,               // 7. Easing function
	
	// ▼ Color Settings (ungrouped)
	SDK_PROCAMP_COLOR_MODE,                // 8. Single/Preset/Custom
	SDK_PROCAMP_LINE_COLOR,                // 9. Single color picker
	SDK_PROCAMP_COLOR_PRESET,              // 10. Preset selection popup
	SDK_PROCAMP_CUSTOM_COLOR_1,            // 11-18. Custom colors 1-8
	SDK_PROCAMP_CUSTOM_COLOR_2,
	SDK_PROCAMP_CUSTOM_COLOR_3,
	SDK_PROCAMP_CUSTOM_COLOR_4,
	SDK_PROCAMP_CUSTOM_COLOR_5,
	SDK_PROCAMP_CUSTOM_COLOR_6,
	SDK_PROCAMP_CUSTOM_COLOR_7,
	SDK_PROCAMP_CUSTOM_COLOR_8,
	
	// ▼ Appearance (ungrouped)
	SDK_PROCAMP_LINE_THICKNESS,            // 19. Line thickness (px)
	SDK_PROCAMP_LINE_LENGTH,               // 20. Line length (px)
	SDK_PROCAMP_LINE_ANGLE,                // 21. Line angle (degrees)
	SDK_PROCAMP_LINE_CAP,                  // 22. Line cap style
	SDK_PROCAMP_LINE_TAIL_FADE,            // 23. Tail fade amount
	
	// ▼ Position & Spawn - Line Origin (grouped topic)
	SDK_PROCAMP_POSITION_HEADER,           // 24. Topic start
	SDK_PROCAMP_SPAWN_SOURCE,              // 25. Spawn source
	SDK_PROCAMP_LINE_ALPHA_THRESH,         // 26. Alpha threshold
	SDK_PROCAMP_LINE_ORIGIN_MODE,          // 27. Wind origin mode
	SDK_PROCAMP_ANIM_PATTERN,              // 28. Animation pattern
	SDK_PROCAMP_LINE_START_TIME,           // 29. Start time
	SDK_PROCAMP_LINE_DURATION,             // 30. Duration
	SDK_PROCAMP_LINE_DEPTH_STRENGTH,       // 31. Depth strength
	SDK_PROCAMP_CENTER_GAP,                // 32. Center gap
	SDK_PROCAMP_ORIGIN_OFFSET_X,           // 33. Origin Offset X
	SDK_PROCAMP_ORIGIN_OFFSET_Y,           // 34. Origin Offset Y
	SDK_PROCAMP_LINE_SPAWN_SCALE_X,        // 35. Spawn scale X
	SDK_PROCAMP_LINE_SPAWN_SCALE_Y,        // 36. Spawn scale Y
	SDK_PROCAMP_LINE_SPAWN_ROTATION,       // 37. Spawn rotation
	SDK_PROCAMP_LINE_SHOW_SPAWN_AREA,      // 38. Show spawn area
	SDK_PROCAMP_LINE_SPAWN_AREA_COLOR,     // 39. Spawn area color
	SDK_PROCAMP_POSITION_TOPIC_END,        // 40. Topic end
	
	// ▼ Shadow (grouped topic)
	SDK_PROCAMP_SHADOW_HEADER,             // 41. Topic start
	SDK_PROCAMP_SHADOW_ENABLE,             // 42. Shadow on/off
	SDK_PROCAMP_SHADOW_COLOR,              // 43. Shadow color
	SDK_PROCAMP_SHADOW_OFFSET_X,           // 44. Shadow offset X
	SDK_PROCAMP_SHADOW_OFFSET_Y,           // 45. Shadow offset Y
	SDK_PROCAMP_SHADOW_OPACITY,            // 46. Shadow opacity
	SDK_PROCAMP_SHADOW_TOPIC_END,          // 47. Topic end
	
	// ▼ Motion Blur (grouped topic)
	SDK_PROCAMP_MOTION_BLUR_HEADER,        // 48. Topic start
	SDK_PROCAMP_MOTION_BLUR_ENABLE,        // 49. Motion blur on/off
	SDK_PROCAMP_MOTION_BLUR_SAMPLES,       // 50. Motion blur samples
	SDK_PROCAMP_MOTION_BLUR_STRENGTH,      // 51. Motion blur strength
	SDK_PROCAMP_MOTION_BLUR_TOPIC_END,     // 52. Topic end
	
	// ▼ Advanced (grouped topic)
	SDK_PROCAMP_ADVANCED_HEADER,           // 53. Topic start
	SDK_PROCAMP_LINE_AA,                   // 54. Anti-aliasing
	SDK_PROCAMP_HIDE_ELEMENT,              // 55. Hide element
	SDK_PROCAMP_BLEND_MODE,                // 56. Blend mode
	SDK_PROCAMP_ADVANCED_TOPIC_END,        // 57. Topic end
	
	// ▼ Linkage (grouped topic)
	SDK_PROCAMP_LINKAGE_HEADER,            // 58. Topic start
	SDK_PROCAMP_LENGTH_LINKAGE,            // 59. Length linkage mode
	SDK_PROCAMP_LENGTH_LINKAGE_RATE,       // 60. Length linkage rate
	SDK_PROCAMP_THICKNESS_LINKAGE,         // 61. Thickness linkage mode
	SDK_PROCAMP_THICKNESS_LINKAGE_RATE,    // 62. Thickness linkage rate
	SDK_PROCAMP_TRAVEL_LINKAGE,            // 63. Travel linkage mode
	SDK_PROCAMP_TRAVEL_LINKAGE_RATE,       // 64. Travel linkage rate
	SDK_PROCAMP_LINKAGE_TOPIC_END,         // 65. Topic end
	
	// Hidden parameters (for backwards compatibility)
	SDK_PROCAMP_LINE_ALLOW_MIDPLAY,        // 66. (hidden)
	SDK_PROCAMP_LINE_COLOR_R,              // 67. (hidden)
	SDK_PROCAMP_LINE_COLOR_G,              // 68. (hidden)
	SDK_PROCAMP_LINE_COLOR_B,              // 69. (hidden)
};

// Compile-time validation
static_assert(sizeof(PARAM_DISPLAY_ORDER) / sizeof(PARAM_DISPLAY_ORDER[0]) == SDK_PROCAMP_NUM_PARAMS,
              "PARAM_DISPLAY_ORDER must contain exactly SDK_PROCAMP_NUM_PARAMS entries");

/**
 * Get the display index for a parameter ID
 * Returns -1 if not found (should never happen if array is complete)
 */
inline int GetParamDisplayIndex(int paramId)
{
	for (int i = 0; i < SDK_PROCAMP_NUM_PARAMS; ++i)
	{
		if (PARAM_DISPLAY_ORDER[i] == paramId)
		{
			return i;
		}
	}
	return -1; // Not found
}

/**
 * Get the parameter ID at a specific display position
 */
inline int GetParamIdAtDisplayIndex(int displayIndex)
{
	if (displayIndex >= 0 && displayIndex < SDK_PROCAMP_NUM_PARAMS)
	{
		return PARAM_DISPLAY_ORDER[displayIndex];
	}
	return -1; // Invalid index
}

#endif // SDK_PROCAMP_PARAM_ORDER_H
