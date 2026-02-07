/**
 * SDK_ProcAmp_ParamOrder_Example.h
 * 
 * EXAMPLE: How to Reorder Parameters
 * 
 * This file demonstrates how to reorder parameters by simply changing
 * the PARAM_DISPLAY_ORDER array. The example shows moving the Shadow
 * group before the Motion Blur group.
 * 
 * BEFORE: Motion Blur → Shadow → Advanced
 * AFTER:  Shadow → Motion Blur → Advanced
 * 
 * To apply this reordering:
 * 1. Copy the EXAMPLE_DISPLAY_ORDER array below
 * 2. Replace PARAM_DISPLAY_ORDER in SDK_ProcAmp_ParamOrder.h
 * 3. Rebuild the project
 * 4. Parameters will appear in the new order
 * 
 * Note: Enum IDs in SDK_ProcAmp.h remain unchanged!
 */

#ifndef SDK_PROCAMP_PARAM_ORDER_EXAMPLE_H
#define SDK_PROCAMP_PARAM_ORDER_EXAMPLE_H

#include "SDK_ProcAmp.h"

// ====================================================================
// EXAMPLE REORDERING: Shadow before Motion Blur
// ====================================================================

static const int EXAMPLE_DISPLAY_ORDER[] = {
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
	
	// ============================================================
	// REORDERED: Shadow now appears BEFORE Motion Blur
	// ============================================================
	
	// ▼ Shadow (grouped topic) - MOVED UP
	SDK_PROCAMP_SHADOW_HEADER,             // 41. Topic start
	SDK_PROCAMP_SHADOW_ENABLE,             // 42. Shadow on/off
	SDK_PROCAMP_SHADOW_COLOR,              // 43. Shadow color
	SDK_PROCAMP_SHADOW_OFFSET_X,           // 44. Shadow offset X
	SDK_PROCAMP_SHADOW_OFFSET_Y,           // 45. Shadow offset Y
	SDK_PROCAMP_SHADOW_OPACITY,            // 46. Shadow opacity
	SDK_PROCAMP_SHADOW_TOPIC_END,          // 47. Topic end
	
	// ▼ Motion Blur (grouped topic) - NOW AFTER SHADOW
	SDK_PROCAMP_MOTION_BLUR_HEADER,        // 48. Topic start
	SDK_PROCAMP_MOTION_BLUR_ENABLE,        // 49. Motion blur on/off
	SDK_PROCAMP_MOTION_BLUR_SAMPLES,       // 50. Motion blur samples
	SDK_PROCAMP_MOTION_BLUR_STRENGTH,      // 51. Motion blur strength
	SDK_PROCAMP_MOTION_BLUR_TOPIC_END,     // 52. Topic end
	
	// ============================================================
	
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

// ====================================================================
// ANOTHER EXAMPLE: Swap Line Count and Line Lifetime
// ====================================================================

static const int EXAMPLE_SWAP_COUNT_LIFETIME[] = {
	SDK_PROCAMP_INPUT,                     // 0. Input layer (required first)
	
	// Top-level parameters (no group)
	SDK_PROCAMP_EFFECT_PRESET,             // 1. Effect preset selector
	SDK_PROCAMP_LINE_SEED,                 // 2. Random seed
	
	// ▼ Basic Settings (ungrouped) - COUNT AND LIFETIME SWAPPED
	SDK_PROCAMP_LINE_LIFETIME,             // 3. Line lifetime (NOW FIRST)
	SDK_PROCAMP_LINE_COUNT,                // 4. Number of lines (NOW SECOND)
	SDK_PROCAMP_LINE_INTERVAL,             // 5. Spawn interval (frames)
	SDK_PROCAMP_LINE_TRAVEL,               // 6. Travel distance (px)
	SDK_PROCAMP_LINE_EASING,               // 7. Easing function
	
	// ... rest remains the same ...
	// (Omitted for brevity - in real usage, include all parameters)
};

// ====================================================================
// EXAMPLE: Move Easing to Top of Basic Settings
// ====================================================================

static const int EXAMPLE_EASING_FIRST[] = {
	SDK_PROCAMP_INPUT,                     // 0. Input layer (required first)
	
	// Top-level parameters (no group)
	SDK_PROCAMP_EFFECT_PRESET,             // 1. Effect preset selector
	SDK_PROCAMP_LINE_SEED,                 // 2. Random seed
	
	// ▼ Basic Settings (ungrouped) - EASING NOW FIRST
	SDK_PROCAMP_LINE_EASING,               // 3. Easing function (MOVED UP)
	SDK_PROCAMP_LINE_COUNT,                // 4. Number of lines
	SDK_PROCAMP_LINE_LIFETIME,             // 5. Line lifetime (frames)
	SDK_PROCAMP_LINE_INTERVAL,             // 6. Spawn interval (frames)
	SDK_PROCAMP_LINE_TRAVEL,               // 7. Travel distance (px)
	
	// ... rest remains the same ...
};

// ====================================================================
// VALIDATION NOTES
// ====================================================================

/*
 * Key Points Demonstrated:
 * 
 * 1. ENUM IDs NEVER CHANGE
 *    - SDK_PROCAMP_LINE_COUNT is always ID 3 in SDK_ProcAmp.h
 *    - SDK_PROCAMP_LINE_LIFETIME is always ID 4 in SDK_ProcAmp.h
 *    - These are stable for backwards compatibility
 * 
 * 2. DISPLAY ORDER IS FLEXIBLE
 *    - Count can appear before or after Lifetime in UI
 *    - Groups can be reordered freely
 *    - Only the array order changes, not the enum definitions
 * 
 * 3. BACKWARDS COMPATIBILITY MAINTAINED
 *    - params[SDK_PROCAMP_LINE_COUNT] always accesses the same parameter
 *    - Old project files continue to work
 *    - Saved preset values map correctly
 * 
 * 4. GROUP INTEGRITY REQUIRED
 *    - Topic HEADER/TOPIC_END must enclose parameters
 *    - Cannot split a topic group
 *    - All parameters in a topic must move together
 * 
 * 5. COMPILE-TIME SAFETY
 *    - static_assert validates complete parameter coverage
 *    - Missing or duplicate parameters cause build errors
 *    - Cannot accidentally break parameter system
 */

#endif // SDK_PROCAMP_PARAM_ORDER_EXAMPLE_H
