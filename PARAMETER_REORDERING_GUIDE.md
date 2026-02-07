# Parameter Reordering Guide

## Overview

This guide explains how to easily reorder user parameters in the Windy Lines effect plugin.

## Problem We Solved

Previously, reordering parameters required:
1. Manually editing the large `ParamsSetup()` function in `SDK_ProcAmp_CPU.cpp`
2. Moving blocks of `PF_ADD_*` macro calls around
3. Carefully maintaining topic group boundaries
4. Risk of breaking existing projects if enum IDs changed

## Solution: Centralized Display Order

We created `SDK_ProcAmp_ParamOrder.h` which defines parameter display order in one place:

```cpp
static const int PARAM_DISPLAY_ORDER[] = {
    SDK_PROCAMP_INPUT,
    SDK_PROCAMP_EFFECT_PRESET,
    SDK_PROCAMP_LINE_SEED,
    SDK_PROCAMP_LINE_COUNT,
    // ... and so on
};
```

## How It Works

### Key Insight
Adobe's Premiere Pro SDK determines parameter display order by the **sequence of `PF_ADD_*` calls**, not by the enum values. The enum ID (last parameter in `PF_ADD_*` macros) only determines which parameter slot is being filled.

### System Architecture

1. **Enum IDs** (in `SDK_ProcAmp.h`): Stable integer IDs that never change
   - Used for parameter access: `params[SDK_PROCAMP_LINE_COUNT]->u.fs_d.value`
   - Stored in project files for backwards compatibility
   - Example: `SDK_PROCAMP_LINE_COUNT = 3`

2. **Display Order** (in `SDK_ProcAmp_ParamOrder.h`): Determines UI presentation
   - Array of enum IDs in desired display order
   - Can be freely rearranged without breaking compatibility
   - Example: Position 0 displays `PARAM_DISPLAY_ORDER[0]` parameter

3. **Parameter Registration**: Links the two together
   - `ParamsSetup()` iterates through `PARAM_DISPLAY_ORDER`
   - Calls appropriate `PF_ADD_*` macro for each parameter
   - Result: Parameters appear in the order specified by `PARAM_DISPLAY_ORDER`

## How to Reorder Parameters

### Simple Example: Swap Two Parameters

To swap "Line Count" and "Line Lifetime":

**Before:**
```cpp
static const int PARAM_DISPLAY_ORDER[] = {
    SDK_PROCAMP_INPUT,
    SDK_PROCAMP_EFFECT_PRESET,
    SDK_PROCAMP_LINE_SEED,
    SDK_PROCAMP_LINE_COUNT,        // Position 3
    SDK_PROCAMP_LINE_LIFETIME,     // Position 4
    SDK_PROCAMP_LINE_INTERVAL,
    // ...
};
```

**After:**
```cpp
static const int PARAM_DISPLAY_ORDER[] = {
    SDK_PROCAMP_INPUT,
    SDK_PROCAMP_EFFECT_PRESET,
    SDK_PROCAMP_LINE_SEED,
    SDK_PROCAMP_LINE_LIFETIME,     // Position 3 (now lifetime first)
    SDK_PROCAMP_LINE_COUNT,        // Position 4 (now count second)
    SDK_PROCAMP_LINE_INTERVAL,
    // ...
};
```

Save, rebuild, and test. The parameters will now appear in the new order!

### Complex Example: Move an Entire Topic Group

To move the Shadow group before Motion Blur:

**Before:**
```cpp
// ... other parameters ...

// Motion Blur group
SDK_PROCAMP_MOTION_BLUR_HEADER,
SDK_PROCAMP_MOTION_BLUR_ENABLE,
SDK_PROCAMP_MOTION_BLUR_SAMPLES,
SDK_PROCAMP_MOTION_BLUR_STRENGTH,
SDK_PROCAMP_MOTION_BLUR_TOPIC_END,

// Shadow group
SDK_PROCAMP_SHADOW_HEADER,
SDK_PROCAMP_SHADOW_ENABLE,
SDK_PROCAMP_SHADOW_COLOR,
SDK_PROCAMP_SHADOW_OFFSET_X,
SDK_PROCAMP_SHADOW_OFFSET_Y,
SDK_PROCAMP_SHADOW_OPACITY,
SDK_PROCAMP_SHADOW_TOPIC_END,

// ... more parameters ...
```

**After:**
```cpp
// ... other parameters ...

// Shadow group (moved up)
SDK_PROCAMP_SHADOW_HEADER,
SDK_PROCAMP_SHADOW_ENABLE,
SDK_PROCAMP_SHADOW_COLOR,
SDK_PROCAMP_SHADOW_OFFSET_X,
SDK_PROCAMP_SHADOW_OFFSET_Y,
SDK_PROCAMP_SHADOW_OPACITY,
SDK_PROCAMP_SHADOW_TOPIC_END,

// Motion Blur group (now below)
SDK_PROCAMP_MOTION_BLUR_HEADER,
SDK_PROCAMP_MOTION_BLUR_ENABLE,
SDK_PROCAMP_MOTION_BLUR_SAMPLES,
SDK_PROCAMP_MOTION_BLUR_STRENGTH,
SDK_PROCAMP_MOTION_BLUR_TOPIC_END,

// ... more parameters ...
```

## Important Rules

### 1. SDK_PROCAMP_INPUT Must Be First
The input layer parameter must always be at index 0:
```cpp
static const int PARAM_DISPLAY_ORDER[] = {
    SDK_PROCAMP_INPUT,  // ← MUST BE FIRST
    // ... rest of parameters
};
```

### 2. Topic Groups Must Stay Together
When reordering, a topic's HEADER, content parameters, and TOPIC_END must move as a unit:

**✅ Correct:**
```cpp
SDK_PROCAMP_SHADOW_HEADER,      // Start of group
SDK_PROCAMP_SHADOW_ENABLE,      // Content
SDK_PROCAMP_SHADOW_COLOR,       // Content
SDK_PROCAMP_SHADOW_OFFSET_X,    // Content
SDK_PROCAMP_SHADOW_OFFSET_Y,    // Content
SDK_PROCAMP_SHADOW_OPACITY,     // Content
SDK_PROCAMP_SHADOW_TOPIC_END,   // End of group
```

**❌ Incorrect:**
```cpp
SDK_PROCAMP_SHADOW_HEADER,
SDK_PROCAMP_SHADOW_ENABLE,
// ❌ Missing other shadow parameters
SDK_PROCAMP_SHADOW_TOPIC_END,
```

### 3. Include All Parameters Exactly Once
The compile-time validation ensures all parameters are included:
```cpp
static_assert(sizeof(PARAM_DISPLAY_ORDER) / sizeof(PARAM_DISPLAY_ORDER[0]) == SDK_PROCAMP_NUM_PARAMS,
              "PARAM_DISPLAY_ORDER must contain exactly SDK_PROCAMP_NUM_PARAMS entries");
```

If you forget a parameter or include it twice, you'll get a compile error.

## Testing Your Changes

### Recommended Test Procedure

1. **Compile Test**
   ```bash
   # Build the project
   # If it compiles successfully, the array is complete and valid
   ```

2. **Visual Test**
   - Open Adobe Premiere Pro
   - Apply the Windy Lines effect to a clip
   - Check the Effects Control panel
   - Verify parameters appear in the new order

3. **Backwards Compatibility Test**
   - Open an existing project file that uses Windy Lines
   - Verify all parameter values are preserved
   - Verify the effect renders correctly

4. **Interaction Test**
   - Change various parameter values
   - Verify the effect responds correctly
   - Verify preset application still works

## Benefits of This System

### ✅ Centralized Management
- All display order logic in one file
- Easy to visualize entire parameter structure
- Single source of truth

### ✅ Safety
- Compile-time validation prevents missing parameters
- Enum IDs never change (backwards compatibility)
- Parameter access code unchanged

### ✅ Maintainability
- Adding new parameters: Just add to array
- Reordering: Simple cut-and-paste
- Clear documentation of structure

### ✅ Flexibility
- Display order independent from logical grouping
- Can experiment with different arrangements
- Easy to A/B test different organizations

## Advanced: Adding New Parameters

When adding a new parameter:

1. **Add enum ID** to `SDK_ProcAmp.h`
   ```cpp
   enum {
       // ... existing parameters ...
       SDK_PROCAMP_NEW_PARAMETER,  // New ID assigned automatically
       SDK_PROCAMP_NUM_PARAMS
   };
   ```

2. **Add to display order** in `SDK_ProcAmp_ParamOrder.h`
   ```cpp
   static const int PARAM_DISPLAY_ORDER[] = {
       // ... other parameters ...
       SDK_PROCAMP_NEW_PARAMETER,  // Insert at desired position
       // ... more parameters ...
   };
   ```

3. **Add registration code** to `ParamsSetup()` (following existing patterns)

4. **Implement rendering logic** in render functions

## Troubleshooting

### Compile Error: "PARAM_DISPLAY_ORDER must contain exactly SDK_PROCAMP_NUM_PARAMS entries"

**Cause:** Array size mismatch
**Solution:** 
- Count parameters in enum (SDK_ProcAmp.h)
- Count entries in PARAM_DISPLAY_ORDER array
- Find missing or duplicate entry

### Parameters Don't Appear in New Order

**Cause:** ParamsSetup not using PARAM_DISPLAY_ORDER yet
**Solution:** Ensure ParamsSetup function is updated to use the array (implementation TBD)

### Old Project Files Show Wrong Values

**Cause:** Enum IDs were changed
**Solution:** Never change enum values! Only change display order array.

## Future Enhancements

Potential improvements to this system:

1. **Automated ParamsSetup Generation**
   - Script to generate ParamsSetup from declarative definition
   - Eliminates manual PF_ADD_* calls

2. **Topic Validation**
   - Compile-time check for matching HEADER/TOPIC_END pairs
   - Runtime validation of topic structure

3. **Parameter Metadata System**
   - Single definition for each parameter (name, type, range, group)
   - Reduces code duplication

4. **Visual Parameter Editor**
   - GUI tool to reorder parameters
   - Drag-and-drop interface

## References

- **SDK_ProcAmp.h** - Parameter enum definitions
- **SDK_ProcAmp_ParamOrder.h** - Display order array (this system)
- **SDK_ProcAmp_CPU.cpp** - ParamsSetup function
- **SDK_ProcAmp_Notes.json** - Project documentation and history
- **Adobe Premiere Pro SDK Documentation** - Parameter system details

---

**Version:** 1.0  
**Date:** 2026-02-07  
**Author:** Parameter Reordering System Implementation
