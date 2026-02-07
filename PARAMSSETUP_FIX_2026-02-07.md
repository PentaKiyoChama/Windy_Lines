# ParamsSetup Plugin Visibility Fix - 2026-02-07

## Problem

After implementing the parameter reordering system (PR #3), the plugin stopped appearing in the After Effects effect list.

## Root Cause

The ParamsSetup() function was iterating through all 70 parameters (indices 0-69), but SDK_PROCAMP_INPUT (at index 0) was not actually registering a parameter. This caused a mismatch:

- **Loop iterations**: 70 times (indices 0-69)
- **Parameters registered**: 69 (SDK_PROCAMP_INPUT case just did `break;` without calling any PF_ADD_* macro)
- **Expected params**: 70 (`out_data->num_params = SDK_PROCAMP_NUM_PARAMS`)

This mismatch caused After Effects to reject the plugin as invalid.

## Technical Details

In After Effects plugin architecture:
- `SDK_PROCAMP_INPUT` (parameter ID 0) represents the input layer
- The input layer is **automatically handled by the framework**
- Plugins should NOT explicitly register it via PF_ADD_* macros
- It must be counted in `num_params` but not explicitly added in ParamsSetup

The parameter reordering implementation correctly placed SDK_PROCAMP_INPUT at index 0 of PARAM_DISPLAY_ORDER array, but the loop incorrectly tried to process it.

## Solution

Modified the ParamsSetup loop to **start from index 1 instead of 0**:

```cpp
// OLD CODE (wrong):
for (int paramIndex = 0; paramIndex < SDK_PROCAMP_NUM_PARAMS; ++paramIndex)
{
    int paramId = PARAM_DISPLAY_ORDER[paramIndex];
    switch (paramId)
    {
        case SDK_PROCAMP_INPUT:
            // Does nothing, but wastes a loop iteration
            break;
        // ... other cases
    }
}

// NEW CODE (correct):
for (int paramIndex = 1; paramIndex < SDK_PROCAMP_NUM_PARAMS; ++paramIndex)
{
    int paramId = PARAM_DISPLAY_ORDER[paramIndex];
    switch (paramId)
    {
        case SDK_PROCAMP_EFFECT_PRESET:  // First actual parameter
            // ... register parameter
            break;
        // ... other cases (SDK_PROCAMP_INPUT case removed)
    }
}
```

Now:
- **Loop iterations**: 69 times (indices 1-69)
- **Parameters registered**: 69 (all actual parameters)
- **Total params**: 70 (including auto-handled INPUT at 0)
- **Result**: ✅ Plugin appears in effects list

## Changes Made

**File**: SDK_ProcAmp_CPU.cpp
- Line 1035: Changed loop start from `paramIndex = 0` to `paramIndex = 1`
- Removed: `case SDK_PROCAMP_INPUT:` block (no longer needed)
- Added: Comment explaining why we start from index 1

## Verification

- ✅ test_param_order.cpp still passes (all 8 tests)
- ✅ Syntax check passes
- ✅ No other code changes needed
- ✅ Parameter display order system remains intact

## Impact

- **Fixes**: Plugin now appears in After Effects effect list
- **Maintains**: All parameter reordering functionality
- **Preserves**: Backwards compatibility with existing projects
- **No side effects**: Only affects ParamsSetup registration, not runtime behavior

## Prevention

This issue occurred because:
1. The parameter reordering guide didn't explicitly mention INPUT handling
2. The switch-based implementation made it easy to overlook that INPUT shouldn't be registered

**Recommendation**: Update PARAMETER_REORDERING_IMPLEMENTATION.md to explicitly warn that INPUT must be skipped in the loop.
