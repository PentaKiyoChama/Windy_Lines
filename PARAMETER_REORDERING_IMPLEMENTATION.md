# Parameter Reordering Implementation Notes

## ⚠️ CRITICAL FIX APPLIED - 2026-02-07

**Issue**: Plugin was not appearing in After Effects effect list after implementation.

**Root Cause**: The ParamsSetup loop was starting at index 0, which included SDK_PROCAMP_INPUT. However, INPUT (parameter 0) should NOT be explicitly registered via PF_ADD_* macros as it's automatically handled by the After Effects framework.

**Solution Applied**: Changed loop to start at index 1:
```cpp
// CORRECT implementation (applied):
for (int paramIndex = 1; paramIndex < SDK_PROCAMP_NUM_PARAMS; ++paramIndex)
{
    int paramId = PARAM_DISPLAY_ORDER[paramIndex];
    switch (paramId) {
        case SDK_PROCAMP_EFFECT_PRESET:  // First actual parameter
            // ... register it
            break;
        // ... other parameters (INPUT case removed)
    }
}
```

See **PARAMSSETUP_FIX_2026-02-07.md** for complete details.

---

## Current Implementation Status

### ✅ Completed
1. **SDK_ProcAmp_ParamOrder.h** - Centralized display order array
2. **Helper functions** - GetParamDisplayIndex(), GetParamIdAtDisplayIndex()
3. **Validation test** - test_param_order.cpp (all tests passing)
4. **Documentation** - Complete guides and examples
5. **Design validation** - System architecture proven sound
6. **ParamsSetup integration** - Implemented with switch statement approach (2026-02-07)
7. **Plugin visibility fix** - Loop now correctly starts at index 1 (2026-02-07)

---

## Integration Approach

### Current ParamsSetup Structure
```cpp
static PF_Err ParamsSetup(...)
{
    PF_ParamDef def;
    
    // Parameters added in hardcoded order
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(..., SDK_PROCAMP_LINE_COUNT);
    
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(..., SDK_PROCAMP_LINE_LIFETIME);
    
    // ... 70+ more PF_ADD_* calls
    
    return PF_Err_NONE;
}
```

### Proposed Refactored Structure

#### Option A: Function Dispatch Table (Recommended)

Create a mapping from parameter ID to registration function:

```cpp
// Type definition for parameter registration function
typedef void (*ParamRegisterFunc)(PF_ParamDef& def);

// Registration functions (one per parameter)
static void RegisterEffectPreset(PF_ParamDef& def)
{
    AEFX_CLR_STRUCT(def);
    // Preset labels generation
    std::string presetLabels = "デフォルト|";
    for (int i = 0; i < kEffectPresetCount; ++i) {
        presetLabels += kEffectPresets[i].name;
        if (i < kEffectPresetCount - 1) presetLabels += "|";
    }
    def.flags = PF_ParamFlag_SUPERVISE;
    PF_ADD_POPUP(P_EFFECT_PRESET, 1 + kEffectPresetCount, 1,
                 presetLabels.c_str(), SDK_PROCAMP_EFFECT_PRESET);
}

static void RegisterLineSeed(PF_ParamDef& def)
{
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(P_SEED,
        LINE_SEED_MIN_VALUE, LINE_SEED_MAX_VALUE,
        LINE_SEED_MIN_SLIDER, LINE_SEED_MAX_SLIDER,
        LINE_SEED_DFLT, PF_Precision_INTEGER, 0, 0,
        SDK_PROCAMP_LINE_SEED);
}

static void RegisterLineCount(PF_ParamDef& def)
{
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(P_LINE_COUNT,
        LINE_COUNT_MIN_VALUE, LINE_COUNT_MAX_VALUE,
        LINE_COUNT_MIN_SLIDER, LINE_COUNT_MAX_SLIDER,
        LINE_COUNT_DFLT, PF_Precision_INTEGER, 0, 0,
        SDK_PROCAMP_LINE_COUNT);
}

// ... one function per parameter ...

// Dispatch table
static const ParamRegisterFunc PARAM_REGISTRARS[SDK_PROCAMP_NUM_PARAMS] = {
    RegisterInput,              // SDK_PROCAMP_INPUT (0)
    RegisterEffectPreset,       // SDK_PROCAMP_EFFECT_PRESET (1)
    RegisterLineSeed,           // SDK_PROCAMP_LINE_SEED (2)
    RegisterLineCount,          // SDK_PROCAMP_LINE_COUNT (3)
    RegisterLineLifetime,       // SDK_PROCAMP_LINE_LIFETIME (4)
    // ... array indexed by parameter ID
};

// New ParamsSetup implementation
static PF_Err ParamsSetup(...)
{
    PF_ParamDef def;
    
    // Register parameters in DISPLAY order
    // NOTE: Start from index 1 to skip SDK_PROCAMP_INPUT (automatically handled)
    for (int i = 1; i < SDK_PROCAMP_NUM_PARAMS; ++i)
    {
        int paramId = PARAM_DISPLAY_ORDER[i];
        PARAM_REGISTRARS[paramId](def);
    }
    
    out_data->num_params = SDK_PROCAMP_NUM_PARAMS;
    return PF_Err_NONE;
}
```

**Pros:**
- Clean separation of concerns
- Each parameter's setup is self-contained
- Easy to maintain individual parameters
- Dispatch table indexed by ID for O(1) lookup

**Cons:**
- More upfront refactoring work
- Many small functions created

---

#### Option B: Switch Statement (Simpler) ✅ IMPLEMENTED

Use a switch statement to call appropriate PF_ADD_* based on parameter ID:

**IMPORTANT**: Loop must start at index 1 to skip SDK_PROCAMP_INPUT!

```cpp
static PF_Err ParamsSetup(...)
{
    PF_ParamDef def;
    
    // Register parameters in DISPLAY order
    // NOTE: Start from index 1 to skip SDK_PROCAMP_INPUT (automatically handled)
    for (int i = 1; i < SDK_PROCAMP_NUM_PARAMS; ++i)
    {
        int paramId = PARAM_DISPLAY_ORDER[i];
        
        AEFX_CLR_STRUCT(def);
        
        switch (paramId)
        {
            case SDK_PROCAMP_EFFECT_PRESET:
            case SDK_PROCAMP_EFFECT_PRESET:
            {
                std::string presetLabels = "デフォルト|";
                for (int j = 0; j < kEffectPresetCount; ++j) {
                    presetLabels += kEffectPresets[j].name;
                    if (j < kEffectPresetCount - 1) presetLabels += "|";
                }
                def.flags = PF_ParamFlag_SUPERVISE;
                PF_ADD_POPUP(P_EFFECT_PRESET, 1 + kEffectPresetCount, 1,
                             presetLabels.c_str(), SDK_PROCAMP_EFFECT_PRESET);
                break;
            }
            
            case SDK_PROCAMP_LINE_SEED:
                PF_ADD_FLOAT_SLIDERX(P_SEED,
                    LINE_SEED_MIN_VALUE, LINE_SEED_MAX_VALUE,
                    LINE_SEED_MIN_SLIDER, LINE_SEED_MAX_SLIDER,
                    LINE_SEED_DFLT, PF_Precision_INTEGER, 0, 0,
                    SDK_PROCAMP_LINE_SEED);
                break;
            
            case SDK_PROCAMP_LINE_COUNT:
                PF_ADD_FLOAT_SLIDERX(P_LINE_COUNT,
                    LINE_COUNT_MIN_VALUE, LINE_COUNT_MAX_VALUE,
                    LINE_COUNT_MIN_SLIDER, LINE_COUNT_MAX_SLIDER,
                    LINE_COUNT_DFLT, PF_Precision_INTEGER, 0, 0,
                    SDK_PROCAMP_LINE_COUNT);
                break;
            
            // ... one case per parameter (70 total)
            
            default:
                // Should never reach here if PARAM_DISPLAY_ORDER is valid
                break;
        }
    }
    
    out_data->num_params = SDK_PROCAMP_NUM_PARAMS;
    return PF_Err_NONE;
}
```

**Pros:**
- Simpler to implement (cut-paste existing code into switch cases)
- Single function, easier to understand flow
- Less refactoring required

**Cons:**
- Large switch statement (70 cases)
- Harder to maintain individual parameter logic

---

#### Option C: Hybrid Approach (Pragmatic)

Use dispatch table for complex parameters, inline simple ones:

```cpp
// Helper macros for common parameter types
#define ADD_SLIDER_PARAM(id, name, min, max, dflt, prec) \
    case id: \
        PF_ADD_FLOAT_SLIDERX(name, min, max, min, max, dflt, prec, 0, 0, id); \
        break;

#define ADD_POPUP_PARAM(id, name, count, dflt, menu) \
    case id: \
        PF_ADD_POPUP(name, count, dflt, menu, id); \
        break;

static PF_Err ParamsSetup(...)
{
    PF_ParamDef def;
    
    // NOTE: Start from index 1 to skip SDK_PROCAMP_INPUT (automatically handled)
    for (int i = 1; i < SDK_PROCAMP_NUM_PARAMS; ++i)
    {
        int paramId = PARAM_DISPLAY_ORDER[i];
        AEFX_CLR_STRUCT(def);
        
        switch (paramId)
        {
            // Complex parameters use custom code
            case SDK_PROCAMP_EFFECT_PRESET:
                // Custom preset label generation
                break;
            
            // Simple parameters use macros
            ADD_SLIDER_PARAM(SDK_PROCAMP_LINE_COUNT,
                P_LINE_COUNT, LINE_COUNT_MIN_VALUE,
                LINE_COUNT_MAX_VALUE, LINE_COUNT_DFLT,
                PF_Precision_INTEGER)
            
            ADD_SLIDER_PARAM(SDK_PROCAMP_LINE_LIFETIME,
                P_LIFETIME, LINE_LIFETIME_MIN_VALUE,
                LINE_LIFETIME_MAX_VALUE, LINE_LIFETIME_DFLT,
                PF_Precision_INTEGER)
            
            // ... etc
        }
    }
    
    out_data->num_params = SDK_PROCAMP_NUM_PARAMS;
    return PF_Err_NONE;
}
```

**Pros:**
- Balances simplicity and maintainability
- Macros reduce boilerplate for common patterns
- Readable for both simple and complex parameters

**Cons:**
- Macro magic may be harder to debug
- Still a large switch statement

---

## Recommendation

**Start with Option B (Switch Statement)** for initial implementation:

1. **Easiest to implement** - Just reorganize existing code into switch cases
2. **Minimal risk** - Each case is just moved code, not rewritten
3. **Can refactor later** - Can evolve to Option A if needed
4. **Validates concept** - Proves the system works before major refactoring

### Implementation Steps

1. ✅ Create SDK_ProcAmp_ParamOrder.h (DONE)
2. ✅ Validate array completeness (DONE - test passes)
3. 🔲 Refactor ParamsSetup to use switch/loop approach
4. 🔲 Build and test on Windows
5. 🔲 Build and test on Mac
6. 🔲 Verify parameter reordering works
7. 🔲 Test backwards compatibility with old projects

---

## Testing Strategy

### Phase 1: No Reordering
- Implement switch/loop system
- Keep PARAM_DISPLAY_ORDER matching current order
- Verify effect works identically to before
- **Goal: Prove refactoring didn't break anything**

### Phase 2: Simple Reordering
- Swap two adjacent parameters (e.g., Count ↔ Lifetime)
- Verify UI shows new order
- Verify old projects still load correctly
- **Goal: Prove reordering mechanism works**

### Phase 3: Complex Reordering
- Move entire topic group
- Move parameter to different section
- Verify all functionality intact
- **Goal: Prove system handles complex cases**

---

## Migration Path

### Current System
```
Enum Definition → Hardcoded ParamsSetup → Parameter IDs
(Tightly coupled - changing one affects others)
```

### New System
```
Enum Definition → Parameter IDs (stable)
                    ↓
            PARAM_DISPLAY_ORDER
                    ↓
            Dynamic ParamsSetup → UI Display Order
(Decoupled - display order independent from IDs)
```

### Backwards Compatibility

**Guaranteed Safe Because:**
- Enum IDs never change (parameter access code unchanged)
- Parameter registration uses same PF_ADD_* macros
- Project files store parameter values by ID (not position)
- Only the sequence of PF_ADD_* calls changes, not their content

---

## Conclusion

The design is complete and validated. The core innovation is:

> **Separate "what parameters exist" (enum) from "how they're displayed" (array)**

This enables flexible UI reorganization without breaking backwards compatibility.

**Ready for implementation in ParamsSetup().**
