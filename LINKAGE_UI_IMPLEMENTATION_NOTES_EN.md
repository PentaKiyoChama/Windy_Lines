# Linkage UI Reorganization - Implementation Notes

## Overview

This PR reorganizes the linkage settings UI to improve user experience by placing linkage controls directly above their corresponding parameters and implementing conditional visibility.

## What Changed

### Before
```
Basic Settings:
  - Line Count
  - Lifetime
  - Interval
  - Travel Distance
  - Easing

Appearance:
  - Thickness
  - Length
  - Angle
  ...

Linkage Settings (separate topic at bottom):
  - Length Linkage
  - Length Linkage Rate (%)
  - Thickness Linkage  
  - Thickness Linkage Rate (%)
  - Travel Linkage
  - Travel Linkage Rate (%)
```

### After
```
Basic Settings:
  - Line Count
  - Lifetime
  - Interval
  - Travel Linkage ← MOVED HERE
  - Travel Linkage Rate (%) ← conditionally visible
  - Travel Distance ← conditionally visible
  - Easing

Appearance:
  - Thickness Linkage ← MOVED HERE
  - Thickness Linkage Rate (%) ← conditionally visible
  - Thickness ← conditionally visible
  - Length Linkage ← MOVED HERE
  - Length Linkage Rate (%) ← conditionally visible
  - Length ← conditionally visible
  - Angle
  ...

Linkage Settings topic: REMOVED
```

## Conditional Visibility Logic

| Linkage Mode | Actual Value Parameter | Linkage Rate (%) |
|--------------|------------------------|------------------|
| Off (1)      | Visible ✓             | Hidden ✗         |
| Width (2)    | Hidden ✗              | Visible ✓        |
| Height (3)   | Hidden ✗              | Visible ✓        |

### Example User Flow

**Setting Thickness to link with element width:**
1. User changes "Thickness Linkage" from "Off" → "Element Width"
2. UI automatically:
   - Hides "Thickness (px)" parameter
   - Shows "Thickness Linkage Rate (%)" parameter
3. User adjusts "Thickness Linkage Rate (%)" to 5% (= 5% of element width)

**Reverting to fixed thickness:**
1. User changes "Thickness Linkage" from "Element Width" → "Off"
2. UI automatically:
   - Hides "Thickness Linkage Rate (%)" parameter  
   - Shows "Thickness (px)" parameter
3. User adjusts "Thickness (px)" to fixed value like 3.0px

## Technical Implementation

### Files Modified

1. **SDK_ProcAmp.h**
   - Updated parameter enum indices
   - Moved linkage parameters to new positions
   - Removed old linkage topic group definitions

2. **SDK_ProcAmp_CPU.cpp**
   - Reorganized `ParamsSetup()` function to add linkage parameters before their corresponding value parameters
   - Added conditional visibility logic to `UpdatePseudoGroupVisibility()`
   - Added change handlers in `PF_Cmd_USER_CHANGED_PARAM` for linkage parameters
   - Set `PF_ParamFlag_SUPERVISE` flag on linkage popup parameters

### Key Code Changes

```cpp
// UpdatePseudoGroupVisibility() - Added conditional visibility
const int thicknessLinkage = params[SDK_PROCAMP_THICKNESS_LINKAGE]->u.pd.value;
const bool thicknessIsLinked = (thicknessLinkage == 2 || thicknessLinkage == 3);
setVisible(SDK_PROCAMP_LINE_THICKNESS, !thicknessIsLinked);
setVisible(SDK_PROCAMP_THICKNESS_LINKAGE_RATE, thicknessIsLinked);
```

```cpp
// PF_Cmd_USER_CHANGED_PARAM - Added change detection
if (changedExtra && (changedExtra->param_index == SDK_PROCAMP_THICKNESS_LINKAGE ||
                     changedExtra->param_index == SDK_PROCAMP_LENGTH_LINKAGE ||
                     changedExtra->param_index == SDK_PROCAMP_TRAVEL_LINKAGE))
{
    UpdatePseudoGroupVisibility(in_data, params);
    out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
}
```

## Testing Requirements

### Critical Tests

1. **Parameter Order Verification**
   - [ ] Load plugin in After Effects/Premiere Pro
   - [ ] Verify linkage controls appear directly above their corresponding parameters
   - [ ] Verify old "Linkage Settings" section is gone

2. **Visibility Toggle Tests**
   - [ ] Test Thickness: Off → Width → Height → Off
   - [ ] Test Length: Off → Width → Height → Off  
   - [ ] Test Travel: Off → Width → Height → Off
   - [ ] Verify correct parameters show/hide in each state

3. **Rendering Functionality**
   - [ ] Verify linked mode renders correctly (values scale with element size)
   - [ ] Verify fixed mode renders correctly (values stay constant)
   - [ ] Verify linkage rate changes reflect immediately

4. **Backward Compatibility**
   - [ ] Open projects created with old version
   - [ ] Verify parameter values are preserved
   - [ ] Verify rendering output is identical

### Edge Cases

- [ ] Rapid linkage mode switching
- [ ] Linkage changes during keyframe animation
- [ ] Preset application
- [ ] Copy/paste effects
- [ ] Undo/redo operations

## Build Instructions

**Note:** This project requires the Adobe After Effects SDK to build.

### Windows
```bash
cd Win
# Open SDK_ProcAmp.sln in Visual Studio
# Build solution
```

### Mac
```bash
cd Mac
# Open SDK_ProcAmp.xcodeproj in Xcode
# Build project
```

## Known Issues / Limitations

1. **Breaking Change**: Parameter indices have changed. While name-based parameter mapping should maintain compatibility, thorough testing is required.

2. **SDK Dependency**: Building requires Adobe AE SDK which is not included in this repository.

## Documentation

See `LINKAGE_UI_IMPLEMENTATION_MEMO.md` for detailed Japanese documentation including:
- Complete parameter index mapping
- Technical implementation details
- Troubleshooting guide
- Future enhancement ideas

## Questions?

If you encounter issues or have questions about the implementation, please refer to:
- `LINKAGE_UI_IMPLEMENTATION_MEMO.md` (detailed Japanese documentation)
- `SDK_ProcAmp_Notes.json` (project-wide technical notes)
- `SDK_ProcAmp_DevGuide.md` (development guide)
