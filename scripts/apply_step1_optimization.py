#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Step 1 optimization: Pre-compute depthAlpha and invDenom

NOTE: This script contains a hardcoded absolute path that needs to be updated
for your environment. Update the 'filepath' variable below to point to your
SDK_ProcAMP_CPU.cpp file, or run this script from the project root and use:
    filepath = os.path.join(os.path.dirname(__file__), '..', 'SDK_ProcAmp_CPU.cpp')
"""

import re
import os

# TODO: Update this path for your environment or use relative path
filepath = r"c:\Users\Owner\Desktop\Premiere_Pro_24.0_C_Win_SDK\Premiere_Pro_24.0_C++_Win_SDK\Premiere_Pro_24.0_SDK\Examples\Projects\GPUVideoFilter\Windy_Lines\SDK_ProcAmp_CPU.cpp"
# Alternative: Use relative path from script location:
# filepath = os.path.join(os.path.dirname(__file__), '..', 'SDK_ProcAmp_CPU.cpp')

with open(filepath, 'r', encoding='cp932') as f:
    content = f.read()

# Step 1: Add fields to LineDerived struct (around line 166)
old_struct = '''	float lineVelocity;
};'''

new_struct = '''	float lineVelocity;
	float depthAlpha;      // Pre-computed depth fade alpha
	float invDenom;        // Pre-computed 1 / (2.0f * halfLen) for tail fade
};'''

content = content.replace(old_struct, new_struct)
print("✓ Added depthAlpha and invDenom fields to LineDerived struct")

# Step 2: Add pre-computation after line velocity calculation (around line 1670)
old_precomp = '''		ld.lineVelocity = fmaxf(fminf(rnorm, 1.0f), -1.0f);
		allLinesDerived[lineIdx] = ld;'''

new_precomp = '''		ld.lineVelocity = fmaxf(fminf(rnorm, 1.0f), -1.0f);
		
		// Pre-compute expensive per-pixel calculations once per line
		const float depthScale = DepthScale(ld.depth, lineDepthStrength);
		const float depthFadeT = saturate((depthScale - 0.2f) / (0.6f - 0.2f));
		ld.depthAlpha = 0.05f + 0.95f * depthFadeT;
		const float denom = (2.0f * ld.halfLen) > 0.0001f ? (2.0f * ld.halfLen) : 0.0001f;
		ld.invDenom = 1.0f / denom;
		
		allLinesDerived[lineIdx] = ld;'''

content = content.replace(old_precomp, new_precomp)
print("✓ Added pre-computation code after line velocity calculation")

# Step 3: Replace depthAlpha calculation in rendering loop (around line 1813)
old_depth = '''				const float depthScale = DepthScale(ld.depth, lineDepthStrength);
				const float depthFadeT = saturate((depthScale - 0.2f) / (0.6f - 0.2f));
				const float depthAlpha = 0.05f + 0.95f * depthFadeT;

				const float dx = (x + 0.5f) - ld.centerX;
				const float dy = (y + 0.5f) - ld.centerY;
				float coverage = 0.0f;'''

new_depth = '''				const float depthAlpha = ld.depthAlpha;  // Use pre-computed

				const float dx = (x + 0.5f) - ld.centerX;
				const float dy = (y + 0.5f) - ld.centerY;
				float coverage = 0.0f;'''

content = content.replace(old_depth, new_depth)
print("✓ Replaced depthAlpha calculation with pre-computed value")

# Step 4: Replace all sdenom/denom usages in shadow rendering
# Shadow rendering locations (lines ~1826, ~1899, ~1935)
pattern1 = re.compile(
    r'(\t+)const float sdenom = \(2\.0f \* ld\.halfLen\) > 0\.0001f \? \(2\.0f \* ld\.halfLen\) : 0\.0001f;\n'
    r'(\t+)const float stailT = saturate\(\((sp(?:x|xSample)) \+ ld\.halfLen\) / sdenom\);',
    re.MULTILINE
)
replacement1 = r'\1const float stailT = saturate((\3 + ld.halfLen) * ld.invDenom);  // Use pre-computed'
content, n1 = pattern1.subn(replacement1, content)
print(f"✓ Replaced {n1} shadow sdenom calculations with pre-computed invDenom")

# Step 5: Replace denom in motion blur (line ~1981)
# This needs special handling - denom declaration is separate from usage
# First, remove the denom declaration line
content = re.sub(
    r'\t{5}const float denom = \(2\.0f \* ld\.halfLen\) > 0\.0001f \? \(2\.0f \* ld\.halfLen\) : 0\.0001f;\n',
    '',
    content
)
print("✓ Removed motion blur denom declaration")

# Then replace the tailT calculation that uses it
content = re.sub(
    r'(\t+)const float tailT = saturate\(\(pxSample \+ ld\.halfLen\) / denom\);',
    r'\1const float tailT = saturate((pxSample + ld.halfLen) * ld.invDenom);  // Use pre-computed',
    content
)
print("✓ Replaced motion blur tailT calculation")

# Step 6: Replace denom in non-motion-blur rendering (lines ~2056, ~2092)
pattern2 = re.compile(
    r'(\t+)const float denom = \(2\.0f \* ld\.halfLen\) > 0\.0001f \? \(2\.0f \* ld\.halfLen\) : 0\.0001f;\n'
    r'(\t+)const float tailT = saturate\(\(px \+ ld\.halfLen\) / denom\);',
    re.MULTILINE
)
replacement2 = r'\1const float tailT = saturate((px + ld.halfLen) * ld.invDenom);  // Use pre-computed'
content, n2 = pattern2.subn(replacement2, content)
print(f"✓ Replaced {n2} main rendering denom calculations with pre-computed invDenom")

with open(filepath, 'w', encoding='cp932') as f:
    f.write(content)

print("\n✓✓✓ Step 1 optimization complete! ✓✓✓")
print("\nChanges made:")
print("  - Added depthAlpha and invDenom fields to LineDerived struct")
print("  - Added pre-computation during line generation")
print("  - Replaced depthAlpha calculation in rendering loop")
print(f"  - Replaced {n1} shadow rendering sdenom calculations")
print("  - Replaced motion blur denom calculation")
print(f"  - Replaced {n2} main rendering denom calculations")
print("\nTotal replacements:")
print(f"  - sdenom: {n1} locations")
print(f"  - denom: {n2 + 1} locations (motion blur + main rendering)")
