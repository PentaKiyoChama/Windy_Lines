#!/usr/bin/env python3
"""Fix AA=0 optimization for SDK_ProcAmp.cl"""

import os

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
cl_file = os.path.join(script_dir, 'SDK_ProcAmp.cl')

# Read file
with open(cl_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern 1: Line coverage calculation (around line 527-530)
old_pattern1 = '''					if (aa > 0.0f) {
						float tt = fmin(fmax((dist - aa) / (0.0f - aa), 0.0f), 1.0f);
						coverage = tt * tt * (3.0f - 2.0f * tt) * tailFade * depthAlpha;
					}
					
					// Shadow: same calculation with offset position'''

new_pattern1 = '''					if (aa > 0.0f) {
						float tt = fmin(fmax((dist - aa) / (0.0f - aa), 0.0f), 1.0f);
						coverage = tt * tt * (3.0f - 2.0f * tt) * tailFade * depthAlpha;
					} else {
						// No anti-aliasing: simple distance test (optimized)
						coverage = (dist <= 0.0f) ? (tailFade * depthAlpha) : 0.0f;
					}
					
					// Shadow: same calculation with offset position'''

# Pattern 2: Shadow coverage calculation (around line 555-559)
old_pattern2 = '''						if (aa > 0.0f) {
							float tt = fmin(fmax((sdist - aa) / (0.0f - aa), 0.0f), 1.0f);
							shadowCoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * depthAlpha;
						}
					}
				
					// Draw shadow first (before line) using the same coverage calculation'''

new_pattern2 = '''						if (aa > 0.0f) {
							float tt = fmin(fmax((sdist - aa) / (0.0f - aa), 0.0f), 1.0f);
							shadowCoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * depthAlpha;
						} else {
							// No anti-aliasing: simple distance test (optimized)
							shadowCoverage = (sdist <= 0.0f) ? (stailFade * depthAlpha) : 0.0f;
						}
					}
				
					// Draw shadow first (before line) using the same coverage calculation'''

# Apply replacements
modified = False
if old_pattern1 in content:
    content = content.replace(old_pattern1, new_pattern1)
    print("✓ Pattern 1 replaced (line coverage)")
    modified = True
else:
    print("✗ Pattern 1 not found")

if old_pattern2 in content:
    content = content.replace(old_pattern2, new_pattern2)
    print("✓ Pattern 2 replaced (shadow coverage)")
    modified = True
else:
    print("✗ Pattern 2 not found")

# Write back
if modified:
    with open(cl_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nFile updated: {cl_file}")
else:
    print("\nNo changes made")
