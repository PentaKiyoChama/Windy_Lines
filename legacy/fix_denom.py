#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix remaining denom replacements"""

filepath = r"c:\Users\Owner\Desktop\Premiere_Pro_24.0_C_Win_SDK\Premiere_Pro_24.0_C++_Win_SDK\Premiere_Pro_24.0_SDK\Examples\Projects\GPUVideoFilter\Windy_Lines\OST_WindyLines_CPU.cpp"

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix 1: Motion blur denom declaration (line 1981, 0-indexed 1980)
if 'const float denom = (2.0f * ld.halfLen) > 0.0001f ? (2.0f * ld.halfLen) : 0.0001f;' in lines[1980]:
    lines[1980] = ''  # Remove the line
    print("Removed motion blur denom declaration at line 1981")

# Fix 2: Motion blur tailT usage (line 2015, 0-indexed 2014) - already done
# Check if already fixed
if '/ denom);' in lines[2014]:
    lines[2014] = lines[2014].replace('/ denom);', '* ld.invDenom);  // Use pre-computed')
    print("Fixed motion blur tailT at line 2015")

# Fix 3: Non-blur denom declaration (line 2056, 0-indexed 2055)
if 'const float denom = (2.0f * ld.halfLen) > 0.0001f ? (2.0f * ld.halfLen) : 0.0001f;' in lines[2055]:
    lines[2055] = ''  # Remove the line
    print("Removed non-blur denom declaration at line 2056")

# Fix 4: Non-blur tailT usage (line 2057, 0-indexed 2056)
if '/ denom);' in lines[2056]:
    lines[2056] = lines[2056].replace('/ denom);', '* ld.invDenom);  // Use pre-computed')
    print("Fixed non-blur tailT at line 2057")

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Done!")
