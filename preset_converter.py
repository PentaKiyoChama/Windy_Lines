#!/usr/bin/env python3
"""
Preset Converter: TSV to C++ Effect Preset Array
Usage: python preset_converter.py presets.tsv
Output: C++ code for SDK_ProcAmp.h
"""

import sys
import csv

def parse_tsv(filepath):
    """Parse TSV file and return list of preset dictionaries"""
    presets = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            presets.append(row)
    return presets

def format_preset_cpp(preset):
    """Convert a preset dictionary to C++ struct initializer"""
    # Parse values
    name = preset['name']
    count = int(float(preset['count']))
    lifetime = float(preset['lifetime'])
    travel = float(preset['travel'])
    thickness = float(preset['thickness'])
    length = float(preset['length'])
    angle = float(preset['angle'])
    tailFade = float(preset['tailFade'])
    aa = float(preset['aa'])
    originMode = int(float(preset['originMode']))
    spawnScaleX = float(preset['spawnScaleX'])
    spawnScaleY = float(preset['spawnScaleY'])
    originOffsetX = float(preset['originOffsetX'])
    originOffsetY = float(preset['originOffsetY'])
    interval = float(preset['interval'])
    animPattern = int(float(preset['animPattern']))
    centerGap = float(preset['centerGap'])
    easing = int(float(preset['easing']))
    startTime = float(preset['startTime'])
    duration = float(preset['duration'])
    blendMode = int(float(preset['blendMode']))
    depthStrength = float(preset['depthStrength'])
    lineCap = int(float(preset['lineCap']))
    colorMode = int(float(preset['colorMode']))
    colorPreset = int(float(preset['colorPreset']))
    spawnSource = int(float(preset['spawnSource']))
    hideElement = int(float(preset['hideElement'])) != 0
    
    # Format as C++ struct initializer
    cpp = f'\t// {name}\n'
    cpp += f'\t{{ "{name}",\n'
    cpp += f'\t  {count}, {lifetime}f, {travel}f,\n'
    cpp += f'\t  {thickness}f, {length}f, {angle}f, {tailFade}f, {aa}f,\n'
    cpp += f'\t  {originMode}, {spawnScaleX}f, {spawnScaleY}f, {originOffsetX}f, {originOffsetY}f, {interval}f,\n'
    cpp += f'\t  {animPattern}, {centerGap}f, {easing}, {startTime}f, {duration}f,\n'
    cpp += f'\t  {blendMode}, {depthStrength}f,\n'
    cpp += f'\t  {lineCap}, {colorMode}, {colorPreset}, {spawnSource}, {"true" if hideElement else "false"}\n'
    cpp += '\t}'
    
    return cpp

def generate_cpp_array(presets):
    """Generate complete C++ array code"""
    cpp = 'static const EffectPreset kEffectPresets[] =\n{\n'
    
    preset_codes = []
    for preset in presets:
        preset_codes.append(format_preset_cpp(preset))
    
    cpp += ',\n'.join(preset_codes)
    cpp += '\n};\n'
    cpp += f'\nstatic const int kEffectPresetCount = static_cast<int>(sizeof(kEffectPresets) / sizeof(kEffectPresets[0]));\n'
    
    return cpp

def main():
    if len(sys.argv) < 2:
        print("Usage: python preset_converter.py presets.tsv")
        print("\nExpected TSV format (with header row):")
        print("name\tcount\tlifetime\ttravel\tthickness\tlength\tangle\ttailFade\taa\t...")
        sys.exit(1)
    
    tsv_file = sys.argv[1]
    
    try:
        presets = parse_tsv(tsv_file)
        cpp_code = generate_cpp_array(presets)
        
        # Output to stdout
        print("// Generated C++ code - Copy this into SDK_ProcAmp.h")
        print("// Replace the existing kEffectPresets array\n")
        print(cpp_code)
        
        # Also save to file
        output_file = 'presets_generated.cpp'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cpp_code)
        print(f"\n// Code also saved to: {output_file}")
        print(f"// Total presets: {len(presets)}")
        
    except FileNotFoundError:
        print(f"Error: File '{tsv_file}' not found")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing column in TSV: {e}")
        print("Make sure the TSV has all required columns with correct names")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
