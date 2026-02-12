#!/usr/bin/env python3
"""
Extract Color Presets from OST_WindyLines.h to TSV

This script extracts the hardcoded color preset definitions from OST_WindyLines.h
and generates a color_presets.tsv file. This is a one-time migration tool to
convert the existing hardcoded presets to the new TSV-based system.

Usage: python extract_color_presets.py
Output: color_presets.tsv
"""

import re
import csv

# Mapping of C++ preset names to IDs and Japanese names
PRESET_MAPPING = [
    (1, 'kRainbow', 'レインボー', 'Rainbow'),
    (2, 'kPastel', 'パステルレインボー', 'RainbowPastel'),
    (3, 'kForest', '森', 'Forest'),
    (4, 'kCyber', 'サイバー', 'Cyber'),
    (5, 'kHazard', '警告', 'Hazard'),
    (6, 'kSakura', '桜', 'Sakura'),
    (7, 'kDesert', '砂漠', 'Desert'),
    (8, 'kStarDust', '星屑', 'StarDust'),
    (9, 'kWakaba', '若葉', 'Wakaba'),
    (10, 'kDangerZone', '危険地帯', 'DangerZone'),
    (11, 'kYoen', '妖艶', 'Yoen'),
    (12, 'kSokai', '爽快', 'Sokai'),
    (13, 'kDreamy', '夢幻の風', 'DreamyWind'),
    (14, 'kSunset', '夕焼け', 'Sunset'),
    (15, 'kOcean', '海', 'Ocean'),
    (16, 'kAutumn', '秋', 'Autumn'),
    (17, 'kSnow', '雪', 'Snow'),
    (18, 'kDeepSea', '深海', 'DeepSea'),
    (19, 'kMorningDew', '朝露', 'MorningDew'),
    (20, 'kNightSky', '夜空', 'NightSky'),
    (21, 'kFlame', '炎', 'Flame'),
    (22, 'kEarth', '大地', 'Earth'),
    (23, 'kJewel', '宝石', 'Jewel'),
    (24, 'kPastel2', 'パステル2', 'Pastel2'),
    (25, 'kCityNight', '夜の街', 'CityNight'),
    (26, 'kMoonlight', '月光', 'Moonlight'),
    (27, 'kDazzlingLight', '眩光', 'DazzlingLight'),
    (28, 'kNeonBlast', 'ネオンブラスト', 'NeonBlast'),
    (29, 'kToxicSwamp', '毒沼', 'ToxicSwamp'),
    (30, 'kCosmicStorm', '宇宙嵐', 'CosmicStorm'),
    (31, 'kLavaFlow', '溶岩流', 'LavaFlow'),
    (32, 'kGold', '金', 'Gold'),
    (33, 'kMonochrome', 'モノクロ', 'Monochrome'),
]

def extract_preset_colors(header_content, preset_cpp_name):
    """
    Extract 8 colors from a preset definition in the header file
    
    Example:
    const PresetColor kRainbow[8] = {
        {255, 255, 0, 0}, {255, 255, 128, 0}, ...
    };
    """
    # Pattern to match the preset array definition
    pattern = rf'const\s+PresetColor\s+{re.escape(preset_cpp_name)}\s*\[8\]\s*=\s*\{{([^}}]+)\}}'
    
    match = re.search(pattern, header_content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find preset: {preset_cpp_name}")
    
    color_data = match.group(1)
    
    # Extract all {a, r, g, b} patterns
    color_pattern = r'\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}'
    colors = re.findall(color_pattern, color_data)
    
    if len(colors) != 8:
        raise ValueError(f"Expected 8 colors for {preset_cpp_name}, found {len(colors)}")
    
    # Convert to list of tuples (a, r, g, b)
    return [(int(a), int(r), int(g), int(b)) for a, r, g, b in colors]

def main():
    # Read OST_WindyLines.h
    header_path = 'OST_WindyLines.h'
    try:
        with open(header_path, 'r', encoding='utf-8') as f:
            header_content = f.read()
    except FileNotFoundError:
        print(f"✗ Error: {header_path} not found")
        print("  Please run this script from the repository root directory")
        return 1
    
    # Extract all presets
    extracted_presets = []
    
    print(f"Extracting color presets from {header_path}...")
    
    for preset_id, cpp_name, name_jp, name_en in PRESET_MAPPING:
        try:
            colors = extract_preset_colors(header_content, cpp_name)
            extracted_presets.append({
                'id': preset_id,
                'name': name_jp,
                'name_en': name_en,
                'colors': colors
            })
            print(f"  ✓ [{preset_id:2d}] {name_jp:12s} ({name_en})")
        except Exception as e:
            print(f"  ✗ [{preset_id:2d}] {name_jp:12s} ({name_en}) - Error: {e}")
    
    # Write to TSV
    output_file = 'color_presets.tsv'
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Header row
        header = ['id', 'name', 'name_en'] + [f'color{i}' for i in range(1, 9)]
        writer.writerow(header)
        
        # Data rows
        for preset in extracted_presets:
            row = [
                preset['id'],
                preset['name'],
                preset['name_en']
            ]
            # Add colors in "a,r,g,b" format
            for a, r, g, b in preset['colors']:
                row.append(f'{a},{r},{g},{b}')
            writer.writerow(row)
    
    print(f"\n✓ Successfully extracted {len(extracted_presets)} presets")
    print(f"✓ Written to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review {output_file} to ensure all data is correct")
    print(f"  2. Run: python color_preset_converter.py")
    print(f"  3. Replace the hardcoded presets in OST_WindyLines.h with:")
    print(f"     #include \"OST_WindyLines_ColorPresets.h\"")
    
    return 0

if __name__ == '__main__':
    exit(main())
