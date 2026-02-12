#!/usr/bin/env python3
"""
Debug Marker Sync Tool
Synchronizes ENABLE_DEBUG_RENDER_MARKERS value from OST_WindyLines.h to all GPU files.
Usage: python sync_debug_markers.py
"""

import os
import re

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def extract_debug_marker_value(h_file_path):
    """Extract ENABLE_DEBUG_RENDER_MARKERS value from OST_WindyLines.h"""
    with open(h_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find #define ENABLE_DEBUG_RENDER_MARKERS <value>
    match = re.search(r'#define\s+ENABLE_DEBUG_RENDER_MARKERS\s+(\d+)', content)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("ENABLE_DEBUG_RENDER_MARKERS not found in OST_WindyLines.h")

def update_gpu_file(file_path, value):
    """Update ENABLE_DEBUG_RENDER_MARKERS value in a GPU file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the value while keeping the comment
    pattern = r'(#define\s+ENABLE_DEBUG_RENDER_MARKERS\s+)\d+'
    replacement = rf'\g<1>{value}'
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    script_dir = get_script_dir()
    
    # Files to sync
    h_file = os.path.join(script_dir, 'OST_WindyLines.h')
    gpu_files = [
        os.path.join(script_dir, 'OST_WindyLines.cu'),
        os.path.join(script_dir, 'OST_WindyLines.cl'),
        # HLSL not used for Premiere Pro (After Effects only), moved to legacy/
        # os.path.join(script_dir, 'OST_WindyLines.hlsl'),
    ]
    
    try:
        # Get value from header file
        debug_value = extract_debug_marker_value(h_file)
        print(f"OST_WindyLines.h: ENABLE_DEBUG_RENDER_MARKERS = {debug_value}")
        
        # Update GPU files
        for gpu_file in gpu_files:
            if os.path.exists(gpu_file):
                if update_gpu_file(gpu_file, debug_value):
                    print(f"✓ Updated {os.path.basename(gpu_file)}")
                else:
                    print(f"- No change needed in {os.path.basename(gpu_file)}")
            else:
                print(f"✗ File not found: {gpu_file}")
        
        print(f"\nSync completed! All files now have ENABLE_DEBUG_RENDER_MARKERS = {debug_value}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())