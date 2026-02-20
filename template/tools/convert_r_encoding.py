#!/usr/bin/env python3
"""
.r ファイルのエンコーディング変換ツール

PiPLリソースの .r ファイルは Rez (Mac) で処理する際に Shift-JIS が必要。
編集時は UTF-8 が便利なため、相互変換ツールを用意。

使い方:
    python convert_r_encoding.py --to-sjis __TPL_MATCH_NAME__.r
    python convert_r_encoding.py --to-utf8 __TPL_MATCH_NAME__.r
    python convert_r_encoding.py --check __TPL_MATCH_NAME__.r
"""

import argparse
import sys
import os


def detect_encoding(filepath):
    """ファイルのエンコーディングを推定"""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    try:
        data.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        pass
    
    try:
        data.decode('shift_jis')
        return 'shift_jis'
    except UnicodeDecodeError:
        pass
    
    return 'unknown'


def convert(filepath, from_enc, to_enc):
    """エンコーディング変換"""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    text = data.decode(from_enc)
    
    with open(filepath, 'wb') as f:
        f.write(text.encode(to_enc))
    
    print(f"Converted: {filepath} ({from_enc} → {to_enc})")


def main():
    parser = argparse.ArgumentParser(description='.r file encoding converter')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to-sjis', action='store_true', help='Convert to Shift-JIS (for Rez)')
    group.add_argument('--to-utf8', action='store_true', help='Convert to UTF-8 (for editing)')
    group.add_argument('--check', action='store_true', help='Check current encoding')
    parser.add_argument('file', help='Target .r file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: {args.file} not found")
        sys.exit(1)
    
    current = detect_encoding(args.file)
    
    if args.check:
        print(f"{args.file}: {current}")
        return
    
    if args.to_sjis:
        if current == 'shift_jis':
            print(f"{args.file} is already Shift-JIS")
        else:
            convert(args.file, current, 'shift_jis')
    
    if args.to_utf8:
        if current == 'utf-8':
            print(f"{args.file} is already UTF-8")
        else:
            convert(args.file, current, 'utf-8')


if __name__ == '__main__':
    main()
