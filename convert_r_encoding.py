#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PiPL .r ファイル エンコーディング変換スクリプト

Xcode の Rez コンパイラは UTF-8 BOM を処理できないため、
.r ファイルのエンコーディングを管理するスクリプトです。

使い方:
  python3 convert_r_encoding.py --to-sjis     # Rez ビルド用 (Shift-JIS化)
  python3 convert_r_encoding.py --to-utf8     # 編集用 (UTF-8化, BOM無し)
  python3 convert_r_encoding.py --check       # 現在のエンコーディング確認
  python3 convert_r_encoding.py --hex-dump    # 日本語文字列のバイト確認

注意:
  - Xcode でビルドする前に --to-sjis を実行してください
  - pbxproj に OTHER_REZFLAGS = "-script Japanese" が必要です
  - UTF-8 でビルドする場合は BOM 無しで、-script Japanese は不要です
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
R_FILE = os.path.join(SCRIPT_DIR, 'OST_WindyLines.r')

# 検索対象の日本語文字列
JAPANESE_STRINGS = ['風を感じるライン', 'おしゃれテロップ']


def detect_encoding(filepath):
    """ファイルのエンコーディングを推定"""
    with open(filepath, 'rb') as f:
        head = f.read(3)

    if head[:3] == b'\xEF\xBB\xBF':
        return 'utf-8-sig'  # UTF-8 with BOM

    # Try UTF-8 first
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Check if any Japanese is present and valid UTF-8
        for s in JAPANESE_STRINGS:
            if s in content:
                return 'utf-8'
        # If no Japanese found, could be ASCII or SJIS
    except UnicodeDecodeError:
        pass

    # Try Shift-JIS
    try:
        with open(filepath, 'r', encoding='shift_jis') as f:
            content = f.read()
        for s in JAPANESE_STRINGS:
            if s in content:
                return 'shift_jis'
    except UnicodeDecodeError:
        pass

    return 'unknown'


def convert_file(filepath, from_enc, to_enc):
    """ファイルのエンコーディングを変換"""
    with open(filepath, 'r', encoding=from_enc) as f:
        content = f.read()

    # 日本語文字列が含まれているか確認
    for s in JAPANESE_STRINGS:
        if s not in content:
            print(f"  警告: '{s}' がファイルに見つかりません", file=sys.stderr)

    with open(filepath, 'w', encoding=to_enc, newline='\n') as f:
        f.write(content)


def hex_dump_strings(filepath):
    """日本語文字列のバイト表現を表示"""
    enc = detect_encoding(filepath)
    if enc == 'unknown':
        print("エンコーディング不明")
        return

    read_enc = enc if enc != 'utf-8-sig' else 'utf-8-sig'
    with open(filepath, 'r', encoding=read_enc) as f:
        content = f.read()

    with open(filepath, 'rb') as f:
        raw = f.read()

    print(f"ファイルエンコーディング: {enc}")
    print()

    for s in JAPANESE_STRINGS:
        if s not in content:
            print(f"'{s}': 未検出")
            continue

        utf8_bytes = s.encode('utf-8')
        sjis_bytes = s.encode('shift_jis')

        print(f"'{s}':")
        print(f"  UTF-8  ({len(utf8_bytes):2d} bytes): {' '.join(f'{b:02X}' for b in utf8_bytes)}")
        print(f"  SJIS   ({len(sjis_bytes):2d} bytes): {' '.join(f'{b:02X}' for b in sjis_bytes)}")

        # Rez hex escape formats
        rez_utf8 = ''.join(f'\\0x{b:02X}' for b in utf8_bytes)
        rez_sjis = ''.join(f'\\0x{b:02X}' for b in sjis_bytes)
        print(f"  Rez UTF-8: \"{rez_utf8}\"")
        print(f"  Rez SJIS : \"{rez_sjis}\"")

        # ファイル内での実際のバイト列を探す
        for enc_name, enc_bytes in [('UTF-8', utf8_bytes), ('SJIS', sjis_bytes)]:
            pos = raw.find(enc_bytes)
            if pos >= 0:
                print(f"  ファイル内 ({enc_name}): offset 0x{pos:04X} で検出")
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]

    if cmd == '--check':
        enc = detect_encoding(R_FILE)
        print(f"エンコーディング: {enc}")
        with open(R_FILE, 'rb') as f:
            head = f.read(10)
        print(f"先頭バイト: {' '.join(f'{b:02X}' for b in head)}")
        return 0

    elif cmd == '--to-sjis':
        enc = detect_encoding(R_FILE)
        if enc == 'shift_jis':
            print("既に Shift-JIS です")
            return 0
        if enc in ('utf-8', 'utf-8-sig'):
            print(f"変換: {enc} → shift_jis")
            convert_file(R_FILE, enc, 'shift_jis')
            print(f"完了: {R_FILE}")
            print("Xcode ビルド設定: OTHER_REZFLAGS = \"-script Japanese\" が必要です")
            return 0
        print(f"エンコーディング不明: {enc}", file=sys.stderr)
        return 1

    elif cmd == '--to-utf8':
        enc = detect_encoding(R_FILE)
        if enc == 'utf-8':
            print("既に UTF-8 (BOM無し) です")
            return 0
        if enc in ('shift_jis', 'utf-8-sig'):
            print(f"変換: {enc} → utf-8 (BOM無し)")
            convert_file(R_FILE, enc, 'utf-8')
            print(f"完了: {R_FILE}")
            print("注意: UTF-8 でビルドする場合は OTHER_REZFLAGS の -script Japanese を外してください")
            return 0
        print(f"エンコーディング不明: {enc}", file=sys.stderr)
        return 1

    elif cmd == '--hex-dump':
        hex_dump_strings(R_FILE)
        return 0

    else:
        print(f"不明なコマンド: {cmd}")
        print(__doc__)
        return 1


if __name__ == '__main__':
    sys.exit(main())
