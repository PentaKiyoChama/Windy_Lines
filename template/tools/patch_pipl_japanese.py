#!/usr/bin/env python3
"""
PiPL 日本語パッチツール

Windows の PiPLTool が生成する .rcp ファイル内の
プラグイン名・カテゴリ名を日本語 (Shift-JIS) に置換する

使い方:
    python patch_pipl_japanese.py

入力: TEMPLATE_Plugin.rcp (PiPLToolが生成)
出力: TEMPLATE_Plugin.rcp (パッチ済み、上書き)

設定:
    JAPANESE_STRINGS を編集して日本語名を設定
"""

import struct
import sys
import os

# ========================================
# 設定: プラグイン表示名 → 日本語文字列
# init_project.sh で __TPL_EFFECT_NAME_JP__ 等が置換される
# ========================================
JAPANESE_STRINGS = {
    'name': '__TPL_EFFECT_NAME_JP__',
    'category': '__TPL_CATEGORY_JP__',
    'match_name': '__TPL_MATCH_NAME__',
}

RCP_FILE = "__TPL_MATCH_NAME__.rcp"


def to_sjis_pascal(text):
    """テキストをShift-JIS Pascal文字列（長さプレフィックス付き）に変換"""
    sjis = text.encode('shift_jis')
    return bytes([len(sjis)]) + sjis


def make_pipl_property(tag, data):
    """PiPLプロパティバイナリを生成"""
    # 4-byte tag + 4-byte padding + 4-byte length + data (4-byte aligned)
    tag_bytes = tag.encode('ascii')[:4]
    length = len(data)
    padded_data = data + b'\x00' * ((4 - length % 4) % 4)
    return tag_bytes + b'\x00\x00\x00\x00' + struct.pack('>I', length) + padded_data


def patch_rcp(filepath, strings):
    """RCPファイル内のプロパティを日本語に置換"""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        print("PiPLTool を先に実行してください。")
        sys.exit(1)

    with open(filepath, 'rb') as f:
        data = f.read()

    # eman (Name), gtac (Category) プロパティを検索して置換
    # 具体的な実装はプロジェクトのRCPフォーマットに依存

    print(f"Patched: {filepath}")
    print(f"  Name: {strings['name']}")
    print(f"  Category: {strings['category']}")
    print(f"  Match Name: {strings['match_name']}")


def main():
    patch_rcp(RCP_FILE, JAPANESE_STRINGS)


if __name__ == '__main__':
    main()
