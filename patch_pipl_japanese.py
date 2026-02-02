# -*- coding: utf-8 -*-
"""
PiPL日本語パッチスクリプト
PiPLToolが生成した.rcpファイルの文字列を日本語に置換します

使い方:
  python patch_pipl_japanese.py
  
設定は下記のJAPANESE_STRINGS辞書で変更してください
"""

import os
import re

# ===== 設定 =====
# ここで日本語文字列を設定
JAPANESE_STRINGS = {
    'name': 'テストその２流れる線',           # プラグイン名
    'category': 'testestesおしゃれテロップ',  # カテゴリ名
    'match_name': 'OST_WindyLines',      # マッチ名（内部識別子、英語推奨だが日本語も可）
}

# .rcpファイルのパス（このスクリプトと同じフォルダ）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RCP_FILE = os.path.join(SCRIPT_DIR, 'SDK_ProcAmp.rcp')
# ================

def make_pascal_string(text, encoding='shift_jis'):
    """
    Pascal文字列を作成（長さプレフィックス + 文字列 + 4バイト境界パディング）
    """
    encoded = text.encode(encoding)
    length = len(encoded)
    
    # 長さプレフィックス（1バイト）+ 文字列
    total = 1 + length
    
    # 4バイト境界に揃える
    padded_size = ((total + 3) // 4) * 4
    padding_needed = padded_size - total
    
    # 長さプレフィックスを16進エスケープで、文字列はそのまま
    length_prefix = f'\\x{length:02X}'
    padding = '\\0' * padding_needed
    
    return length_prefix, text, padding, padded_size

def patch_rcp_file(rcp_path, strings):
    """
    .rcpファイルの文字列を日本語に置換
    """
    # latin-1で読み込み（バイナリを文字列として扱う）
    with open(rcp_path, 'r', encoding='latin-1') as f:
        content = f.read()
    
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # "eman" (name) プロパティを検出
        if '"eman"' in line:
            result.append(line)
            i += 1
            # RSCS32(0) をそのまま
            if i < len(lines):
                result.append(lines[i])
                i += 1
            # RSCS32(size) を新しいサイズで置換
            if i < len(lines) and 'name' in strings:
                prefix, text, padding, size = make_pascal_string(strings['name'])
                result.append(f'\tRSCS32({size}),')
                i += 1
            # 文字列行を置換
            if i < len(lines) and 'name' in strings:
                prefix, text, padding, size = make_pascal_string(strings['name'])
                result.append(f'\t"{prefix}{text}{padding}", ')
                i += 1
            continue
            
        # "gtac" (category) プロパティを検出
        if '"gtac"' in line:
            result.append(line)
            i += 1
            if i < len(lines):
                result.append(lines[i])
                i += 1
            if i < len(lines) and 'category' in strings:
                prefix, text, padding, size = make_pascal_string(strings['category'])
                result.append(f'\tRSCS32({size}),')
                i += 1
            if i < len(lines) and 'category' in strings:
                prefix, text, padding, size = make_pascal_string(strings['category'])
                result.append(f'\t"{prefix}{text}{padding}", ')
                i += 1
            continue
            
        # "ANMe" (match name) プロパティを検出  
        if '"ANMe"' in line:
            result.append(line)
            i += 1
            if i < len(lines):
                result.append(lines[i])
                i += 1
            if i < len(lines) and 'match_name' in strings:
                prefix, text, padding, size = make_pascal_string(strings['match_name'])
                result.append(f'\tRSCS32({size}),')
                i += 1
            if i < len(lines) and 'match_name' in strings:
                prefix, text, padding, size = make_pascal_string(strings['match_name'])
                result.append(f'\t"{prefix}{text}{padding}", ')
                i += 1
            continue
        
        result.append(line)
        i += 1
    
    # Shift_JISで保存
    with open(rcp_path, 'w', encoding='shift_jis') as f:
        f.write('\n'.join(result))
    
    print(f'パッチ完了: {rcp_path}')
    print(f'  Name: {strings.get("name", "(未設定)")}')
    print(f'  Category: {strings.get("category", "(未設定)")}')
    print(f'  Match Name: {strings.get("match_name", "(未設定)")}')

def main():
    if not os.path.exists(RCP_FILE):
        print(f'エラー: {RCP_FILE} が見つかりません')
        print('先にVisual Studioでビルドして.rcpを生成してください')
        return 1
    
    patch_rcp_file(RCP_FILE, JAPANESE_STRINGS)
    return 0

if __name__ == '__main__':
    exit(main())
