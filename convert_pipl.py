# -*- coding: utf-8 -*-
"""
PiPL変換スクリプト - 日本語対応版
.rファイルから.rcpファイルを生成する際に日本語を正しく処理します

使い方:
  python convert_pipl.py <input.rr> <output.rcp> [--name "名前"] [--category "カテゴリ"] [--match-name "マッチ名"]
"""

import sys
import os
import subprocess
import re

def string_to_sjis_hex(s):
    """文字列をShift_JISの16進エスケープに変換"""
    try:
        sjis_bytes = s.encode('shift_jis')
        return ''.join(f'\\x{b:02X}' for b in sjis_bytes)
    except UnicodeEncodeError:
        print(f"警告: '{s}' をShift_JISにエンコードできません", file=sys.stderr)
        return s

def process_rcp_content(content, name=None, category=None, match_name=None):
    """
    .rcpファイルの内容を処理して日本語を正しいShift_JISバイトに置換
    """
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # "eman" (name) プロパティを検出
        if '"eman"' in line and name:
            result.append(line)
            i += 1
            # RSCS32(0) をスキップ
            if i < len(lines):
                result.append(lines[i])
                i += 1
            # RSCS32(size) - サイズを計算して置換
            if i < len(lines):
                sjis_bytes = name.encode('shift_jis')
                # 長さプレフィックス(1) + 文字列 + パディング(4バイト境界)
                total_len = 1 + len(sjis_bytes)
                padded_len = ((total_len + 3) // 4) * 4
                result.append(f'\tRSCS32({padded_len}),')
                i += 1
            # 文字列行を置換
            if i < len(lines):
                sjis_hex = string_to_sjis_hex(name)
                padding = '\\0' * (padded_len - 1 - len(sjis_bytes))
                result.append(f'\t"\\x{len(sjis_bytes):02X}{sjis_hex}{padding}", ')
                i += 1
            continue
            
        # "gtac" (category) プロパティを検出
        if '"gtac"' in line and category:
            result.append(line)
            i += 1
            if i < len(lines):
                result.append(lines[i])
                i += 1
            if i < len(lines):
                sjis_bytes = category.encode('shift_jis')
                total_len = 1 + len(sjis_bytes)
                padded_len = ((total_len + 3) // 4) * 4
                result.append(f'\tRSCS32({padded_len}),')
                i += 1
            if i < len(lines):
                sjis_hex = string_to_sjis_hex(category)
                padding = '\\0' * (padded_len - 1 - len(sjis_bytes))
                result.append(f'\t"\\x{len(sjis_bytes):02X}{sjis_hex}{padding}", ')
                i += 1
            continue
            
        # "ANMe" (match name) プロパティを検出  
        if '"ANMe"' in line and match_name:
            result.append(line)
            i += 1
            if i < len(lines):
                result.append(lines[i])
                i += 1
            if i < len(lines):
                sjis_bytes = match_name.encode('shift_jis')
                total_len = 1 + len(sjis_bytes)
                padded_len = ((total_len + 3) // 4) * 4
                result.append(f'\tRSCS32({padded_len}),')
                i += 1
            if i < len(lines):
                sjis_hex = string_to_sjis_hex(match_name)
                padding = '\\0' * (padded_len - 1 - len(sjis_bytes))
                result.append(f'\t"\\x{len(sjis_bytes):02X}{sjis_hex}{padding}", ')
                i += 1
            continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)

def main():
    if len(sys.argv) < 3:
        print("使い方: python convert_pipl.py <input.rr> <output.rcp> [--name 名前] [--category カテゴリ] [--match-name マッチ名]")
        sys.exit(1)
    
    input_rr = sys.argv[1]
    output_rcp = sys.argv[2]
    
    # オプション引数をパース
    name = None
    category = None
    match_name = None
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--name' and i + 1 < len(sys.argv):
            name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--category' and i + 1 < len(sys.argv):
            category = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--match-name' and i + 1 < len(sys.argv):
            match_name = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # PiPLToolのパスを環境変数から取得
    ae_sdk_path = os.environ.get('AE_SDK_BASE_PATH', '')
    pipltool = os.path.join(ae_sdk_path, 'Examples', 'Resources', 'PiPLTool.exe')
    
    if not os.path.exists(pipltool):
        print(f"エラー: PiPLToolが見つかりません: {pipltool}", file=sys.stderr)
        sys.exit(1)
    
    # 一時ファイルに出力
    temp_rcp = output_rcp + '.tmp'
    
    # PiPLToolを実行
    result = subprocess.run([pipltool, input_rr, temp_rcp], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"PiPLToolエラー: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    
    # 生成された.rcpを読み込んで日本語を置換
    with open(temp_rcp, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # 日本語を置換
    if name or category or match_name:
        content = process_rcp_content(content, name, category, match_name)
    
    # 最終ファイルに出力
    with open(output_rcp, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 一時ファイルを削除
    os.remove(temp_rcp)
    
    print(f"変換完了: {output_rcp}")

if __name__ == '__main__':
    main()
