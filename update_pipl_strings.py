#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PiPL 日本語文字列 統合更新スクリプト

Mac (.r) と Windows (.rcp) の PiPL ファイルの日本語文字列を一括更新します。
hex escape の手動計算は不要 — 下記の JAPANESE_STRINGS を編集するだけで OK。

使い方:
  python3 update_pipl_strings.py              # 両方更新
  python3 update_pipl_strings.py --mac        # .r のみ更新
  python3 update_pipl_strings.py --windows    # .rcp のみ更新
  python3 update_pipl_strings.py --dry-run    # 変更内容を表示（実際には書き込まない）
  python3 update_pipl_strings.py --show       # 現在の設定と hex escape を表示

注意:
  - .r ファイルの BOM は自動で除去されます
  - Match Name は ASCII のまま変更不要（内部識別子のため）
"""

import os
import re
import sys

# ============================================================
#  ★ ここを編集するだけで Mac / Windows 両方に反映されます ★
# ============================================================
JAPANESE_STRINGS = {
    'name':       '風を感じるライン',    # Premiere Pro 上のプラグイン名
    'category':   'おしゃれテロップ',    # Premiere Pro 上のカテゴリ名
    'match_name': 'OST_WindyLines',      # 内部識別子（ASCII推奨、変更非推奨）
}
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
R_FILE   = os.path.join(SCRIPT_DIR, 'OST_WindyLines.r')
RCP_FILE = os.path.join(SCRIPT_DIR, 'OST_WindyLines.rcp')


# ── ユーティリティ ──────────────────────────────────

def to_sjis_bytes(text: str) -> bytes:
    """日本語テキスト → Shift-JIS バイト列"""
    return text.encode('shift_jis')


def to_utf8_bytes(text: str) -> bytes:
    """日本語テキスト → UTF-8 バイト列"""
    return text.encode('utf-8')


def to_rez_hex(data: bytes) -> str:
    r"""バイト列 → Rez hex escape 文字列  (例: \0x95\0x97...)"""
    return ''.join(f'\\0x{b:02X}' for b in data)


def to_c_hex(data: bytes) -> str:
    r"""バイト列 → C hex escape 文字列  (例: \x95\x97...)"""
    return ''.join(f'\\x{b:02X}' for b in data)


def show_info():
    """現在の設定と各エンコーディングの hex escape を表示"""
    print('=' * 60)
    print('  PiPL 日本語文字列 設定一覧')
    print('=' * 60)
    for key, text in JAPANESE_STRINGS.items():
        print(f'\n  [{key}] "{text}"')
        if key == 'match_name':
            ascii_bytes = text.encode('ascii', errors='replace')
            print(f'    ASCII ({len(ascii_bytes):2d} bytes): {" ".join(f"{b:02X}" for b in ascii_bytes)}')
            continue
        sjis = to_sjis_bytes(text)
        utf8 = to_utf8_bytes(text)
        print(f'    Shift-JIS ({len(sjis):2d} bytes): {" ".join(f"{b:02X}" for b in sjis)}')
        print(f'    UTF-8     ({len(utf8):2d} bytes): {" ".join(f"{b:02X}" for b in utf8)}')
        print(f'    Rez SJIS : "{to_rez_hex(sjis)}"')
        print(f'    Rez UTF-8: "{to_rez_hex(utf8)}"')
    print()


# ── Mac (.r ファイル) 更新 ─────────────────────────

def strip_bom(data: bytes) -> bytes:
    """UTF-8 BOM があれば除去"""
    if data[:3] == b'\xEF\xBB\xBF':
        return data[3:]
    return data


def update_r_file(dry_run: bool = False) -> bool:
    """
    .r ファイルの Name / Category hex escape を更新する。
    BOM があれば自動除去。
    """
    if not os.path.exists(R_FILE):
        print(f'[SKIP] {R_FILE} が見つかりません')
        return False

    with open(R_FILE, 'rb') as f:
        raw = f.read()

    # BOM 除去
    cleaned = strip_bom(raw)
    had_bom = len(cleaned) < len(raw)
    content = cleaned.decode('utf-8', errors='replace')

    changes = []

    def _update_block(block_name, text, content_str):
        """
        Name { ... } または Category { ... } ブロック内の
        hex escape 文字列とコメントを更新する。
        行ベースで安全に処理。
        """
        sjis = to_sjis_bytes(text)
        utf8 = to_utf8_bytes(text)
        rez_sjis = to_rez_hex(sjis)
        rez_utf8 = to_rez_hex(utf8)

        lines = content_str.split('\n')
        result = []
        i = 0
        found = False

        while i < len(lines):
            line = lines[i]

            # block_name { を検出 (例: "Name {" or "Category {")
            if re.match(rf'^\s*{block_name}\s*\{{', line):
                found = True
                result.append(line)
                i += 1

                # ブロック内を処理（"}" で閉じるまで）
                wrote_comment = False
                wrote_string = False
                wrote_utf8 = False

                while i < len(lines):
                    bline = lines[i]

                    # ブロック終了
                    if re.match(r'^\s*\}', bline):
                        # まだ書いてなければここで書く
                        if not wrote_comment:
                            result.append(f'\t\t\t/* Shift-JIS hex: "{text}" ({len(sjis)} bytes) */')
                        if not wrote_string:
                            result.append(f'\t\t\t"{rez_sjis}"')
                        if not wrote_utf8:
                            result.append(f'\t\t\t/* UTF-8 alternative ({len(utf8)} bytes):')
                            result.append(f'\t\t\t   "{rez_utf8}" */')
                        result.append(bline)
                        i += 1
                        break

                    # Shift-JIS コメント行 → 新しいコメントで置換
                    if 'Shift-JIS hex' in bline or 'Shift-JIS' in bline:
                        result.append(f'\t\t\t/* Shift-JIS hex: "{text}" ({len(sjis)} bytes) */')
                        wrote_comment = True
                        i += 1
                        continue

                    # hex escape 文字列行 ("\0x..." で始まる)
                    if re.match(r'^\s*"\\0x', bline):
                        result.append(f'\t\t\t"{rez_sjis}"')
                        wrote_string = True
                        i += 1
                        continue

                    # UTF-8 alternative コメント（複数行の可能性）
                    if 'UTF-8 alternative' in bline:
                        result.append(f'\t\t\t/* UTF-8 alternative ({len(utf8)} bytes):')
                        result.append(f'\t\t\t   "{rez_utf8}" */')
                        wrote_utf8 = True
                        i += 1
                        # コメント閉じ "*/" が次の行にある場合はスキップ
                        while i < len(lines) and '*/' not in lines[i - 1]:
                            i += 1
                        continue

                    result.append(bline)
                    i += 1
                continue

            result.append(line)
            i += 1

        return '\n'.join(result), found

    # ----- Name ブロック -----
    name_text = JAPANESE_STRINGS['name']
    new_content, n1 = _update_block('Name', name_text, content)
    if n1:
        sjis_n = to_sjis_bytes(name_text)
        changes.append(f'Name → "{name_text}" (SJIS {len(sjis_n)} bytes)')

    # ----- Category ブロック -----
    cat_text = JAPANESE_STRINGS['category']
    new_content, n2 = _update_block('Category', cat_text, new_content)
    if n2:
        sjis_c = to_sjis_bytes(cat_text)
        changes.append(f'Category → "{cat_text}" (SJIS {len(sjis_c)} bytes)')

    if had_bom:
        changes.append('BOM を除去しました')

    if not changes:
        print(f'[Mac .r] 変更なし')
        return True

    print(f'[Mac .r] {R_FILE}')
    for c in changes:
        print(f'  ✓ {c}')

    if dry_run:
        print('  (dry-run: 実際のファイルは変更されません)')
        return True

    # UTF-8 (BOM なし) で書き出し — コメント内の日本語を保持
    with open(R_FILE, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)

    print('  → 保存完了')
    return True


# ── Windows (.rcp ファイル) 更新 ────────────────────

def make_rcp_pascal_string(text: str, encoding: str = 'shift_jis'):
    """
    .rcp 用の Pascal 文字列を生成
    Returns: (padded_size, formatted_string)
    """
    encoded = text.encode(encoding)
    length = len(encoded)
    total = 1 + length              # 長さプレフィックス + 文字列
    padded_size = ((total + 3) // 4) * 4
    padding_needed = padded_size - total

    # C hex escape 形式
    length_hex = f'\\x{length:02X}'
    text_hex = to_c_hex(encoded)
    padding = '\\0' * padding_needed

    return padded_size, f'"{length_hex}{text_hex}{padding}"'


def update_rcp_file(dry_run: bool = False) -> bool:
    """
    .rcp ファイルの Name / Category / MatchName を更新する。
    """
    if not os.path.exists(RCP_FILE):
        print(f'[SKIP] {RCP_FILE} が見つかりません')
        return False

    with open(RCP_FILE, 'r', encoding='latin-1') as f:
        content = f.read()

    lines = content.split('\n')
    result = []
    changes = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # "eman" (name) プロパティ
        if '"eman"' in line:
            result.append(line)
            i += 1
            if i < len(lines):
                result.append(lines[i])  # RSCS32(0)
                i += 1
            if i < len(lines):
                padded_size, formatted = make_rcp_pascal_string(JAPANESE_STRINGS['name'])
                result.append(f'\tRSCS32({padded_size}),')
                i += 1
            if i < len(lines):
                _, formatted = make_rcp_pascal_string(JAPANESE_STRINGS['name'])
                result.append(f'\t{formatted}, ')
                i += 1
                changes.append(f'Name → "{JAPANESE_STRINGS["name"]}"')
            continue

        # "gtac" (category) プロパティ
        if '"gtac"' in line:
            result.append(line)
            i += 1
            if i < len(lines):
                result.append(lines[i])
                i += 1
            if i < len(lines):
                padded_size, formatted = make_rcp_pascal_string(JAPANESE_STRINGS['category'])
                result.append(f'\tRSCS32({padded_size}),')
                i += 1
            if i < len(lines):
                _, formatted = make_rcp_pascal_string(JAPANESE_STRINGS['category'])
                result.append(f'\t{formatted}, ')
                i += 1
                changes.append(f'Category → "{JAPANESE_STRINGS["category"]}"')
            continue

        # "ANMe" (match name) プロパティ
        if '"ANMe"' in line:
            result.append(line)
            i += 1
            if i < len(lines):
                result.append(lines[i])
                i += 1
            if i < len(lines):
                padded_size, formatted = make_rcp_pascal_string(
                    JAPANESE_STRINGS['match_name'], encoding='ascii')
                result.append(f'\tRSCS32({padded_size}),')
                i += 1
            if i < len(lines):
                _, formatted = make_rcp_pascal_string(
                    JAPANESE_STRINGS['match_name'], encoding='ascii')
                result.append(f'\t{formatted}, ')
                i += 1
                changes.append(f'Match Name → "{JAPANESE_STRINGS["match_name"]}"')
            continue

        result.append(line)
        i += 1

    if not changes:
        print(f'[Windows .rcp] 変更なし')
        return True

    print(f'[Windows .rcp] {RCP_FILE}')
    for c in changes:
        print(f'  ✓ {c}')

    if dry_run:
        print('  (dry-run: 実際のファイルは変更されません)')
        return True

    # Shift-JIS で保存（Windows RC コンパイラ用）
    with open(RCP_FILE, 'w', encoding='shift_jis') as f:
        f.write('\n'.join(result))

    print('  → 保存完了')
    return True


# ── メイン ──────────────────────────────────────

def main():
    args = set(sys.argv[1:])

    if '--help' in args or '-h' in args:
        print(__doc__)
        return 0

    if '--show' in args:
        show_info()
        return 0

    dry_run = '--dry-run' in args
    do_mac = '--mac' in args
    do_win = '--windows' in args or '--win' in args

    # フラグなし → 両方
    if not do_mac and not do_win:
        do_mac = True
        do_win = True

    print()
    print('╔════════════════════════════════════════╗')
    print('║  PiPL 日本語文字列 統合更新            ║')
    print('╚════════════════════════════════════════╝')
    print()

    ok = True
    if do_mac:
        ok = update_r_file(dry_run) and ok
        print()
    if do_win:
        ok = update_rcp_file(dry_run) and ok
        print()

    if ok:
        print('完了！')
        if not dry_run and do_mac:
            print()
            print('ヒント: .r ファイルを VS Code で開き直した場合、BOM が再付与される')
            print('        ことがあります。ビルド前にこのスクリプトを再実行すると安全です。')
    else:
        print('一部のファイルで問題がありました')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
