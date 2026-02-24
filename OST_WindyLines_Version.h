/*******************************************************************/
/*                                                                 */
/*  OST_WindyLines - Version Management                            */
/*  Copyright (c) 2026 Kiyoto Nakamura. All rights reserved.       */
/*                                                                 */
/*******************************************************************/

#ifndef OST_WINDYLINES_VERSION_H
#define OST_WINDYLINES_VERSION_H

// ===== 内部バージョン（Premiere Pro エフェクト名・プロジェクト互換性用） =====
// 変更するとエフェクト名が変わり、既存プロジェクトとの互換性が失われるため注意
#define OST_WINDYLINES_VERSION_MAJOR 53
#define OST_WINDYLINES_VERSION_MINOR 0
#define OST_WINDYLINES_VERSION_BUILD 0

// 文字列化マクロ
#define OST_WINDYLINES_STRINGIFY(x) #x
#define OST_WINDYLINES_TOSTRING(x) OST_WINDYLINES_STRINGIFY(x)

// 内部バージョン文字列
#define OST_WINDYLINES_VERSION_SHORT "v" OST_WINDYLINES_TOSTRING(OST_WINDYLINES_VERSION_MAJOR)

// エフェクト名（Premiere Pro UI表示用 — 変更すると既存プロジェクトが壊れます）
#define OST_WINDYLINES_EFFECT_NAME "流れる線 " OST_WINDYLINES_VERSION_SHORT

// ===== 公開バージョン（配布・ライセンスAPI・ユーザー表示用） =====
// リリース時はここだけ更新すればOK
#define OST_WINDYLINES_PUBLIC_VERSION_MAJOR 1
#define OST_WINDYLINES_PUBLIC_VERSION_MINOR 0
#define OST_WINDYLINES_PUBLIC_VERSION_PATCH 1

#define OST_WINDYLINES_VERSION_FULL \
    "v" OST_WINDYLINES_TOSTRING(OST_WINDYLINES_PUBLIC_VERSION_MAJOR) "." \
        OST_WINDYLINES_TOSTRING(OST_WINDYLINES_PUBLIC_VERSION_MINOR) "." \
        OST_WINDYLINES_TOSTRING(OST_WINDYLINES_PUBLIC_VERSION_PATCH)

#endif // OST_WINDYLINES_VERSION_H
