/*******************************************************************/
/*                                                                 */
/*  __TPL_MATCH_NAME__ - Version Management                        */
/*  Copyright (c) __TPL_YEAR__ __TPL_AUTHOR__. All rights reserved.*/
/*                                                                 */
/*******************************************************************/

#ifndef __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_H
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_H

// ===== 内部バージョン（Premiere Pro エフェクト名・プロジェクト互換性用） =====
// 変更するとエフェクト名が変わり、既存プロジェクトとの互換性が失われるため注意
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_MAJOR 1
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_MINOR 0
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_BUILD 0

// 文字列化マクロ
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___STRINGIFY(x) #x
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(x) __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___STRINGIFY(x)

// 内部バージョン文字列
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_SHORT \
    "v" __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_MAJOR)

// エフェクト名（Premiere Pro UI表示用 — 変更すると既存プロジェクトが壊れます）
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___EFFECT_NAME \
    "__TPL_EFFECT_NAME_JP__ " __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_SHORT

// ===== 公開バージョン（配布・ライセンスAPI・ユーザー表示用） =====
// リリース時はここだけ更新すればOK
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PUBLIC_VERSION_MAJOR 1
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PUBLIC_VERSION_MINOR 0
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PUBLIC_VERSION_PATCH 0

#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_FULL \
    "v" __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PUBLIC_VERSION_MAJOR) "." \
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PUBLIC_VERSION_MINOR) "." \
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PUBLIC_VERSION_PATCH)

#endif // __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_H
