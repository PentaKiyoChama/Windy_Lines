/*******************************************************************/
/*                                                                 */
/*  __TPL_MATCH_NAME__ - Version Management                        */
/*  Copyright (c) __TPL_YEAR__ __TPL_AUTHOR__. All rights reserved.*/
/*                                                                 */
/*******************************************************************/

#ifndef __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_H
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_H

// ========== バージョン番号（ここだけ変更すればOK） ==========
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_MAJOR 1
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_MINOR 0
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_BUILD 0

// 文字列化マクロ
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___STRINGIFY(x) #x
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(x) __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___STRINGIFY(x)

// バージョン文字列（例: "v1"）
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_SHORT \
    "v" __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_MAJOR)

// フルバージョン文字列（例: "v1.0.0"）
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_FULL \
    "v" __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_MAJOR) "." \
    __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_MINOR) "." \
    __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___TOSTRING(__TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_BUILD)

// エフェクト名（UI表示用）
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___EFFECT_NAME \
    "__TPL_EFFECT_NAME_JP__ " __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_SHORT

#endif // __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_H
