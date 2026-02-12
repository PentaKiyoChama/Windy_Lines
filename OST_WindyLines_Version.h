/*******************************************************************/
/*                                                                 */
/*                   SDK ProcAmp Version Management                */
/*                                                                 */
/*******************************************************************/

#ifndef OST_WINDYLINES_VERSION_H
#define OST_WINDYLINES_VERSION_H

// バージョン番号（ここだけ変更すればOK）
#define OST_WINDYLINES_VERSION_MAJOR 53
#define OST_WINDYLINES_VERSION_MINOR 0
#define OST_WINDYLINES_VERSION_BUILD 0

// 文字列化マクロ
#define OST_WINDYLINES_STRINGIFY(x) #x
#define OST_WINDYLINES_TOSTRING(x) OST_WINDYLINES_STRINGIFY(x)

// バージョン文字列（例: "v53"）
#define OST_WINDYLINES_VERSION_SHORT "v" OST_WINDYLINES_TOSTRING(OST_WINDYLINES_VERSION_MAJOR)

// フルバージョン文字列（例: "v53.0.0"）
#define OST_WINDYLINES_VERSION_FULL "v" OST_WINDYLINES_TOSTRING(OST_WINDYLINES_VERSION_MAJOR) "." OST_WINDYLINES_TOSTRING(OST_WINDYLINES_VERSION_MINOR) "." OST_WINDYLINES_TOSTRING(OST_WINDYLINES_VERSION_BUILD)

// エフェクト名（UI表示用）
#define OST_WINDYLINES_EFFECT_NAME "流れる線 " OST_WINDYLINES_VERSION_SHORT

#endif // OST_WINDYLINES_VERSION_H
