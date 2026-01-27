/*******************************************************************/
/*                                                                 */
/*                   SDK ProcAmp Version Management                */
/*                                                                 */
/*******************************************************************/

#ifndef SDK_PROCAMP_VERSION_H
#define SDK_PROCAMP_VERSION_H

// バージョン番号（ここだけ変更すればOK）
#define SDK_PROCAMP_VERSION_MAJOR 52
#define SDK_PROCAMP_VERSION_MINOR 0
#define SDK_PROCAMP_VERSION_BUILD 0

// 文字列化マクロ
#define SDK_PROCAMP_STRINGIFY(x) #x
#define SDK_PROCAMP_TOSTRING(x) SDK_PROCAMP_STRINGIFY(x)

// バージョン文字列（例: "v52"）
#define SDK_PROCAMP_VERSION_SHORT "v" SDK_PROCAMP_TOSTRING(SDK_PROCAMP_VERSION_MAJOR)

// フルバージョン文字列（例: "v52.0.0"）
#define SDK_PROCAMP_VERSION_FULL "v" SDK_PROCAMP_TOSTRING(SDK_PROCAMP_VERSION_MAJOR) "." SDK_PROCAMP_TOSTRING(SDK_PROCAMP_VERSION_MINOR) "." SDK_PROCAMP_TOSTRING(SDK_PROCAMP_VERSION_BUILD)

// エフェクト名（UI表示用）
#define SDK_PROCAMP_EFFECT_NAME "SDK ProcAmp " SDK_PROCAMP_VERSION_SHORT

#endif // SDK_PROCAMP_VERSION_H
