/*******************************************************************/
/*                                                                 */
/*  __TPL_MATCH_NAME__ - Main Header                               */
/*  GPU Video Filter Plugin for Adobe Premiere Pro                 */
/*                                                                 */
/*  Copyright (c) __TPL_YEAR__ __TPL_AUTHOR__. All rights reserved.*/
/*                                                                 */
/*  This plugin was developed using the Adobe Premiere Pro SDK.    */
/*  Portions based on SDK sample code:                             */
/*    Copyright 2012 Adobe Systems Incorporated.                   */
/*    Used in accordance with the Adobe Developer SDK License.     */
/*                                                                 */
/*  This software is not affiliated with or endorsed by Adobe.     */
/*                                                                 */
/*******************************************************************/

#ifndef __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___H
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___H

#include "AEConfig.h"
#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "A.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "Param_Utils.h"
#include "PrSDKAESupport.h"

#include <math.h>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <ctime>

// ========== GPU RENDERING CONTROL ==========
// 1 = GPU有効 (CUDA/OpenCL/Metal), 0 = CPU専用（テスト用）
#define ENABLE_GPU_RENDERING 1

// ========== DEBUG RENDER MARKERS ==========
// 1 = 左上隅にGPU/CPU表示, 0 = 無効（リリース時は0に）
#define ENABLE_DEBUG_RENDER_MARKERS 0

// ========== DEBUG LOGGING ==========
#if defined(_DEBUG) || defined(DEBUG)
static std::mutex sLogMutex;
static void WriteLog(const char* format, ...)
{
    std::lock_guard<std::mutex> lock(sLogMutex);
#ifdef _WIN32
    const char* logPath = "C:\\Temp\\__TPL_MATCH_NAME___Log.txt";
#else
    const char* logPath = "/tmp/__TPL_MATCH_NAME___Log.txt";
#endif
    FILE* fp = nullptr;
#ifdef _WIN32
    errno_t err = fopen_s(&fp, logPath, "a");
    if (err != 0 || !fp) return;
#else
    fp = fopen(logPath, "a");
    if (!fp) return;
#endif
    time_t now = time(nullptr);
    char timeStr[64];
    struct tm timeInfo;
#ifdef _WIN32
    localtime_s(&timeInfo, &now);
#else
    localtime_r(&now, &timeInfo);
#endif
    strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &timeInfo);
    fprintf(fp, "[%s] ", timeStr);
    va_list args;
    va_start(args, format);
    vfprintf(fp, format, args);
    va_end(args);
    fprintf(fp, "\n");
    fflush(fp);
    fclose(fp);
}
#endif

#if defined(_DEBUG) || defined(DEBUG)
    #define DebugLog WriteLog
#else
    #define DebugLog(...)  // リリースビルドではno-op
#endif


/*******************************************************************/
/*  パラメータID定義                                                */
/*  NOTE: 順序を変更すると既存プロジェクトとの互換性が壊れるため     */
/*        新規パラメータは末尾に追加すること                         */
/*******************************************************************/
enum
{
    __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___INPUT = 0,

    // ▼ サンプルパラメータ（プロジェクトに合わせて置き換え）
    __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_AMOUNT,       // 1. エフェクト量
    __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_COLOR,        // 2. カラー
    __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_POPUP,        // 3. ポップアップ選択

    // -- ここにパラメータを追加 --

    __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___NUM_PARAMS
};


/*******************************************************************/
/*  パラメータのデフォルト値・範囲                                   */
/*  パターン: NAME_MIN_VALUE, NAME_MAX_VALUE,                       */
/*            NAME_MIN_SLIDER, NAME_MAX_SLIDER, NAME_DFLT           */
/*******************************************************************/

// Amount (サンプルスライダー)
#define PARAM_AMOUNT_MIN_VALUE      0.0
#define PARAM_AMOUNT_MAX_VALUE      100.0
#define PARAM_AMOUNT_MIN_SLIDER     0.0
#define PARAM_AMOUNT_MAX_SLIDER     100.0
#define PARAM_AMOUNT_DFLT           50.0

// Popup (サンプルポップアップ)
#define PARAM_POPUP_DFLT            1


/*******************************************************************/
/*  Enum定義（ポップアップメニュー等で使用）                         */
/*******************************************************************/

// サンプルモード（1-based: Premiere Proのポップアップは1始まり）
enum SampleMode
{
    SAMPLE_MODE_A = 1,
    SAMPLE_MODE_B = 2,
    SAMPLE_MODE_C = 3
};


/*******************************************************************/
/*  共通構造体                                                      */
/*******************************************************************/

// AE Effect バージョン情報
#define MAJOR_VERSION       1
#define MINOR_VERSION       0
#define BUG_VERSION         0
#define STAGE_VERSION       PF_Stage_DEVELOP
#define BUILD_VERSION       0


/*******************************************************************/
/*  CPU-GPU間共有データ（必要に応じて拡張）                          */
/*  GPUはclipTimeしか取得できないため、CPUで取得した情報を             */
/*  staticマップで共有する設計パターン                                */
/*******************************************************************/
#ifdef __cplusplus
#include <unordered_map>
#include <mutex>

struct SharedClipData
{
    static std::unordered_map<csSDK_int64, csSDK_int64> clipStartMap;
    static std::mutex mapMutex;

    static void SetClipStart(csSDK_int64 clipOffset, csSDK_int64 clipStartFrame)
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        clipStartMap[clipOffset] = clipStartFrame;
    }

    static csSDK_int64 GetClipStart(csSDK_int64 clipOffset)
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        auto it = clipStartMap.find(clipOffset);
        return (it != clipStartMap.end()) ? it->second : -1;
    }
};
#endif // __cplusplus

#endif // __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___H
