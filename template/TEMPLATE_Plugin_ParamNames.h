/*******************************************************************/
/*                                                                 */
/*  __TPL_MATCH_NAME__ - Parameter Name Management                 */
/*  UTF-8 → Shift-JIS 自動エンコーディング変換                     */
/*                                                                 */
/*  仕組み:                                                        */
/*  - ソースコードは常に UTF-8 で記述                               */
/*  - Windows: コンパイラオプションで Shift_JIS に自動変換           */
/*      vcxproj に /source-charset:utf-8 /execution-charset:shift_jis */
/*  - Mac: ランタイムで iconv を使って Shift_JIS に変換              */
/*                                                                 */
/*******************************************************************/

#ifndef __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_NAMES_H
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_NAMES_H

#include <cstring>
#include <string>
#include <unordered_map>
#include <mutex>

#ifdef _WIN32
    // Windows: コンパイラオプションで自動変換されるため、そのまま返す
    #define PARAM(str) (str)
    #define PARAM_MENU(str) (str)
#else
    // Mac: ランタイム変換
    #include <iconv.h>

    class ParamNameConverter
    {
    public:
        static ParamNameConverter& Instance()
        {
            static ParamNameConverter instance;
            return instance;
        }

        const char* ToShiftJIS(const char* utf8)
        {
            if (!utf8 || !*utf8) return utf8;

            std::lock_guard<std::mutex> lock(mMutex);

            auto it = mCache.find(utf8);
            if (it != mCache.end()) return it->second.c_str();

            std::string sjis = ConvertUTF8ToShiftJIS(utf8);
            mCache[utf8] = sjis;
            return mCache[utf8].c_str();
        }

    private:
        ParamNameConverter() = default;
        ~ParamNameConverter() = default;
        ParamNameConverter(const ParamNameConverter&) = delete;
        ParamNameConverter& operator=(const ParamNameConverter&) = delete;

        std::string ConvertUTF8ToShiftJIS(const char* utf8)
        {
            iconv_t cd = iconv_open("SHIFT_JIS", "UTF-8");
            if (cd == (iconv_t)-1) return utf8;

            size_t inLeft = strlen(utf8);
            size_t outSize = inLeft * 2 + 1;
            std::string result(outSize, '\0');

            char* inPtr = const_cast<char*>(utf8);
            char* outPtr = &result[0];
            size_t outLeft = outSize;

            size_t ret = iconv(cd, &inPtr, &inLeft, &outPtr, &outLeft);
            iconv_close(cd);

            if (ret == (size_t)-1) return utf8;
            result.resize(outSize - outLeft);
            return result;
        }

        std::mutex mMutex;
        std::unordered_map<std::string, std::string> mCache;
    };

    #define PARAM(str)      (ParamNameConverter::Instance().ToShiftJIS(str))
    #define PARAM_MENU(str) (ParamNameConverter::Instance().ToShiftJIS(str))
#endif

// ========================================
// パラメータ名定義（UTF-8で記述）
// ここを編集するだけで両プラットフォームに反映される
// ========================================

namespace ParamNames
{
    // ---- サンプルパラメータ（プロジェクトに合わせて置き換え） ----
    constexpr const char* AMOUNT       = "エフェクト量";
    constexpr const char* COLOR        = "カラー";
    constexpr const char* MODE         = "モード";

    // ---- メニュー文字列（"|"区切り） ----
    constexpr const char* MODE_MENU    = "モードA|モードB|モードC";

    // ---- イージングメニュー（共通で使いまわし可能） ----
    constexpr const char* EASING_MENU  =
        "リニア|スムースステップ|スムーサーステップ|"
        "サインイン|サインアウト|サインインアウト|サインアウトイン|"
        "二次イン|二次アウト|二次インアウト|二次アウトイン|"
        "三次イン|三次アウト|三次インアウト|三次アウトイン|"
        "サークルイン|サークルアウト|サークルインアウト|サークルアウトイン|"
        "バックイン|バックアウト|バックインアウト|"
        "エラスティックイン|エラスティックアウト|エラスティックインアウト|"
        "バウンスイン|バウンスアウト|バウンスインアウト";
}

// ========================================
// 便利なショートカットマクロ
// P_XXX = PARAM(ParamNames::XXX) の省略形
// PM_XXX = PARAM_MENU(ParamNames::XXX_MENU) の省略形
// ========================================

#define P_AMOUNT     PARAM(ParamNames::AMOUNT)
#define P_COLOR      PARAM(ParamNames::COLOR)
#define P_MODE       PARAM(ParamNames::MODE)
#define PM_MODE      PARAM_MENU(ParamNames::MODE_MENU)
#define PM_EASING    PARAM_MENU(ParamNames::EASING_MENU)

#endif // __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_NAMES_H
