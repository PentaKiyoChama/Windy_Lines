/**
 * SDK_ProcAmp_ParamNames.h
 * 
 * Version: v51 - DevGuide approach: clipTime-based with GPU fallback estimation
 * 
 * パラメータ名の一元管理と自動エンコーディング変換
 * 
 * 使用方法:
 * 1. このヘッダーをインクルード
 * 2. PARAM("日本語名") マクロでパラメータ名を取得
 * 
 * 仕組み:
 * - ソースコードは常に UTF-8 で記述
 * - Windows: コンパイラが Shift_JIS に自動変換
 * - Mac: ランタイムで iconv を使って Shift_JIS に変換
 */

#ifndef SDK_PROCAMP_PARAM_NAMES_H
#define SDK_PROCAMP_PARAM_NAMES_H

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
    
    /**
     * UTF-8 から Shift_JIS への変換を行うシングルトンクラス
     * 変換結果はキャッシュされるため、同じ文字列の再変換は行われない
     */
    class ParamNameConverter
    {
    public:
        static ParamNameConverter& Instance()
        {
            static ParamNameConverter instance;
            return instance;
        }
        
        /**
         * UTF-8 文字列を Shift_JIS に変換
         * @param utf8 UTF-8 エンコードされた文字列
         * @return Shift_JIS に変換された文字列（キャッシュされた静的文字列へのポインタ）
         */
        const char* ToShiftJIS(const char* utf8)
        {
            if (!utf8 || !*utf8)
            {
                return utf8;
            }
            
            std::lock_guard<std::mutex> lock(mMutex);
            
            // キャッシュを確認
            auto it = mCache.find(utf8);
            if (it != mCache.end())
            {
                return it->second.c_str();
            }
            
            // 変換を実行
            std::string sjis = ConvertUTF8ToShiftJIS(utf8);
            
            // キャッシュに保存
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
            if (cd == (iconv_t)-1)
            {
                // 変換失敗時は元の文字列を返す
                return utf8;
            }
            
            size_t inLeft = strlen(utf8);
            size_t outSize = inLeft * 2 + 1; // Shift_JIS は最大2バイト/文字
            std::string result(outSize, '\0');
            
            char* inPtr = const_cast<char*>(utf8);
            char* outPtr = &result[0];
            size_t outLeft = outSize;
            
            size_t ret = iconv(cd, &inPtr, &inLeft, &outPtr, &outLeft);
            iconv_close(cd);
            
            if (ret == (size_t)-1)
            {
                // 変換失敗時は元の文字列を返す
                return utf8;
            }
            
            result.resize(outSize - outLeft);
            return result;
        }
        
        std::mutex mMutex;
        std::unordered_map<std::string, std::string> mCache;
    };
    
    // Mac用マクロ: ランタイム変換
    #define PARAM(str) (ParamNameConverter::Instance().ToShiftJIS(str))
    #define PARAM_MENU(str) (ParamNameConverter::Instance().ToShiftJIS(str))
#endif

// ========================================
// パラメータ名定義（UTF-8で記述）
// ここを編集するだけで両プラットフォームに反映
// ========================================

namespace ParamNames
{
    constexpr const char* LINE_COUNT        = "22線の数";
    constexpr const char* LIFETIME          = "寿命(fps)";
    constexpr const char* TRAVEL            = "移動距離(px)";
    constexpr const char* SEED              = "ランダムシード";
    constexpr const char* COLOR_MODE        = "カラーモード";
    constexpr const char* COLOR             = "色";
    constexpr const char* COLOR_PRESET      = "プリセット";
    constexpr const char* CUSTOM_1          = "カスタム1";
    constexpr const char* CUSTOM_2          = "カスタム2";
    constexpr const char* CUSTOM_3          = "カスタム3";
    constexpr const char* CUSTOM_4          = "カスタム4";
    constexpr const char* CUSTOM_5          = "カスタム5";
    constexpr const char* CUSTOM_6          = "カスタム6";
    constexpr const char* CUSTOM_7          = "カスタム7";
    constexpr const char* CUSTOM_8          = "カスタム8";
    constexpr const char* THICKNESS         = "太さ(px)";
    constexpr const char* LENGTH            = "長さ(px)";
    constexpr const char* LINE_CAP          = "端のスタイル";
    constexpr const char* ANGLE             = "線の角度";
    constexpr const char* AA                = "アンチエイリアス";
    constexpr const char* TAIL_FADE         = "テールフェード";
    constexpr const char* POSITION_HEADER   = "線の起点";
    constexpr const char* ORIGIN_MODE       = "起点モード";      
    constexpr const char* INTERVAL          = "インターバル(fps)";  
    constexpr const char* SPAWN_SCALE_X     = "スケールX(%)";    
    constexpr const char* SPAWN_SCALE_Y     = "スケールY(%)";    
    constexpr const char* SPAWN_ROTATION    = "範囲の回転";           
    constexpr const char* SHOW_SPAWN        = "目安の表示";            
    constexpr const char* SPAWN_COLOR       = "目安の色";              
    constexpr const char* OFFSET_X          = "オフセットX(px)";
    constexpr const char* OFFSET_Y          = "オフセットY(px)";
    constexpr const char* ANIM_PATTERN      = "進行方向";        
    constexpr const char* CENTER_GAP        = "中央ギャップ";    
    constexpr const char* EASING            = "イージング";
    constexpr const char* START_TIME        = "開始時間(fps)";
    constexpr const char* DURATION          = "継続時間(fps)";
    constexpr const char* SHADOW            = "シャドウ";
    constexpr const char* SHADOW_ENABLE     = "影を有効化";      
    constexpr const char* SHADOW_COLOR      = "影色";            
    constexpr const char* SHADOW_OFFSET_X   = "オフセットX";  
    constexpr const char* SHADOW_OFFSET_Y   = "オフセットY";  
    constexpr const char* SHADOW_OPACITY    = "不透明度";       
    constexpr const char* ADVANCED_HEADER   = "詳細";
    constexpr const char* HIDE_ELEMENT      = "要素を隠す";      
    constexpr const char* BLEND_MODE        = "合成モード";      
    constexpr const char* DEPTH_STRENGTH    = "奥行き強度";     
    constexpr const char* SPAWN_SOURCE      = "範囲ソース";
    constexpr const char* SPAWN_SOURCE_CHOICES = "画面全体|元要素";
    constexpr const char* ALPHA_THRESH      = "元要素のアルファしきい値"; 
    constexpr const char* MOTION_BLUR       = "モーションブラー";
    constexpr const char* BLUR_SAMPLES      = "ブラー品質(サンプル数)"; 
    constexpr const char* BLUR_STRENGTH     = "ブラー強度";
    constexpr const char* EFFECT_PRESET     = "エフェクトプリセット";
    constexpr const char* DEFAULT           = "デフォルト";
    constexpr const char* COLOR_MODE_MENU   = "単色|プリセット|カスタム";
    constexpr const char* LINE_CAP_MENU     = "フラット|ラウンド";
    constexpr const char* COLOR_PRESET_MENU = 
        "レインボー|パステル|フォレスト|サイバー|ハザード|"
        "さくら|デザート|スターダスト|若葉|フレイム|"
        "妖艶|リフレッシュ|ドリーミー|サンセット|オーシャン|"
        "オータム|スノー|深海|パーティー|ナイトスカイ|"
        "アメジスト|花畑|ジュエル|パステル2|アースカラー|"
        "ムーンライト|きらめく光|ネオン|警告|オーロラ|"
        "溶岩|ゴールド|モノクローム";
    constexpr const char* ORIGIN_MODE_MENU  = "中央|前方|後方"; 
    constexpr const char* ANIM_PATTERN_MENU = "標準|半反転|分割"; 
    constexpr const char* EASING_MENU       = 
        "リニア|サインイン|サインアウト|サインインアウト|"
        "二次イン|二次アウト|二次インアウト|"
        "三次イン|三次アウト|三次インアウト"; 
    constexpr const char* BLEND_MODE_MENU   = "背面|前面|背面と前面|アルファ"; 
}

// ========================================
// 便利なショートカットマクロ
// ========================================

// パラメータ名取得（自動変換付き）
#define P_LINE_COUNT        PARAM(ParamNames::LINE_COUNT)
#define P_LIFETIME          PARAM(ParamNames::LIFETIME)
#define P_TRAVEL            PARAM(ParamNames::TRAVEL)
#define P_SEED              PARAM(ParamNames::SEED)
#define P_COLOR_MODE        PARAM(ParamNames::COLOR_MODE)
#define P_COLOR             PARAM(ParamNames::COLOR)
#define P_COLOR_PRESET      PARAM(ParamNames::COLOR_PRESET)
#define P_CUSTOM_1          PARAM(ParamNames::CUSTOM_1)
#define P_CUSTOM_2          PARAM(ParamNames::CUSTOM_2)
#define P_CUSTOM_3          PARAM(ParamNames::CUSTOM_3)
#define P_CUSTOM_4          PARAM(ParamNames::CUSTOM_4)
#define P_CUSTOM_5          PARAM(ParamNames::CUSTOM_5)
#define P_CUSTOM_6          PARAM(ParamNames::CUSTOM_6)
#define P_CUSTOM_7          PARAM(ParamNames::CUSTOM_7)
#define P_CUSTOM_8          PARAM(ParamNames::CUSTOM_8)
#define P_THICKNESS         PARAM(ParamNames::THICKNESS)
#define P_LENGTH            PARAM(ParamNames::LENGTH)
#define P_LINE_CAP          PARAM(ParamNames::LINE_CAP)
#define P_ANGLE             PARAM(ParamNames::ANGLE)
#define P_AA                PARAM(ParamNames::AA)
#define P_TAIL_FADE         PARAM(ParamNames::TAIL_FADE)
#define P_POSITION_HEADER   PARAM(ParamNames::POSITION_HEADER)
#define P_ORIGIN_MODE       PARAM(ParamNames::ORIGIN_MODE)
#define P_INTERVAL          PARAM(ParamNames::INTERVAL)
#define P_SPAWN_SCALE_X     PARAM(ParamNames::SPAWN_SCALE_X)
#define P_SPAWN_SCALE_Y     PARAM(ParamNames::SPAWN_SCALE_Y)
#define P_SPAWN_ROTATION    PARAM(ParamNames::SPAWN_ROTATION)
#define P_SHOW_SPAWN        PARAM(ParamNames::SHOW_SPAWN)
#define P_SPAWN_COLOR       PARAM(ParamNames::SPAWN_COLOR)
#define P_OFFSET_X          PARAM(ParamNames::OFFSET_X)
#define P_OFFSET_Y          PARAM(ParamNames::OFFSET_Y)
#define P_ANIM_PATTERN      PARAM(ParamNames::ANIM_PATTERN)
#define P_CENTER_GAP        PARAM(ParamNames::CENTER_GAP)
#define P_EASING            PARAM(ParamNames::EASING)
#define P_START_TIME        PARAM(ParamNames::START_TIME)
#define P_DURATION          PARAM(ParamNames::DURATION)
#define P_SHADOW            PARAM(ParamNames::SHADOW)
#define P_SHADOW_ENABLE     PARAM(ParamNames::SHADOW_ENABLE)
#define P_SHADOW_COLOR      PARAM(ParamNames::SHADOW_COLOR)
#define P_SHADOW_OFFSET_X   PARAM(ParamNames::SHADOW_OFFSET_X)
#define P_SHADOW_OFFSET_Y   PARAM(ParamNames::SHADOW_OFFSET_Y)
#define P_SHADOW_OPACITY    PARAM(ParamNames::SHADOW_OPACITY)
#define P_ADVANCED_HEADER   PARAM(ParamNames::ADVANCED_HEADER)
#define P_HIDE_ELEMENT      PARAM(ParamNames::HIDE_ELEMENT)
#define P_BLEND_MODE        PARAM(ParamNames::BLEND_MODE)
#define P_DEPTH_STRENGTH    PARAM(ParamNames::DEPTH_STRENGTH)
#define P_SPAWN_SOURCE      PARAM(ParamNames::SPAWN_SOURCE)
#define P_SPAWN_SOURCE_CHOICES PARAM(ParamNames::SPAWN_SOURCE_CHOICES)
#define P_ALPHA_THRESH      PARAM(ParamNames::ALPHA_THRESH)
#define P_MOTION_BLUR       PARAM(ParamNames::MOTION_BLUR)
#define P_BLUR_SAMPLES      PARAM(ParamNames::BLUR_SAMPLES)
#define P_BLUR_STRENGTH     PARAM(ParamNames::BLUR_STRENGTH)
#define P_EFFECT_PRESET     PARAM(ParamNames::EFFECT_PRESET)
#define P_DEFAULT           PARAM(ParamNames::DEFAULT)

// メニュー項目取得（自動変換付き）
#define PM_COLOR_MODE       PARAM_MENU(ParamNames::COLOR_MODE_MENU)
#define PM_LINE_CAP         PARAM_MENU(ParamNames::LINE_CAP_MENU)
#define PM_COLOR_PRESET     PARAM_MENU(ParamNames::COLOR_PRESET_MENU)
#define PM_ORIGIN_MODE      PARAM_MENU(ParamNames::ORIGIN_MODE_MENU)
#define PM_ANIM_PATTERN     PARAM_MENU(ParamNames::ANIM_PATTERN_MENU)
#define PM_EASING           PARAM_MENU(ParamNames::EASING_MENU)
#define PM_BLEND_MODE       PARAM_MENU(ParamNames::BLEND_MODE_MENU)

#endif // SDK_PROCAMP_PARAM_NAMES_H
