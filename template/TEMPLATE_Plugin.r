// __TPL_MATCH_NAME__.r
// PiPL Resource Definition for Adobe Premiere Pro
//
// NOTE: 日本語文字列はShift-JIS 16進エスケープで記述する
//       → tools/patch_pipl_japanese.py で自動パッチ推奨

resource 'PiPL' (16000) {
    {
        Kind {
            AEEffect
        },
        // Name: プラグイン表示名
        // TODO: patch_pipl_japanese.py で日本語に置換
        Name {
            "__TPL_MATCH_NAME__"
        },
        // Category: Premiere Proのエフェクトカテゴリ
        // TODO: patch_pipl_japanese.py で日本語に置換
        Category {
            "__TPL_CATEGORY_JP__"
        },
#ifdef AE_OS_WIN
        #ifdef AE_PROC_INTELx64
        CodeWin64X86 {
            "__TPL_MATCH_NAME___GPU"
        },
        #endif
#else
        #ifdef AE_OS_MAC
        CodeMacIntel64 {
            "__TPL_MATCH_NAME___GPU"
        },
        CodeMacARM64 {
            "__TPL_MATCH_NAME___GPU"
        },
        #endif
#endif
        AE_PiPL_Version {
            2,
            0
        },
        AE_Effect_Spec_Version {
            PF_PLUG_IN_VERSION,
            PF_PLUG_IN_SUBVERS
        },
        AE_Effect_Version {
            524288    /* 8.0 = (MAJOR << 19) | (MINOR << 15) | 0 */
        },
        AE_Effect_Match_Name {
            "__TPL_MATCH_NAME__"
        },
        AE_Reserved_Info {
            8
        },
    }
};
