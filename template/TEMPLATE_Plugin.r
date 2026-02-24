// __TPL_MATCH_NAME__.r
// PiPL Resource Definition for Adobe Premiere Pro
//
// NOTE: 日本語文字列はShift-JIS 16進エスケープで記述する
//       → tools/patch_pipl_japanese.py で自動パッチ推奨

#include "AEConfig.h"
#include "AE_EffectVers.h"
#include <AE_General.r>

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
        // NOTE: .r では ASCII を維持し、patch_pipl_japanese.py で日本語へ置換
        Category {
            "__TPL_CATEGORY_ASCII__"
        },
#ifdef AE_OS_WIN
        #ifdef AE_PROC_INTELx64
        CodeWin64X86 {
            "EffectMain"
        },
        #endif
#else
        CodeMacIntel64 {
            "EffectMain"
        },
        CodeMacARM64 {
            "EffectMain"
        },
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
        AE_Effect_Info_Flags {
            0
        },
        AE_Effect_Global_OutFlags {
            0x44
        },
        AE_Effect_Global_OutFlags_2 {
            0x100
        },
        AE_Effect_Match_Name {
            "__TPL_MATCH_NAME__"
        },
        AE_Reserved_Info {
            8
        },
    }
};
