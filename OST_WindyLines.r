/*******************************************************************/
/*                                                                 */
/*  OST_WindyLines - Particle Line Effect Plugin                   */
/*  for Adobe Premiere Pro                                         */
/*                                                                 */
/*  Copyright (c) 2026 Kiyoto Nakamura. All rights reserved.       */
/*                                                                 */
/*  This plugin was developed using the Adobe Premiere Pro SDK.    */
/*  Portions based on SDK sample code:                             */
/*    Copyright 2012 Adobe Systems Incorporated.                   */
/*    Used in accordance with the Adobe Developer SDK License.     */
/*                                                                 */
/*  This software is not affiliated with or endorsed by Adobe.     */
/*                                                                 */
/*******************************************************************/

#include "AEConfig.h"
#include "AE_EffectVers.h"
#include <AE_General.r>


resource 'PiPL' (16000) {
	{	/* array properties: 11 elements */
		/* [1] */
		Kind {
			AEEffect
		},
		/* [2] */
		Name {
			/* Shift-JIS hex: "風を感じるライン" (16 bytes) */
			"\0x95\0x97\0x82\0xF0\0x8A\0xB4\0x82\0xB6\0x82\0xE9\0x83\0x89\0x83\0x43\0x83\0x93"
			/* UTF-8 alternative (24 bytes):
			   "\0xE9\0xA2\0xA8\0xE3\0x82\0x92\0xE6\0x84\0x9F\0xE3\0x81\0x98\0xE3\0x82\0x8B\0xE3\0x83\0xA9\0xE3\0x82\0xA4\0xE3\0x83\0xB3" */
		},
		/* [3] */
		Category {
			/* Shift-JIS hex: "おしゃれテロップ" (16 bytes) */
			"\0x82\0xA8\0x82\0xB5\0x82\0xE1\0x82\0xEA\0x83\0x65\0x83\0x8D\0x83\0x62\0x83\0x76"
			/* UTF-8 alternative (24 bytes):
			   "\0xE3\0x81\0x8A\0xE3\0x81\0x97\0xE3\0x82\0x83\0xE3\0x82\0x8C\0xE3\0x83\0x86\0xE3\0x83\0xAD\0xE3\0x83\0x83\0xE3\0x83\0x97" */
		},

		/* [4] */
#ifdef AE_OS_WIN
		CodeWin64X86 {"EffectMain"},
#else
		CodeMacIntel64 {"EffectMain"},
		CodeMacARM64 {"EffectMain"},
#endif
		/* [5] */
		AE_PiPL_Version {
			2,
			0
		},
		/* [6] */
		AE_Effect_Spec_Version {
			PF_PLUG_IN_VERSION,
			PF_PLUG_IN_SUBVERS
		},
		/* [7] */
		AE_Effect_Version {
			524288 
		},
		/* [8] */
		AE_Effect_Info_Flags {
			0
		},
		/* [9] */
		AE_Effect_Global_OutFlags {
			0x44
		},
		AE_Effect_Global_OutFlags_2 {
			0x100
		},
		/* [10] */
		AE_Effect_Match_Name {
			"OST_WindyLines"
		},
		/* [11] */
		AE_Reserved_Info {
			8
		}
	}
};
