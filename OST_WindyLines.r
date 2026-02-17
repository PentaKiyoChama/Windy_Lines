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
			"風を感じるライン"
		},
		/* [3] */
		Category {
			"OST"
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
