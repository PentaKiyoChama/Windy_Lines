/*******************************************************************/
/*                                                                 */
/*  __TPL_MATCH_NAME__ - CPU Renderer (Fallback)                   */
/*  GPU が使えない環境用のCPUレンダリングパス                       */
/*                                                                 */
/*  Copyright (c) __TPL_YEAR__ __TPL_AUTHOR__. All rights reserved.*/
/*                                                                 */
/*******************************************************************/

#include "__TPL_MATCH_NAME__.h"
#include "__TPL_MATCH_NAME___Version.h"
#include "__TPL_MATCH_NAME___ParamNames.h"
#include "__TPL_MATCH_NAME___Common.h"

#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "Param_Utils.h"
#include "PrSDKAESupport.h"

#include <math.h>
#include <string.h>


/*******************************************************************/
/*  静的データの実体化                                              */
/*******************************************************************/
#ifdef __cplusplus
std::unordered_map<csSDK_int64, csSDK_int64> SharedClipData::clipStartMap;
std::mutex SharedClipData::mapMutex;
#endif


/*******************************************************************/
/*  エントリーポイント                                              */
/*******************************************************************/
static PF_Err About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_SPRINTF(out_data->return_msg,
        "%s %s\n%s",
        "__TPL_EFFECT_NAME_JP__",
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_FULL,
        "(c) __TPL_YEAR__ __TPL_AUTHOR__");
    return PF_Err_NONE;
}


static PF_Err GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION,
                                       STAGE_VERSION, BUILD_VERSION);

    out_data->out_flags =
        PF_OutFlag_PIX_INDEPENDENT |
        PF_OutFlag_SEND_UPDATE_PARAMS_UI;

    out_data->out_flags2 =
        PF_OutFlag2_SUPPORTS_SMART_RENDER |
        PF_OutFlag2_FLOAT_COLOR_AWARE |
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

#if ENABLE_GPU_RENDERING
    out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
#endif

    return err;
}


static PF_Err ParamsSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    // ---- パラメータ1: エフェクト量（スライダー） ----
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDER(
        P_AMOUNT,                      // パラメータ名
        PARAM_AMOUNT_MIN_VALUE,        // 有効最小値
        PARAM_AMOUNT_MAX_VALUE,        // 有効最大値
        PARAM_AMOUNT_MIN_SLIDER,       // スライダー最小値
        PARAM_AMOUNT_MAX_SLIDER,       // スライダー最大値
        0,                             // カーブ許容値
        PARAM_AMOUNT_DFLT,             // デフォルト値
        1,                             // 精度
        0,                             // 表示精度
        0,                             // フラグ
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_AMOUNT);

    // ---- パラメータ2: カラー ----
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(
        P_COLOR,
        255, 255, 255,                 // デフォルト RGB
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_COLOR);

    // ---- パラメータ3: ポップアップ ----
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        P_MODE,
        3,                             // 選択肢の数
        PARAM_POPUP_DFLT,              // デフォルト値
        PM_MODE,                       // メニュー文字列（"|"区切り）
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_POPUP);

    out_data->num_params = __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___NUM_PARAMS;

    return err;
}


/*******************************************************************/
/*  CPUレンダリング本体                                             */
/*  NOTE: GPUが使えない場合のフォールバック。                        */
/*  SmartRender 経由で呼ばれる。                                    */
/*******************************************************************/
static PF_Err SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    PF_RenderRequest req = extra->input->output_request;
    PF_CheckoutResult result;

    ERR(extra->cb->checkout_layer(
        in_data->effect_ref,
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___INPUT,
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &result));

    UnionLRect(&result.result_rect, &extra->output->result_rect);
    UnionLRect(&result.max_result_rect, &extra->output->max_result_rect);

    return err;
}


static PF_Err SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    // 入力レイヤーチェックアウト
    PF_EffectWorld* input_worldP = nullptr;
    ERR(extra->cb->checkout_layer_pixels(
        in_data->effect_ref,
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___INPUT,
        &input_worldP));

    PF_EffectWorld* output_worldP = nullptr;
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

    if (!err && input_worldP && output_worldP)
    {
        // パラメータ取得
        PF_ParamDef paramAmount, paramColor, paramPopup;
        AEFX_CLR_STRUCT(paramAmount);
        AEFX_CLR_STRUCT(paramColor);
        AEFX_CLR_STRUCT(paramPopup);

        ERR(PF_CHECKOUT_PARAM(in_data,
            __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_AMOUNT,
            in_data->current_time, in_data->time_step, in_data->time_scale,
            &paramAmount));
        ERR(PF_CHECKOUT_PARAM(in_data,
            __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_COLOR,
            in_data->current_time, in_data->time_step, in_data->time_scale,
            &paramColor));
        ERR(PF_CHECKOUT_PARAM(in_data,
            __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_POPUP,
            in_data->current_time, in_data->time_step, in_data->time_scale,
            &paramPopup));

        float amount = (float)paramAmount.u.fs_d.value;
        int mode = paramPopup.u.pd.value;

        // ========================================
        // TODO: ここにCPUレンダリングロジックを実装
        // input_worldP → output_worldP にピクセルを書き込む
        // ========================================

        // サンプル: 入力をそのまま出力にコピー
        if (!err)
        {
            PF_Pixel8* inRow;
            PF_Pixel8* outRow;
            int width = output_worldP->width;
            int height = output_worldP->height;

            for (int y = 0; y < height; y++)
            {
                inRow  = (PF_Pixel8*)((char*)input_worldP->data  + y * input_worldP->rowbytes);
                outRow = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                memcpy(outRow, inRow, width * sizeof(PF_Pixel8));
            }
        }

        // パラメータチェックイン
        ERR(PF_CHECKIN_PARAM(in_data, &paramAmount));
        ERR(PF_CHECKIN_PARAM(in_data, &paramColor));
        ERR(PF_CHECKIN_PARAM(in_data, &paramPopup));
    }

    return err;
}


/*******************************************************************/
/*  メインディスパッチャー                                          */
/*******************************************************************/
DllExport
PF_Err PluginMain(
    PF_Cmd cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err err = PF_Err_NONE;

    try
    {
        switch (cmd)
        {
            case PF_Cmd_ABOUT:
                err = About(in_data, out_data, params, output);
                break;

            case PF_Cmd_GLOBAL_SETUP:
                err = GlobalSetup(in_data, out_data, params, output);
                break;

            case PF_Cmd_PARAMS_SETUP:
                err = ParamsSetup(in_data, out_data, params, output);
                break;

            case PF_Cmd_SMART_PRE_RENDER:
                err = SmartPreRender(in_data, out_data,
                    reinterpret_cast<PF_PreRenderExtra*>(extra));
                break;

            case PF_Cmd_SMART_RENDER:
                err = SmartRender(in_data, out_data,
                    reinterpret_cast<PF_SmartRenderExtra*>(extra));
                break;

            default:
                break;
        }
    }
    catch (PF_Err& thrown_err)
    {
        err = thrown_err;
    }

    return err;
}
