# エージェント向け実装マニュアル

> **対象**: テンプレートから派生プラグインを実装する AI エージェント  
> **前提**: `DERIVATION_CHECKLIST.md` のビルド・PiPL 関連チェックは完了済みであること  
> **目的**: テンプレートの TODO / スタブを正しく埋める際に陥りやすい罠を網羅する

---

## 目次

1. [GPU レンダリングパイプラインの全体像](#1-gpu-レンダリングパイプラインの全体像)
2. [GetFrameDependencies — 入力フレーム要求の必須実装](#2-getframedependencies--入力フレーム要求の必須実装)
3. [Render — 入出力バッファの正しい取得方法](#3-render--入出力バッファの正しい取得方法)
4. [Metal 初期化 — バンドル ID の完全一致](#4-metal-初期化--バンドル-id-の完全一致)
5. [Metal ディスパッチ — バッファインデックス規約](#5-metal-ディスパッチ--バッファインデックス規約)
6. [ProcAmpParams — 四重同期ルール](#6-procampparams--四重同期ルール)
7. [ピクセルフォーマットと BGRA アクセス](#7-ピクセルフォーマットと-bgra-アクセス)
8. [テンプレート GPU.cpp の既知の誤り](#8-テンプレート-gpucpp-の既知の誤り)
9. [デバッグログパターン](#9-デバッグログパターン)
10. [GPU パラメータ取得の罠 — Popup と Color](#10-gpu-パラメータ取得の罠--popup-と-color)
11. [クリップ時間・長さの取得 — VideoSegmentSuite](#11-クリップ時間長さの取得--videosegmentsuite)
12. [実装完了チェックリスト](#12-実装完了チェックリスト)

---

## 1. GPU レンダリングパイプラインの全体像

Premiere Pro の GPU フィルタープラグインは以下の順序でメソッドが呼ばれる:

```
Initialize()              ← 1回だけ。Metal パイプライン構築
    ↓
GetFrameDependencies()    ← 毎フレーム。必要な入力フレームを宣言
    ↓
Precompute() [optional]   ← CPU側で計算した情報をGPUに渡す場合
    ↓
Render()                  ← 毎フレーム。GPU カーネルをディスパッチ
    ↓
Shutdown()                ← アンロード時。パイプラインキャッシュ解放
```

**重要**: `GetFrameDependencies` が正しく入力フレームを要求しないと、`Render` に渡される `inFrames` が空になるか、入力データが提供されない。

---

## 2. GetFrameDependencies — 入力フレーム要求の必須実装

### ❌ テンプレートの誤り（致命的）

```cpp
prSuiteError GetFrameDependencies(
    const PrGPUFilterRenderParams* inRenderParams,
    csSDK_int32* ioQueryIndex,
    PrGPUFilterFrameDependency* outFrameDependency)
{
    return suiteError_NoError;  // ← 何も要求していない！
}
```

テンプレートは `suiteError_NoError` を返すだけで、入力フレームを一切要求していない。
この場合 Premiere Pro は入力データを提供しないため、**Render で空のバッファが渡される**。

### ✅ 正しい実装

```cpp
prSuiteError GetFrameDependencies(
    const PrGPUFilterRenderParams* inRenderParams,
    csSDK_int32* ioQueryIndex,
    PrGPUFilterFrameDependency* outFrameRequirements)
{
    if (*ioQueryIndex == 0)
    {
        outFrameRequirements->outDependencyType = PrGPUDependency_InputFrame;
        outFrameRequirements->outTrackID = 0;
        outFrameRequirements->outSequenceTime = inRenderParams->inSequenceTime;
        return suiteError_NoError;
    }
    return suiteError_NotImplemented;  // ← これ以上のフレームは不要
}
```

### ルール
- `*ioQueryIndex == 0` で **最低 1 フレーム（現在の入力フレーム）を要求**する
- 複数フレームが必要な場合（時間ブラー等）は `*ioQueryIndex == 1, 2, ...` で追加要求
- 不要なインデックスには `suiteError_NotImplemented` を返して終了を伝える
- 引数名 `outFrameDependency` ではなく `outFrameRequirements` が正式名称（SDK ヘッダ参照）

---

## 3. Render — 入出力バッファの正しい取得方法

### ❌ テンプレートの誤り

テンプレートの Render は以下が欠落している:
- `GetGPUPPixData()` による GPU バッファポインタ取得
- `GetPixelFormat()` / `GetBounds()` / `GetRowBytes()` による出力フレーム情報取得
- 入力バッファと出力バッファの分離

### ✅ 正しい実装パターン

```cpp
prSuiteError Render(
    const PrGPUFilterRenderParams* inRenderParams,
    const PPixHand* inFrames,
    csSDK_size_t inFrameCount,
    PPixHand* outFrame)
{
    if (inFrameCount < 1) return suiteError_InvalidParms;

    PrTime clipTime = inRenderParams->inClipTime;

    // ======== 入力バッファ ========
    void* inFrameData = nullptr;
    mGPUDeviceSuite->GetGPUPPixData(inFrames[0], &inFrameData);

    // ======== 出力バッファ ========
    void* outFrameData = nullptr;
    mGPUDeviceSuite->GetGPUPPixData(*outFrame, &outFrameData);

    // ======== フレーム情報 ========
    PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
    mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);

    prRect bounds = {};
    mPPixSuite->GetBounds(*outFrame, &bounds);
    int width  = bounds.right - bounds.left;
    int height = bounds.bottom - bounds.top;

    csSDK_int32 rowBytes = 0;
    mPPixSuite->GetRowBytes(*outFrame, &rowBytes);
    int pitch = rowBytes / GetGPUBytesPerPixel(pixelFormat);
    int is16f = (pixelFormat != PrPixelFormat_GPU_BGRA_4444_32f) ? 1 : 0;

    // ======== パラメータ構築 ========
    ProcAmpParams params = {};
    params.mPitch  = pitch;
    params.m16f    = is16f;
    params.mWidth  = width;
    params.mHeight = height;
    // ... GetParam() で各パラメータを取得 ...

    // ======== GPU ディスパッチ ========
    // ... (後述) ...
}
```

### 重要なポイント
| 項目 | 値の取得元 | 注意 |
|------|-----------|------|
| `inFrameData` | `mGPUDeviceSuite->GetGPUPPixData(inFrames[0], ...)` | `outFrame` ではない |
| `outFrameData` | `mGPUDeviceSuite->GetGPUPPixData(*outFrame, ...)` | ポインタ演算に注意 |
| `width` / `height` | `mPPixSuite->GetBounds(*outFrame, ...)` | `inRenderParams->inRenderWidth` ではない |
| `pitch` | `rowBytes / GetGPUBytesPerPixel(pixelFormat)` | バイト単位ではなくピクセル単位 |
| `clipTime` | `inRenderParams->inClipTime` | パラメータ取得のタイムスタンプ |

### パラメータ取得

```cpp
// Int32 (ドロップダウン等)
params.mShape = GetParam(PARAM_ID_SHAPE, clipTime).mInt32;

// Float64 (スライダー等)
params.mPadding = (float)GetParam(PARAM_ID_PADDING, clipTime).mFloat64;

// Bool (チェックボックス)
params.mEnabled = GetParam(PARAM_ID_ENABLED, clipTime).mBool ? 1 : 0;

// ⚠️ Color パラメータ（最も躓きやすいポイント）
// Premiere Pro の色パラメータは mInt32 ではなく、
// mMemoryPtr (PF_Pixel*) または mInt64 で返される。
// mInt32 で取得すると不正な色になる。
{
    PrParam cp = GetParam(PARAM_ID_COLOR, clipTime);
    if (cp.mType == kPrParamType_PrMemoryPtr && cp.mMemoryPtr)
    {
        const PF_Pixel* px = reinterpret_cast<const PF_Pixel*>(cp.mMemoryPtr);
        params.mColorR = px->red   / 255.0f;
        params.mColorG = px->green / 255.0f;
        params.mColorB = px->blue  / 255.0f;
    }
    else if (cp.mType == kPrParamType_Int64)
    {
        csSDK_uint64 c = static_cast<csSDK_uint64>(cp.mInt64);
        params.mColorR = ((c >> 32) & 0xFFFF) / 65535.0f;
        params.mColorG = ((c >> 16) & 0xFFFF) / 65535.0f;
        params.mColorB = (c & 0xFFFF) / 65535.0f;
    }
    else  // kPrParamType_Int32 フォールバック
    {
        csSDK_uint32 c = static_cast<csSDK_uint32>(cp.mInt32);
        params.mColorR = ((c >> 16) & 0xFF) / 255.0f;
        params.mColorG = ((c >> 8)  & 0xFF) / 255.0f;
        params.mColorB = (c & 0xFF) / 255.0f;
    }
}
```

> **注意**: `GetParam(...).mInt32` で直接色を取得するのは間違い。必ず `PrParam` 変数に受けて `mType` を判定すること。

---

## 4. Metal 初期化 — バンドル ID の完全一致

### ❌ よくある間違い

```objc
// Info.plist の CFBundleIdentifier: com.OshareTelop.OSTSpeechBubble
// コード内で以下のように書いてしまう:
NSBundle* bundle = [NSBundle bundleWithIdentifier:@"com.osharetelop.OST-SpeechBubble"];
// → nil が返る（大文字小文字・ハイフン等が不一致）
```

`NSBundle bundleWithIdentifier:` は **大文字小文字を含め完全一致** でないと `nil` を返す。
`nil` が返ると `bundlePath` も `nil` → `metalLibPath` も `nil` → `fileExistsAtPath:nil` は `NO` → Initialize が `suiteError_Fail` を返す → **GPU レンダリング無効化（サイレント失敗）**。

### ✅ 確認方法

```bash
# Info.plist からバンドル ID を確認
/usr/libexec/PlistBuddy -c "Print CFBundleIdentifier" Mac/OST_XXX-Info.plist

# project.pbxproj から確認
grep PRODUCT_BUNDLE_IDENTIFIER Mac/OST_XXX.xcodeproj/project.pbxproj
```

### ✅ 正しい Initialize パターン

```objc
prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData)
{
    PrGPUFilterBase::Initialize(ioInstanceData);
    if (mDeviceIndex >= kMaxDevices) return suiteError_Fail;

#if HAS_METAL
    if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal)
    {
        if (!sMetalPipelineStateCache[mDeviceIndex])
        @autoreleasepool {
            prSuiteError result = suiteError_NoError;

            // ⚠️ Info.plist の CFBundleIdentifier と完全一致させること！
            NSString* pluginBundlePath = [[NSBundle bundleWithIdentifier:
                @"com.YourCompany.YourPlugin"] bundlePath];

            NSString* metalLibPath = [pluginBundlePath
                stringByAppendingPathComponent:
                @"Contents/Resources/MetalLib/__TPL_MATCH_NAME__.metalLib"];

            if (!(metalLibPath &&
                  [[NSFileManager defaultManager] fileExistsAtPath:metalLibPath]))
                return suiteError_Fail;

            NSError* error = nil;
            id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;
            id<MTLLibrary> library = [[device newLibraryWithFile:metalLibPath
                                                          error:&error] autorelease];
            result = CheckForMetalError(error);

            if (result == suiteError_NoError)
            {
                NSString* name = [NSString stringWithCString:"PluginKernel"
                                                    encoding:NSUTF8StringEncoding];
                id<MTLFunction> function =
                    [[library newFunctionWithName:name] autorelease];

                if (!function) return suiteError_Fail;

                sMetalPipelineStateCache[mDeviceIndex] =
                    [device newComputePipelineStateWithFunction:function
                                                         error:&error];
                result = CheckForMetalError(error);
            }
            return result;
        }
        return suiteError_NoError;
    }
#endif
    return suiteError_NoError;
}
```

### Metal ヘッダー

テンプレートに `<Metal/Metal.h>` の include が欠落している。Mac ビルドに必須:

```cpp
#if !_WIN32
    #include <OpenCL/cl.h>
    #include <Metal/Metal.h>    // ← テンプレートに欠落
#endif
```

---

## 5. Metal ディスパッチ — バッファインデックス規約

### バッファ配置規約

| Index | 内容 | 型 |
|-------|------|-----|
| 0 | 入力フレーム (`inBuf`) | `device const float4*` |
| 1 | 出力フレーム (`outBuf`) | `device float4*` |
| 2 | パラメータ (`params`) | `constant ProcAmpParams*` |

### ✅ 正しいディスパッチコード

```objc
#if HAS_METAL
if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal)
{
    @autoreleasepool {
        id<MTLDevice> device = (id<MTLDevice>)mDeviceInfo.outDeviceHandle;

        id<MTLComputePipelineState> pso = sMetalPipelineStateCache[mDeviceIndex];
        if (!pso) return suiteError_Fail;

        id<MTLBuffer> parameterBuffer = [[device newBufferWithBytes:&params
            length:sizeof(ProcAmpParams)
            options:MTLResourceStorageModeManaged] autorelease];

        id<MTLCommandQueue> queue =
            (id<MTLCommandQueue>)mDeviceInfo.outCommandQueueHandle;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];

        id<MTLBuffer> srcBuffer = (id<MTLBuffer>)inFrameData;
        id<MTLBuffer> dstBuffer = (id<MTLBuffer>)outFrameData;

        MTLSize threadsPerGroup = {
            [pso threadExecutionWidth], 16, 1
        };
        MTLSize numThreadgroups = {
            DivideRoundUp(width,  threadsPerGroup.width),
            DivideRoundUp(height, threadsPerGroup.height),
            1
        };

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:srcBuffer       offset:0 atIndex:0];  // input
        [encoder setBuffer:dstBuffer       offset:0 atIndex:1];  // output
        [encoder setBuffer:parameterBuffer offset:0 atIndex:2];  // params
        [encoder dispatchThreadgroups:numThreadgroups
                threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        [commandBuffer commit];

        return suiteError_NoError;
    }
}
#endif
```

### よくある間違い
- **入力と出力に同じバッファを使う**: `inFrameData` と `outFrameData` は常に別のポインタ。同じバッファを index 0 と 1 に設定すると入力が上書きされる
- **パイプラインステートの null チェック漏れ**: Initialize が失敗している場合 `pso` が nil でクラッシュする
- **`@autoreleasepool` の欠落**: Metal オブジェクトのリークを防ぐために必須

### ⚠️ アニメーション変形時の入力バッファ レースコンディション

**問題**: ピクセル座標をスケーリング・回転などで変換して入力バッファから「別の座標」を読む場合、
`inBuf` と `outBuf` が同一の物理バッファを指していることがある。
Metal の threadgroup は任意順序で実行されるため、あるスレッドが変換先座標から読むとき、
そのピクセルが既に別の threadgroup によって上書きされている→ **水平ノイズバンド（threadgroup 境界のアーティファクト）** が発生する。

**症状**: アニメーション（スケール変形など）の瞬間にだけ、横縞状のノイズバンドが出現する。
変形なしのフレームでは正常に描画される。

**根本原因**: Premiere Pro GPU SDK は inBuf / outBuf が同一メモリを指す場合がある。
通常の `pixel = ReadFloat4(inBuf, y * pitch + x)` では自分の座標のみ読むため問題にならないが、
逆変換で「別のピクセル座標」から読む場合はレースコンディションが起きる。

**解決策**: カーネル実行前に Blit Copy で入力バッファの一時コピーを作成し、カーネルにはそのコピーを渡す。

```objc
// アニメーション変形時: 入力バッファのクリーンコピーを作成
id<MTLBuffer> readBuffer = srcBuffer;
if (needsTransform) {  // animScale != 1.0f など
    id<MTLBuffer> tempBuffer = [[device newBufferWithLength:srcBuffer.length
                                 options:MTLResourceStorageModePrivate] autorelease];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromBuffer:srcBuffer sourceOffset:0
                toBuffer:tempBuffer destinationOffset:0
                    size:srcBuffer.length];
    [blit endEncoding];
    readBuffer = tempBuffer;
}

id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
[encoder setBuffer:readBuffer offset:0 atIndex:0];   // clean copy
[encoder setBuffer:dstBuffer  offset:0 atIndex:1];   // output
[encoder setBuffer:paramBuf   offset:0 atIndex:2];   // params
```

**パフォーマンス**: Blit Copy は GPU 内メモリ間コピーのため非常に高速（1920x1080 32bit で約 0.1ms）。
変形が不要なフレームでは条件分岐でスキップされるため、通常時のコストはゼロ。

**CUDA での注意**: CUDA は inBuf / outBuf が常に別ポインタであることが保証されるケースが多いが、
安全のため同様のパターンを検討すること。

---

## 6. ProcAmpParams — 四重同期ルール

パラメータ構造体は以下の **4 箇所で完全にフィールド順序が一致** している必要がある:

| ファイル | 定義場所 | 形式 |
|---------|---------|------|
| `_GPU.cpp` | `typedef struct { ... } ProcAmpParams;` | C構造体 |
| `.cl` | `GF_UNMARSHAL_USING_STRUCT` マクロ内 | 構造体アンマーシャル |
| `.cu` | `typedef struct { ... } ProcAmpParams;` + カーネル引数 | C構造体 |
| `.metal` | `struct ProcAmpParams { ... };` | Metal構造体 |

### フィールド追加手順

1. `_GPU.cpp` の `ProcAmpParams` にフィールドを追加
2. `.cl` の `GF_UNMARSHAL_USING_STRUCT` 内に **同じ順序で** 同名フィールドを追加
3. `.cu` の `ProcAmpParams` に **同じ順序で** 追加
4. `.metal` の `ProcAmpParams` に **同じ順序で** 追加
5. `_GPU.cpp` の `Render()` 内で `GetParam()` → `params.mNewField = ...` を追加

### ⚠️ 典型的な失敗
- 1 箇所だけフィールドを追加して他を忘れる → バイトオフセットがずれて全パラメータが破壊される
- `int` と `float` を混在させた時のアライメント問題 → 各フィールドは 4 バイト境界に配置すること
- GPU カーネル側の共通ヘッダ（`_Common.h`）のユーティリティ関数を変更した場合、`.cu` と `.cl` にも手動で同等の変更が必要（三重同期ルール）

---

## 7. ピクセルフォーマットと BGRA アクセス

### Premiere Pro GPU ピクセルフォーマット

Premiere Pro の GPU パスでは **BGRA** 順のピクセルフォーマットが使われる:

| `float4` コンポーネント | チャンネル |
|----------------------|---------|
| `.x` | Blue |
| `.y` | Green |
| `.z` | Red |
| `.w` | Alpha |

### ❌ よくある間違い

```c
// RGBA 前提で書いてしまう
float4 color = (float4)(red, green, blue, alpha);
```

### ✅ 正しいアクセス

```c
// BGRA 順
float4 pixel = ReadFloat4(inBuf, y * inPitch + x, !!is16f);
float blue  = pixel.x;
float green = pixel.y;
float red   = pixel.z;
float alpha = pixel.w;

// 書き込み時も BGRA 順
float4 outPixel = (float4)(blue, green, red, alpha);
WriteFloat4(outPixel, outBuf, y * outPitch + x, !!is16f);
```

### ReadFloat4 / WriteFloat4

SDK 提供のマクロ/関数。16f (half-float) と 32f の両方に対応:

```c
float4 ReadFloat4(const float* buf, int address, int is16f);
void WriteFloat4(float4 pixel, float* buf, int address, int is16f);
```

- `address` = `y * pitch + x` （ピクセル単位、バイト単位ではない）
- `is16f` = `1` なら 16bit half-float、`0` なら 32bit float
- テンプレートにあった `GF_PIXEL_FLOAT` / `GF_LOAD_PIXEL` / `GF_STORE_PIXEL` は **SDK に存在しない架空のマクロ**

---

## 8. テンプレート GPU.cpp の既知の誤り

テンプレートファイル `TEMPLATE_Plugin_GPU.cpp` には以下の問題がある。
派生時に **必ず修正** すること:

### 8-1. `<Metal/Metal.h>` の include が欠落

```cpp
// テンプレート:
#else
    #include <OpenCL/cl.h>
#endif

// 修正後:
#else
    #include <OpenCL/cl.h>
    #include <Metal/Metal.h>
#endif
```

### 8-2. GetFrameDependencies が入力フレームを要求していない

→ [セクション 2](#2-getframedependencies--入力フレーム要求の必須実装) を参照

### 8-3. Render 内のバッファ取得が TODO のまま

→ [セクション 3](#3-render--入出力バッファの正しい取得方法) を参照

### 8-4. GPU フレームワーク判定 API が間違っている

```cpp
// ❌ テンプレートの誤り（このメンバは存在しない）
if (inRenderParams->inRenderGPUType == PrGPURenderType_CUDA)
if (inRenderParams->inRenderGPUType == PrGPURenderType_Metal)

// ✅ 正しい判定方法
if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_Metal)
```

`PrGPUDeviceFramework` は `mDeviceInfo` のメンバであり、`inRenderParams` にはない。

### 8-5. Metal の Initialize / Shutdown / パイプラインキャッシュが未実装

テンプレートには Metal パイプライン構築コードがない。
→ [セクション 4](#4-metal-初期化--バンドル-id-の完全一致)、[セクション 5](#5-metal-ディスパッチ--バッファインデックス規約) を参照

### 8-6. Metal パイプラインキャッシュの宣言

```cpp
// クラス外（ファイルスコープ）に配置
#if HAS_METAL
static id<MTLComputePipelineState> sMetalPipelineStateCache[kMaxDevices] = {};
#endif
```

`static` メンバではなくファイルスコープの配列として宣言する。
`Shutdown()` で `[sMetalPipelineStateCache[inIndex] release]` して `nil` にリセット。

---

## 9. デバッグログパターン

GPU レンダリングが動作しない場合、サイレント失敗が多い。
以下のパターンで `/tmp/` にファイルベースのログを出力すると効果的:

### ログ関数

```cpp
#define OST_GPU_DEBUG_LOG 1  // リリース時は 0

#if OST_GPU_DEBUG_LOG
static std::mutex sGpuLogMutex;
static void GpuLog(const char* fmt, ...)
{
    std::lock_guard<std::mutex> lock(sGpuLogMutex);
    FILE* f = fopen("/tmp/OST_XXX_GPU.log", "a");
    if (!f) return;
    time_t t = time(nullptr);
    struct tm tm;
    localtime_r(&t, &tm);
    fprintf(f, "[%02d:%02d:%02d] ", tm.tm_hour, tm.tm_min, tm.tm_sec);
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fprintf(f, "\n");
    fclose(f);
}
#else
#define GpuLog(...) ((void)0)
#endif
```

### 推奨ログポイント

| メソッド | ログ内容 |
|---------|---------|
| `Initialize` | バンドルパス, metallib パス, パイプライン作成結果 |
| `GetFrameDependencies` | `ioQueryIndex` の値 |
| `Render` | `inFrameCount`, `inFrameData`/`outFrameData` のポインタ (null チェック), `width`/`height`/`pitch`, 主要パラメータ値, Metal dispatch のスレッド数 |
| `Shutdown` | デバイスインデックス |

### ログ確認

```bash
# ログクリア → Premiere Pro で操作 → ログ確認
rm -f /tmp/OST_XXX_GPU.log
# ... Premiere Pro で操作 ...
cat /tmp/OST_XXX_GPU.log
```

### ⚠️ mutex 名の衝突

`_GPU.cpp` の中で `static std::mutex` を使う場合、ヘッダー（`.h`）で定義済みの同名変数と衝突しないよう **一意な名前** にすること:

```cpp
// ❌ .h に sLogMutex があると衝突
static std::mutex sLogMutex;

// ✅ GPU 用に一意な名前
static std::mutex sGpuLogMutex;
```

---

## 10. GPU パラメータ取得の罠 — Popup と Color

このセクションは **エージェントが最も躓きやすい 2 大ポイント** を解説する。

### 10-1. Popup パラメータの 0-based / 1-based 問題

#### 背景

Premiere Pro の `PF_ADD_POPUP` は **1-based** で選択肢を定義する:

```cpp
// CPU側の ParamsSetup (1-based)
PF_ADD_POPUP(P_SHAPE, 3, 1, PM_SHAPE, PARAM_ID_SHAPE);
//                        ^ default=1 (最初の選択肢)
```

しかし GPU 側の `GetParam().mInt32` は **0-based** の値を返す:

| UI 表示 | CPU (PF_ADD_POPUP) | GPU GetParam().mInt32 |
|----------|-------------------|---------------------|
| 四角     | 1                 | **0**               |
| 楽円     | 2                 | **1**               |
| 多角     | 3                 | **2**               |

#### ❗ 典型的な失敗

```cpp
// ❌ .h の enum が 1-based、カーネルが 0-based、GPU.cpp がそのまま渡す
// → 全ての popup が 1 つずれる
enum BubbleShape { SHAPE_RECT = 1, SHAPE_ELLIPSE = 2 };
params.mShape = GetParam(PARAM_ID_SHAPE, clipTime).mInt32; // 0 or 1
// カーネルで inShape == 2 をチェック → 一度もマッチしない
```

#### ✅ 正しい対処法（方法 A: GPU 側で +1）

`.h` の enum と CPU 側のロジックを 1-based のまま保ち、GPU 取得時に +1:

```cpp
// GPU.cpp — popup 取得時に 0-based → 1-based に変換
params.mShape     = GetParam(PARAM_ID_SHAPE, clipTime).mInt32 + 1;
params.mTailSide  = GetParam(PARAM_ID_TAIL_SIDE, clipTime).mInt32 + 1;
params.mPattern   = GetParam(PARAM_ID_PATTERN, clipTime).mInt32 + 1;
params.mAnimation = GetParam(PARAM_ID_ANIMATION, clipTime).mInt32 + 1;
```

これにより `.h` の enum、CPU 側の switch、`ComputeTailVertices` 等の共有関数が全て 1-based で統一される。

#### ✅ 正しい対処法（方法 B: 全て 0-based に統一）

WindyLines 方式。enum を 0-based にし、カーネルも 0-based で比較:

```cpp
enum BubbleShape { SHAPE_RECT = 0, SHAPE_ELLIPSE = 1 };
// GPU.cpp
params.mShape = GetParam(PARAM_ID_SHAPE, clipTime).mInt32; // そのまま
// カーネル
if (inShape == 1) { /* 楽円 */ }
```

→ CPU 側の `PF_ADD_POPUP` の default も変更が必要なので注意。

### 10-2. Color パラメータの型問題

#### ❗ 最も多いエージェントの失敗パターン

```cpp
// ❌ そのまま mInt32 で色を取得（間違い）
csSDK_int32 c = GetParam(PARAM_ID_COLOR, clipTime).mInt32;
params.mColorR = ((c >> 16) & 0xFF) / 255.0f;
```

Premiere Pro の GPU `GetParam()` が色パラメータをどの型で返すかは **バージョンや環境によって異なる**。
`mInt32` で取得すると不正な色（緑、青等）が表示される。

#### ✅ 正しい取得パターン（必ず型判定する）

```cpp
// ✅ PrParam の mType を判定して、適切なフィールドで色を取得
{
    PrParam cp = GetParam(PARAM_ID_COLOR, clipTime);
    if (cp.mType == kPrParamType_PrMemoryPtr && cp.mMemoryPtr)
    {
        // 最も一般的: PF_Pixel ポインタ経由
        const PF_Pixel* px = reinterpret_cast<const PF_Pixel*>(cp.mMemoryPtr);
        params.mColorR = px->red   / 255.0f;
        params.mColorG = px->green / 255.0f;
        params.mColorB = px->blue  / 255.0f;
    }
    else if (cp.mType == kPrParamType_Int64)
    {
        // 48bit パッキング (16bit/ch)
        csSDK_uint64 c = static_cast<csSDK_uint64>(cp.mInt64);
        params.mColorR = ((c >> 32) & 0xFFFF) / 65535.0f;
        params.mColorG = ((c >> 16) & 0xFFFF) / 65535.0f;
        params.mColorB = (c & 0xFFFF) / 65535.0f;
    }
    else  // kPrParamType_Int32 フォールバック
    {
        // 24bit パッキング (8bit/ch)
        csSDK_uint32 c = static_cast<csSDK_uint32>(cp.mInt32);
        params.mColorR = ((c >> 16) & 0xFF) / 255.0f;
        params.mColorG = ((c >> 8)  & 0xFF) / 255.0f;
        params.mColorB = (c & 0xFF) / 255.0f;
    }
}
```

#### 重要なルール

- **必ず `PrParam` 変数に受けて `mType` を判定すること**
- `GetParam(...).mInt32` と直接アクセスするのは **禁止**
- 色パラメータが複数ある場合は、ヘルパー関数を作るとよい:

```cpp
static void ExtractColorParam(const PrParam& cp, float& r, float& g, float& b)
{
    if (cp.mType == kPrParamType_PrMemoryPtr && cp.mMemoryPtr)
    {
        const PF_Pixel* px = reinterpret_cast<const PF_Pixel*>(cp.mMemoryPtr);
        r = px->red / 255.0f; g = px->green / 255.0f; b = px->blue / 255.0f;
    }
    else if (cp.mType == kPrParamType_Int64)
    {
        csSDK_uint64 c = static_cast<csSDK_uint64>(cp.mInt64);
        r = ((c >> 32) & 0xFFFF) / 65535.0f;
        g = ((c >> 16) & 0xFFFF) / 65535.0f;
        b = (c & 0xFFFF) / 65535.0f;
    }
    else
    {
        csSDK_uint32 c = static_cast<csSDK_uint32>(cp.mInt32);
        r = ((c >> 16) & 0xFF) / 255.0f;
        g = ((c >> 8) & 0xFF) / 255.0f;
        b = (c & 0xFF) / 255.0f;
    }
}
```

---

## 11. クリップ時間・長さの取得 — VideoSegmentSuite

アニメーション（イン/アウト）やパターンの時間変化を実装するには、
**クリップ先頭からの相対時間**と**クリップの長さ（秒）**が必要になる。
これらは GPU フィルタープラグインでは自明に取得できず、複数の罠がある。

### 11-1. 3つの致命的な罠

#### ❌ 罠1: `inClipTime` はクリップ相対時間ではない

```cpp
// ❌ inClipTime をそのまま「クリップ先頭からの時間」として使う
PrTime clipTime = inRenderParams->inClipTime;
double sec = (double)clipTime / 254016000000.0;
// → 1時間地点にインポイントがあるメディアでは sec ≈ 3600
```

`inClipTime` は **メディアソースの絶対時間** を返す。
ソースのタイムコードが `01:00:00:00` から始まるメディアでは、
クリップの先頭フレームで `inClipTime ≈ 3600秒分の ticks`。

→ **「クリップの先頭から何秒目か」は `inClipTime` だけでは分からない。**

#### ❌ 罠2: `TrackItemStartAsTicks / TrackItemEndAsTicks` はトリム後の長さではない

```cpp
// ❌ TrackItemEnd - TrackItemStart をクリップの「表示長さ」として使う
double clipDurSec = (TrackItemEndAsTicks - TrackItemStartAsTicks) / ticksPerSec;
// → メディアソースのイン/アウトポイント間の長さが返る
// → タイムラインでトリムしてもこの値は変わらない！
```

**実測値の例:**
- タイムラインで 5 秒にトリムしたクリップ → `TrackItemEnd - TrackItemStart = 10.67秒`
- これはメディアファイルの使用区間であり、タイムライン上の表示長さではない

#### ❌ 罠3: パラメータが変わらないフレームは再描画されない

```
再生時にアニメーションが動かない（スクラブでは動く）
```

Premiere Pro はパラメータが変化しない場合、レンダリング結果をキャッシュして再利用する。
時間依存アニメーションはパラメータのキーフレームがないため、全フレームが「同じ結果」と判断される。

### 11-2. 正しい取得方法（3つの API を組み合わせる）

| 必要な値 | 取得元 API | 説明 |
|---------|-----------|------|
| **クリップ相対時間** | `AcquireNodeForTime` の `outSegmentOffset` | セグメント内オフセット。シーケンス配置に依存しない |
| **トリム後クリップ長** | `GetSegmentInfo` の `segEnd - segStart` | セグメントのタイムライン上の実際の長さ |
| **キャッシュ無効化** | `PF_OutFlag_NON_PARAM_VARY` + カーネルパラメータにハッシュ | フレームごとに異なるキャッシュキーを生成 |

```
AcquireNodeForTime(segmentsID, sequenceTime, ...)
    → outSegmentOffset = クリップ先頭からの相対時間 (ticks)

GetSegmentInfo(segmentsID, segIndex, &segStart, &segEnd, ...)
    → segEnd - segStart = トリム後クリップ長 (ticks)

TickRate = 254016000000 ticks/秒
```

### 11-3. 前提: `PrGPUFilterBase` の利用可能リソース

`PrGPUFilterBase` を継承した GPU フィルタークラスには、以下が最初から利用可能:

| メンバ | 型 | 用途 |
|--------|------|------|
| `mTimelineID` | `PrTimelineID` | 現在のシーケンスID |
| `mNodeID` | `csSDK_int32` | このエフェクトインスタンスのノードID |
| `mVideoSegmentSuite` | `PrSDKVideoSegmentSuite*` | ノードグラフ探索API |

これらは `Initialize()` 時に SDK が自動取得済みなので、追加の Suite 取得は不要。

### 11-4. 完全な実装コード

#### Render() 内での呼び出し

```cpp
prSuiteError Render(...)
{
    PrTime sequenceTime = inRenderParams->inSequenceTime;
    PrTime clipTime = inRenderParams->inClipTime;  // キャッシュハッシュ用
    const double kTicksPerSecond = 254016000000.0;

    // クリップ相対時間とクリップ長を取得
    PrTime segOffsetTicks = 0, segDurationTicks = 0;
    FindClipBoundsTicks(sequenceTime, &segOffsetTicks, &segDurationTicks);

    // クリップ先頭からの相対時間（秒）
    double clipTimeSec = (double)segOffsetTicks / kTicksPerSecond;
    if (clipTimeSec < 0.0) clipTimeSec = 0.0;

    // クリップの長さ（セグメントのタイムライン上の長さ = トリム後の長さ）
    double clipDurSec = (double)segDurationTicks / kTicksPerSecond;

    // カーネルに渡す
    params.mClipTimeSec = (float)clipTimeSec;

    // GPUキャッシュ無効化: フレームごとに異なる値を設定
    params.mSeqTimeHash = (float)((clipTime / 1000000) % 10000000);
    // ...
}
```

#### FindClipBoundsTicks — segOffset + セグメント長の二重取得

```cpp
void FindClipBoundsTicks(PrTime sequenceTime,
                         PrTime* outSegOffset,
                         PrTime* outSegDuration)
{
    *outSegOffset = 0;
    *outSegDuration = 0;
    if (!mVideoSegmentSuite) return;

    csSDK_int32 segmentsID = 0;
    prSuiteError err = mVideoSegmentSuite->AcquireVideoSegmentsID(
        mTimelineID, &segmentsID);
    if (err != suiteError_NoError) return;

    // ① AcquireNodeForTime でクリップ内オフセットを取得
    csSDK_int32 rootNodeID = 0;
    PrTime segOffset = 0;
    err = mVideoSegmentSuite->AcquireNodeForTime(
        segmentsID, sequenceTime, &rootNodeID, &segOffset);

    if (err == suiteError_NoError && rootNodeID != 0) {
        *outSegOffset = segOffset;
        mVideoSegmentSuite->ReleaseVideoNodeID(rootNodeID);
    }

    // ② GetSegmentInfo でセグメントのタイムライン上の長さを取得
    csSDK_int32 numSegments = 0;
    mVideoSegmentSuite->GetSegmentCount(segmentsID, &numSegments);
    for (csSDK_int32 i = 0; i < numSegments; i++) {
        PrTime segStart = 0, segEnd = 0, segOff = 0;
        prPluginID segHash = {};
        mVideoSegmentSuite->GetSegmentInfo(
            segmentsID, i, &segStart, &segEnd, &segOff, &segHash);
        if (sequenceTime >= segStart && sequenceTime < segEnd) {
            *outSegDuration = segEnd - segStart;
            break;
        }
    }

    mVideoSegmentSuite->ReleaseVideoSegmentsID(segmentsID);
}
```

### 11-5. GPUキャッシュ無効化の必須パターン

時間依存のレンダリングでは、キャッシュ対策を **2箇所** で行う必要がある。

#### ① CPU 側: `PF_OutFlag_NON_PARAM_VARY`

```cpp
// GlobalSetup() 内
out_data->out_flags =
    PF_OutFlag_PIX_INDEPENDENT |
    PF_OutFlag_SEND_UPDATE_PARAMS_UI |
    PF_OutFlag_NON_PARAM_VARY;  // ← これが必須
```

このフラグは「パラメータが変化しなくても出力が変わる」ことをホストに通知する。
これがないと、Premiere Pro は **パラメータが同一のフレームをキャッシュから再利用**し、
再生時にアニメーションが動かない（スクラブでは動く）症状が出る。

#### ② GPU 側: `mSeqTimeHash` でキャッシュキーを変える

```cpp
// ProcAmpParams 構造体（カーネルパラメータの末尾に追加）
float   mClipTimeSec;    // 柄アニメーション用
float   mSeqTimeHash;    // GPUキャッシュ無効化用

// Render() 内で設定
params.mSeqTimeHash = (float)((clipTime / 1000000) % 10000000);
```

`mSeqTimeHash` はカーネル内では使わないが、**カーネルパラメータの一部として渡す**ことで、
Premiere の GPU キャッシュキーがフレームごとに変わり、再レンダリングが強制される。

⚠️ Metal はバッファ丸ごと渡すため、`mSeqTimeHash` は **CPU-only section ではなく、
カーネル引数セクションに配置**する。`.cl` / `.cu` にも同じ位置に追加が必要。

```opencl
// .cl / .cu のカーネル引数末尾
((float)(inClipTimeSec))
((float)(inSeqTimeHash)),  // ← カーネル内では未使用だが引数として必要
```

### 11-6. 重要な注意点

| 項目 | 詳細 |
|------|------|
| **TickRate** | `254016000000` ticks/秒 (Premiere Pro 標準) |
| **`inClipTime` の正体** | メディアソースの絶対タイムコード位置。クリップ相対時間ではない |
| **`segOffset` の正体** | `AcquireNodeForTime` が返す、セグメント内のクリップ相対オフセット。シーケンス配置に依存しない |
| **`TrackItemStart/End` の正体** | メディアソースのイン/アウトポイント（ticks）。タイムラインでのトリムは反映されない |
| **トリム後のクリップ長** | `GetSegmentInfo` の `segEnd - segStart` が正確な値。クリップをトリムするとこの値が追従する |
| **リソース解放** | `AcquireVideoSegmentsID` → `ReleaseVideoSegmentsID`、`AcquireNodeForTime` → `ReleaseVideoNodeID` を必ずペアで呼ぶ |
| **キャッシュ無効化** | `PF_OutFlag_NON_PARAM_VARY`（CPU側）と `mSeqTimeHash`（GPU側）の **両方** が必要 |
| **UX 原則** | ユーザーはクリップの長さを変えるだけでイン/アウトがレスポンシブに追従すべき。追加パラメータでの時間指定は不要 |

### 11-7. デバッグ時の確認値

以下のログ出力パターンで、3つの値が正常か確認できる:

```cpp
GpuLog("ClipTiming: segOffset=%lld segDur=%lld clipTimeSec=%.4f clipDurSec=%.2f",
       (long long)segOffsetTicks, (long long)segDurationTicks, clipTimeSec, clipDurSec);
```

**正常な出力例（5秒クリップをシーケンスの2秒地点から配置）:**
```
ClipTiming: segOffset=0       segDur=1270096128000 clipTimeSec=0.0000 clipDurSec=5.00   ← 先頭
ClipTiming: segOffset=127009  segDur=1270096128000 clipTimeSec=0.0500 clipDurSec=5.00   ← 再生中
ClipTiming: segOffset=126xxx  segDur=1270096128000 clipTimeSec=4.9500 clipDurSec=5.00   ← 末尾付近
```

**異常な出力例（旧手法の場合）:**
```
ClipTiming: clipTimeSec=3600.28  ← inClipTime をそのまま使った場合（メディア絶対時間）
ClipTiming: clipDurSec=10.67     ← TrackItemEnd-Start を使った場合（トリム前のメディア長）
```

### 11-8. ユースケース例

#### イン/アウトアニメーション

```cpp
const float animDuration = 0.5f; // 秒

// イン: クリップ先頭から animDuration 秒で easeOutCubic
if (animType == ANIM_IN || animType == ANIM_IN_OUT) {
    float t = (float)clipTimeSec / animDuration;
    t = clamp(t, 0.0f, 1.0f);
    float inv = 1.0f - t;
    animScale = 1.0f - inv * inv * inv; // easeOutCubic
}

// アウト: クリップ末尾の animDuration 秒で easeInCubic
if (animType == ANIM_OUT || animType == ANIM_IN_OUT) {
    if (clipDurSec > 0.0) {
        double remainSec = clipDurSec - clipTimeSec;
        if (remainSec < (double)animDuration) {
            float t2 = (float)(remainSec / (double)animDuration);
            t2 = clamp(t2, 0.0f, 1.0f);
            float outScale = t2 * t2 * t2; // easeInCubic
            animScale *= outScale;
        }
    }
}
```

#### パターン柄の時間移動

```cpp
// カーネル内 (OpenCL/Metal)
float animOffset = inClipTimeSec * inPatternAnimSpeed;
float2 dir = (float2)(cos(inPatternAngle), sin(inPatternAngle));
float px = localX + dir.x * animOffset;
float py = localY + dir.y * animOffset;
```

---

## 12. 実装完了チェックリスト

派生プラグインの GPU レンダリングを実装した後、以下を確認:

### ビルド前

- [ ] `<Metal/Metal.h>` が include されている
- [ ] バンドル ID が `Info.plist` / `project.pbxproj` / ソースコード内で完全一致
- [ ] `ProcAmpParams` のフィールド順が `_GPU.cpp` / `.cl` / `.cu` / `.metal` で一致
- [ ] `GetFrameDependencies` で入力フレームを要求している
- [ ] `Render` で `mGPUDeviceSuite->GetGPUPPixData()` を使って入出力バッファを取得
- [ ] `Render` で入力 (`inFrames[0]`) と出力 (`*outFrame`) を **別々に** 取得
- [ ] GPU フレームワーク判定が `mDeviceInfo.outDeviceFramework` を使用
- [ ] Metal パイプラインキャッシュが宣言・初期化・解放されている
- [ ] カーネル内のピクセルアクセスが BGRA 順
- [ ] **Popup パラメータに `+1` して 1-based に変換している**（または enum/カーネルが 0-based で統一）
- [ ] **Color パラメータを `PrParam.mType` 判定で取得している**（`.mInt32` 直接アクセス禁止）

### クリップ時間・アニメーション関連

- [ ] **`inClipTime` を直接使用していない**（メディアソース絶対時刻であり、クリップ相対時刻ではない）
- [ ] **`TrackItemStartAsTicks` / `TrackItemEndAsTicks` をクリップ長として使用していない**（トリム前のメディアソース範囲を返す）
- [ ] クリップ相対時刻は `AcquireNodeForTime` の `segOffset` から取得している
- [ ] トリム後のクリップ長は `GetSegmentInfo` の `segEnd - segStart` から取得している
- [ ] アニメーション使用時は `PF_OutFlag_NON_PARAM_VARY` が GlobalSetup で設定されている
- [ ] `mSeqTimeHash` がカーネル引数に渡されている（GPU キャッシュ無効化用）
- [ ] `mClipTimeSec` と `mSeqTimeHash` が `ProcAmpParams` / `.cl` / `.cu` で全て一致している
- [ ] tick → 秒 変換で `254016000000` (kTicksPerSecond) を使用している

### ピクセル変形・バッファ安全性

- [ ] **カーネルで「自座標以外」のピクセルを読む処理がある場合、Blit Copy で入力バッファの一時コピーを作成している**（→ Section 5「レースコンディション」参照）
- [ ] バイリニア補間のサンプル座標がフレーム範囲外になる場合、透明 `(0,0,0,0)` を返すか境界クランプしている
- [ ] 変形不要なフレーム（`animScale == 1.0f` 等）では Blit Copy をスキップしてパフォーマンスを維持している

### ビルド後

```bash
# バイナリにエントリポイントがあるか
nm -gU ".../Contents/MacOS/OST_XXX" | grep EffectMain

# metallib が存在するか
ls -la ".../Contents/Resources/MetalLib/OST_XXX.metalLib"

# PiPL rsrc が存在するか
ls -la ".../Contents/Resources/OST_XXX.rsrc"
```

### 動作確認

1. Premiere Pro にプラグインが表示されること
2. エフェクト適用後にプレビューが真っ黒にならないこと
3. パラメータを操作して描画が変化すること
4. 問題がある場合は `/tmp/OST_XXX_GPU.log` を確認

---

## 付録: テンプレートからの置換対応表

| テンプレート変数 | 置換例 |
|----------------|-------|
| `__TPL_MATCH_NAME__` | `OST_SpeechBubble` |
| `__TPL_UPPER_PREFIX__` | `OST` |
| `__TPL_UPPER_PLUGIN__` | `SPEECH_BUBBLE` |
| バンドル ID | `com.OshareTelop.OSTSpeechBubble` |
| Metal カーネル名 | `PluginKernel` |
| ログファイルパス | `/tmp/OST_SpeechBubble_GPU.log` |
