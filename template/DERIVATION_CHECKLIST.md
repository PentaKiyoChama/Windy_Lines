# テンプレート派生プロジェクト チェックリスト

SpeechBubble プロジェクト作成時に発覚した不備をもとに作成。
次回の派生プロジェクトで同じ問題を踏まないためのガイド。

> **関連ドキュメント**: GPU レンダリング実装の詳細は [`AGENT_IMPLEMENTATION_GUIDE.md`](AGENT_IMPLEMENTATION_GUIDE.md) を参照。
> テンプレートの TODO / スタブを正しく埋める際の罠を網羅したエージェント向け実装マニュアル。

---

## 1. PiPL リソース (.r ファイル)

### 1-1. 必須 include が欠落していた

```r
// ✅ 正しい（角括弧 <> を使用）
#include "AEConfig.h"
#include "AE_EffectVers.h"
#include <AE_General.r>
```

- `AEConfig.h`, `AE_EffectVers.h`, `AE_General.r` がないと Rez が `PiPL` 型を認識できない
- `AE_General.r` は `""` でなく `<>` で include（SDK の REZ_SEARCH_PATHS で解決させる）

### 1-2. エントリポイント名が間違っていた

```r
// ❌ テンプレートの誤り
CodeMacARM64 {"OST_SpeechBubble_GPU"}

// ✅ 正しい（CPU エントリポイント名 = EffectMain）
CodeMacARM64 {"EffectMain"}
```

- PiPL の `CodeMacIntel64` / `CodeMacARM64` / `CodeWin64X86` には **CPU 側のエントリポイント関数名** を指定する
- GPU クラス名（`OST_XXX_GPU`）ではない
- WindyLines 含め全動作プラグインが `"EffectMain"` を使用

### 1-3. 不要な `#ifdef AE_OS_MAC` ネストがあった

```r
// ❌ 不要なネスト（Mac ビルドでも AE_OS_MAC が未定義だと Code が出力されない）
#else
    #ifdef AE_OS_MAC
    CodeMacIntel64 {"EffectMain"},
    CodeMacARM64 {"EffectMain"},
    #endif
#endif

// ✅ 正しい（#else で十分）
#else
    CodeMacIntel64 {"EffectMain"},
    CodeMacARM64 {"EffectMain"},
#endif
```

### 1-4. OutFlags 系プロパティが欠落していた

```r
// ✅ 以下を必ず含めること
AE_Effect_Info_Flags {
    0
},
AE_Effect_Global_OutFlags {
    0x44
},
AE_Effect_Global_OutFlags_2 {
    0x100
},
```

- これらがないと Premiere Pro がプラグインの能力を正しく判定できない

---

## 2. CPU エントリポイント (_CPU.cpp)

### 2-1. 関数名が `EffectMain` でなかった

```cpp
// ❌ テンプレートの誤り
DllExport
PF_Err PluginMain(...)

// ✅ 正しい
extern "C" DllExport
PF_Err EffectMain(...)
```

### 2-2. `extern "C"` が欠落していた

- `DllExport` マクロは `__attribute__((visibility("default")))` のみ
- **C リンケージを自分で付ける必要がある**（`extern "C"`）
- これがないとシンボルが C++ マングル名 `__Z10EffectMain...` になり、Premiere Pro が見つけられない
- WindyLines は `extern "C" DllExport PF_Err EffectMain(...)` と記述している

### 2-3. PF_PixelFormatSuite 登録が欠落していた

```cpp
// ✅ GlobalSetup 内に追加必須
if (in_data->appl_id == 'PrMr')
{
    AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite(
        in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);
    (*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
    (*pixelFormatSuite->AddSupportedPixelFormat)(
        in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
}
```

- Premiere Pro GPU プラグインでは必須
- `PrSDKAESupport.h` に定義されている

---

## 3. Xcode プロジェクト (project.pbxproj)

### 3-1. PBXRezBuildPhase の files が空だった

```
// ❌ テンプレートの状態
CC000004 /* Rez */ = {
    isa = PBXRezBuildPhase;
    files = (
    );  // ← .r ファイルが入っていない
};
```

**対処法（2 つの選択肢）：**

#### 方法 A: PBXRezBuildPhase に .r を登録する（WindyLines 方式）

1. `PBXBuildFile` セクションに追加:
   ```
   AA000007 /* OST_XXX.r in Rez */ = {
       isa = PBXBuildFile;
       fileRef = BB000007 /* OST_XXX.r */;
   };
   ```
2. `PBXRezBuildPhase` の files に追加:
   ```
   files = (
       AA000007 /* OST_XXX.r in Rez */,
   );
   ```

> ⚠️ Xcode が "Build Carbon Resources build phases are no longer supported" と警告を出すが、
> 実際には動作する（Xcode 16.2 時点で確認済み）

#### 方法 B: Shell Script で Rez を実行する

PBXRezBuildPhase を削除し、PBXShellScriptBuildPhase に置き換え:

```bash
set -e
REZ_SOURCE="${SRCROOT}/../OST_XXX.r"
RSRC_DIR="${TARGET_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/Resources"
RSRC_OUT="${RSRC_DIR}/${PRODUCT_NAME}.rsrc"
mkdir -p "$RSRC_DIR"
/usr/bin/Rez \
  "$REZ_SOURCE" \
  -o "$RSRC_OUT" \
  -useDF \
  -I "${AE_SDK_BASE_PATH}/Examples/Headers" \
  -I "${AE_SDK_BASE_PATH}/Examples/Resources" \
  -I "${PREMIERE_SDK_BASE_PATH}/Examples/Headers" \
  -I "${PREMIERE_SDK_BASE_PATH}/Examples/Resources"
```

**重要ポイント:**
- 出力先は `Contents/Resources/${PRODUCT_NAME}.rsrc`（データフォーク = `-useDF`）
- バイナリ本体に `-useDF` で書くとバイナリが破壊される
- バイナリのリソースフォーク（xattr）に書くと Deploy の `xattr -cr` で消える

### 3-2. Deploy スクリプトの `xattr -cr` がリソースフォークを消していた

```bash
# ❌ 全 xattr 削除（リソースフォークも消える）
xattr -cr "$BUILT"

# ✅ quarantine のみ削除（リソースフォークを保持）
find "$BUILT" -exec xattr -d com.apple.quarantine {} \; 2>/dev/null || true
```

- `com.apple.ResourceFork` は macOS 上でリソースフォークを保持する xattr
- `xattr -cr` で一括削除すると PiPL が消失する
- 方法 B（.rsrc ファイル）を使う場合はデータフォークなので影響を受けない

---

## 4. GPU カーネル (.cu / .cl)

### 4-1. 存在しないマクロを使用していた

```c
// ❌ SDK に存在しないマクロ
GF_PIXEL_FLOAT pixel;
GF_LOAD_PIXEL(pixel, ...);
GF_STORE_PIXEL(pixel, ...);

// ✅ SDK 標準の API
float4 pixel = ReadFloat4(inBuf, y * inPitch + x, !!in16f);
WriteFloat4(pixel, outBuf, y * outPitch + x, !!in16f);
```

### 4-2. ピクセルメンバーアクセスが間違っていた

```c
// ❌ RGBA 前提
pixel.red, pixel.green, pixel.blue, pixel.alpha

// ✅ BGRA (float4: x=B, y=G, z=R, w=A)
pixel.z  // Red
pixel.y  // Green
pixel.x  // Blue
pixel.w  // Alpha
```

### 4-3. カーネル引数に `inPitch` / `in16f` が欠落していた

- `ReadFloat4` / `WriteFloat4` が pitch と 16f フラグを必要とする
- `_GPU.cpp` の ProcAmpParams 構造体と三重同期が必要

---

## 5. テンプレート改善 TODO

テンプレート自体に以下の修正を反映すべき:

- [ ] `.r` ファイルに 3 つの include を含めておく
- [ ] `.r` のエントリポイントを `"EffectMain"` にする
- [ ] `.r` に OutFlags 系プロパティを含める
- [ ] `_CPU.cpp` のエントリポイントを `extern "C" DllExport PF_Err EffectMain(...)` にする
- [ ] `_CPU.cpp` の GlobalSetup に PF_PixelFormatSuite 登録を含める
- [ ] `project.pbxproj` の PBXRezBuildPhase に .r ファイル参照を含める（または Shell Script 方式にする）
- [ ] Deploy スクリプトで `xattr -cr` を使わない
- [ ] GPU カーネルテンプレートで `ReadFloat4`/`WriteFloat4` と BGRA アクセスを使う
- [ ] 派生スクリプトで .r ファイル内のプラグイン名・マッチ名の置換を行う
- [ ] `_GPU.cpp` に `<Metal/Metal.h>` の include を追加
- [ ] `_GPU.cpp` の `GetFrameDependencies` で入力フレームを要求する実装に修正
- [ ] `_GPU.cpp` の `Render` で `GetGPUPPixData` による入出力バッファ取得を実装
- [ ] `_GPU.cpp` の GPU フレームワーク判定を `mDeviceInfo.outDeviceFramework` に修正
- [ ] `_GPU.cpp` に Metal パイプライン初期化 / キャッシュ / Shutdown を実装
- [ ] `_GPU.cpp` のバンドル ID をテンプレート変数化（`__TPL_BUNDLE_ID__`）
- [ ] 派生時に `AGENT_IMPLEMENTATION_GUIDE.md` をプロジェクトの `docs/` にコピー

---

## クイック検証コマンド

派生プロジェクトをビルド後、以下で確認:

```bash
# 1. エントリポイントが C リンケージか確認
nm -gU ".../Contents/MacOS/OST_XXX" | grep EffectMain
# → "_EffectMain" なら OK、"__Z10EffectMain..." なら extern "C" が欠落

# 2. PiPL rsrc ファイルが存在し中身があるか確認
ls -la ".../Contents/Resources/OST_XXX.rsrc"
# → 600+ バイトあれば OK

# 3. rsrc の中身を確認
xxd ".../Contents/Resources/OST_XXX.rsrc" | grep "8BIM"
# → kind, name, catg, mi64, ma64, ePVR, eSVR, eVER, eINF, eGLO, eGL2, eMNA, aeFL が見えれば OK

# 4. WindyLines との構造差分
diff <(find ".../OST_WindyLines.plugin" -type f | sort | sed 's/WindyLines/XXX/g') \
     <(find ".../OST_XXX.plugin" -type f | sort | sed 's/XXX_NAME/XXX/g')
```
