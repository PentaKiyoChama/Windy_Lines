# Mac版 ライセンス認証実装の詳細（Windows移植用リファレンス）

最終更新: 2026年2月13日
対象: OST_WindyLines v53

---

## 概要

Mac版では以下の仕組みでライセンス認証によるウォーターマーク表示制御を実現し、
**実機テストで有効/無効の自動切替を確認済み**。

- サーバー側で有効/無効を切り替えると、**最大10分以内にプラグインが自動反映**
- ユーザーは何も操作しなくてよい

---

## アーキテクチャ（Mac版で動作確認済み）

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│ 自社サイトAPI    │────▶│ ローカルキャッシュ │────▶│ プラグイン     │
│ authorized:bool │     │ license_cache_v1 │     │ 描画判定      │
└─────────────────┘     └──────────────────┘     └──────────────┘
        ▲                       ▲
        │                       │
   バックグラウンド          0.2秒毎に
   curl (system())          ファイル読込
   キャッシュ期限切れ時
```

### フロー詳細

1. **レンダー毎**: `RefreshLicenseAuthenticatedState(false)` が呼ばれる
2. 0.2秒のデバウンス後、`LoadLicenseAuthenticatedFromCache()` でキャッシュファイル読込
3. キャッシュが有効 → `sLicenseAuthenticated` を更新 → 描画判定に使用
4. キャッシュが無効（期限切れ or ファイルなし）→ `sLicenseAuthenticated=false`（未認証）→ `TriggerBackgroundCacheRefresh()` を呼ぶ
5. バックグラウンドでAPI問い合わせ → キャッシュファイル書き換え
6. 次のレンダーで新しいキャッシュが読み込まれ、状態が反映

---

## 1. キャッシュファイル

### パス
- macOS: `~/Library/Application Support/OST/WindyLines/license_cache_v1.txt`
- Windows: `%APPDATA%\OST\WindyLines\license_cache_v1.txt`

### 形式（プレーンテキスト、key=value）
```
authorized=true
reason=ok
validated_unix=1770963000
cache_expire_unix=1770963600
license_key_masked=
machine_id_hash=auto_refresh
```

### キャッシュ読込ロジック（CPU/GPU共通）
```
1. ファイルを開く（複数候補パスを順に試す）
2. authorized= と cache_expire_unix= を読み取る
3. 両方存在しなければ → return false（キャッシュ無効）
4. cache_expire_unix <= 現在時刻 → return false（期限切れ）
5. outAuthenticated = authorized の値を返す
```

---

## 2. CPU側認証ロジック（OST_WindyLines_CPU.cpp）

### 関数一覧と役割

| 関数名 | 役割 |
|--------|------|
| `GetLicenseCachePaths()` | キャッシュファイルの候補パスリストを返す |
| `LoadLicenseAuthenticatedFromCache()` | キャッシュ読込→期限チェック→authorized値返却 |
| `RefreshLicenseAuthenticatedState(force)` | 0.2秒デバウンスでキャッシュ再読込、期限切れなら自動更新発動 |
| `TriggerBackgroundCacheRefresh()` | バックグラウンドAPIコール→キャッシュ書換（Mac: system+curl） |
| `IsLicenseAuthenticated()` | atomic bool を読むだけ（レンダー中安全） |

### 呼び出しポイント
- `GlobalSetup()`: `RefreshLicenseAuthenticatedState(true)` — 起動時に強制読込
- `SmartRender()` 内: `RefreshLicenseAuthenticatedState(false)` — レンダー毎（デバウンス付き）
- `SmartRender()` 内ピクセルループ: `if (!IsLicenseAuthenticated())` でウォーターマーク描画

### 自動更新の仕組み（Mac版 — 動作確認済み）

キャッシュ期限切れ時に `system()` でバックグラウンドシェルを起動:

```sh
(resp=$(/usr/bin/curl -s -m 10 -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action":"verify","product":"OST_WindyLines","plugin_version":"v53.0.0","platform":"mac"}' \
  'https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test' 2>/dev/null); \
if /usr/bin/printf '%s' "$resp" | /usr/bin/grep -q '"authorized"'; then \
  if /usr/bin/printf '%s' "$resp" | /usr/bin/grep -q '"authorized"[[:space:]]*:[[:space:]]*true'; then \
    auth=true; reason=ok; \
  else \
    auth=false; reason=denied; \
  fi; \
  now=$(/bin/date +%s); expire=$((now + 600)); \
  /bin/mkdir -p '<cache_dir>'; \
  tmp=$(/usr/bin/mktemp /tmp/ost_wl_cache_XXXXXX); \
  /usr/bin/printf 'authorized=%s\nreason=%s\n...' "$auth" "$reason" "$now" "$expire" > "$tmp" && \
  /bin/mv "$tmp" '<cache_path>'; \
fi) >/dev/null 2>&1 &
```

**ポイント**:
- 末尾の `&` でバックグラウンド実行（レンダースレッドをブロックしない）
- `/usr/bin/curl` 等フルパス指定（Premiere Pro環境ではPATHが制限される）
- TTLは600秒（10分）
- 最小60秒間隔で発動（連続呼び出し防止）
- `atomic<bool> sAutoRefreshInProgress` で多重発動防止

### Windows版の課題

Windows版では `CreateProcessA + powershell.exe` を試したが、
Premiere Pro内のレンダースレッドから実行すると**画面が真っ暗になる問題**が発生。
現在は無効化されている（DebugLogのみ）。

**Windows版で検討すべき代替案**:

1. **WinHTTPを直接使う（最推奨）**
   - `WinHttpOpen` → `WinHttpConnect` → `WinHttpOpenRequest` → `WinHttpSendRequest`
   - 別スレッド（`std::thread` + detach）で実行すればレンダーをブロックしない
   - PowerShellプロセス起動が不要なので軽量

2. **URLDownloadToFile / URLOpenBlockingStream**
   - COM/URLMonベースだがPOST対応が面倒

3. **バッチファイル + CreateProcess**
   - PowerShellの代わりに `curl.exe`（Win10以降標準搭載）を使う
   - `CREATE_NO_WINDOW | DETACHED_PROCESS` で起動
   - PowerShellより軽量で問題が起きにくい可能性あり

---

## 3. GPU側認証ロジック（OST_WindyLines_GPU.cpp）

### Mac版の元実装（CPUフォールバック方式）
```cpp
// Render() 先頭
if (!IsGpuLicenseAuthenticated()) {
    return suiteError_Fail;  // → CPUにフォールバック → CPU側でウォーターマーク描画
}
```
- macOSではこれで正常動作（CPUレンダーにフォールバックしてウォーターマーク適用）

### Windows版での問題
- `suiteError_Fail` を返すと**画面が真っ暗**になる（CPUフォールバックが正しく動かない）
- Windows版エージェントがGPUフォールバックを無効化し、GPU各経路にウォーターマーク直接描画を追加して解決

### 現在の構成（Windows版エージェント対応後）
- CUDA経路: `WatermarkOverlay_CUDA()` で直接描画
- OpenCL経路: ウォーターマークカーネルで直接描画
- Metal経路: ウォーターマークカーネルで直接描画（macOS専用）
- GPU各経路で `if (!IsGpuLicenseAuthenticated())` チェック後にウォーターマーク合成

---

## 4. APIレスポンス形式

### エンドポイント
```
POST https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test
Content-Type: application/json
```

### リクエスト
```json
{
  "action": "verify",
  "product": "OST_WindyLines",
  "plugin_version": "v53.0.0",
  "platform": "mac"
}
```
- `license_key` は現在省略可能（サーバー側でキーなし対応済み）

### レスポンス（有効時）
```json
{
  "status": "success",
  "response": {
    "authorized": true
  }
}
```

### レスポンス（無効時）
```json
{
  "status": "success",
  "response": {
    "authorized": false
  }
}
```

### パース優先順位（activate_license_cache.py と同じ）
1. トップレベル `authorized` (boolean)
2. `response.authorized` (boolean)
3. `response.Parameter 1` (string: "authorized"/"true" → 有効)

---

## 5. Mac版で確認済みのテスト結果

| テスト | サーバー側 | 期待動作 | 結果 |
|--------|-----------|---------|------|
| 有効状態 | authorized: true | ウォーターマーク非表示 | PASS |
| 無効状態 | authorized: false | ウォーターマーク表示 | PASS |
| 有効→無効 自動反映 | true → false切替後放置 | 10分以内に表示 | PASS |
| 無効→有効 自動反映 | false → true切替後放置 | 10分以内に非表示 | PASS（ユーザー確認済み） |

---

## 6. Windows版で未解決の課題

### サーバー有効/無効の自動切替が動かない
- 原因: `TriggerBackgroundCacheRefresh()` のWindows実装が無効化されている
- キャッシュが期限切れになっても新しいAPI結果を取得できない
- 手動で `activate_license_cache.py` を実行すれば反映されるが、自動ではない

### 推奨修正方針
1. `TriggerBackgroundCacheRefresh()` の `#else`（_WIN32）ブロックを実装する
2. WinHTTP API を `std::thread` + detach で呼ぶのが最も安全
3. または `curl.exe`（Win10標準搭載）を `CreateProcessA` で呼ぶ

### WinHTTP実装の骨格（参考）
```cpp
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

// 別スレッドで実行
std::thread([cachePath, cacheDir, ttlSec]() {
    HINTERNET hSession = WinHttpOpen(L"OST_WindyLines/1.0", 
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, NULL, NULL, 0);
    HINTERNET hConnect = WinHttpConnect(hSession, 
        L"penta.bubbleapps.io", INTERNET_DEFAULT_HTTPS_PORT, 0);
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST",
        L"/version-test/api/1.1/wf/ppplugin_test", NULL, NULL, NULL,
        WINHTTP_FLAG_SECURE);
    
    std::string body = "{\"action\":\"verify\",\"product\":\"OST_WindyLines\","
        "\"plugin_version\":\"" OST_WINDYLINES_VERSION_FULL "\",\"platform\":\"win\"}";
    
    WinHttpSendRequest(hRequest, L"Content-Type: application/json",
        -1, (LPVOID)body.c_str(), body.size(), body.size(), 0);
    WinHttpReceiveResponse(hRequest, NULL);
    
    // レスポンス読み取り → "authorized":true/false を grep → キャッシュ書込
    // ... (Mac版のシェルスクリプトと同等のロジック)
    
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
}).detach();
```

### curl.exe方式（より簡単、Win10以降）
```cpp
std::string cmd = "cmd.exe /c \"curl.exe -s -m 10 -X POST "
    "-H \"Content-Type: application/json\" "
    "-d \"{...}\" \"" + endpoint + "\" > %TEMP%\\ost_wl_resp.tmp "
    "&& powershell -NoProfile -Command \"...parse and write cache...\"\"";

STARTUPINFOA si = {}; si.cb = sizeof(si);
si.dwFlags = STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
PROCESS_INFORMATION pi = {};
CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE,
    CREATE_NO_WINDOW | DETACHED_PROCESS, nullptr, nullptr, &si, &pi);
CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
```

---

## 7. 補助ツール

### activate_license_cache.py（手動即時反映）
```bash
python3 ./activate_license_cache.py \
  --endpoint "https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test"
```
- ライセンスキーは省略可能
- API応答をローカルキャッシュに即時書込
- 自動更新が動かない場合の手動代替

### set_license_cache_state.py（デバッグ用強制切替）
```bash
python3 ./set_license_cache_state.py --state authorized --ttl 86400
python3 ./set_license_cache_state.py --state unauthorized --ttl 86400
```
