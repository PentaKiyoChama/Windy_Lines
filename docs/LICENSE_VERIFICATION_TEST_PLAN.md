# ライセンス検証パターン一覧（ペルソナ別）

## 前提：現在のセキュリティ仕様

| 項目 | 値 |
|------|-----|
| TTL（認証済み / 拒否共通） | 600秒（10分） |
| オフライン猶予 | 3600秒（1時間） |
| 署名方式 | DJB2ダブルハッシュ + XOR難読化salt |
| 署名フィールド名 | `cache_signature` |
| ポストアクティベーション | 5秒間隔 × 2分間 |
| キャッシュ再読み間隔 | 200ms（レンダー呼出毎） |
| API最小呼出間隔 | 60秒（rapid中は5秒） |

### キャッシュファイルパス
```
Mac: ~/Library/Application Support/OshareTelop/license_cache_v1.txt
Win: %APPDATA%\OshareTelop\license_cache_v1.txt
```

### ライセンス実装の差分一覧

| 要素 | CPU レンダラー | GPU レンダラー | 備考 |
|------|-----------|-----------|------|
| キャッシュファイル | 共通（`license_cache_v1.txt`） | 共通（同じファイル） | 読み書きの競合はファイル単位のアトミック性に依存 |
| ライセンス実装 | `OST_WindyLines_CPU.cpp` に全ロジック | `OST_WindyLines_License.h` 経由で CPU 関数を呼出し | **GPU は独立実装ではなく共有** |
| 署名検証 | `ComputeCacheSignature` | 同左（共有） | `License.h` → `RefreshLicenseAuthenticatedState` 経由 |
| API 呼出し | `TriggerBackgroundCacheRefresh` | 同左（共有） | GPU からも CPU 側の同一関数が呼ばれる |
| HTTP 実装 (Mac) | `std::thread` + `popen("curl ...")` | 同左（共有） | 同じ |
| HTTP 実装 (Win) | `std::thread` + `WinHTTP` | 同左（共有） | 同じ |
| MID 生成 | `GetMachineIdHash` | 同左（共有） | 同一関数 |

| 要素 | Mac | Windows | 備考 |
|------|-----|---------|------|
| キャッシュパス | `~/Library/Application Support/OshareTelop/` | `%APPDATA%\OshareTelop\` | |
| HTTP | `popen("curl ...")` | `WinHTTP` API | 完全に異なる実装 |
| MID 構成 | `hostname\|mac\|8` | `hostname\|win\|8` | platform 文字列が異なる |
| デバッグログ | Console.app / `log stream` | `OutputDebugStringA` → DebugView | |
| ファイル操作 | `rm` / `cat` | `del` / `type` | |

### 略語
- **WM** = ウォーターマーク
- **PP** = Premiere Pro
- **TTL** = キャッシュ有効期限（10分）

---

## ペルソナ A：正規ユーザー（サブスク有効、Bubble に MID 登録済み） `[Mac / CPU]`

### A-1. 初回起動（キャッシュなし）

> **所要時間：約 2 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `rm ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 2 | PP を起動（または既に起動中ならそのまま） | — |
| 3 | タイムライン上のクリップに WindyLines エフェクトを適用 | — |
| 4 | **確認①**: プレビューに WM が **表示される** | — |
| 5 | **そのまま 60〜90 秒待つ**（バックグラウンド API が走る） | 60〜90秒 |
| 6 | タイムラインのインジケータを 1 フレーム前後に動かして再描画させる | — |
| 7 | **確認②**: WM が **消えている** | — |
| 8 | ターミナルで `cat ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 9 | **確認③**: `authorized=true` と `cache_signature=` が存在する | — |

---

### A-2. 10 分経過後の自動再検証（オフライン猶予で維持）

> **所要時間：約 2 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `python3 set_license_cache_state.py --state authorized --ttl 1` | — |
| 2 | **3 秒待つ**（TTL を確実に超過させる） | 3秒 |
| 3 | PP でタイムラインを動かして再描画 | — |
| 4 | **確認①**: WM が **表示されない**（オフライン猶予 1h 以内） | — |
| 5 | **60〜90 秒待つ**（バックグラウンドで再検証が走る） | 60〜90秒 |
| 6 | ターミナルで `cat ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 7 | **確認②**: `validated_unix` と `cache_expire_unix` が更新され、`cache_signature` も新しい値に変わっている | — |

---

### A-3. Wi-Fi 一時切断（オフライン猶予内：30 分）

> **所要時間：約 15 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `python3 set_license_cache_state.py --state authorized --ttl 600` | — |
| 2 | PP でプレビュー → WM なしを確認 | — |
| 3 | **Mac の Wi-Fi を OFF にする**（メニューバー → Wi-Fi → 切） | — |
| 4 | **12 分待つ**（TTL 10分を超過） | 12分 |
| 5 | PP でタイムラインを動かして再描画 | — |
| 6 | **確認①**: WM が **表示されない**（猶予 1h 以内） | — |
| 7 | Wi-Fi を ON に戻す | — |
| 8 | **60〜90 秒待つ**（再接続後のバックグラウンド再検証） | 60〜90秒 |
| 9 | **確認②**: キャッシュが新しい TTL で更新されている | — |

---

### A-4. 長時間オフライン（猶予超過：1h 以上）

> **所要時間：約 1 時間 15 分（実時間放置）** or スクリプトで即時テスト

#### 方法 A：実時間テスト
| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | `python3 set_license_cache_state.py --state authorized --ttl 600` | — |
| 2 | Mac の Wi-Fi を OFF にする | — |
| 3 | **1 時間 5 分放置する** | 65分 |
| 4 | PP でタイムラインを動かして再描画 | — |
| 5 | **確認**: WM が **表示される** | — |
| 6 | Wi-Fi を ON に戻して復帰確認（60〜90秒後 WM 消失） | — |

#### 方法 B：スクリプトで即時テスト（validated_unix を古い値で署名ごと生成）
| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | 以下のコマンドを実行（validated を 2h 前、TTL=1 で即期限切れキャッシュを生成）: | — |
| | `python3 set_license_cache_state.py --state authorized --ttl 1 --validated-ago 7200` | — |
| 2 | PP でタイムラインを動かして再描画 | — |
| 3 | **確認**: WM が **表示される**（validated から 1h 超過） | — |

---

## ペルソナ B：正規ユーザー（サブスク解約直後） `[Mac / CPU]`

### B-1. 解約後 → 最大 10 分で検知

> **所要時間：約 12 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `python3 set_license_cache_state.py --state authorized --ttl 600` | — |
| 2 | PP でプレビュー → WM なしを確認 | — |
| 3 | **Bubble 管理画面**で対象ユーザーの authorized を `no` に変更 | — |
| 4 | **ここから時計スタート** | — |
| 5 | PP でそのまま作業を続ける（WM なしのまま） | — |
| 6 | **10 分待つ**（TTL 失効タイミング） | 10分 |
| 7 | TTL 切れ → バックグラウンドで API 再検証 → authorized=false に更新 | — |
| 8 | PP でタイムラインを動かして再描画 | — |
| 9 | **確認**: WM が **表示される** | — |
| 10 | ターミナルで `cat ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 11 | **確認**: `authorized=false` に更新されている | — |

---

### B-2. 解約後に再契約 → 10 分以内に復帰

> **所要時間：約 3 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `python3 set_license_cache_state.py --state unauthorized --ttl 1` | — |
| 2 | **3 秒待つ**（TTL 超過） | 3秒 |
| 3 | **Bubble 管理画面**で対象ユーザーの authorized を `yes` に変更 | — |
| 4 | PP でタイムラインを動かして再描画 | — |
| 5 | （TTL 切れ → バックグラウンドで API 再検証が走る） | — |
| 6 | **60〜90 秒待つ** | 60〜90秒 |
| 7 | PP でタイムラインを動かして再描画 | — |
| 8 | **確認**: WM が **消えている** | — |

---

## ペルソナ C：未課金ユーザー（Free 版、Bubble に MID 未登録） `[Mac / CPU]`

### C-1. 常時ウォーターマーク

> **所要時間：約 3 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `rm ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 2 | PP でエフェクトを適用してプレビュー | — |
| 3 | **確認①**: WM が **表示される** | — |
| 4 | **2 分待つ** | 2分 |
| 5 | PP でタイムラインを動かして再描画 | — |
| 6 | **確認②**: WM が **まだ表示されている**（API は authorized=false を返し続ける） | — |
| 7 | ターミナルで `cat ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 8 | **確認③**: `authorized=false` のキャッシュが生成されている | — |

---

## ペルソナ D：新規ユーザー（初めてアクティベートするユーザー） `[Mac / CPU]`

### D-1. 「アクティベート…」ボタン押下 → ブラウザ遷移

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `rm ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 2 | PP でエフェクトを適用 | — |
| 3 | エフェクトコントロールパネルの「アクティベート」ポップアップをクリック | — |
| 4 | 「アクティベート...」（2番目の項目）を選択 | — |
| 5 | **確認①**: デフォルトブラウザが開き、URL に以下が含まれる: | — |
| | `?token=（32桁hex）&mid=（16桁hex）&product=OST_WindyLines&ver=（バージョン）` | — |
| 6 | **確認②**: ブラウザのページにトークン・MID が表示される | — |

---

### D-2. ブラウザでアクティベート完了 → 即座にウォーターマーク消失

> **所要時間：約 30 秒**（D-1 の直後に実行）

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | D-1 で開いたブラウザページで「アクティベート」ボタンを押す | — |
| 2 | ブラウザ側で「アクティベーション成功」を確認 | — |
| 3 | PP に切り替える（⌘+Tab） | — |
| 4 | **5〜10 秒待つ**（rapid mode: 5秒間隔でチェック中） | 5〜10秒 |
| 5 | PP でタイムラインを 1 フレーム動かして再描画 | — |
| 6 | **確認①**: WM が **消えている** | — |
| 7 | ターミナルで `cat ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 8 | **確認②**: `authorized=true` + `cache_signature=` が存在する | — |

---

### D-3. ブラウザでアクティベートせずに閉じた → 2 分で rapid mode 終了

> **所要時間：約 3 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | D-1 の手順 1〜4 を実行（ブラウザが開く） | — |
| 2 | ブラウザを**アクティベートせずに閉じる** | — |
| 3 | PP でプレビュー → WM 表示のまま | — |
| 4 | **2 分 10 秒待つ**（rapid window 120秒 + α） | 2分10秒 |
| 5 | PP でタイムラインを動かして再描画 | — |
| 6 | **確認**: WM が **まだ表示されている**（rapid 終了、通常 60秒間隔に戻る） | — |

---

## ペルソナ E：カジュアル改ざんユーザー `[Mac / CPU]`

### E-1. authorized=true に手書き変更

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `python3 set_license_cache_state.py --state unauthorized --ttl 600` | — |
| 2 | PP でプレビュー → WM 表示を確認 | — |
| 3 | テキストエディタで `~/Library/Application Support/OshareTelop/license_cache_v1.txt` を開く | — |
| 4 | `authorized=false` を `authorized=true` に書き換えて保存 | — |
| 5 | PP でタイムラインを動かして再描画 | — |
| 6 | **確認**: WM が **まだ表示されている**（署名不一致で unauthorized 扱い） | — |

---

### E-2. cache_expire_unix を遠い未来に変更 → 署名不一致で検知

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `python3 set_license_cache_state.py --state authorized --ttl 5` | — |
| 2 | テキストエディタでキャッシュファイルを開く | — |
| 3 | `cache_expire_unix=（元の値）` を `cache_expire_unix=9999999999` に書き換えて保存 | — |
| 4 | **7 秒待つ**（元の TTL 5秒を超過） | 7秒 |
| 5 | PP でタイムラインを動かして再描画 | — |
| 6 | **確認**: WM が **表示される**（expire は署名対象に含まれるため署名不一致） | — |

---

### E-3. cache_signature 行を削除

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `python3 set_license_cache_state.py --state authorized --ttl 600` | — |
| 2 | PP でプレビュー → WM なしを確認 | — |
| 3 | テキストエディタでキャッシュファイルを開く | — |
| 4 | `cache_signature=xxxx...` の行を**削除**して保存 | — |
| 5 | PP でタイムラインを動かして再描画 | — |
| 6 | **確認**: WM が **表示される**（署名なし = unauthorized 扱い） | — |

---

### E-4. 別マシンのキャッシュファイルをコピー

> **所要時間：約 2 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | テキストエディタでキャッシュファイルを開く | — |
| 2 | `machine_id_hash=（現在の値）` を `machine_id_hash=aaaa1111bbbb2222` に書き換えて保存 | — |
| 3 | ※ これは「別マシンからコピーしてきた」のと同等 | — |
| 4 | PP でタイムラインを動かして再描画 | — |
| 5 | **確認**: WM が **表示される**（MID 不一致） | — |
| 6 | **60〜90 秒待つ**（自分の MID で API 再検証） | 60〜90秒 |
| 7 | PP でタイムラインを動かして再描画 | — |
| 8 | **確認**: WM 状態が正しい MID での認証結果に従っている | — |

---

## ペルソナ F：高度な改ざんユーザー `[Mac / CPU]`

### F-1. バイナリから salt を特定して署名を再計算

> **テスト方法：机上検証**（実行任意）

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `strings /Library/Application\ Support/Adobe/Common/Plug-ins/7.0/MediaCore/OST_WindyLines.plugin/Contents/MacOS/OST_WindyLines \| grep SALT` | — |
| 2 | **確認**: salt 文字列が **見つからない**（XOR 難読化済み） | — |
| 3 | 逆アセンブラ（Hopper / Ghidra 等）でバイナリを解析し、XOR デコードルーチンを特定する必要がある | — |
| | **リスク評価**: 逆アセンブラ＋暗号解析スキルが必要。一般ユーザーには到達不可能。 | |
| | **将来の対策**: サーバー発行 JWT、コード署名検証 | |

---

### F-2. MITM プロキシで API レスポンスを偽装

> **テスト方法：机上検証**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | Charles Proxy 等で HTTPS 通信を傍受しようとする | — |
| 2 | **確認**: HTTPS 証明書検証により通常は失敗 | — |
| | **リスク評価**: 低（ルート証明書インストール + プロキシ設定が必要） | |

---

## ペルソナ G：プラグインアップデート後のユーザー `[Mac / CPU]`

### G-1. v1.0 → v1.1 へアップデート後もキャッシュが引き継がれる

> **所要時間：約 5 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `python3 set_license_cache_state.py --state authorized --ttl 600` | — |
| 2 | PP でエフェクト適用 → WM なしを確認 | — |
| 3 | ターミナルで `cat ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` → 内容をメモ | — |
| 4 | PP を終了する | — |
| 5 | **プラグインを新しいビルドで上書きインストール**（`Mac/install_plugin.sh`） | — |
| 6 | PP を再起動し、エフェクト適用済みクリップをプレビュー | — |
| 7 | **確認①**: WM が **表示されない**（キャッシュファイルはプラグインバイナリとは別パスなので残る） | — |
| 8 | ターミナルでキャッシュ確認 → 手順 3 でメモした内容と一致する | — |
| 9 | **60〜90 秒待つ**（バックグラウンド再検証が走る） | 60〜90秒 |
| 10 | **確認②**: キャッシュが新しい `validated_unix` で更新されている（plugin_version が変わっても API は authorized を返す） | — |

> **⚠️ 署名フォーマット変更時の注意**: 署名ペイロードの構成が変更されたビルド（例: `expire_unix` 追加）にアップデートした場合、旧ビルドで生成されたキャッシュは署名不一致となる。この場合、手順 7 で **一時的に WM が表示される** が、60〜90 秒後のバックグラウンド再検証で自動回復する（手順 10 で WM 消失を確認）。これは正常動作であり、ユーザー体験上は「アップデート後 1〜2 分で元に戻る」だけの影響。

---

## ペルソナ H：複数 Mac 所有ユーザー（同一アカウント） `[Mac / CPU]`

### H-1. 2 台目の Mac に初回インストール

> **所要時間：約 2 分**（2 台目のマシンで実行）

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | **2 台目の Mac** にプラグインをインストール | — |
| 2 | PP を起動、エフェクトを適用 | — |
| 3 | **確認①**: WM が **表示される**（この MID は Bubble 未登録） | — |
| 4 | エフェクトコントロールから「アクティベート...」を選択 | — |
| 5 | ブラウザの URL で `mid=` パラメータを確認 → **1 台目とは異なる MID** であること | — |
| 6 | ブラウザでアクティベートを実行（Bubble に 2 台目の MID が登録される） | — |
| 7 | **5〜10 秒待つ**（rapid mode） | 5〜10秒 |
| 8 | PP でタイムラインを動かして再描画 | — |
| 9 | **確認②**: WM が **消えている** | — |

### H-2. 1 台目のキャッシュを 2 台目にコピーしても無効

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | 1 台目の `license_cache_v1.txt` を USB メモリ等で 2 台目にコピー | — |
| 2 | 2 台目のキャッシュパスに上書き保存 | — |
| 3 | 2 台目の PP でタイムラインを動かして再描画 | — |
| 4 | **確認**: WM が **表示される**（MID 不一致 → 署名検証で machine_id_hash が合わない） | — |

> **補足**: E-4 と原理は同じだが、こちらは「実際にユーザーがやりそうな」自然なシナリオ。

---

## ペルソナ I：共有 Mac の別 macOS アカウント利用者 `[Mac / CPU]`

### I-1. 同一 Mac・別アカウント → キャッシュは完全独立

> **所要時間：約 3 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | **ユーザー A のアカウント**でログイン中に `python3 set_license_cache_state.py --state authorized --ttl 600` | — |
| 2 | PP でプレビュー → WM なしを確認 | — |
| 3 | macOS で**ユーザー B のアカウント**に切り替え（ファストユーザスイッチ or ログアウト→ログイン） | — |
| 4 | ユーザー B で PP を起動、エフェクトを適用 | — |
| 5 | **確認①**: WM が **表示される**（ユーザー B の `~/Library/...` にはキャッシュがない） | — |
| 6 | ターミナルで `cat ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 7 | **確認②**: ファイルが存在しない、または `authorized=false`（ユーザー A のキャッシュとは独立） | — |

> **ポイント**: `~/Library` はアカウントごとに異なるため、ライセンスの共有は不可能。各ユーザーが個別にアクティベートする必要がある。

---

## ペルソナ J：GPU レンダラーで利用するユーザー `[Mac / GPU]`

> **背景**: GPU レンダラーは `OST_WindyLines_License.h` 経由で CPU 側の `RefreshLicenseAuthenticatedState` / `IsLicenseAuthenticated` を直接呼び出す（共有実装）。キャッシュファイル・署名検証・API 呼出しすべて CPU と同一コードパス。テストの目的は「GPU レンダーパスからの呼出しが正しく動作すること」の確認。

### J-1. GPU レンダラーでの基本動作（キャッシュなし→生成→WM消失）

> **所要時間：約 2 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | ターミナルで `rm ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt` | — |
| 2 | PP のエフェクト設定で **GPU レンダラーを選択**（Metal / CUDA / OpenCL） | — |
| 3 | クリップにエフェクトを適用してプレビュー | — |
| 4 | **確認①**: WM が **表示される** | — |
| 5 | **60〜90 秒待つ**（GPU 経由で `TriggerBackgroundCacheRefresh` が走る） | 60〜90秒 |
| 6 | タイムラインを動かして再描画 | — |
| 7 | **確認②**: WM が **消えている** | — |
| 8 | キャッシュ確認 → `authorized=true` + `cache_signature=` が存在 | — |

### J-2. CPU で作成したキャッシュを GPU レンダラーが読めるか

> **所要時間：約 30 秒**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | エフェクト設定で **CPU レンダラー**を選択 | — |
| 2 | `python3 set_license_cache_state.py --state authorized --ttl 600` | — |
| 3 | PP でプレビュー → WM なしを確認 | — |
| 4 | エフェクト設定を **GPU レンダラー**に切り替え | — |
| 5 | PP でプレビュー | — |
| 6 | **確認**: WM が **表示されない**（同じキャッシュを読めている） | — |

> **ポイント**: CPU と GPU は同じ `license_cache_v1.txt` と同一の検証関数を共有。テストの目的は GPU レンダーパスからの呼出し経路が正しく接続されていることの確認。

### J-3. GPU レンダラーでの改ざん検知（authorized 書換え）

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | `python3 set_license_cache_state.py --state unauthorized --ttl 600` | — |
| 2 | エフェクト設定で **GPU レンダラー**を選択 | — |
| 3 | テキストエディタで `authorized=false` → `authorized=true` に書き換えて保存 | — |
| 4 | PP でタイムラインを動かして再描画 | — |
| 5 | **確認**: WM が **表示される**（CPU と同一の `ComputeCacheSignature` で署名不一致） | — |

> **ポイント**: E-1 の GPU 版。共有実装だが GPU レンダーパスからの呼出し経路で正しく検知できるかを確認。

---

## ペルソナ K：Windows ユーザー `[Win / CPU]`

> **背景**: Windows では HTTP が `WinHTTP` API、パスが `%APPDATA%`、MID が `hostname|win|8`。Mac とは完全に異なる実装。

### K-1. Windows での初回起動（キャッシュなし）

> **所要時間：約 2 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | PowerShell で `Remove-Item "$env:APPDATA\OshareTelop\license_cache_v1.txt" -ErrorAction SilentlyContinue` | — |
| 2 | PP を起動、クリップに WindyLines エフェクトを適用 | — |
| 3 | **確認①**: WM が **表示される** | — |
| 4 | **60〜90 秒待つ**（WinHTTP で API 呼出し） | 60〜90秒 |
| 5 | タイムラインを動かして再描画 | — |
| 6 | **確認②**: WM が **消えている** | — |
| 7 | PowerShell で `Get-Content "$env:APPDATA\OshareTelop\license_cache_v1.txt"` | — |
| 8 | **確認③**: `authorized=true` + `cache_signature=` + `machine_id_hash=` が存在 | — |

### K-2. Windows での MID が Mac と異なることの確認

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | K-1 のキャッシュから `machine_id_hash=` の値をメモ | — |
| 2 | Mac 側のキャッシュから `machine_id_hash=` の値をメモ | — |
| 3 | **確認**: 値が **異なる**（MID に `\|mac\|` vs `\|win\|` が含まれるため） | — |

> **ポイント**: Mac と Windows はそれぞれ別 MID として Bubble に登録される。同一ホスト名のデュアルブート環境でも別ライセンス扱い。

### K-3. Windows での改ざん検知

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | PowerShell で `python set_license_cache_state.py --state unauthorized --ttl 600` | — |
| 2 | メモ帳で `%APPDATA%\OshareTelop\license_cache_v1.txt` を開く | — |
| 3 | `authorized=false` → `authorized=true` に書き換えて保存 | — |
| 4 | PP でタイムラインを動かして再描画 | — |
| 5 | **確認**: WM が **表示される**（署名不一致） | — |

> **ポイント**: E-1 の Windows 版。同じロジックだが `#ifdef _WIN32` 内のパス解決が正しく動作するかの確認。

---

## ペルソナ L：Mac⇔Win クロスプラットフォームユーザー `[Mac⇔Win]`

### L-1. Mac のキャッシュを Windows にコピーしても無効

> **所要時間：約 1 分**

| 手順 | 操作 | 待ち時間 |
|------|------|---------|
| 1 | Mac の `~/Library/Application Support/OshareTelop/license_cache_v1.txt` を USB 等で Windows にコピー | — |
| 2 | `%APPDATA%\OshareTelop\license_cache_v1.txt` に保存 | — |
| 3 | Windows の PP でタイムラインを動かして再描画 | — |
| 4 | **確認**: WM が **表示される** | — |

> **理由**: MID が `hostname|mac|8` vs `hostname|win|8` で異なるため `machine_id_hash` が不一致。たとえ同じホスト名のデュアルブートでもキャッシュの互換性はない。

---

## 検証チェックリスト

| # | ペルソナ | テスト名 | OS | レンダラー | 所要時間 | 期待結果 | 確認済 |
|---|---------|---------|-----|---------|---------|---------|--------|
| A-1 | 正規 | 初回起動（キャッシュなし） | Mac | CPU | 2分 | WM表示→60s後消失 | ☐ |
| A-2 | 正規 | TTL切れ後再検証 | Mac | CPU | 2分 | 猶予維持→再検証成功 | ☐ |
| A-3 | 正規 | Wi-Fi切断（猶予内） | Mac | CPU | 15分 | WMなし | ☐ |
| A-4 | 正規 | オフライン1h超 | Mac | CPU | 65分 | WM表示 | ☐ |
| B-1 | 解約 | 解約後検知 | Mac | CPU | 12分 | 10分後WM表示 | ☐ |
| B-2 | 再契約 | 再契約復帰 | Mac | CPU | 3分 | 90s後WM消失 | ☐ |
| C-1 | Free | 常時WM | Mac | CPU | 3分 | WM常時表示 | ☐ |
| D-1 | 新規 | アクティベートボタン | Mac | CPU | 1分 | ブラウザ起動+URL確認 | ☐ |
| D-2 | 新規 | アクティベート完了 | Mac | CPU | 30秒 | 5〜10s後WM消失 | ☐ |
| D-3 | 新規 | アクティベート中断 | Mac | CPU | 3分 | rapid終了後も変化なし | ☐ |
| E-1 | 改ざん | authorized書換え | Mac | CPU | 1分 | 署名NG→WM表示 | ☐ |
| E-2 | 改ざん | expire延長 | Mac | CPU | 1分 | 署名NG→WM表示 | ☐ |
| E-3 | 改ざん | 署名行削除 | Mac | CPU | 1分 | WM表示 | ☐ |
| E-4 | 改ざん | 別マシンコピー | Mac | CPU | 2分 | MID不一致→WM表示 | ☐ |
| F-1 | 高度 | salt特定+再計算 | — | — | 机上 | strings不可（XOR難読化） | ☐ |
| F-2 | 高度 | MITM偽装 | — | — | 机上 | HTTPS検証で失敗 | ☐ |
| G-1 | 更新 | プラグインアップデート後 | Mac | CPU | 5分 | キャッシュ継続→再検証 | ☐ |
| H-1 | 複数Mac | 2台目初回インストール | Mac | CPU | 2分 | 別MID→アクティベート必要 | ☐ |
| H-2 | 複数Mac | キャッシュコピー無効 | Mac | CPU | 1分 | MID不一致→WM表示 | ☐ |
| I-1 | 共有Mac | 別アカウントは独立 | Mac | CPU | 3分 | 各自キャッシュ独立 | ☐ |
| J-1 | GPU | 初回起動（GPU） | Mac | **GPU** | 2分 | WM表示→60s後消失 | ☐ |
| J-2 | GPU | CPUキャッシュをGPUで読む | Mac | **GPU** | 30秒 | WMなし（キャッシュ共有） | ☐ |
| J-3 | GPU | 改ざん検知（GPU） | Mac | **GPU** | 1分 | 署名NG→WM表示 | ☐ |
| K-1 | Win | 初回起動（Win） | **Win** | CPU | 2分 | WM表示→60s後消失 | ☐ |
| K-2 | Win | MIDがMacと異なる | **Win** | CPU | 1分 | MID値が異なる | ☐ |
| K-3 | Win | 改ざん検知（Win） | **Win** | CPU | 1分 | 署名NG→WM表示 | ☐ |
| L-1 | クロス | Macキャッシュ→Winコピー | Mac⇔Win | CPU | 1分 | MID不一致→WM表示 | ☐ |

### 推奨テスト順序

#### Phase 1：Mac + CPU（約 40 分）

```
--- 基本動作（約 10 分）---
1. C-1 (3分)    — Free版ウォーターマーク確認
2. A-1 (2分)    — キャッシュなし→生成→WM消失
3. D-1 (1分)    — アクティベートボタン→ブラウザ遷移
4. D-2 (30秒)   — アクティベート完了→即時反映
5. I-1 (3分)    — 共有Macの別アカウント独立確認

--- 改ざん耐性（約 5 分）---
6. E-1 (1分)    — authorized書換え→署名NG
7. E-3 (1分)    — 署名行削除→WM表示
8. E-4 (2分)    — 別マシンキャッシュコピー→MID不一致
9. E-2 (1分)    — expire延長→署名NG確認

--- TTL・解約サイクル（約 20 分）---
10. A-2 (2分)   — TTL切れ→オフライン猶予で維持→再検証
11. B-1 (12分)  — 解約→10分で検知
12. B-2 (3分)   — 再契約→90s復帰
13. D-3 (3分)   — アクティベート中断→rapid終了

--- アップデート・複数台（約 8 分）---
14. G-1 (5分)   — プラグイン更新後キャッシュ継続
15. H-1 (2分)   — 2台目Mac初回セットアップ
16. H-2 (1分)   — キャッシュコピー無効確認
```

#### Phase 2：GPU レンダラー差分（約 4 分）

```
17. J-1 (2分)   — GPUで初回起動→キャッシュ生成→WM消失
18. J-2 (30秒)  — CPUキャッシュをGPUで読めるか
19. J-3 (1分)   — GPUでも改ざん検知できるか
```

#### Phase 3：Windows 差分（約 5 分）

```
20. K-1 (2分)   — Winで初回起動（WinHTTP動作確認）
21. K-2 (1分)   — WinのMIDがMacと異なること
22. K-3 (1分)   — Winでも改ざん検知
23. L-1 (1分)   — Macキャッシュ→Winコピー無効
```

#### Phase 4：長時間テスト（時間があれば）

```
24. A-3 (15分)  — Wi-Fi切断テスト
25. A-4 (65分)  — オフライン猶予超過テスト
```

---

## テスト用コマンド集

### Mac (zsh / bash)

```bash
# === キャッシュ操作 ===

# 署名付き authorized キャッシュ生成（TTL 10分）
python3 set_license_cache_state.py --state authorized --ttl 600

# 署名付き unauthorized キャッシュ生成（TTL 10分）
python3 set_license_cache_state.py --state unauthorized --ttl 600

# 署名付き authorized キャッシュ生成（TTL 5秒 = すぐ切れる）
python3 set_license_cache_state.py --state authorized --ttl 5

# キャッシュ内容確認
cat ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt

# キャッシュ削除
rm ~/Library/Application\ Support/OshareTelop/license_cache_v1.txt

# === API 直接テスト ===

# authorized な MID でテスト
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action":"verify","product":"OST_WindyLines","plugin_version":"1.0.0","platform":"mac","machine_id":"test_mid_001"}' \
  'https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test'

# 未登録 MID でテスト
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action":"verify","product":"OST_WindyLines","plugin_version":"1.0.0","platform":"mac","machine_id":"unknown_mid"}' \
  'https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test'

# === デバッグ ===

# Console.app でプラグインのログを確認（Mac）
# Console.app → フィルタに「[License]」と入力
# または
log stream --predicate 'eventMessage contains "[License]"' --level debug
```

### Windows (PowerShell)

```powershell
# === キャッシュ操作 ===

# 署名付き authorized キャッシュ生成（TTL 10分）
python set_license_cache_state.py --state authorized --ttl 600

# 署名付き unauthorized キャッシュ生成
python set_license_cache_state.py --state unauthorized --ttl 600

# キャッシュ内容確認
Get-Content "$env:APPDATA\OshareTelop\license_cache_v1.txt"

# キャッシュ削除
Remove-Item "$env:APPDATA\OshareTelop\license_cache_v1.txt" -ErrorAction SilentlyContinue

# === API 直接テスト ===

# authorized な MID でテスト
Invoke-RestMethod -Method Post -Uri 'https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test' `
  -ContentType 'application/json' `
  -Body '{"action":"verify","product":"OST_WindyLines","plugin_version":"1.0.0","platform":"win","machine_id":"test_mid_001"}'

# === デバッグ ===

# DebugView (Sysinternals) を起動して [License] フィルタでログを確認
# DL: https://learn.microsoft.com/en-us/sysinternals/downloads/debugview
```
