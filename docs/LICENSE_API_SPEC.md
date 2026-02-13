# ライセンス認証 API 最小仕様（v1.0 推奨）

最終更新: 2026年2月13日
対象: OST_WindyLines v1.0

---

## 目的

- v1.0 で必要十分な認証フローを最小構成で定義する。
- `未認証時のみウォーターマーク表示` を安全に制御する。
- レンダー処理中の通信を禁止し、パフォーマンス劣化を避ける。

---

## 実装方針（推奨）

- 通信は `GlobalSetup` 相当の初期化タイミング、またはユーザー明示操作時のみ実行する。
- `Render` / `SmartRender` 中はネットワークアクセスしない。
- サーバー到達不可時は `未認証` 扱い（fail-closed）にする。
- 認証結果はローカル保存し、保存TTL内はオフラインでも継続利用可能にする。
- v1.0 現在は「ユーザー明示操作時」を `activate_license_cache.py` で実現する。

---

## エンドポイント

- Method: `POST`
- URL（検証済みの到達先形式）:
  - `https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test`
- 備考:
  - 末尾 `initialize` は付けない。
  - v1.0 では単一POSTエンドポイント運用を前提にする。

---

## Request（JSON）

必須:
- `action`: 文字列。`"verify"`
- `license_key`: 文字列
- `machine_id`: 文字列（端末固有IDのハッシュ推奨）
- `product`: 文字列。`"OST_WindyLines"`
- `plugin_version`: 文字列（例: `"1.0.0"`）
- `platform`: 文字列（`"win"` / `"mac"`）
- `timestamp_utc`: 文字列（ISO8601）

任意:
- `nonce`: 文字列（リプレイ対策用）

例:

```json
{
  "action": "verify",
  "license_key": "WL-XXXX-XXXX-XXXX",
  "machine_id": "3f7c...",
  "product": "OST_WindyLines",
  "plugin_version": "1.0.0",
  "platform": "mac",
  "timestamp_utc": "2026-02-13T04:40:00Z"
}
```

---

## Response（JSON）

必須:
- `authorized`: 真偽値
- `reason`: 文字列（`"ok"`, `"invalid_key"`, `"revoked"`, `"expired"`, `"over_limit"`, `"server_error"` など）
- `server_time_utc`: 文字列（ISO8601）
- `cache_ttl_sec`: 数値（ローカル有効秒）

推奨（セキュリティ強化）:
- `license_id`: 文字列
- `entitlements`: 配列（将来拡張用）
- `expires_at_utc`: 文字列または `null`
- `signature`: 文字列（応答改ざん検知。v1.1以降で必須化可）

成功例:

```json
{
  "authorized": true,
  "reason": "ok",
  "server_time_utc": "2026-02-13T04:40:01Z",
  "cache_ttl_sec": 86400,
  "license_id": "lic_abc123",
  "expires_at_utc": null
}
```

失敗例:

```json
{
  "authorized": false,
  "reason": "invalid_key",
  "server_time_utc": "2026-02-13T04:40:01Z",
  "cache_ttl_sec": 600
}
```

---

## HTTP ステータス運用

- `200`: 認証判定結果をJSONで返す（成功/失敗ともにここへ集約）
- `400`: リクエスト不正（JSON欠落/型不正）
- `401/403`: 認可失敗（必要時）
- `429`: レート制限
- `5xx`: サーバー異常

クライアント判定（v1.0）:
- `200` かつ `authorized=true` → 認証済み（ウォーターマーク非表示）
- 上記以外 → 未認証（ウォーターマーク表示）

---

## ローカル保存仕様（v1.0）

保存項目:
- `authorized`
- `reason`
- `validated_at_utc`
- `cache_expire_at_utc`
- `license_key_masked`（平文保存しない）
- `machine_id_hash`

保存先:
- macOS: `~/Library/Application Support/OST/WindyLines/license_cache_v1.txt`
- Windows: `%APPDATA%\\OST\\WindyLines\\license_cache_v1.txt`

保存フォーマット（key=value）:
- `authorized=true|false`
- `cache_expire_unix=<unix epoch sec>`
- `reason=<text>`
- `license_key_masked=<masked key>`
- `machine_id_hash=<sha256>`

---

## 最小状態遷移

1. プラグイン起動
2. ローカルキャッシュ有効ならその結果を採用
3. キャッシュ期限切れならAPI `verify` をPOST
4. `authorized=true` なら `sLicenseAuthenticated=true`
5. それ以外は `sLicenseAuthenticated=false`

---

## v1.0 受け入れ条件

- 未認証: 常にウォーターマーク表示
- 認証済み: ウォーターマーク非表示
- API到達不可: 未認証扱い（クラッシュしない）
- Render中通信なし
- 連続起動時にキャッシュが有効活用される

---

## 次実装タスク（この仕様に対応）

- [x] プラグイン側: 認証ステート管理（初期化時ロード）
- [x] 明示操作ツール: API応答をローカル保存（activate_license_cache.py）
- [ ] APIクライアント: POST実装（タイムアウト/リトライ最小）
- [ ] ローカル保存: Win実機での書込確認
- [ ] UI/メッセージ: 未認証理由の表示文言（最小）
- [ ] 検証: 未認証/認証/通信断の3ケース
