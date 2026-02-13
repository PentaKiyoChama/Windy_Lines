# ライセンスAPI疎通 PoC（Go/No-Go 判定）

最終更新: 2026年2月13日

## 目的

- Premiere プラグイン内実装の前に、`自社サイト API へ通信できるか` を先に判定する。
- 通信NGなら、サーバー連携実装は見送り（オフラインキー方式へ切替）。

## 1) エンドポイント準備

最低1つ、軽量な確認用エンドポイントを用意してください。

例:
- `POST /api/1.1/wf/ppplugin_test`（`initialize` なし）
- `GET /api/license/ping`（200/204を返す）
- 認証必須でも可（401/403が返れば「到達可」と判定可能）

## 2) 疎通確認（ターミナル）

```bash
cd /Users/kiyotonakamura/Desktop/Windy_Lines
chmod +x ./verify_license_connectivity.sh

# 認証なし（GET）
./verify_license_connectivity.sh "https://example.com/api/license/ping"

# 認証なし（POST / workflow）
./verify_license_connectivity.sh "https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test" POST '{}'

# 認証あり（任意）
./verify_license_connectivity.sh "https://example.com/api/license/verify" "YOUR_API_KEY" POST '{"license_key":"YOUR_KEY"}'
```

## 3) 判定基準

- Go（実装継続）
  - HTTP `200/204`、または `401/403`（= サーバー到達はできている）
- No-Go（実装見送り）
  - タイムアウト、DNS失敗、TLS失敗、`5xx` が継続

## 4) Premiere実機での最終確認（推奨）

- Premiere起動中の環境で同じコマンドを再実行する。
- 会社/学校ネットワーク、VPN、プロキシ環境で再試験する。

## 5) No-Go時の方針

- サーバー通信による認証実装は停止。
- v1.0 はオフラインキー検証（ローカルのみ）で運用。
