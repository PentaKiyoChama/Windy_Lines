# ライセンス有効化 Runbook（v1.0 最小運用）

最終更新: 2026年2月13日

## 目的

- API 応答をローカルキャッシュへ保存し、プラグイン起動時の認証状態へ反映する。
- Render中通信を避けるため、認証は明示操作（このスクリプト実行）で行う。
- 本番運用では、認証/未認証の切替は `自社サイトAPI応答` でのみ反映する。

## 実行コマンド

```bash
cd /Users/kiyotonakamura/Desktop/Windy_Lines
python3 ./activate_license_cache.py \
  --endpoint "https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test"
```

ライセンスキーを使う場合（任意）:

```bash
python3 ./activate_license_cache.py \
  --license-key "WL-XXXX-XXXX-XXXX" \
  --endpoint "https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test"
```

Bearer トークンが必要な場合:

```bash
python3 ./activate_license_cache.py \
  --license-key "WL-XXXX-XXXX-XXXX" \
  --endpoint "https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test" \
  --api-key "YOUR_API_KEY"
```

## 結果の見方

- `authorized: true` → 認証済みキャッシュ保存（次回起動でウォーターマーク非表示）
- `authorized: false` → 未認証キャッシュ保存（ウォーターマーク表示）

注: APIレスポンスの `authorized`（boolean）を基準に反映する。

## キャッシュ保存先

- macOS: `~/Library/Application Support/OshareTelop/license_cache_v1.txt`
- Windows: `%APPDATA%\OshareTelop\license_cache_v1.txt`

## 運用手順（最小）

1. ユーザーが `activate_license_cache.py` を実行（自社サイトAPIへPOST）
2. API `authorized` 結果がローカルキャッシュへ保存される
3. Premiereで再描画（または再起動）
4. ウォーターマーク表示状態を確認

## デバッグ用: ローカル強制切替（本番運用では非推奨）

未認証を強制:

```bash
python3 ./set_license_cache_state.py --state unauthorized --ttl 1800
```

認証済みを強制:

```bash
python3 ./set_license_cache_state.py --state authorized --ttl 1800
```

確認フロー:

1. `unauthorized` を実行
2. Premiere再起動してウォーターマーク表示を確認
3. `authorized` を実行
4. Premiere再起動してウォーターマーク非表示を確認

注意:

- `--ttl 300` のような短い値は、再起動や確認中に期限切れになりやすいです。
- 検証時は `--ttl 1800` 以上（推奨 `86400`）を使ってください。

反映されない場合（重要）:

- Premiereのプレビューキャッシュが残ると、認証状態を切り替えても見た目が更新されないことがあります。
- プラグイン側の認証再読込は短周期（約0.2秒）なので、通常はほぼ即時で切り替わります。
- 次のいずれかを実施してください。
  - タイムラインで1フレーム以上移動して再描画
  - エフェクトパラメータを一度変更して戻す（強制再描画）
  - Sequence のレンダーファイルを削除して再生

## トラブル時

- HTTPが200でも `authorized` 未返却なら、API側レスポンスに `authorized` / `reason` / `cache_ttl_sec` を追加する。
- 強制的に未認証に戻す場合は、キャッシュファイルを削除してPremiere再起動。
