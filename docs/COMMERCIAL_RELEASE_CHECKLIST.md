# OST_WindyLines 商用リリース最終チェック（現行方針）

最終更新: 2026年2月13日
対象: v1.0 初回有料配布

---

## 方針確定

- [x] 認証連動表示: 未認証時のみウォーターマーク表示（A方針）
- [x] 表示方式: 手動トグルなし（認証状態で自動切替）
- [x] 認証状態の反映元: 自社サイトAPIレスポンス（authorized）
- [x] 配布方式: インストーラーなし、ZIP手動配布
- [x] Windows署名: 初期リリースでは実施しない
- [x] 文字品質: 現在のウォーターマーク見た目を採用

---

## 実装確認（完了）

- [x] 著作権ヘッダー整理（自作コードとSDK由来表記の整理）
- [x] Releaseでデバッグログ無効化
- [x] 未認証ウォーターマーク描画をCPU側に実装
- [x] 起動時の認証ステート初期化（ローカルキャッシュ読込）
- [x] 文字マスク自動生成スクリプト整備
- [x] U/V中立値修正（ピンク被り対策）
- [x] ダウンサンプル時のサイズ追従

---

## リリース前の残タスク（これだけ）

- [x] 自社ライセンスAPI疎通PoCで Go 判定（2026-02-13 / POST 200）
- [x] 認証API最小仕様の固定（docs/LICENSE_API_SPEC.md）
- [x] ライセンス有効化Runbookで実機確認（2026-02-13）
- [ ] Win ReleaseビルドしてPremiereで適用確認
- [x] Mac Releaseビルド成功（2026-02-13 / xcodebuild）
- [x] Mac Premiereで適用確認（2026-02-13）
- [x] 未認証状態: ウォーターマークが表示されること（2026-02-13）
- [x] 認証状態: ウォーターマークが非表示になること（2026-02-13）
- [ ] プレビュー解像度 Full / 1/2 / 1/4 で表示バランス確認
- [ ] ZIP同梱物の最終確認（plugin, INSTALL, README, EULA）

---

## 配布パッケージ構成（手動配布）

Windows ZIP:
- OST_WindyLines.aex
- INSTALL.txt
- README.txt
- EULA.txt

macOS ZIP:
- OST_WindyLines.plugin
- INSTALL.txt
- README.txt
- EULA.txt

---

## 参照ドキュメント

- docs/WATERMARK_MASK_GENERATION.md
- docs/DISTRIBUTION_GUIDE.md
- docs/EULA.md
- docs/LICENSE_CONNECTIVITY_POC.md
- docs/LICENSE_API_SPEC.md
- docs/LICENSE_ACTIVATION_RUNBOOK.md

補助スクリプト:
- set_license_cache_state.py（デバッグ用のローカル切替。運用では非推奨）

---

責任者: Kiyoto Nakamura