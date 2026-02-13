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
- docs/LICENSE_IMPLEMENTATION_MAC.md

補助スクリプト:
- activate_license_cache.py（手動即時キャッシュ更新。自動更新が動かない場合の代替）
- set_license_cache_state.py（デバッグ用のローカル切替。運用では非推奨）

---

## 法的・コンプライアンス チェック（リリース前必須）

| 項目 | リスク | 対応状況 | 必要な対応 |
|------|--------|---------|-----------|
| プライバシーポリシー/EULAへの通信記載 | **中** | [ ] 未対応 | バックグラウンドでサーバーと定期通信する旨を明記 |
| machine_id送信時のデータ保護告知 | **中** | [ ] 未対応 | activate_license_cache.py使用時、端末識別子送信の旨を記載 |
| Adobe SDK利用規約の確認 | **低〜中** | [ ] 未確認 | プラグイン内からの外部通信・子プロセス生成の制約を確認 |
| 販売ページへの認証要件明記 | **低** | [ ] 未対応 | 「サーバー認証が必要」「接続不可時はウォーターマーク表示」を明記 |
| コマンドインジェクション対策 | **低** | [x] 現状問題なし | URL/パスが動的になる場合はサニタイズ必要 |

### 詳細

1. **バックグラウンド通信の告知義務**
   - プラグインは `system()` (Mac) / 未実装 (Win) でユーザーに通知なくHTTP通信を行う
   - 送信データ: product, plugin_version, platform（個人情報ではないがIPアドレスがサーバーに記録される）
   - GDPR・個人情報保護法等でネットワーク通信の事前告知が必要
   - → EULA/プライバシーポリシーに「認証のためサーバーと定期通信する」旨を記載すること

2. **machine_id の収集**
   - activate_license_cache.py はホスト名+MACアドレス+OS情報のハッシュを送信
   - GDPR下では「仮名化された個人データ」に分類されうる
   - C++側の自動更新では machine_id を送信していないため現状はOK
   - → 手動ツールを本番利用する場合はプライバシーポリシーへの記載が必要

3. **消費者保護の観点**
   - サーバー側で一方的に無効化できる仕組み = 購入後のリモート機能制限
   - → 販売ページ・EULAに認証要件と返金ポリシーを明記すること

---

責任者: Kiyoto Nakamura