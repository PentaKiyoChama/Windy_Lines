# EULAバージョン管理と同意ログ設計（Bubble向け）

このドキュメントは、EULAの法的証拠性を高めるための最小構成を定義します。
本設計は、ライセンサーが配布する **すべてのプラグインで共通EULAを使う** 前提です。

## 1. 現行バージョン

- EULAバージョン: `v1.5.0-common`
- EULA本文(Markdown): `docs/EULA.md`
- EULA本文(HTML): `docs/EULA.html`

---

## 2. Bubbleのデータ構造

## Data Type: `EulaVersion`

- `version` (text) 例: `v1.3.0-common`
- `scope` (text) 例: `all_plugins`
- `effective_at` (date)
- `is_active` (yes/no)
- `title` (text) 例: `エンドユーザーライセンス契約 (EULA)`
- `html_body` (text) ※ `docs/EULA.html` の本文
- `markdown_body` (text, optional) ※ `docs/EULA.md` の原文保存用
- `change_summary` (text) 例: `賠償責任上限額・返金不可・データ保持期間・存続条項・地位承継を追加`

## Data Type: `EulaAcceptance`

- `user` (User)
- `eula_version` (EulaVersion)
- `accepted_at` (date)
- `accepted` (yes/no) ※ 通常 true
- `ip_address` (text, optional)
- `user_agent` (text, optional)
- `locale` (text, optional)
- `plugin_identifier` (text, optional) ※ どのプラグイン画面で同意したかの記録用

## Data Type: `User`（追加フィールド）

- `latest_accepted_eula_version` (EulaVersion)
- `latest_accepted_eula_at` (date)

> `latest_accepted_eula_version` は共通EULAの同意版を示すため、全プラグインで共有して利用します。

---

## 3. 同意フロー（最小実装）

1. 画面表示時に `EulaVersion` の `is_active = yes` を1件取得
2. Userの `latest_accepted_eula_version` と比較
3. 未同意またはバージョン不一致なら、EULAモーダルを表示
4. 「同意する」チェック必須 + 「同意して続行」ボタン
5. ボタン押下時に `EulaAcceptance` を新規作成
6. Userの `latest_accepted_eula_version` / `latest_accepted_eula_at` を更新

---

## 4. 再同意判定ルール

- `User.latest_accepted_eula_version.version != ActiveEula.version` の場合に再同意必須
- 判定はログイン時と重要機能利用前の2箇所で実施推奨

---

## 5. バージョン更新ルール

- 軽微な文言修正: `v1.3.1-common`（patch更新）
- 条項追加/削除: `v1.4.0-common`（minor更新）
- 利用条件の大幅変更: `v2.0.0-common`（major更新）

更新時の必須作業:

1. `docs/EULA.md` と `docs/EULA.html` の内容を同期
2. 版番号を両ファイルで同一に更新
3. `EulaVersion` に新レコード追加し `is_active = yes`
4. 旧版 `is_active = no`
5. 全ユーザーに再同意フラグを自動適用（バージョン比較で判定）

---

## 6. 監査・証跡の最低要件

- だれが（user）
- いつ（accepted_at）
- どの版に（eula_version.version）
- どの画面経由で同意したか（必要なら source フィールド追加）

この4点を保持すると、後から説明可能性が大幅に上がります。
