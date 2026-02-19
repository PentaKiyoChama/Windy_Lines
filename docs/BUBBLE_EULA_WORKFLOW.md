# Bubble実装用 WF 定義（EULA同意・再同意）

この手順は `docs/EULA_VERSIONING_AND_DATA_MODEL.md` のデータ構造を前提にしています。
このWFは、配布するすべてのプラグインで共通EULAを使う前提です。

## 前提（Data Type）

- `EulaVersion`
  - `version` (text)
  - `scope` (text) 例: `all_plugins`
  - `is_active` (yes/no)
  - `html_body` (text)
- `EulaAcceptance`
  - `user` (User)
  - `eula_version` (EulaVersion)
  - `accepted_at` (date)
  - `accepted` (yes/no)
  - `ip_address` (text, optional)
  - `user_agent` (text, optional)
  - `plugin_identifier` (text, optional)
- `User`
  - `latest_accepted_eula_version` (EulaVersion)
  - `latest_accepted_eula_at` (date)

---

## 画面構成（最小）

- Popup: `Popup EULA`
- HTML Element: `HTML EULA Body`
- Checkbox: `Checkbox Agree`（文言: 「EULAに同意します」）
- Button: `Button Agree`
- Button: `Button Cancel`

`HTML EULA Body` には次を設定:

`Search for EulaVersions:first item (is_active = yes):html_body`

---

## WF-1: ページロード時に同意判定

**Event**: `When Page is loaded`

1. `Set state`（optional）
   - `activeEula = Search for EulaVersions:first item (is_active = yes)`
2. `Only when` 条件:
   - `Current User is logged in`
   - `activeEula is not empty`
   - `Current User's latest_accepted_eula_version is empty`
     **or**
     `Current User's latest_accepted_eula_version's version != activeEula's version`
3. Action: `Show Popup EULA`
4. Action: `Set Checkbox Agree to no`

---

## WF-2: 同意ボタン（必須）

**Event**: `When Button Agree is clicked`

### Step A（未チェック時のガード）
- `Only when Checkbox Agree is no`
- Action: `Show message`（Alert）
  - 例: 「続行するにはEULAへの同意が必要です」
- `Terminate this workflow`

### Step B（同意保存）
- `Only when Checkbox Agree is yes`
- Action: `Create a new EulaAcceptance`
  - `user = Current User`
  - `eula_version = Search for EulaVersions:first item (is_active = yes)`
  - `accepted_at = Current date/time`
  - `accepted = yes`
  - `ip_address = Current User's last IP`（取得できる場合のみ）
  - `user_agent = Browser user agent`（プラグイン等で取得する場合）

### Step C（Userへ最新同意を反映）
- Action: `Make changes to Current User`
  - `latest_accepted_eula_version = Search for EulaVersions:first item (is_active = yes)`
  - `latest_accepted_eula_at = Current date/time`

### Step D（モーダルを閉じる）
- Action: `Hide Popup EULA`

---

## WF-3: 非同意時

**Event**: `When Button Cancel is clicked`

推奨どちらか:

- A) `Log the user out` → ログイン画面へ遷移
- B) `Go to page`（EULA必須でない説明ページ）

---

## WF-4: 重要操作前の再同意チェック（推奨）

例: ダウンロードボタン、購入処理、プラグイン有効化など。

**Event**: `When Button Download is clicked`（など）

- `Only when`:
  - `Current User's latest_accepted_eula_version is empty`
    **or**
  - `Current User's latest_accepted_eula_version's version != Search for EulaVersions:first item (is_active = yes):version`
- Action: `Show Popup EULA`
- Action: `Terminate this workflow`

同意済みの場合のみ本来の処理を続行。

---

## WF-5: EULA改定時（運用WF）

1. 新しい `EulaVersion` を作成（`version` を更新）
2. 新版に `is_active = yes`
3. 旧版は `is_active = no`
4. 既存ユーザーは次回アクセス時に WF-1 で自動再同意

---

## 実装メモ

- `is_active = yes` の `EulaVersion` は常に1件に保つ
- 同意ログ（`EulaAcceptance`）は削除しない（証跡）
- 版番号は `EULA.md` / `EULA.html` / `eula_version.json` と一致させる
