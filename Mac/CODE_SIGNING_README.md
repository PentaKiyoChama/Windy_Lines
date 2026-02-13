# macOS 署名・公証ガイド

**作成日**: 2026年2月13日  
**対象**: OST_WindyLines プラグイン

---

## 概要

macOS で配布するプラグインには以下が必要です:

1. **コード署名** — Developer ID Application 証明書による署名
2. **公証 (Notarization)** — Apple による自動セキュリティチェック
3. **インストーラー** — .pkg パッケージの作成

---

## 準備: 必要なもの

| 項目 | 費用 | 備考 |
|---|---|---|
| Apple Developer Program | $99/年 | 必須 |
| Developer ID Application 証明書 | 無料 | Developer Program に含まれる |
| Developer ID Installer 証明書 | 無料 | (オプション) パッケージ署名用 |
| App-Specific Password | 無料 | 公証用 |

---

## 手順

### Step 1: 初期セットアップ

```bash
cd /Users/kiyotonakamura/Desktop/Windy_Lines/Mac
./codesign_setup.sh
```

このスクリプトが以下を確認・設定します:
- Apple Developer Program の登録状態
- コード署名証明書の有無
- Team ID の取得
- App-Specific Password の Keychain 保存
- 署名設定ファイル (`codesign_config.sh`) の生成

**所要時間**: 初回は約10-15分 (証明書作成を含む)

---

### Step 2: プラグインへの署名

```bash
./codesign_plugin.sh
```

処理内容:
1. 既存の署名を削除
2. Developer ID Application 証明書で署名
3. 署名の検証

**対象**: `build/Debug/OST_WindyLines.plugin`

---

### Step 3: Apple 公証

```bash
./notarize_plugin.sh
```

処理内容:
1. プラグインを ZIP 化
2. Apple サーバーに公証申請
3. 公証結果の待機 (通常 5-10分)
4. 公証チケットのステープル

**公証ステータス**: https://developer.apple.com/account に表示されます

---

### Step 4: 配布パッケージ作成

```bash
cd build/Debug
zip -r OST_WindyLines_v1.0_macOS.zip OST_WindyLines.plugin
```

処理内容:
1. 公証済みプラグインを ZIP 化
2. INSTALL.txt / EULA.txt を同梱

**出力**: `build/OST_WindyLines_v1.0_macOS.zip`

**配布方法**: 手動インストール（ユーザーが MediaCore フォルダにコピー）

---

## トラブルシューティング

### 問題: 証明書が見つからない

```bash
security find-identity -v -p codesigning
```

結果が `0 valid identities found` の場合:
1. https://developer.apple.com/account/resources/certificates/list にアクセス
2. Developer ID Application 証明書を作成
3. ダウンロードしてダブルクリックでインストール

---

### 問題: 公証に失敗する

**エラー例**: `The software asset is not notarized`

原因:
- プラグインが署名されていない
- 署名が無効
- ハードニング (hardened runtime) 未対応

確認:
```bash
codesign -dv --verbose=4 build/Debug/OST_WindyLines.plugin
```

`flags=0x10000(runtime)` が表示されていることを確認。

---

### 問題: Keychain プロファイルが無効

```bash
xcrun notarytool list --keychain-profile AC_PASSWORD
```

エラーが出る場合、再設定:
```bash
xcrun notarytool store-credentials AC_PASSWORD \
    --apple-id "your@email.com" \
    --team-id "YOUR_TEAM_ID" \
    --password "xxxx-xxxx-xxxx-xxxx"
```

---

## 署名状態の確認

### 現在の署名情報

```bash
codesign -dv build/Debug/OST_WindyLines.plugin
```

### 公証状態の確認

```bash
spctl --assess --type execute -vv build/Debug/OST_WindyLines.plugin
```

**期待される出力** (公証済み):
```
build/Debug/OST_WindyLines.plugin: accepted
source=Notarized Developer ID
```

### ステープル状態の確認

```bash
xcrun stapler validate build/Debug/OST_WindyLines.plugin
```

---

## 証明書の有効期限

Developer ID 証明書の有効期限は **5年** です。

有効期限の確認:
```bash
security find-certificate -c "Developer ID Application" -p | openssl x509 -noout -dates
```

---

## 自動化 (CI/CD)

GitHub Actions で署名・公証を自動化する場合:

```yaml
- name: Import Certificate
  env:
    CERTIFICATE_BASE64: ${{ secrets.MACOS_CERTIFICATE }}
    CERTIFICATE_PASSWORD: ${{ secrets.MACOS_CERTIFICATE_PASSWORD }}
  run: |
    echo "$CERTIFICATE_BASE64" | base64 --decode > certificate.p12
    security create-keychain -p actions temp.keychain
    security import certificate.p12 -k temp.keychain -P "$CERTIFICATE_PASSWORD" -T /usr/bin/codesign
    security set-key-partition-list -S apple-tool:,apple: -s -k actions temp.keychain

- name: Sign Plugin
  run: |
    codesign --deep --force --verify --verbose \
      --sign "Developer ID Application: Kiyoto Nakamura (TEAM_ID)" \
      --options runtime \
      build/Release/OST_WindyLines.plugin
```

---

## 参考リンク

- [Apple Developer - Notarizing macOS Software](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Apple Developer - Certificate Types](https://developer.apple.com/support/certificates/)
- [Code Signing Guide](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/Introduction/Introduction.html)

---

**Copyright (c) 2026 Kiyoto Nakamura. All rights reserved.**
