# OST_WindyLines 配布準備ガイド

**作成日**: 2026年2月13日  
**対象**: コード署名、ライセンス認証、インストーラー

---

## 1. コード署名

### 1-1. Windows: 署名不要（手動配布）

**方針**: コード署名なしで配布

**理由**:
- 初期リリースでは証明書コスト（年間$200-400）を削減
- 手動インストール方式なら署名なしでも動作
- Windows Defender SmartScreen 警告は初回のみ

**ユーザー向け案内**:
```
初回実行時に「Windows によって PC が保護されました」と表示される場合:
1. "詳細情報" をクリック
2. "実行" をクリック
```

**将来的に署名する場合**:
- ユーザー数が増えた段階で検討
- DigiCert/Sectigo から EV 証明書取得
- 年間 $200-400 の継続コスト

---

### 1-2. macOS コード署名 + 公証 (必須)

**必要なもの**: Apple Developer ID (年間 $99)

**手順**:
1. [Apple Developer Program](https://developer.apple.com/programs/) に登録
2. 「Developer ID Application」証明書を作成
3. Xcode または `codesign` コマンドで署名:
   ```bash
   codesign --deep --force --verify --verbose \
     --sign "Developer ID Application: Kiyoto Nakamura (TEAM_ID)" \
     --options runtime \
     OST_WindyLines.plugin
   ```
4. Apple 公証 (Notarization):
   ```bash
   # ZIP 化
   ditto -c -k --keepParent OST_WindyLines.plugin OST_WindyLines.zip
   
   # 公証申請
   xcrun notarytool submit OST_WindyLines.zip \
     --apple-id "your@email.com" \
     --team-id "TEAM_ID" \
     --password "@keychain:AC_PASSWORD" \
     --wait
   
   # ステープル
   xcrun stapler staple OST_WindyLines.plugin
   ```

---

## 2. ライセンス認証 (自社サイト独自販売)

### 2-1. 推奨アーキテクチャ

```
[購入サイト] → ライセンスキー発行 → メール送信
                ↓
[プラグイン] → 初回起動時にキー入力 → サーバー認証
                ↓
[認証サーバー] → キー検証 → アクティベーション記録
```

### 2-2. 実装方式の選択肢

| 方式 | 複雑度 | セキュリティ | 備考 |
|---|---|---|---|
| **A. Gumroad License API** | 低 | 中 | Gumroad販売+API認証。実装最小限 |
| **B. 自社サーバー認証** | 高 | 高 | 完全制御。サーバー運用必要 |
| **C. オフラインキー** | 中 | 低 | マシンID+暗号署名。サーバー不要 |

### 2-3. 推奨: 方式C (オフラインキー) + シンプル検証

初期リリースでは、サーバー不要のオフラインライセンスキーが最も確実:

**キー形式**: `OSTW-XXXX-XXXX-XXXX-XXXX`

**実装概要** (OST_WindyLines_CPU.cpp の `GlobalSetup` に追加):
```cpp
// ライセンス検証 (概要)
// 1. レジストリ/plist からライセンスキーを読み込み
// 2. キーの書式チェック (OSTW-XXXX-XXXX-XXXX-XXXX)
// 3. チェックサム検証 (HMAC-SHA256)
// 4. 未認証の場合、ウォーターマーク描画 or 機能制限
```

**保存先**:
- Windows: `HKCU\Software\OST\WindyLines\LicenseKey`
- macOS: `~/Library/Preferences/com.ost.windylines.plist`

### 2-4. 海賊版対策

- キーの検証ロジックは難読化（文字列を直接埋め込まない）
- 定期的なオンラインチェック（任意、オフラインでも動作）
- バージョンアップ時にキー形式変更で旧クラック無効化

---

## 3. 配布方法（手動インストール）

### 3-1. Windows 配布

**配布形式**: ZIP アーカイブ

**パッケージ内容**:
```
OST_WindyLines_v1.0_Windows.zip
├── OST_WindyLines.aex          # プラグイン本体
├── INSTALL.txt                 # インストール手順
├── EULA.txt                    # 利用規約
└── README.txt                  # 製品説明
```

**インストール手順** (INSTALL.txt に記載):
```
1. Adobe Premiere Pro を終了
2. OST_WindyLines.aex を以下のフォルダにコピー:
   C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\

3. Premiere Pro を起動
4. エフェクトパネルで「流れる線」を確認

※ フォルダが存在しない場合は手動で作成してください
```

**アンインストール**:
```
1. Premiere Pro を終了
2. 以下のファイルを削除:
   C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\OST_WindyLines.aex
```

---

### 3-2. macOS 配布

**配布形式**: ZIP アーカイブ（公証済み）

**パッケージ内容**:
```
OST_WindyLines_v1.0_macOS.zip
├── OST_WindyLines.plugin       # プラグイン本体（公証済み）
├── INSTALL.txt                 # インストール手順
├── EULA.txt                    # 利用規約
└── README.txt                  # 製品説明
```

**インストール手順** (INSTALL.txt に記載):
```
1. Adobe Premiere Pro を終了
2. OST_WindyLines.plugin を以下のフォルダにコピー:
   /Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/

3. セキュリティ警告が出た場合:
   システム設定 > プライバシーとセキュリティ > "許可"

4. Premiere Pro を起動
5. エフェクトパネルで「流れる線」を確認
```

**アンインストール**:
```
1. Premiere Pro を終了
2. 以下を削除:
   /Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/OST_WindyLines.plugin
```

---

### 3-3. 配布パッケージ作成手順

#### Windows
```bash
cd Win/build/Release
mkdir OST_WindyLines_v1.0_Windows
cp OST_WindyLines.aex OST_WindyLines_v1.0_Windows/
cp ../../docs/EULA.md OST_WindyLines_v1.0_Windows/EULA.txt
# INSTALL.txt と README.txt を作成
zip -r OST_WindyLines_v1.0_Windows.zip OST_WindyLines_v1.0_Windows/
```

#### macOS
```bash
cd Mac/build/Debug
mkdir OST_WindyLines_v1.0_macOS
cp -R OST_WindyLines.plugin OST_WindyLines_v1.0_macOS/
cp ../../../docs/EULA.md OST_WindyLines_v1.0_macOS/EULA.txt
# INSTALL.txt と README.txt を作成
zip -r OST_WindyLines_v1.0_macOS.zip OST_WindyLines_v1.0_macOS/
```

---

### 3-4. 配布チェックリスト

- [ ] Windows: Release ビルド完了
- [ ] macOS: Release ビルド + 公証完了
- [ ] INSTALL.txt 作成（日本語・英語）
- [ ] README.txt 作成（製品説明）
- [ ] EULA.txt の配置
- [ ] ZIP アーカイブ作成
- [ ] 手動インストールテスト（クリーンな環境）
- [ ] Premiere Pro で動作確認

---

## 4. 決済・販売サイト

### 自社サイト販売の構成

| コンポーネント | 推奨サービス | 備考 |
|---|---|---|
| 決済 | Stripe / PayPal | 国際対応、手数料 3.6%前後 |
| ダウンロード配信 | Gumroad / SendOwl | ダウンロードリンク自動発行 |
| ライセンスキー発行 | 自作 or Keygen.sh | 購入時に自動生成 |
| サポート | メール / Discord | 初期はメール、規模拡大後にDiscord |

---

**次のステップ**:
1. Apple Developer Program 登録 (macOS署名に必須)
2. Windows EV コード署名証明書の取得
3. ライセンスキー検証ロジックの実装
4. NSIS / pkgbuild によるインストーラー作成
5. 販売サイトの構築

--- ← **申請済み、承認待ち**
2. ライセンスキー検証ロジックの実装
3. Release ビルドの動作確認（Win/Mac）
4. 配布パッケージ（ZIP）の