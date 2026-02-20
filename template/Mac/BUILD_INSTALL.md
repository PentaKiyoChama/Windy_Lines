# __TPL_MATCH_NAME__ — Mac ビルド & インストール手順

## 前提条件

- macOS 13+ (Ventura以降)
- Xcode 14+
- Adobe Premiere Pro SDK（`/path/to/Adobe Premiere Pro SDK/`）
- Apple Developer ID 証明書（配布時）

## ビルド手順

### 1. Xcode プロジェクト作成

SDK の `Examples/Projects/GPUVideoFilter/` をベースに新規 Xcode プロジェクトを作成するか、
既存プロジェクトを複製して以下を修正:

- **Product Name**: `__TPL_MATCH_NAME__`
- **Bundle Identifier**: `com.__TPL_VENDOR__.__TPL_MATCH_NAME__`
- **Architectures**: arm64（Apple Silicon）
- **Info.plist**: `Mac/__TPL_MATCH_NAME__-Info.plist` を使用
- **Prefix Header**: `Mac/__TPL_MATCH_NAME__-Prefix.pch` を使用
- **Wrapper Extension**: `plugin`

### 2. ソースファイル追加

以下をプロジェクトに追加:
- `__TPL_MATCH_NAME___CPU.cpp`
- `__TPL_MATCH_NAME___GPU.cpp`
- `__TPL_MATCH_NAME__.cl`
- `__TPL_MATCH_NAME__.metal`
- 全 `.h` ファイル

### 3. ビルド

```bash
cd Mac
xcodebuild clean -configuration Debug ARCHS=arm64
xcodebuild -configuration Debug ARCHS=arm64
```

### 4. インストール

```bash
cd Mac
./install_plugin.sh
```

→ `/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/` にコピー

### 5. Premiere Pro 再起動

## コード署名 & 公証（配布時）

### 初回セットアップ

1. `codesign_config.sh` を編集（証明書ID、Team ID、Apple ID）
2. `xcrun notarytool store-credentials AC_PASSWORD` でパスワード保存

### 署名 → 公証

```bash
cd Mac
./codesign_plugin.sh    # 署名
./notarize_plugin.sh    # Apple公証 (5-10分)
```

### パッケージ作成

```bash
cd Mac
./package_cross_platform.sh
```

## トラブルシューティング

| 問題 | 対処 |
|------|------|
| `codesign` 失敗 | Keychain Access で証明書を確認 |
| 公証リジェクト | `xcrun notarytool log <id> --keychain-profile AC_PASSWORD` でログ確認 |
| プラグイン認識されない | `CFBundlePackageType` = `eFKT`, `CFBundleSignature` = `FXTC` を確認 |
| Metal カーネルエラー | `.metal` が `.cl` を正しく `#include` しているか確認 |
