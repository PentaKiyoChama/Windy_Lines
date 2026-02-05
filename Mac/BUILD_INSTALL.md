# SDK_ProcAmp ビルド & インストール手順

## 1. クリーンビルド
```bash
cd Mac
xcodebuild clean -configuration Debug ARCHS=arm64
xcodebuild -configuration Debug ARCHS=arm64
```

## 2. プラグインのインストール
```bash
./install_plugin.sh
```

管理者パスワードを入力すると、プラグインが自動的に以下の場所にインストールされます：
`/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/`

## 3. Premiere Proでテスト
1. Premiere Proを再起動
2. エフェクトプリセットのドロップダウンメニューで日本語が正しく表示されることを確認
3. デバッグマーカー（左上の図形）が表示されないことを確認

## Xcodeでビルドする場合
Xcodeでビルド後、TerminalでMacフォルダに移動して `./install_plugin.sh` を実行してください。

## 注意事項
- Premiere Proを終了してからプラグインをインストールしてください
- インストール後はPremiere Proを再起動する必要があります
