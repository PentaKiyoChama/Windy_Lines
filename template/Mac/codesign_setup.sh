#!/bin/bash
# macOS Code Signing Setup Script
# __TPL_MATCH_NAME__ プラグイン署名準備

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}__TPL_MATCH_NAME__ - macOS 署名準備${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Step 1: Apple Developer Program 登録確認
echo -e "${YELLOW}[Step 1] Apple Developer Program 登録確認${NC}"
echo ""
echo "Apple Developer Program への登録が必要です。"
echo "URL: https://developer.apple.com/programs/"
echo "年間費用: \$99 USD"
echo ""
read -p "既に登録済みですか？ (y/n): " REGISTERED

if [ "$REGISTERED" != "y" ]; then
    echo -e "${RED}✗${NC} まず Apple Developer Program に登録してください。"
    echo "  https://developer.apple.com/programs/"
    exit 1
fi
echo -e "${GREEN}✓${NC} Apple Developer Program 登録確認完了\n"

# Step 2: 証明書の存在確認
echo -e "${YELLOW}[Step 2] コード署名証明書の確認${NC}"
echo ""

CERT_COUNT=$(security find-identity -v -p codesigning | grep "Developer ID Application" | wc -l | xargs)

if [ "$CERT_COUNT" -eq 0 ]; then
    echo -e "${RED}✗${NC} Developer ID Application 証明書が見つかりません。"
    echo ""
    echo "  1. Xcode → Settings → Accounts → Manage Certificates"
    echo "  2. '+' ボタン → 'Developer ID Application' を選択"
    echo "  3. または: https://developer.apple.com/account/resources/certificates/add"
    echo ""
    echo "証明書を作成後、このスクリプトを再実行してください。"
    exit 1
fi

echo -e "${GREEN}✓${NC} Developer ID Application 証明書: ${CERT_COUNT}件\n"

# Step 3: Team ID 取得
echo -e "${YELLOW}[Step 3] Team ID の確認${NC}"
TEAM_ID=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | sed 's/.*(\(.*\))/\1/')
echo -e "${GREEN}✓${NC} Team ID: ${TEAM_ID}\n"

# Step 4: App-Specific Password
echo -e "${YELLOW}[Step 4] App-Specific Password (公証用)${NC}"
echo ""
echo "公証（Notarization）には App-Specific Password が必要です。"
echo "  1. https://appleid.apple.com にログイン"
echo "  2. セキュリティ → App用パスワード → パスワードを生成"
echo ""
read -p "App-Specific Password を Keychain に保存しますか？ (y/n): " SAVE_PWD

if [ "$SAVE_PWD" == "y" ]; then
    read -p "Apple ID (メールアドレス): " APPLE_ID
    echo ""
    echo "以下のコマンドでパスワードを Keychain に保存します:"
    echo "  xcrun notarytool store-credentials \"__TPL_MATCH_NAME__\" --apple-id \"${APPLE_ID}\" --team-id \"${TEAM_ID}\""
    echo ""
    xcrun notarytool store-credentials "__TPL_MATCH_NAME__" --apple-id "${APPLE_ID}" --team-id "${TEAM_ID}"
    echo -e "${GREEN}✓${NC} 認証情報を Keychain に保存しました\n"
fi

# Step 5: codesign_config.sh の生成
echo -e "${YELLOW}[Step 5] 署名設定ファイルの生成${NC}"

SIGNING_IDENTITY=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | awk -F'"' '{print $2}')

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/codesign_config.sh"

cat > "$CONFIG_FILE" << EOF
#!/bin/bash
# 自動生成: $(date +%Y-%m-%d)
# __TPL_MATCH_NAME__ Code Signing Configuration

SIGNING_IDENTITY="${SIGNING_IDENTITY}"
TEAM_ID="${TEAM_ID}"
NOTARIZE_PROFILE="__TPL_MATCH_NAME__"
PLUGIN_NAME="__TPL_MATCH_NAME__"
BUNDLE_ID="com.__TPL_VENDOR__.__TPL_MATCH_NAME_NOUNDERSCORE__"
EOF

chmod +x "$CONFIG_FILE"
echo -e "${GREEN}✓${NC} 設定ファイル生成: ${CONFIG_FILE}\n"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}セットアップ完了！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "次のステップ:"
echo "  1. ビルド:       xcodebuild -configuration Debug ARCHS=arm64"
echo "  2. 署名:         ./codesign_plugin.sh"
echo "  3. 公証:         ./notarize_plugin.sh"
echo "  4. インストール: ./install_plugin.sh"
