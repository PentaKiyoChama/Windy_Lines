#!/bin/bash
# __TPL_MATCH_NAME__ — コード署名設定
# codesign_setup.sh で自動生成される
# DO NOT EDIT MANUALLY

export CODESIGN_IDENTITY="Developer ID Application: __TPL_AUTHOR__ (TEAM_ID_HERE)"
export TEAM_ID="TEAM_ID_HERE"
export APPLE_ID="your-apple-id@example.com"
export KEYCHAIN_PROFILE="AC_PASSWORD"
export ENTITLEMENTS_PATH="Mac/__TPL_MATCH_NAME__.entitlements.plist"
