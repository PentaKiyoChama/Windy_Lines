#!/bin/bash
# Auto-install script for OST_WindyLines plugin
# This script copies the built plugin to Adobe MediaCore folder with admin privileges

INSTALL_DIR="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
SOURCE_PLUGIN="$PWD/build/Debug/OST_WindyLines.plugin"

echo "Installing plugin from: ${SOURCE_PLUGIN}"
echo "To: ${INSTALL_DIR}"

# Check if plugin exists
if [ ! -d "${SOURCE_PLUGIN}" ]; then
    echo "❌ Error: Plugin not found at ${SOURCE_PLUGIN}"
    echo "Please build the project first using: xcodebuild -configuration Debug ARCHS=arm64"
    exit 1
fi

# Use sudo to copy with admin privileges
echo "Installing plugin (requires administrator password)..."
sudo rm -rf "${INSTALL_DIR}/OST_WindyLines.plugin" && \
sudo ditto "${SOURCE_PLUGIN}" "${INSTALL_DIR}/OST_WindyLines.plugin"

if [ $? -eq 0 ]; then
    echo "✅ Plugin installed successfully to ${INSTALL_DIR}"
    ls -la "${INSTALL_DIR}/"
else
    echo "⚠️ Installation cancelled or failed"
    exit 1
fi
