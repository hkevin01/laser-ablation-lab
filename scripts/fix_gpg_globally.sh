#!/bin/bash
# Fix GPG signing issues globally for VS Code and Git

echo "🔧 Fixing GPG signing issues globally..."

# Disable GPG signing in git global config
echo "Disabling GPG signing in git global configuration..."
git config --global commit.gpgsign false
git config --global --unset-all gpg.program 2>/dev/null || true
git config --global --unset-all core.gpgCommand 2>/dev/null || true

# Update VS Code global settings
VSCODE_SETTINGS="$HOME/.config/Code/User/settings.json"
if [ -f "$VSCODE_SETTINGS" ]; then
    echo "Updating VS Code global settings..."
    cp "$VSCODE_SETTINGS" "$VSCODE_SETTINGS.backup.$(date +%Y%m%d-%H%M%S)"
    
    # Use jq if available, otherwise use sed
    if command -v jq &> /dev/null; then
        jq '.["git.enableCommitSigning"] = false | .["git.alwaysSignOff"] = false' "$VSCODE_SETTINGS" > "$VSCODE_SETTINGS.tmp" && mv "$VSCODE_SETTINGS.tmp" "$VSCODE_SETTINGS"
    else
        sed -i 's/"git.enableCommitSigning": true/"git.enableCommitSigning": false/' "$VSCODE_SETTINGS"
        if ! grep -q "git.alwaysSignOff" "$VSCODE_SETTINGS"; then
            sed -i '/"git.enableCommitSigning": false/a \ \ "git.alwaysSignOff": false,' "$VSCODE_SETTINGS"
        fi
    fi
else
    echo "VS Code settings file not found at $VSCODE_SETTINGS"
fi

# Also check VS Code Insiders
VSCODE_INSIDERS_SETTINGS="$HOME/.config/Code - Insiders/User/settings.json"
if [ -f "$VSCODE_INSIDERS_SETTINGS" ]; then
    echo "Updating VS Code Insiders global settings..."
    cp "$VSCODE_INSIDERS_SETTINGS" "$VSCODE_INSIDERS_SETTINGS.backup.$(date +%Y%m%d-%H%M%S)"
    sed -i 's/"git.enableCommitSigning": true/"git.enableCommitSigning": false/' "$VSCODE_INSIDERS_SETTINGS"
fi

echo "✅ GPG signing disabled globally!"
echo "📋 Summary:"
echo "   - Git global commit.gpgsign: $(git config --global commit.gpgsign)"
echo "   - VS Code settings updated"
echo ""
echo "🔄 Please restart VS Code for changes to take effect."
