#!/usr/bin/env bash
# Install weco from local repo with env vars baked in for local development.
set -e

# Install globally via uv (with langfuse extra)
uv tool install --force --editable "$(cd "$(dirname "$0")" && pwd)[langfuse]"

# Find where uv installed the `weco` script
WECO_BIN=$(which weco)
BIN_DIR=$(dirname "$WECO_BIN")

# Back up the original entry point and replace with a wrapper
mv "$WECO_BIN" "$WECO_BIN-real"
cat > "$WECO_BIN" <<'EOF'
#!/usr/bin/env bash
export WECO_BASE_URL=http://localhost:8000/v1
export WECO_DASHBOARD_URL=http://localhost:3000
exec "$(dirname "$0")/weco-real" "$@"
EOF
chmod +x "$WECO_BIN"

echo "Installed weco (dev) → $WECO_BIN"
echo "  WECO_BASE_URL=http://localhost:8000/v1"
echo "  WECO_DASHBOARD_URL=http://localhost:3000"
