#!/usr/bin/env bash
# Launch Chrome with HTML-in-Canvas (drawElementImage / layoutSubtree) enabled.
# Requires a recent Chromium (e.g. Chrome Canary 147+). Feature name in source:
# third_party/blink/.../runtime_enabled_features.json5 → "CanvasDrawElement"
#
# Usage:
#   1. Quit all Chrome/Canary windows (otherwise macOS may reuse a running
#      process and ignore these args).
#   2. ./scripts/chrome-html-in-canvas.sh
#      Optional URL: ./scripts/chrome-html-in-canvas.sh 'https://html-in-canvas.dev/demos/'
set -euo pipefail

FEATURES="CanvasDrawElement"
URL="${1:-about:blank}"

for app in "Google Chrome Canary" "Google Chrome"; do
  if [[ -d "/Applications/${app}.app" ]]; then
    echo "Launching: ${app} with --enable-features=${FEATURES}"
    open -na "/Applications/${app}.app" --args \
      --enable-features="${FEATURES}" \
      "${URL}"
    exit 0
  fi
done

echo "error: Install Google Chrome Canary or Google Chrome under /Applications" >&2
exit 1
