#!/usr/bin/env bash
# Build the COLMAP Rust web app: wasm-pack compiles colmap-web (release, --target web) into
# web/pkg, then web/dist is assembled from index.html + pkg/ + .nojekyll — the exact tree the
# Playwright smoke test serves and the Pages deploy (.github/workflows/deploy.yml) publishes.
# Needs wasm-pack and the wasm32-unknown-unknown target. Usage: web/build.sh (from anywhere).
set -euo pipefail

WEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$WEB_DIR/.." && pwd)"

cd "$ROOT_DIR"
wasm-pack build colmap-web --target web --release --no-typescript --out-dir ../web/pkg

DIST="$WEB_DIR/dist"
rm -rf "$DIST"
mkdir -p "$DIST/pkg"
cp "$WEB_DIR/index.html" "$DIST/"
# Only what the page loads; wasm-pack's package.json/README/.gitignore stay out of the site.
cp "$WEB_DIR/pkg/colmap_web.js" "$WEB_DIR/pkg/colmap_web_bg.wasm" "$DIST/pkg/"
touch "$DIST/.nojekyll"
echo "web/dist ready: $(du -sh "$DIST" | cut -f1)"
