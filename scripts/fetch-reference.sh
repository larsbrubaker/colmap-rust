#!/usr/bin/env bash
# Clones COLMAP at the commit pinned in REFERENCE into cpp-reference/ (git-ignored).
# Reading material for porting only; nothing in the build uses it.
set -euo pipefail
cd "$(dirname "$0")/.."
read -r TAG COMMIT < REFERENCE
if [ ! -d cpp-reference/.git ]; then
  git clone --filter=blob:none https://github.com/colmap/colmap.git cpp-reference
fi
git -C cpp-reference fetch --quiet origin "$COMMIT"
git -C cpp-reference checkout --quiet --detach "$COMMIT"
echo "cpp-reference at COLMAP $TAG ($COMMIT)"
