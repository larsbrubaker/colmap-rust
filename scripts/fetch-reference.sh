#!/usr/bin/env bash
# Clones COLMAP at the commit pinned in REFERENCE into cpp-reference/, and Ceres Solver at the
# tag colmap-sharp's solver follows (2.2.0) into ceres-reference/ (both git-ignored).
# Reading material for porting only; nothing in the build uses either.
set -euo pipefail
cd "$(dirname "$0")/.."
read -r TAG COMMIT < REFERENCE
if [ ! -d cpp-reference/.git ]; then
  git clone --filter=blob:none https://github.com/colmap/colmap.git cpp-reference
fi
git -C cpp-reference fetch --quiet origin "$COMMIT"
git -C cpp-reference checkout --quiet --detach "$COMMIT"
echo "cpp-reference at COLMAP $TAG ($COMMIT)"

CERES_TAG=2.2.0
if [ ! -d ceres-reference/.git ]; then
  git clone --quiet --filter=blob:none --branch "$CERES_TAG" --depth 1 \
    https://github.com/ceres-solver/ceres-solver.git ceres-reference
fi
echo "ceres-reference at Ceres Solver $CERES_TAG ($(git -C ceres-reference rev-parse --short HEAD))"
