#!/usr/bin/env bash
# Creates oracle/.venv with the pinned pycolmap wheel. Test-fixture generation only:
# the Rust crates and their tests never call Python.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/python -c "import pycolmap; print('pycolmap', pycolmap.__version__)"
