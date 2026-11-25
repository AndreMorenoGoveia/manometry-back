#!/usr/bin/env bash
# Re-copy the vendored upstream modules. See app/vendor/README.md.
set -euo pipefail

CLASSIFICATION="${1:-../manometry-classification}"
UNIFIER="${2:-../manometry-data-unifier}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$HERE/app/vendor"

cp "$CLASSIFICATION/experiments/chicago.py"        "$VENDOR/chicago.py"
cp "$CLASSIFICATION/experiments/image_chicago.py"  "$VENDOR/image_chicago.py"
cp "$UNIFIER/unifier/images.py"                    "$VENDOR/swallow_crop.py"
cp "$UNIFIER/unifier/rtf.py"                       "$VENDOR/rtf_laudo.py"

# The one documented edit: package-relative import.
sed -i 's/^from chicago import/from .chicago import/' "$VENDOR/image_chicago.py"

echo "vendored from:"
echo "  classification $(git -C "$CLASSIFICATION" rev-parse --short HEAD)"
echo "  unifier        $(git -C "$UNIFIER" rev-parse --short HEAD)"
