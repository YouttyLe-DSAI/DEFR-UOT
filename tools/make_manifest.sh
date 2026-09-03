#!/bin/bash
# Sinh lai SOURCE_MANIFEST tu git. Chay sau khi them hoac xoa file nguon.
set -euo pipefail
cd "$(dirname "$0")/.."
{
    sed -n '1,/^# *bash tools\/make_manifest.sh/p' SOURCE_MANIFEST
    git ls-files '*.py' '*.sh' | LC_ALL=C sort
} > SOURCE_MANIFEST.tmp
mv SOURCE_MANIFEST.tmp SOURCE_MANIFEST
echo "SOURCE_MANIFEST: $(grep -vc '^#' SOURCE_MANIFEST) file"
