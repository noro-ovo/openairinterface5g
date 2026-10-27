#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
#
# Refresh the vendored copy of the fast LDPC decoder (upstream/) from a dec_std source tree, then re-apply the
# OAI-side patches (patches/*.patch, in name order).
#
#   ./sync_from_upstream.sh <dec_std dir>            e.g. /home/user/ldpc_speedup/dec_std (v2) or .../dec_std_v3
#   ./sync_from_upstream.sh --check <dec_std dir>    only report whether upstream/ matches <dec_std dir> + patches
#
# What is copied (everything else in the source tree -- build/, bench/, results/, Makefile -- stays outside OAI):
#   include/*.h, src/*.c, src/*.h            the decoder (all kernels; new files such as a v3 kernel are picked up,
#                                            removed files are removed: the CMake list is a glob over upstream/src)
#   tables/ldpc_tables.c, tables/gen_tables.py   generated base-graph tables and their generator
#   test/test_dec.c, test/tutil.c, test/tutil.h  unit test (target test_ldpc_decoder_fast)
#   README.md                                 upstream documentation
# A patch whose files are absent (e.g. a kernel of another version) is skipped; a patch already contained in the new
# sources (dry-run of the reverse applies) is skipped; a patch that applies neither way stops the sync with an error,
# so a v3 drop is never half-patched silently.
# UPSTREAM.txt records the source path, date and the sha256 of every copied file (before patching).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
SRC=${1:?usage: $0 [--check] <dec_std dir>}
SRC=$(cd "$SRC" && pwd)
for f in include/ldpc_dec.h include/ldpc_dec_oai.h src/ldpc_dec.c src/ldpc_oai_entry.c tables/ldpc_tables.c; do
  [ -f "$SRC/$f" ] || { echo "error: $SRC/$f missing (not a dec_std tree?)" >&2; exit 1; }
done

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/include" "$STAGE/src" "$STAGE/tables" "$STAGE/test"
cp "$SRC"/include/*.h "$STAGE/include/"
cp "$SRC"/src/*.c "$SRC"/src/*.h "$STAGE/src/"
cp "$SRC/tables/ldpc_tables.c" "$STAGE/tables/"
[ -f "$SRC/tables/gen_tables.py" ] && cp "$SRC/tables/gen_tables.py" "$STAGE/tables/"
for f in test_dec.c tutil.c tutil.h; do [ -f "$SRC/test/$f" ] && cp "$SRC/test/$f" "$STAGE/test/"; done
[ -f "$SRC/README.md" ] && cp "$SRC/README.md" "$STAGE/"

# record provenance before patching
{
  echo "source: $SRC"
  if git -C "$SRC" rev-parse HEAD >/dev/null 2>&1; then echo "git: $(git -C "$SRC" describe --always --dirty)"; fi
  echo "synced: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "files (sha256 of the upstream file, before the patches in patches/):"
  (cd "$STAGE" && find . -type f | sort | sed 's|^\./||' | xargs sha256sum)
} > "$STAGE.upstream.txt"

shopt -s nullglob
for p in "$HERE"/patches/*.patch; do
  present=0
  for t in $(sed -n 's|^+++ b/\([^[:space:]]*\).*|\1|p' "$p"); do [ -f "$STAGE/$t" ] && present=1; done
  if [ $present = 0 ]; then
    echo "skipped $(basename "$p") (its files are not in this upstream version)"
  elif patch -s -p1 -d "$STAGE" --dry-run -N < "$p" >/dev/null 2>&1; then
    patch -s -p1 -d "$STAGE" -N < "$p"
    echo "applied $(basename "$p")"
  elif patch -s -p1 -d "$STAGE" --dry-run -R < "$p" >/dev/null 2>&1; then
    echo "skipped $(basename "$p") (already in the upstream sources)"
  else
    echo "error: $(basename "$p") does not apply to $SRC; update or drop it" >&2
    exit 1
  fi
done
find "$STAGE" -name '*.orig' -delete

if [ $CHECK = 1 ]; then
  if diff -r "$STAGE" "$HERE/upstream" >/dev/null; then echo "upstream/ is in sync with $SRC"; exit 0; fi
  diff -rq "$STAGE" "$HERE/upstream" || true
  exit 1
fi
rm -rf "$HERE/upstream"
mkdir -p "$HERE/upstream"
cp -r "$STAGE"/. "$HERE/upstream/"
mv "$STAGE.upstream.txt" "$HERE/UPSTREAM.txt"
echo "upstream/ refreshed from $SRC ($(find "$HERE/upstream" -type f | wc -l) files); see UPSTREAM.txt"
