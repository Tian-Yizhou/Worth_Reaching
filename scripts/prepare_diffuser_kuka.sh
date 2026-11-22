#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

source_dir=results/external/diffuser-kuka
data_dir=results/datasets/diffuser-kuka
revision=7c4d4da918c38f30241283d1f70d61a9b2293d09
archive="$data_dir/metainfo.tar.gz"
archive_sha=39c11a72be2d3de0ed2722ae3ce1eda039d25d09f4873fecb651b01d06fe79a8
patch="$PWD/scripts/diffuser-kuka-compat.patch"

mkdir -p -- results/external "$data_dir"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --branch kuka https://github.com/jannerm/diffuser.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
  printf 'Expected Diffuser Kuka revision %s at %s\n' "$revision" "$source_dir" >&2
  exit 1
fi
if git -C "$source_dir" apply --unidiff-zero --check "$patch" 2>/dev/null; then
  git -C "$source_dir" apply --unidiff-zero "$patch"
else
  git -C "$source_dir" apply --unidiff-zero --reverse --check "$patch"
fi

if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 2 --output "$archive.partial" \
    'https://www.dropbox.com/s/zofqvtkwpmp4v44/metainfo.tar.gz?dl=1'
  printf '%s  %s\n' "$archive_sha" "$archive.partial" | sha256sum --check
  mv -- "$archive.partial" "$archive"
fi
printf '%s  %s\n' "$archive_sha" "$archive" | sha256sum --check
if [[ ! -f "$data_dir/.extracted-$archive_sha" ]]; then
  uv run --no-sync python - "$archive" <<'PY'
import sys
import tarfile
from pathlib import Path

archive = Path(sys.argv[1])
with tarfile.open(archive) as source:
    source.extractall(archive.parent, filter="data")
PY
  touch -- "$data_dir/.extracted-$archive_sha"
fi
