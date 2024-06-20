#!/bin/bash

DEPS=(
  "s3fs"
  "git+https://github.com/opendatacube/datacube-core@hs-load"
  "git+https://github.com/opendatacube/odc-geo.git@hs-load"
  "git+https://github.com/opendatacube/odc-stac.git@hs-load"

  #"odc-geo>=0.4.7rc1"
  #"odc-stac>=0.3.10rc2"
)

DEPS_ACTIVE_DEV=(
  "git+https://github.com/csiro-easi/emit.git@new-reader-api"
)

EE="${HOME}/.envs/emit"

#################

cmd="${1:-all}"

case "$cmd" in
  all)
    "$EE/bin/python" -m pip install "${DEPS[@]}"
    "$EE/bin/python" -m pip install --no-deps --force-reinstall "${DEPS_ACTIVE_DEV[@]}"
    ;;
  dev)
    "$EE/bin/python" -m pip install --no-deps --force-reinstall "${DEPS_ACTIVE_DEV[@]}"
  ;;
esac

"$EE/bin/python" -m odc.emit info 

printf 'To activate:\n>>> source "%s/bin/activate"\n\n' "$EE"
