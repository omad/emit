#!/bin/bash

DEPS=(
  "s3fs"
  "odc-geo>=0.4.7"
  "odc-stac>=0.3.10rc3"
  "git+https://github.com/opendatacube/datacube-core@1866e55f764e810f581a5f5ff929f65d872bc4f4"

  #"git+https://github.com/opendatacube/odc-stac.git@hs-load"
  #"git+https://github.com/opendatacube/odc-geo.git@hs-load"
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
