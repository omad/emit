#!/bin/bash

export PATH=$HOME/.local/mamba/bin:$PATH
db_dir=~/Data/DB
db_log=/tmp/db.log

db_snapshot_s3="s3://adias-prod-dc-data-projects/odc-hs/db/datacube-latest.tar.xz"

_start() {
  exec pg_ctl -D "$db_dir" -l "$db_log" "$@" start
}

_stop() {
  exec pg_ctl -D "$db_dir" -l "$db_log" "$@" stop
}

_log() {
  tail -f "$db_log"
}

_env() {
  echo "unset DB_PASSWORD"
  echo "unset DB_HOSTNAME"
  echo "unset DB_PORT"
  echo "unset DB_USERNAME"
  echo "unset DB_INDEX_DRIVER"
  echo "unset DB_DATABASE"
  echo "export ODC_EMIT_DB_URL='postgresql:///datacube?host=/tmp'"
  echo "export ODC_ENVIRONMENT=emit"
}

_fetch() {
  local db_url="${1:-$db_snapshot_s3}"
  if [ -d "$db_dir" ]; then
    echo "Database already exists at $db_dir"
    return 1
  fi
  printf "Fetching database\n %s\n %s\n" "$db_url" "$db_dir"
  mkdir -p "$db_dir"
  aws s3 cp "$db_url" - | tar -xJ -C "$db_dir"
}

_snapshot(){
  local out="/tmp/datacube-$(date +%Y%m%dT%H%M).tar.xz"
  printf "Saving to %s\n" "$out"
  (cd $db_dir && tar -c . | xz -zv -4 -T 0) > "$out"
  printf "Database snapshot created at %s\n" "$out"
  printf "To upload to S3, run:\n"
  printf "> aws s3 cp %s %s/\n" "$out" $(dirname "$db_snapshot_s3")
  printf "> aws s3 cp %s %s\n" "$out" "$db_snapshot_s3"
}

main() {
  case "$1" in
  "start")
    shift
    _start "$@"
    ;;
  "stop")
    shift
    _stop "$@"
    ;;
  "log")
    shift
    _log "$@"
    ;;
  "env")
    shift
    _env "$@"
    ;;
  "fetch")
    shift
    _fetch "$@"
    ;;
  "snapshot")
    shift
    _snapshot "$@"
    ;;
  *) echo "Invalid parameter. Please use on of 'start,stop,log,env,fetch,snapshot'." ;;
  esac
}

main "$@"
