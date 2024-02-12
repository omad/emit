"""
Earthdata credentials
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Hashable

import requests
from cachetools import cached

_creds_cache: dict[Hashable, dict[str, Any]] = {}


def earthdata_token(tk=None):
    if tk is not None:
        return tk

    if (tk := os.environ.get("EARTHDATA_TOKEN", None)) is None:
        tk_file = Path.home() / ".safe" / "earth-data.tk"

        if tk_file.exists():
            print(f"Reading from: {tk_file}", file=sys.stderr)
            with tk_file.open("rt", encoding="utf8") as src:
                tk = src.read().strip()

    if tk is None:
        raise RuntimeError(f"please set EARTHDATA_TOKEN= or create: {tk_file}")

    return tk


@cached(_creds_cache)
def _cached_s3_creds(tk=None):
    tk = earthdata_token(tk)
    creds = requests.get(
        "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials",
        headers={"Authorization": f"Bearer {tk}"},
        timeout=20,
    ).json()
    creds["expiration"] = datetime.strptime(creds["expiration"], "%Y-%m-%d %H:%M:%S%z")
    return creds


def fetch_s3_creds(tk=None, fresh=False):
    if fresh:
        _creds_cache.clear()
    creds = _cached_s3_creds(tk)
    t_now = datetime.now(timezone.utc)
    if creds["expiration"] - t_now <= timedelta(seconds=60):
        _creds_cache.clear()
        creds = _cached_s3_creds(tk)
    return creds
