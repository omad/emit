"""
Earthdata credentials
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable

import requests
from cachetools import cached

_creds_cache: dict[Hashable, dict[str, Any]] = {}

LOG = getLogger(__name__)


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
    )
    if not creds:
        raise RuntimeError(f"Failed to fetch S3 credentials: {creds.reason}")

    creds = creds.json()

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


def prep_s3_fs(*, creds: dict[str, Any] | None = None, **kw) -> "s3fs.S3FileSystem":
    # pylint: disable=import-outside-toplevel
    import s3fs

    if creds is None:
        LOG.debug("Fetching S3 credentials")
        creds = fetch_s3_creds()

    s3_opts = {
        "key": creds["accessKeyId"],
        "secret": creds["secretAccessKey"],
        "token": creds["sessionToken"],
        "anon": False,
        **kw,
    }

    fs = s3fs.S3FileSystem(**s3_opts)
    if isinstance(fs.protocol, list):
        # fix for `s3fs < 2023.10.0`
        fs.protocol = tuple(fs.protocol)

    return fs


if TYPE_CHECKING:
    import s3fs
