"""
EMIT helper functions.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Hashable

import numpy as np
import requests
import xarray as xr
from affine import Affine
from cachetools import cached
from odc.geo import geom, wh_, xy_
from odc.geo.gcp import GCPGeoBox, GCPMapping
from odc.geo.geobox import GeoBox
from odc.geo.math import affine_from_pts, quasi_random_r2

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


def fetch_s3_creds(tk=None):
    creds = _cached_s3_creds(tk)
    t_now = datetime.now(timezone.utc)
    if creds["expiration"] - t_now <= timedelta(seconds=60):
        _creds_cache.clear()
        creds = _cached_s3_creds(tk)
    return creds


def gxy(gg):
    return np.asarray(gg.json["coordinates"])


def parse_emit_orbit(item):
    if isinstance(item, str):
        _id = item
    else:
        _id = item.id

    return int(_id.split("_")[-2])


def gbox_from_points(pix, wld, shape):
    crs = wld.crs
    _pix = [xy_(*g.points[0]) for g in pix.geoms]
    _wld = [xy_(*g.points[0]) for g in wld.geoms]
    A = affine_from_pts(_pix, _wld)
    gbox = GeoBox(shape, A, crs)
    return gbox


def gbox_from_pix_lonlat(lon, lat, *, nsamples=100, crs=None, approx=False):
    assert lon.shape == lat.shape

    ix, iy = quasi_random_r2(nsamples, lon.shape).astype("int32").T
    wld = geom.multipoint([(x, y) for x, y in zip(lon[iy, ix], lat[iy, ix])], 4326)
    pix = geom.multipoint([(x + 0.5, y + 0.5) for x, y in zip(ix, iy)], None)

    if crs is not None:
        wld = wld.to_crs(crs)

    if approx:
        return gbox_from_points(pix, wld, lon.shape)

    return GCPGeoBox(lon.shape, GCPMapping(pix, wld))


def ortho_gbox(zarr_meta):
    h, w = zarr_meta["location/glt_x/.zarray"]["shape"]
    tx, sx, _, ty, _, sy = zarr_meta[".zattrs"]["geotransform"]
    return GeoBox(wh_(w, h), Affine(sx, 0, tx, 0, sy, ty), 4326)


def snap_to(x, y, off=0.5):
    def op(x):
        return [int(_x) + off for _x in x]

    return op(x), op(y)


def sample_error_0(gbox, lon, lat, nsamples):
    pix_ = gbox.qr2sample(nsamples).transform(lambda x, y: snap_to(x, y, 0))

    iy, ix = gxy(pix_).astype(int).T
    ww = geom.multipoint([(x, y) for x, y in zip(lon[ix, iy], lat[ix, iy])], 4326)

    pix_c = pix_.transform(lambda x, y: snap_to(x, y, 0.5))

    ee = gxy(pix_c) - gxy(gbox.project(ww))
    pix_error = np.sqrt((ee**2).sum(axis=1))

    return pix_c, ww, pix_error, ee


def sample_error(xx: xr.Dataset, nsamples: int) -> SimpleNamespace:
    gbox = xx.odc.geobox
    lon = xx.lon.data
    lat = xx.lat.data

    pix_ = gbox.qr2sample(nsamples).transform(lambda x, y: snap_to(x, y, 0))
    iy, ix = gxy(pix_).astype(int).T
    pts_p = pix_.transform(lambda x, y: snap_to(x, y, 0.5))

    pts_w = geom.multipoint(list(zip(lon[ix, iy], lat[ix, iy])), 4326)

    ee = gxy(pts_p) - gxy(gbox.project(pts_w))
    pix_error = np.sqrt((ee**2).sum(axis=1))

    return SimpleNamespace(pts_p=pts_p, pts_w=pts_w, pix_error=pix_error, ee=ee)


def mk_error_plot(
    xx: xr.Dataset,
    nsamples: int = 100,
    max_err_axis: float = -1,
    msg: str = "",
) -> SimpleNamespace:
    import seaborn as sns
    from matplotlib import pyplot as plt

    rr = sample_error(xx, nsamples)

    fig, axd = plt.subplot_mosaic(
        [
            ["A", "A", "A", "B", "B"],
            ["A", "A", "A", "B", "B"],
            ["C", "C", "C", "C", "C"],
        ],
        figsize=(8, 8),
    )
    rr.fig = fig
    rr.axd = axd

    info = []
    if epsg := xx.odc.crs.epsg:
        info.append(f"epsg:{epsg}")

    info.append(f"avg:{rr.pix_error.mean():.3f}px")
    if msg:
        info.append(msg)
    info = " ".join(info)

    fig.suptitle(f"Pixel Registration Error, {info}")

    sns.scatterplot(x=rr.ee.T[0], y=rr.ee.T[1], size=rr.pix_error, ax=axd["A"])
    if max_err_axis > 0:
        b = max_err_axis
    else:
        b = max(map(abs, axd["A"].axis()))

    axd["A"].axis([-b, b, -b, b])
    axd["A"].axvline(0, color="k", linewidth=0.3)
    axd["A"].axhline(0, color="k", linewidth=0.3)

    X, Y = gxy(rr.pts_p).T
    sns.heatmap(
        xx.reflectance.isel(band=100),
        cmap="bone",
        alpha=0.7,
        square=True,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=axd["B"],
    )
    with plt.rc_context({"legend.loc": "lower right"}):
        sns.scatterplot(
            x=X,
            y=Y,
            size=rr.pix_error,
            color="c",
            alpha=0.5,
            ax=axd["B"],
        )

    axd["C"].axvline(rr.pix_error.mean(), color="y")
    sns.kdeplot(rr.pix_error, ax=axd["C"], clip=(0, b * 100))
    *_, maxy = axd["C"].axis()
    axd["C"].axis([0, b, 0, maxy])
    fig.tight_layout()
    return rr
