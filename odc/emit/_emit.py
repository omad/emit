"""
EMIT helper functions.
"""

import os
import sys
from collections import namedtuple
import json
import requests
from cachetools import cached
from pathlib import Path
from typing import Hashable, Any, Iterator, Iterable, Literal

from odc.geo.math import affine_from_pts, quasi_random_r2
from odc.geo import xy_, wh_
from odc.geo.geobox import GeoBox
from odc.geo.gcp import GCPMapping, GCPGeoBox
from odc.geo import geom
from affine import Affine

import numpy as np
from .vendor.eosdis_store.dmrpp import to_zarr

ZarrSpecMode = Literal["default"] | Literal["raw"]

ChunkInfo = namedtuple(
    "ChunkInfo",
    ["shape", "dtype", "order", "fill_value", "byte_range", "compressor", "filters"],
)

BBOX_KEYS = (
    "westernmost_longitude",
    "southernmost_latitude",
    "easternmost_longitude",
    "northernmost_latitude",
)

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
def fetch_s3_creds(tk=None):
    tk = earthdata_token(tk)
    return requests.get(
        "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials",
        headers={"Authorization": f"Bearer {tk}"},
        timeout=20,
    ).json()


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


def _parse_band_info(md_store, band=None):
    if band is None:
        all_bands = [n.rsplit("/", 1)[0] for n in md_store if n.endswith("/.zchunkstore")]
        return {b: _parse_band_info(md_store, b) for b in all_bands}

    ii = md_store[f"{band}/.zarray"]

    shape, dtype, order, fill_value, compressor, filters = (
        ii.get(k, None) for k in ("shape", "dtype", "order", "fill_value", "compressor", "filters")
    )
    shape = tuple(shape)

    cc = md_store[f"{band}/.zchunkstore"]

    byte_ranges = {k: slice(ch["offset"], ch["offset"] + ch["size"]) for k, ch in cc.items()}
    if len(byte_ranges) == 1:
        (byte_ranges,) = byte_ranges.values()

    return ChunkInfo(shape, dtype, order, fill_value, byte_ranges, compressor, filters)


def _walk_dirs_up(p: str) -> Iterator[str]:
    while True:
        *d, _ = p.rsplit("/", 1)
        if not d:
            break
        (p,) = d
        yield p


def _find_groups(all_paths: Iterable[str]) -> list[str]:
    # group is any directory without .zarray in it
    all_bands = set()
    all_dirs = set()

    for p in all_paths:
        if p.endswith(".zarray"):
            all_bands.add(p.rsplit("/", 1)[0])

        for _dir in _walk_dirs_up(p):
            if _dir in all_dirs:
                break
            all_dirs.add(_dir)

    return list(all_dirs - all_bands)


def _do_edits(refs):
    dims = {"downtrack": "y", "crosstrack": "x"}
    coords = {
        ("y", "x"): "lon lat",
        ("bands",): "wavelengths",
        ("y", "x", "bands"): "lon lat wavelengths",
    }
    drop_vars = ("location/glt_x", "location/glt_y", "build_dmrpp_metadata")
    flatten_groups = ("location", "sensor_band_parameters")
    skip_zgroups = set(f"{group}/.zgroup" for group in flatten_groups)
    drop_ds_attrs = ("history", "geotransform")

    def _keep(k):
        p, *_ = k.rsplit("/", 1)
        if p in drop_vars:
            return False
        if p in skip_zgroups:
            return False
        return True

    def patch_zattrs(doc, dims, coords):
        A_DIMS = "_ARRAY_DIMENSIONS"
        assert isinstance(doc, dict)

        if A_DIMS not in doc:
            return doc

        doc[A_DIMS] = [dims.get(dim, dim) for dim in doc[A_DIMS]]
        if (coords := coords.get(tuple(doc[A_DIMS]), None)) is not None:
            doc["coordinates"] = coords

        return doc

    def edit_one(k, doc):
        if k.endswith("/.zattrs"):
            doc = patch_zattrs(doc, dims, coords)
        if k == ".zattrs":
            doc = {k: v for k, v in doc.items() if k not in drop_ds_attrs}

        group, *_ = k.rsplit("/", 2)
        if group in flatten_groups:
            k = k[len(group) + 1 :]

        return (k, doc)

    return dict(edit_one(k, doc) for k, doc in refs.items() if _keep(k))


def to_zarr_spec(
    dmrpp_doc: str | bytes,
    url: str | None = None,
    mode: ZarrSpecMode = "default",
) -> dict[str, Any]:
    def to_docs(zz: dict[str, Any]) -> Iterator[tuple[str, str | tuple[str | None, int, int]]]:
        # sorted keys are needed to work around problem in fsspec directory listing 1430
        for k in sorted(zz, key=lambda p: (p.count("/"), p)):
            doc = zz[k]
            if k.endswith("/.zchunkstore"):
                prefix, _ = k.rsplit("/", 1)
                for chunk_key, info in doc.items():
                    yield f"{prefix}/{chunk_key}", (url, info["offset"], info["size"])
            else:
                yield k, json.dumps(doc, separators=(",", ":"))

    zz = to_zarr(dmrpp_doc)
    if mode == "raw":
        zz.update({f"{group}/.zgroup": {"zarr_format": 2} for group in _find_groups(zz)})
    else:
        zz = _do_edits(zz)

    refs = dict(to_docs(zz))
    return {"version": 1, "refs": refs}


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


def sample_error(xx, nsamples):
    gbox = xx.odc.geobox
    lon = xx.lon.data
    lat = xx.lat.data

    pix_ = gbox.qr2sample(nsamples).transform(lambda x, y: snap_to(x, y, 0))

    iy, ix = gxy(pix_).astype(int).T
    ww = geom.multipoint([(x, y) for x, y in zip(lon[ix, iy], lat[ix, iy])], 4326)

    pix_c = pix_.transform(lambda x, y: snap_to(x, y, 0.5))

    ee = gxy(pix_c) - gxy(gbox.project(ww))
    pix_error = np.sqrt((ee**2).sum(axis=1))

    return pix_c, ww, pix_error, ee
