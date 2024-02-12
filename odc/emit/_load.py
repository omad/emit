"""
EMIT loading functions.
"""

from itertools import chain

import numpy as np
import xarray as xr
from affine import Affine
from odc.geo import geom, wh_, xy_
from odc.geo.gcp import GCPGeoBox, GCPMapping
from odc.geo.geobox import GeoBox
from odc.geo.math import affine_from_pts, quasi_random_r2
from odc.geo.xr import xr_coords

from ._md import fs_from_stac_doc


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


def emit_load(
    stac_doc,
    fs,
    *,
    chunks=None,
    asset="RFL",
) -> xr.Dataset:
    # pylint: disable=too-many-locals

    rows_per_chunk: int | None = None

    if isinstance(chunks, dict):
        rows_per_chunk = chunks.get("y")

    rfs = fs_from_stac_doc(
        stac_doc,
        fs,
        rows_per_chunk=rows_per_chunk,
        asset=asset,
    )

    xx = xr.open_dataset(
        rfs.get_mapper(""),
        engine="zarr",
        chunks=chunks,
    )

    xx = xx.assign_coords(
        {
            "y": np.arange(xx.dims["y"]) + 0.5,
            "x": np.arange(xx.dims["x"]) + 0.5,
        }
    )

    if "ortho_x" not in xx.dims:
        return xx

    # construct spatial ref coordinate for glt_x/glt_y
    tx, sx, _, ty, _, sy = xx.attrs["ortho_geotransform"]  # GDAL order
    gbox = GeoBox(xx.glt_x.shape, Affine(sx, 0, tx, 0, sy, ty), 4326)
    ortho_sr_coords = "ortho_spatial_ref"
    oy, ox, ospatial = xr_coords(gbox, ortho_sr_coords).values()
    ox = ox.rename({ox.dims[0]: "ortho_x"})
    oy = oy.rename({oy.dims[0]: "ortho_y"})
    xx = xx.assign_coords(
        {
            "ortho_x": ox,
            "ortho_y": oy,
            ortho_sr_coords: ospatial,
        }
    )

    # record grid_mappings for spatial data vars
    for dv in chain(xx.data_vars.values(), xx.coords.values()):
        if "ortho_y" in dv.dims:
            dv.encoding["grid_mapping"] = ortho_sr_coords
        elif "x" in dv.dims:
            dv.encoding["grid_mapping"] = "spatial_ref"

    return xx
