from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Tuple

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from odc.geo.converters import rio_geobox
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr

# uuid.uuid5(uuid.NAMESPACE_URL, "https://stacspec.org")
UUID_NAMESPACE_STAC = uuid.UUID("55d26088-a6d0-5c77-bf9a-3a7f3c6a6dab")


def _envi_np_mmap(path: str) -> Tuple[Any, float, GeoBox | GCPGeoBox | None]:
    """Read ENVI file as numpy memmap array"""

    with rasterio.open(path) as src:
        nb, ny, nx, dtype, nodata = [src.meta[k] for k in ["count", "height", "width", "dtype", "nodata"]]
        gbox = rio_geobox(src)

        # TODO: assumes bil
        pix = np.memmap(path, mode="r", shape=(ny, nb, nx), dtype=dtype)
        return pix.transpose([0, 2, 1]), float(nodata), gbox


def envi_to_xr(path: str) -> xr.DataArray:
    """Open local ENVI file as xarray.DataArray"""
    data, nodata, gbox = _envi_np_mmap(path)
    assert gbox is not None
    return wrap_xr(data, gbox, nodata=nodata, axis=0, _FillValue=nodata)


def av3_basename(fname):
    fname = fname.rsplit("/", 2)[-1]
    parts = fname.split("_")
    return "_".join(parts[:5])


def av3_timestamp(fname):
    base, *_ = av3_basename(fname).split("_", 2)
    return datetime.strptime(base[3:], "%Y%m%dt%H%M%S")


def av3_extra_coords(fname):
    with rioxarray.open_rasterio(fname, chunks={}) as rr:
        fwhm = rr.fwhm.data.compute()
        wavelength = rr.wavelength.data.compute()

        return {
            "wavelength": xr.DataArray(
                wavelength,
                dims=("wavelength",),
                name="wavelength",
                attrs={"units": "nm"},
            ),
            "fwhm": xr.DataArray(
                fwhm,
                dims=("wavelength",),
                name="fwhm",
                attrs={"units": "nm"},
            ),
        }


def av3_xr_load(base, extra_coords: dict[str, Any] | None = None):
    if extra_coords is None:
        extra_coords = av3_extra_coords(f"{base}_RFL_ORT")

    rfl = envi_to_xr(f"{base}_RFL_ORT").drop_vars("band").swap_dims({"band": "wavelength"}).assign_coords(extra_coords)
    rfl_unc = (
        envi_to_xr(f"{base}_UNC_ORT").drop_vars("band").swap_dims({"band": "wavelength"}).assign_coords(extra_coords)
    )

    atm = envi_to_xr(f"{base}_ATM_ORT").drop_vars("band")

    ds = xr.Dataset(
        {
            "rfl": rfl,
            "rfl_unc": rfl_unc,
            "AOT550": atm.isel(band=0, drop=True),
            "H2OSTR": atm.isel(band=1, drop=True),
        },
        attrs={"id": av3_basename(base)},
    )
    return ds


def av3_mk_dataset(xx: xr.Dataset, zarr_md: dict[str, Any] | None = None) -> dict[str, Any]:
    _id = xx.id
    gbox = xx.odc.geobox
    assert gbox is not None
    assert gbox.crs is not None
    assert gbox.crs.epsg is not None

    _uuid = str(uuid.uuid5(UUID_NAMESPACE_STAC, _id))

    url = f"s3://adias-prod-dc-data-projects/odc-hs/av3/{_id}.zarr"
    dt = av3_timestamp(_id).isoformat() + "Z"

    doc = {
        "$schema": "https://schemas.opendatacube.org/dataset",
        "id": _uuid,
        "product": {"name": "av3_l2a"},
        "location": url,
        # grids
        "crs": f"EPSG:{gbox.crs.epsg}",
        "grids": {
            "default": {
                "shape": list(gbox.shape.yx),
                "transform": list(gbox.transform)[:6],
            }
        },
        # ----------
        "properties": {
            "av3:granule": _id,
            "dtr:start_datetime": dt,
            "dtr:end_datetime": dt,
            "eo:instrument": ["AVIRIS"],
        },
        "measurements": {name: {"layer": name, "driver_data": "rfl"} for name in map(str, xx.data_vars)},
    }

    if zarr_md is not None:
        doc["driver_data"] = {"rfl": {"zarr:metadata": zarr_md["metadata"]}}

    return doc
