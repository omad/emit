from __future__ import annotations

from contextlib import contextmanager
from logging import getLogger
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import zarr.convenience
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox
from odc.geo.overlap import compute_reproject_roi
from odc.geo.roi import NormalizedROI, roi_is_empty
from odc.geo.warp import rio_reproject
from odc.loader import (
    BandKey,
    FixedCoord,
    RasterBandMetadata,
    RasterGroupMetadata,
    RasterLoadParams,
    RasterSource,
    resolve_dst_nodata,
    resolve_src_nodata,
)
from zarr.core import Array as ZarrArray

from ._creds import prep_s3_fs
from ._gcps import geobox_from_zarr
from ._md import fs_from_stac_doc
from .assets import EMIT_WAVELENGTH_VALUES

LOG = getLogger(__name__)


class EmitMD:
    """
    EMIT metadata extractor.
    """

    SUBDATASETS = ["reflectance", "lon", "lat", "elev"]

    FIXED_MD = RasterGroupMetadata(
        bands={
            ("RFL", 1): RasterBandMetadata("float32", -9999.0, dims=("y", "x", "wavelength")),
            ("RFL", 2): RasterBandMetadata("float64", -9999.0),
            ("RFL", 3): RasterBandMetadata("float64", -9999.0),
            ("RFL", 4): RasterBandMetadata("float64", -9999.0),
        },
        aliases={n: [("RFL", i)] for i, n in enumerate(SUBDATASETS, start=1)},
        extra_dims={"wavelength": len(EMIT_WAVELENGTH_VALUES)},
        extra_coords=[
            FixedCoord(
                "wavelength",
                EMIT_WAVELENGTH_VALUES,
                dtype="float32",
                dim="wavelength",
                units="nm",
            )
        ],
    )

    def extract(self, md: Any) -> RasterGroupMetadata:
        assert md is not None
        return self.FIXED_MD

    def driver_data(self, md: Any, band_key: BandKey) -> Any:
        if band_key not in self.FIXED_MD.bands:
            return None

        _, idx = band_key
        data = {k: v for k, v in md.extra_fields.items() if k.startswith("zarr:")}
        data["_subdataset"] = self.SUBDATASETS[idx - 1]
        return data


class _Cache:
    """
    Cache of src GeoBoxes.

    ``url: str -> GCPGeoBox``
    """

    def __init__(self):
        self._gboxes = {}

    def get_geobox(self, href: str, fallback: Callable[[str], GCPGeoBox]) -> GCPGeoBox:
        gbox = self._gboxes.get(href, None)
        if gbox is None:
            LOG.debug("Fetching gbox: %s", href)
            gbox = fallback(href)
            self._gboxes[href] = gbox
            LOG.debug("Done with gbox: %s", href)
        else:
            LOG.debug("Using cached gbox: %s", href)
        return gbox

    def clear(self):
        self._gboxes.clear()

    @staticmethod
    def instance() -> "_Cache":
        return _cache


_cache = _Cache()


class EmitReader:
    """
    EMIT reader.
    """

    class LoaderState:
        """
        Shared state across all readers.
        """

        def __init__(self, is_dask: bool, s3) -> None:
            self.is_dask = is_dask
            self.s3 = s3
            self.finalised = False
            self._cache = _Cache.instance()

        def __getstate__(self) -> dict[str, Any]:
            return {"is_dask": self.is_dask, "s3": self.s3, "finalised": self.finalised}

        def __setstate__(self, state: dict[str, Any]) -> None:
            self.is_dask = state["is_dask"]
            self.s3 = state["s3"]
            self.finalised = state["finalised"]
            self._cache = _Cache.instance()

        def finalise(self) -> Any:
            self.finalised = True
            return self

        def prep(self, src: RasterSource) -> Tuple[ZarrArray, GCPGeoBox]:
            doc = {"href": src.uri, **src.driver_data}
            subdataset = src.subdataset or doc["_subdataset"]
            zz = open_zarr(doc, s3=self.s3, rows_per_chunk=32)
            gbox = self._cache.get_geobox(doc["href"], lambda _: geobox_from_zarr(zz))
            return zz[subdataset], gbox

        def __dask_tokenize__(self):
            return ("odc.emit.EmitReader.LoaderState", self.is_dask, self.s3)

    def __init__(self, src: ZarrArray, gbox: GCPGeoBox, ctx: "EmitReader.LoaderState") -> None:
        self._src = src
        self._gbox = gbox
        self._ctx = ctx

    def __dask_tokenize__(self):
        return ("odc.emit.EmitReader", self._src, *self._ctx.__dask_tokenize__()[1:])

    @staticmethod
    def open(src: RasterSource, ctx: "EmitReader.LoaderState") -> "EmitReader":
        LOG.info("EmitReader.open: %s", src.uri)
        zarr_array, gbox = ctx.prep(src)
        return EmitReader(zarr_array, gbox, ctx)

    @property
    def href(self) -> str:
        try:
            return self._src.chunk_store.fs.target
        except AttributeError:
            return "<???>"

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        dst: Optional[np.ndarray] = None,
    ) -> Tuple[NormalizedROI, np.ndarray]:
        # pylint: disable=too-many-locals
        src = self._src
        band = src.basename
        LOG.info("EmitReader.read: %s@%s", band, self.href)
        src_nodata = resolve_src_nodata(getattr(src, "fill_value", None), cfg)

        postfix_dims = src.shape[2:]
        src_gbox = self._gbox

        assert cfg.dtype is not None
        dst_dtype = np.dtype(cfg.dtype)

        dst_nodata = resolve_dst_nodata(dst_dtype, cfg, src_nodata)
        fill_value = dst_nodata if dst_nodata is not None else 0

        rr = compute_reproject_roi(src_gbox, dst_geobox, padding=16)
        # only slice along y-axis
        roi_src = (rr.roi_src[0], slice(0, src.shape[1]))
        src_gbox = src_gbox[roi_src]

        dst_geobox = dst_geobox[rr.roi_dst]
        if dst is not None:
            _dst = dst[rr.roi_dst]  # type: ignore
        else:
            ny, nx = dst_geobox.shape
            dtype = cfg.dtype or src.dtype
            assert dtype is not None
            _dst = np.full((ny, nx, *postfix_dims), fill_value, dtype=dtype)

        if roi_is_empty(rr.roi_dst):
            LOG.debug("empty roi_dst: %r", rr.roi_dst)
            return (rr.roi_dst, _dst)

        if roi_is_empty(roi_src):
            # no overlap case
            LOG.debug("empty roi_src: %r", roi_src)
            np.copyto(_dst, fill_value)
            return (rr.roi_dst, _dst)

        LOG.debug("native-read.start: %s orig.shape=%s, rows=%d:%d", band, src.shape, roi_src[0].start, roi_src[0].stop)
        _src = src[roi_src]
        LOG.debug("native-read.stop: %s orig.shape=%s, rows=%d:%d", band, src.shape, roi_src[0].start, roi_src[0].stop)

        assert isinstance(_src, np.ndarray)
        LOG.debug(
            "starting reproject: %s %s, nodata=%f=>%f, fill=%f", band, _src.shape, src_nodata, dst_nodata, fill_value
        )

        _dst = rio_reproject(
            _src,
            _dst,
            src_gbox,
            dst_geobox,
            resampling=cfg.resampling,
            dst_nodata=dst_nodata,
            src_nodata=src_nodata,
            ydim=0,
        )
        LOG.debug("done with reproject: %s %s", band, src.shape)

        return rr.roi_dst, _dst


class EmitDriver:
    """
    Reader for EMIT data.
    """

    def __init__(self, *, s3=None, creds=None) -> None:
        self._creds = creds
        self._s3 = s3

    def __dask_tokenize__(self):
        return ("odc.emit.EmitDriver", self._s3, self._creds)

    def new_load(self, chunks: None | Dict[str, int] = None) -> EmitReader.LoaderState:
        s3 = self._s3
        if s3 is None:
            s3 = prep_s3_fs(creds=self._creds)
        LOG.debug("EmitDriver.new_load: chunks=%r", chunks)
        return EmitReader.LoaderState(chunks is not None, s3=s3)

    def finalise_load(self, load_state: EmitReader.LoaderState) -> Any:
        LOG.debug("EmitDriver.finalise_load")
        return load_state.finalise()

    def capture_env(self) -> dict[str, Any]:
        LOG.debug("EmitDriver.capture_env")
        return {}

    @contextmanager
    def restore_env(self, env: Dict[str, Any], load_state: EmitReader.LoaderState) -> Iterator[EmitReader.LoaderState]:
        assert isinstance(env, dict)
        LOG.debug("EmitDriver.restore_env")
        yield load_state

    def open(
        self,
        src: RasterSource,
        ctx: EmitReader.LoaderState,
    ) -> EmitReader:
        return EmitReader.open(src, ctx)

    @property
    def md_parser(self) -> EmitMD:
        return EmitMD()


def open_zarr(
    doc,
    *,
    s3=None,
    creds=None,
    rows_per_chunk: int | None = None,
    factor: int | None = None,
    s3_opts: dict[str, Any] | None = None,
):
    if s3 is None:
        if s3_opts is None:
            s3_opts = {}
        s3 = prep_s3_fs(creds=creds, **s3_opts)

    rfs = fs_from_stac_doc(doc, s3, rows_per_chunk=rows_per_chunk, factor=factor)

    zz = zarr.convenience.open(
        zarr.storage.ConsolidatedMetadataStore(rfs.get_mapper("")),
        "r",
        chunk_store=rfs.get_mapper(""),
    )

    return zz
