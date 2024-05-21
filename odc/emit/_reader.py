from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from logging import getLogger
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

import fsspec
import numpy as np
import zarr.convenience
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox
from odc.geo.overlap import compute_reproject_roi
from odc.geo.roi import roi_is_empty, roi_shape
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
from odc.loader._rio import capture_rio_env, rio_env
from odc.loader.types import ReaderSubsetSelection
from zarr.core import Array as ZarrArray

from ._creds import prep_s3_fs
from ._md import _unjson_chunk, subchunk_consolidated
from .assets import EMIT_WAVELENGTH_VALUES
from .gcps import geobox_from_zarr

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

        def __init__(self, geobox: GeoBox, is_dask: bool, s3) -> None:
            self.geobox = geobox
            self.is_dask = is_dask
            self.s3 = s3
            self.finalised = False
            self._cache = _Cache.instance()

        def __getstate__(self) -> dict[str, Any]:
            return {"geobox": self.geobox, "is_dask": self.is_dask, "s3": self.s3, "finalised": self.finalised}

        def __setstate__(self, state: dict[str, Any]) -> None:
            LOG.debug("EmitReader.LoaderState.__setstate__: %r", state)
            self.geobox = state["geobox"]
            self.is_dask = state["is_dask"]
            self.s3 = state["s3"]
            self.finalised = state["finalised"]
            self._cache = _Cache.instance()

        def finalise(self) -> Any:
            self.finalised = True
            return self

        def prep(self, src: RasterSource) -> Tuple[dict[str, Any], str, GCPGeoBox]:
            doc = {"href": src.uri, **src.driver_data}
            subdataset = src.subdataset or doc["_subdataset"]

            def _fetch_gbox(href: str) -> GCPGeoBox:
                zz = self.open_zarr(doc, href=href)
                return geobox_from_zarr(zz)

            gbox = self._cache.get_geobox(doc["href"], _fetch_gbox)
            return doc, subdataset, gbox

        def open_zarr(self, doc, href: str | None = None, **kw):
            return open_zarr(doc, s3=self.s3, href=href, **kw)

        def __dask_tokenize__(self):
            return ("odc.emit.EmitReader.LoaderState", self.is_dask, self.s3)

    def __init__(self, doc: dict[str, Any], subdataset: str, gbox: GCPGeoBox, ctx: "EmitReader.LoaderState") -> None:
        self._doc = doc
        self._subdataset = subdataset
        self._gbox = gbox
        self._ctx = ctx
        self._src = None

    def __getstate__(self) -> dict[str, Any]:
        return {"_doc": self._doc, "_subdataset": self._subdataset, "_gbox": self._gbox, "_ctx": self._ctx}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._doc = state["_doc"]
        self._subdataset = state["_subdataset"]
        self._gbox = state["_gbox"]
        self._ctx = state["_ctx"]
        self._src = None

    def __dask_tokenize__(self):
        return ("odc.emit.EmitReader", self._doc, self._subdataset, *self._ctx.__dask_tokenize__()[1:])

    @staticmethod
    def open(src: RasterSource, ctx: "EmitReader.LoaderState") -> "EmitReader":
        LOG.info("EmitReader.open: %s", src.uri)
        doc, subdataset, gbox = ctx.prep(src)
        return EmitReader(doc, subdataset, gbox, ctx)

    @property
    def href(self) -> str:
        return self._doc.get("href", "<???>")

    def zarr(self) -> ZarrArray:
        if self._src is None:
            self._src = self._ctx.open_zarr(self._doc, rows_per_chunk=32)[self._subdataset]
        return self._src

    def read(
        self,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        *,
        dst: Optional[np.ndarray] = None,
        selection: Optional[ReaderSubsetSelection] = None,
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        # pylint: disable=too-many-locals
        assert selection is None, "EmitReader does not support subsetting"

        band = self._subdataset
        LOG.info("EmitReader.read: %s@%s", band, self.href)

        src = self.zarr()
        src_nodata = resolve_src_nodata(getattr(src, "fill_value", None), cfg)

        postfix_dims = src.shape[2:]
        src_gbox = self._gbox

        assert cfg.dtype is not None
        dst_dtype = np.dtype(cfg.dtype)

        dst_nodata = resolve_dst_nodata(dst_dtype, cfg, src_nodata)
        fill_value = dst_nodata if dst_nodata is not None else 0

        # align=32 is to align to block boundaries (rows_per_chunk)
        rr = compute_reproject_roi(src_gbox, dst_geobox, padding=16, align=32)
        # only slice along y-axis
        roi_src = (rr.roi_src[0], slice(0, src.shape[1]))
        roi_dst = cast(tuple[slice, slice], rr.roi_dst)
        src_gbox = src_gbox[roi_src]

        dst_geobox = dst_geobox[roi_dst]
        if dst is not None:
            _dst = dst[roi_dst]  # type: ignore
        else:
            ny, nx = dst_geobox.shape
            dtype = cfg.dtype or src.dtype
            assert dtype is not None
            _dst = _np_alloc_yxb((ny, nx, *postfix_dims), dtype=dtype)

        assert _dst.shape[:2] == dst_geobox.shape

        if roi_is_empty(roi_dst):
            LOG.debug("empty roi_dst: %r", rr.roi_dst)
            return (roi_dst, _dst)

        if roi_is_empty(roi_src):
            # no overlap case
            LOG.debug("empty roi_src: %r", roi_src)
            np.copyto(_dst, fill_value)
            return (roi_dst, _dst)

        # Perform read with data transpose for 3D cases
        # (y, x, band) -> (band, y, x) as fas as memory layout is concerned
        LOG.debug("native-read.start: %s orig.shape=%s, rows=%d:%d", band, src.shape, roi_src[0].start, roi_src[0].stop)
        _shape = (*roi_shape(roi_src), *postfix_dims)
        _src = _np_alloc_yxb(_shape, dtype=src.dtype)
        src.get_basic_selection(roi_src, out=_src)
        LOG.debug("native-read.stop: %s orig.shape=%s, rows=%d:%d", band, src.shape, roi_src[0].start, roi_src[0].stop)
        del src

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
        LOG.debug("done with reproject: %s %s", band, _src.shape)

        return roi_dst, _dst


class EmitDriver:
    """
    Reader for EMIT data.
    """

    def __init__(self, *, s3=None, creds=None) -> None:
        self._creds = creds
        self._s3 = s3

    def __dask_tokenize__(self):
        return ("odc.emit.EmitDriver", self._s3, self._creds)

    def new_load(
        self,
        geobox: GeoBox,
        *,
        chunks: None | Dict[str, int] = None,
    ) -> EmitReader.LoaderState:
        LOG.debug("EmitDriver.new_load: chunks=%r", chunks)
        s3 = self._s3
        if s3 is None:
            s3 = prep_s3_fs(creds=self._creds)
        return EmitReader.LoaderState(geobox, chunks is not None, s3=s3)

    def finalise_load(self, load_state: EmitReader.LoaderState) -> Any:
        LOG.debug("EmitDriver.finalise_load")
        return load_state.finalise()

    def capture_env(self) -> dict[str, Any]:
        LOG.debug("EmitDriver.capture_env")
        return capture_rio_env()

    @contextmanager
    def restore_env(self, env: Dict[str, Any], load_state: EmitReader.LoaderState) -> Iterator[EmitReader.LoaderState]:
        assert isinstance(env, dict)
        LOG.debug("EmitDriver.restore_env")
        # rasterio reproject inits GDAL, that can be costly across many threads
        # this should cache GDAL thread context
        with rio_env(**env):
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


def _np_alloc_yxb(shape, **kw):
    """
    For 3D shapes, assume ``shape=(ny,nx,nb)`` annd allocate array of the given shape,
    but with native pixel order being ``b,y,x`` in memory.

    For all other shapes, just allocate array as usual.
    """
    if len(shape) != 3:
        return np.empty(shape, **kw)
    ny, nx, nb = shape
    return np.empty((nb, ny, nx), **kw).transpose(1, 2, 0)


def open_zarr(
    doc,
    *,
    s3=None,
    creds=None,
    rows_per_chunk: int | None = None,
    factor: int | None = None,
    s3_opts: dict[str, Any] | None = None,
    href: str | None = None,
):
    if s3 is None:
        if s3_opts is None:
            s3_opts = {}
        LOG.debug("open_zarr: prepping s3")
        s3 = prep_s3_fs(creds=creds, **s3_opts)
    else:
        LOG.debug("open_zarr: using provided s3: %r", id(s3))

    rfs = fs_from_stac_doc(doc, s3, href=href, rows_per_chunk=rows_per_chunk, factor=factor)

    zz = zarr.convenience.open(
        zarr.storage.ConsolidatedMetadataStore(rfs.get_mapper("")),
        "r",
        chunk_store=rfs.get_mapper(""),
    )

    return zz


def fs_from_stac_doc(doc, fs, *, factor=None, rows_per_chunk=None, asset="RFL", href: str | None = None):
    if assets := doc.get("assets", None):
        src = assets[asset]
    else:
        src = doc

    chunks = {k: _unjson_chunk(v) for k, v in src["zarr:chunks"].items()}
    zmd = deepcopy(src["zarr:metadata"])

    if factor is not None or rows_per_chunk is not None:
        band_dims = {
            k.rsplit("/", 1)[0]: tuple(v["_ARRAY_DIMENSIONS"])
            for k, v in zmd.items()
            if k.endswith("/.zattrs") and "_ARRAY_DIMENSIONS" in v
        }

        for band, dims in band_dims.items():
            if len(dims) >= 2 and dims[0] == "y":
                zmd, chunks = subchunk_consolidated(
                    band,
                    zmd,
                    chunks,
                    factor=factor,
                    rows_per_chunk=rows_per_chunk,
                )

    md_store = {
        ".zmetadata": json.dumps({"zarr_consolidated_format": 1, "metadata": zmd}),
        **chunks,
    }
    if href is None:
        href = src["href"]

    return fsspec.filesystem("reference", fo=md_store, fs=fs, target=href)
