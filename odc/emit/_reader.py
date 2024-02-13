from __future__ import annotations

from typing import Any, ContextManager, Optional, Tuple

import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.roi import NormalizedROI
from odc.stac.loader import RasterBandMetadata, RasterLoadParams, RasterSource


class EmitMD:
    """
    EMIT metadata extractor.
    """

    SUBDATASETS = ("reflectance", "lon", "lat", "elev")
    ASSET_RFL = "RFL"

    def bands(self, md: Any, name: str) -> Tuple[RasterBandMetadata, ...]:
        assert md is not None
        if name == self.ASSET_RFL:
            # reflectance,lon,lat,elev
            return (
                RasterBandMetadata("float32", -9999, dims=("y", "x", "wavelength")),
                RasterBandMetadata("float32", -9999),
                RasterBandMetadata("float32", -9999),
                RasterBandMetadata("float64", -9999),
            )
        return ()

    def aliases(self, md: Any, name: str) -> Tuple[str, ...]:
        assert md is not None
        if name == self.ASSET_RFL:
            return self.SUBDATASETS

        return ()

    def driver_data(self, md: Any, name: str, idx: int) -> Any:
        if name == self.ASSET_RFL:
            data = {k: v for k, v in md.extra_fields.items() if k.startswith("zarr:")}
            data["_idx"] = idx
            data["_subdataset"] = self.SUBDATASETS[idx]
            return data
        return None


class EmitReader:
    """
    Reader for EMIT data.
    """

    class Context:
        """
        EMIT Context manager.
        """

        def __init__(self, parent: "EmitReader", env: dict[str, Any]) -> None:
            self._parent = parent
            self.env = env

        def __enter__(self):
            self._parent._ctx = self

        def __exit__(self, type, value, traceback):
            # pylint: disable=unused-argument,redefined-builtin
            self._parent._ctx = None

    def __init__(self) -> None:
        self._ctx: EmitReader.Context | None = None

    @property
    def md_parser(self) -> EmitMD:
        return EmitMD()

    def capture_env(self) -> dict[str, Any]:
        return {}

    def restore_env(self, env: dict[str, Any]) -> ContextManager[Any]:
        return EmitReader.Context(self, env)

    def read(
        self,
        src: RasterSource,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        dst: Optional[np.ndarray] = None,
        **kw,
    ) -> Tuple[NormalizedROI, np.ndarray]:

        raise RuntimeError("not implemented")


driver = EmitReader()
