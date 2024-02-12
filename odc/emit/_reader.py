from typing import Any, ContextManager, Dict, Optional, Tuple

import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.roi import NormalizedROI
from odc.stac.loader import RasterBandMetadata, RasterLoadParams, RasterSource


class EmitMD:
    """
    EMIT metadata extractor.
    """

    def bands(self, md: Any) -> Tuple[RasterBandMetadata, ...]:
        if md.title != "rfl":
            return ()
        # reflectance,lon,lat,elev
        return (
            RasterBandMetadata("float32", -9999, dims=("y", "x", "wavelength")),
            RasterBandMetadata("float32", -9999),
            RasterBandMetadata("float32", -9999),
            RasterBandMetadata("float64", -9999),
        )

    def aliases(self, md: Any) -> Tuple[str, ...]:
        if md.title != "rfl":
            return ()
        return ("reflectance", "lon", "lat", "elev")

    def driver_data(self, md: Any) -> Any:
        return {k: v for k, v in md.extra_fields.items() if k.startswith("zarr:")}


class EmitReader:
    """
    Reader for EMIT data.
    """

    class Context:
        """
        EMIT Context manager.
        """

        def __init__(self, env) -> None:
            self.env = env

        def __enter__(self):
            pass

        def __exit__(self, type, value, traceback):
            # pylint: disable=unused-argument,redefined-builtin
            pass

    @property
    def md_parser(self):
        return EmitMD()

    def capture_env(self) -> Dict[str, Any]:
        return {}

    def restore_env(self, env: Dict[str, Any]) -> ContextManager[Any]:
        return EmitReader.Context(env)

    def read(
        self,
        src: RasterSource,
        cfg: RasterLoadParams,
        dst_geobox: GeoBox,
        dst: Optional[np.ndarray] = None,
        *,
        ctx: Any = None,
    ) -> Tuple[NormalizedROI, np.ndarray]:
        raise RuntimeError("not implemented")


driver = EmitReader()
