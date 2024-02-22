from ._creds import earthdata_token, fetch_s3_creds, prep_s3_fs
from ._gcps import SampleLoader, gcp_geobox, gen_sample
from ._load import emit_load, gbox_from_pix_lonlat, gbox_from_points, ortho_gbox
from ._md import (
    ZarrSpecMode,
    cmr_to_stac,
    emit_id,
    fs_from_stac_doc,
    parse_emit_orbit,
    subchunk_consolidated,
    to_zarr_spec,
)
from ._plots import gxy, mk_error_plot, review_gcp_sample, sample_error
from ._reader import EmitMD, EmitReader, open_zarr
from ._version import __version__
from ._zict import open_zict, open_zict_json

__all__ = [
    "ZarrSpecMode",
    "SampleLoader",
    "EmitMD",
    "EmitReader",
    "gen_sample",
    "gcp_geobox",
    "earthdata_token",
    "fetch_s3_creds",
    "prep_s3_fs",
    "to_zarr_spec",
    "emit_id",
    "parse_emit_orbit",
    "cmr_to_stac",
    "emit_load",
    "gbox_from_points",
    "gbox_from_pix_lonlat",
    "ortho_gbox",
    "sample_error",
    "mk_error_plot",
    "review_gcp_sample",
    "gxy",
    "subchunk_consolidated",
    "fs_from_stac_doc",
    "open_zict",
    "open_zict_json",
    "stac_store",
    "open_zarr",
    "__version__",
]


def stac_store():
    # pylint: disable=protected-access
    fname = "/tmp/emit.zip"
    if (_cache := getattr(stac_store, "_cache", None)) is not None:
        return _cache

    stac_store._cache = open_zict_json(fname, "r")
    return stac_store._cache
