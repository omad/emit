import fsspec
import numpy as np
from odc.geo.geobox import GeoBox
from odc.loader import RasterLoadParams, RasterSource
from odc.stac import parse_item
from pystac.item import Item as StacItem

from . import cmr_to_stac, fs_from_stac_doc
from ._md import patch_hrefs, remap_to_local_dir
from ._reader import EmitDriver, EmitMD, LoaderState


def test_reader_driver(cmr_sample, dmrpp_sample, data_dir):
    print(f"data dir: {data_dir}")
    doc = cmr_to_stac(cmr_sample, dmrpp_sample)
    doc = patch_hrefs(doc, remap_to_local_dir(data_dir))

    assert isinstance(doc, dict)
    stac_item = StacItem.from_dict(doc)
    assert "RFL" in stac_item.assets

    gbox = GeoBox.from_bbox(doc["bbox"], resolution=0.01).pad_wh(100)

    s3_fake = fsspec.filesystem("file")
    rfs = fs_from_stac_doc(doc, s3_fake)
    assert isinstance(rfs, fsspec.AbstractFileSystem)

    driver = EmitDriver(s3=s3_fake)
    mdp = driver.md_parser
    assert mdp is not None
    assert isinstance(mdp, EmitMD)
    pit = parse_item(stac_item, md_plugin=driver.md_parser)
    assert isinstance(pit["reflectance"], RasterSource)
    assert pit["reflectance"].driver_data is not None
    assert pit["reflectance"].driver_data["_subdataset"] == "reflectance"
    assert pit["elev"].driver_data["_subdataset"] == "elev"

    env = driver.capture_env()
    assert isinstance(env, dict)

    ctx = driver.new_load(gbox, chunks={})
    assert isinstance(ctx, LoaderState)
    assert ctx.geobox is gbox
    assert ctx.s3 is s3_fake

    with driver.restore_env(env, ctx) as ctx_local:
        assert isinstance(ctx_local, LoaderState)
        src = pit["elev"]

        if not s3_fake.exists(src.uri):
            print(f"Missing test file: {src.uri}")
            return

        assert src.meta is not None
        cfg = RasterLoadParams(
            src.meta.data_type,
            src.meta.nodata,
            resampling="average",
            dims=src.meta.dims,
        )

        rdr = driver.open(src, ctx_local)
        assert rdr is not None
        roi, aa = rdr.read(cfg, gbox)
        assert len(roi) == 2
        assert isinstance(aa, np.ndarray)
        assert aa.shape == gbox[roi].shape.yx
        assert aa.dtype == cfg.dtype
