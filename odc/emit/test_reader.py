import fsspec
import numpy as np
from dask.base import is_dask_collection, tokenize
from odc.geo.geobox import GeoBox
from odc.loader import RasterLoadParams, RasterSource
from odc.stac import parse_item
from pystac.item import Item as StacItem

from . import cmr_to_stac, fs_from_stac_doc
from ._md import patch_hrefs, remap_to_local_dir
from ._reader import EmitDriver, EmitMD, EmitReaderDask, LoaderState


def test_reader_driver(cmr_sample, dmrpp_sample, data_dir):
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

    assert driver.dask_reader is not None

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


def test_reader_driver_dask(cmr_sample, dmrpp_sample, data_dir):
    s3_fake = fsspec.filesystem("file")
    driver = EmitDriver(s3=s3_fake)

    doc = cmr_to_stac(cmr_sample, dmrpp_sample)
    doc = patch_hrefs(doc, remap_to_local_dir(data_dir))
    stac_item = StacItem.from_dict(doc)
    pit = parse_item(stac_item, md_plugin=driver.md_parser)

    gbox = GeoBox.from_bbox(doc["bbox"], resolution=0.01).pad_wh(100)

    src = pit["reflectance"]
    ctx = driver.new_load(gbox, chunks={})
    tk = tokenize(src, ctx)

    assert src.meta is not None
    cfg = RasterLoadParams(
        src.meta.data_type,
        src.meta.nodata,
        resampling="average",
        dims=src.meta.dims,
    )

    layer_name = f"reflectance-{tk}"
    idx = (0, 0, 0, 0, 0)
    rdr = driver.dask_reader.open(src, ctx, layer_name=layer_name)
    assert isinstance(rdr, EmitReaderDask)
    chunk_future = rdr.read(cfg, gbox, selection=np.s_[:10], idx=idx)

    assert is_dask_collection(chunk_future)

    if not s3_fake.exists(src.uri):
        print(f"Missing test file: {src.uri}")
        return

    ydim = src.ydim

    _roi, aa = chunk_future.compute(schedule="single-threaded")
    assert isinstance(aa, np.ndarray)
    assert isinstance(_roi, tuple)
    assert len(_roi) == 2
    assert aa.shape[ydim : ydim + 2] == gbox[_roi].shape.yx
    assert aa.shape == (*gbox[_roi].shape.yx, 10)
