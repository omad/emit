import json
from pathlib import Path

import fsspec
import pytest
import xarray as xr
from pystac.item import Item

from . import cmr_to_stac, emit_load, to_zarr_spec
from .vendor.eosdis_store.dmrpp import to_zarr

# pylint: disable=redefined-outer-name


@pytest.fixture
def data_dir():
    yield Path(__file__).parent / "test_data"


@pytest.fixture
def cmr_sample(data_dir):
    path = data_dir / "emit_l2a_rfl_sample.cmr.json"
    with open(path, "rt", encoding="utf8") as f:
        doc = json.load(f)
    yield doc


@pytest.fixture
def cmr_sample_partial(cmr_sample):
    doc = cmr_sample.copy()
    doc.pop("RelatedUrls", "")
    yield doc


@pytest.fixture
def dmrpp_sample(data_dir):
    path = data_dir / "emit_l2a_rfl_sample.dmrpp"
    with open(path, "rt", encoding="utf8") as f:
        doc = f.read()
    yield doc


def test_cmr(cmr_sample, dmrpp_sample):
    doc = cmr_to_stac(cmr_sample)
    assert "id" in doc
    item = Item.from_dict(doc)

    assert len(item.assets) > 0
    assert item.datetime is not None

    doc = cmr_to_stac(cmr_sample, dmrpp_sample)

    # must be JSON serializable
    assert json.dumps(doc) is not None

    assert "id" in doc
    item = Item.from_dict(doc)

    assert "zarr:metadata" in item.assets["RFL"].to_dict()
    assert "zarr:chunks" in item.assets["RFL"].to_dict()

    fs_fake = fsspec.filesystem("reference", fo={})
    xx = emit_load(doc, fs_fake, chunks={"y": 32})
    assert isinstance(xx, xr.Dataset)
    assert "ortho_x" in xx.dims
    assert "ortho_y" in xx.coords


def test_cmr_partial(cmr_sample_partial, dmrpp_sample):
    for dmrpp in (None, dmrpp_sample):
        doc = cmr_to_stac(cmr_sample_partial, dmrpp)
        assert "id" in doc
        item = Item.from_dict(doc)

        assert len(item.assets) == 0
        assert item.datetime is not None


def test_dmrpp(dmrpp_sample):
    zz = to_zarr(dmrpp_sample)
    assert isinstance(zz, dict)
    assert "reflectance/.zarray" in zz
    assert "reflectance/.zchunkstore" in zz
    assert list(zz["reflectance/.zchunkstore"]) == ["0.0.0"]
    assert list(zz["location/lon/.zchunkstore"]) == ["0.0"]


@pytest.mark.parametrize("mode", ("raw", "default"))
@pytest.mark.parametrize("url", (None, "s3://fake/dummy.nc"))
def test_to_zarr_spec(dmrpp_sample, mode, url):
    spec, gbox = to_zarr_spec(dmrpp_sample, url, mode=mode)
    assert gbox is None

    assert "reflectance/0.0.0" in spec
    xx = spec["reflectance/0.0.0"]
    assert isinstance(xx, tuple)
    assert len(xx) == 3
    assert xx[0] == url
    assert isinstance(xx[1], int)
    assert isinstance(xx[2], int)

    fs = fsspec.filesystem("reference", fo=spec)

    xx = xr.open_dataset(
        fs.get_mapper(""),
        engine="zarr",
        backend_kwargs={"consolidated": False},
        chunks={},
    )

    def _json(k):
        return json.loads(fs.cat(k))

    if mode == "raw":
        assert "location/lon/.zarray" in spec
        assert _json("reflectance/.zattrs")["_ARRAY_DIMENSIONS"] == ["downtrack", "crosstrack", "bands"]
        assert "history" in _json(".zattrs")
        assert "geotransform" in _json(".zattrs")
        assert set(xx.data_vars) == set(["reflectance"])
        assert xx.reflectance.dims == ("downtrack", "crosstrack", "bands")
    else:
        assert "lon/.zarray" in spec
        assert "history" not in _json(".zattrs")
        assert "geotransform" not in _json(".zattrs")
        assert _json("reflectance/.zattrs")["coordinates"] == "lon lat wavelength"
        assert _json("reflectance/.zattrs")["_ARRAY_DIMENSIONS"] == ["y", "x", "wavelength"]

        assert set(xx.data_vars) == set(["reflectance", "good_wavelength", "fwhm", "elev", "glt_x", "glt_y"])
        assert set(xx.dims) == set(["y", "x", "wavelength", "ortho_x", "ortho_y"])
        assert set(xx.coords) == set(["lat", "lon", "wavelength"])
        assert xx.lon.shape == xx.lat.shape
        assert xx.lon.shape == xx.reflectance.shape[:2]
        assert xx.reflectance.shape[-1] == xx.good_wavelength.shape[0]
