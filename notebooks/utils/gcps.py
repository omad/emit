import functools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo import geom
from odc.geo.gcp import GCPGeoBox, GCPMapping
from odc.geo.geobox import GeoBox
from odc.geo.math import quasi_random_r2
from odc.geo.roi import polygon_path
from rasterio.transform import GCPTransformer, GroundControlPoint, TransformerBase

from odc.emit import fs_from_stac_doc, gxy

from .zict import _stac_store

# pylint: disable=import-outside-toplevel


class SampleLoader:
    """
    Sample lon/lat/elev.

    """

    DROPS = ["reflectance", "fwhm", "good_wavelengths", "wavelengths"]

    def __init__(self, pts, s3):
        self._pts = pts
        self._s3 = s3

    def _doc_store(self):
        return _stac_store()

    def get(self, _id):
        docs = self._doc_store()
        xx = xr.open_dataset(
            fs_from_stac_doc(docs[_id], self._s3).get_mapper(),
            engine="zarr",
            drop_variables=self.DROPS,
        )
        return {
            "id": _id,
            **_np_extract_sample(xx.lon.data, xx.lat.data, xx.elev.data, self._pts),
        }


def gen_sample(
    n,
    *,
    n_edge="auto",
    pad="auto",
    npix_per_side_target=1280,
):
    if n_edge == "auto":
        n_edge = int(np.round(np.sqrt(n)))

    if pad == "auto":
        pad = 0.25 * (1 / n_edge)
    elif pad > 1:
        pad = pad / npix_per_side_target

    pts = np.vstack(
        [
            quasi_random_r2(n) * (1 - 2 * pad) + pad,
            polygon_path(np.linspace(0, 1, n_edge), closed=False).T,
        ]
    )

    return pts


def sample_to_idx(pts, shape):
    ny, nx = shape
    iy, ix = [np.clip(np.round(a * (n - 1)).astype("int32"), 0, n - 1) for a, n in zip([pts.T[1], pts.T[0]], [ny, nx])]

    return iy, ix


def _np_extract_sample(x, y, z, pts):
    iy, ix = sample_to_idx(pts, x.shape)
    wx, wy, wz = [aa[iy, ix].ravel() for aa in (x, y, z)]

    return {
        "row": iy.tolist(),
        "col": ix.tolist(),
        "x": wx.tolist(),
        "y": wy.tolist(),
        "z": wz.tolist(),
        "shape": x.shape,
    }


def _rio_gcp_transform(gg, tr, crs=4326):
    if gg.crs is None:
        col, row = gxy(gg).T
        return geom.multipoint(list(zip(*tr.xy(row, col))), crs)

    x, y = gxy(gg).T
    r, c = tr.rowcol(x, y, op=lambda a: a + 0.5)
    return geom.multipoint(list(zip(c, r)), None)


def mk_rio_gcp_mapper(tr, crs=4326):
    return functools.partial(_rio_gcp_transform, tr=tr, crs=crs)


def extract_rio_gcps(sample, skip_z=False, pix_offset=0):
    return [
        GroundControlPoint(
            r + pix_offset,
            c + pix_offset,
            x,
            y,
            None if skip_z else z,
            str(_id),
        )
        for _id, (r, c, x, y, z) in enumerate(zip(*[sample[k] for k in ("row", "col", "x", "y", "z")]))
    ]


def rio_gcp_transformer(sample, nsamples=None, skip_z=False, pix_offset=0):
    if nsamples is not None:
        sample = sub_sample(sample, nsamples)
    return GCPTransformer(extract_rio_gcps(sample, skip_z=skip_z, pix_offset=pix_offset))


def gcp_geobox(sample, nsample=None, crs=4326):
    if nsample is not None:
        sample = sub_sample(sample, nsample)

    wld = np.stack([sample["x"], sample["y"]]).T
    pix = np.stack([sample["col"], sample["row"]]).T + 0.5
    mapping = GCPMapping(pix, wld, crs)

    return GCPGeoBox(sample["shape"], mapping)


def compute_error(sample, pt_mapper):
    pts_w = geom.multipoint(list(zip(sample["x"], sample["y"])), 4326)
    pts_p = geom.multipoint([(x + 0.5, y + 0.5) for x, y in zip(sample["col"], sample["row"])], None)
    if isinstance(pt_mapper, TransformerBase):
        project = mk_rio_gcp_mapper(pt_mapper)

    if isinstance(pt_mapper, (GeoBox, GCPGeoBox)):
        project = pt_mapper.project

    ee = gxy(pts_p) - gxy(project(pts_w))
    pix_error = np.sqrt((ee**2).sum(axis=1))
    return SimpleNamespace(
        pts_p=pts_p,
        pts_w=pts_w,
        pix_error=pix_error,
        ee=ee,
        shape=sample["shape"],
    )


def to_pandas(sample):
    xx = pd.DataFrame({k: sample[k] for k in ("row", "col", "x", "y", "z")})
    xx.attrs["id"] = sample["id"]
    xx.attrs["src_shape"] = sample["shape"]
    xx.attrs["ny"] = sample["shape"][0]
    xx.attrs["nx"] = sample["shape"][1]
    return xx


def sub_sample(sample, n, nside=None):
    if isinstance(n, slice):
        roi = n
    else:
        roi = slice(0, n)

    edges = {}
    if nside is not None:
        edges = sample_edge(sample, nside=nside)

    def _proc(v, k):
        if k in ("shape", "id"):
            return v
        if isinstance(v, list):
            return v[roi] + edges.get(k, [])
        return v

    return {k: _proc(v, k) for k, v in sample.items()}


def sample_edge(sample, nside=None):
    pts = np.stack([sample["row"], sample["col"]]).T
    pix = np.asarray([sample["row"], sample["col"]])
    (idx,) = np.where((pix[0] == 0) * (pix[1] == 0))
    idx = int(idx[-1])
    nside_have = pts[idx:].shape[0] // 4

    step = None
    if nside is not None:
        if nside_have > nside:
            step = max(1, int(nside_have // nside))

    def _proc(v, k):
        if k in ("shape", "id"):
            return v
        if not isinstance(v, list):
            return v
        src = v[idx:]
        if step is None:
            return src

        out = []
        for _ in range(4):
            out.extend(src[0:nside_have:step])
            src = src[nside_have:]
        return out

    return {k: _proc(v, k) for k, v in sample.items()}


def _style_ax(ax, mode="all"):
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)

    if mode == "all":
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.RIVERS)


def mk_error_plot(
    rr,
    max_err_axis: float = -1,
    msg: str = "",
    figsize=None,
) -> SimpleNamespace:
    import seaborn as sns
    from matplotlib import pyplot as plt

    ui_mode = "square"
    if rr.shape[0] > rr.shape[1] * 1.3:
        ui_mode = "tall"

    ui = {
        "tall": [
            ["A", "A", "A", "B", "B"],
            ["A", "A", "A", "B", "B"],
            ["C", "C", "C", "C", "C"],
        ],
        "square": [
            ["A", "A", "A", "B", "B", "B"],
            ["A", "A", "A", "B", "B", "B"],
            ["A", "A", "A", "B", "B", "B"],
            ["C", "C", "C", "C", "C", "C"],
        ],
    }

    if figsize is None:
        figsize = {"square": (8, 6), "tall": (8, 8)}[ui_mode]

    fig, axd = plt.subplot_mosaic(ui[ui_mode], figsize=figsize)
    rr.fig = fig
    rr.axd = axd

    info = []
    if epsg := rr.pts_w.crs.epsg:
        info.append(f"epsg:{epsg}")

    info.append(f"avg:{rr.pix_error.mean():.3f}px")
    if msg:
        info.append(msg)
    info = " ".join(info)

    fig.suptitle(f"Pixel Registration Error, {info}")

    sns.scatterplot(x=rr.ee.T[0], y=rr.ee.T[1], size=rr.pix_error, ax=axd["A"])
    if max_err_axis > 0:
        b = max_err_axis
    else:
        b = max(map((lambda x: float(abs(x))), axd["A"].axis()))

    axd["A"].axis([-b, b, -b, b])
    axd["A"].set_aspect(1)
    axd["A"].axvline(0, color="k", linewidth=0.3)
    axd["A"].axhline(0, color="k", linewidth=0.3)
    axd["A"].set_aspect(1)

    X, Y = gxy(rr.pts_p).T
    with plt.rc_context({"legend.loc": "lower right"}):
        ax = axd["B"]
        ax.set_aspect(1)
        sns.scatterplot(
            x=X,
            y=Y,
            s=20,
            c=rr.pix_error,
            alpha=1,
            ax=ax,
        )
        ny, nx = rr.shape
        ax.axis([0, nx, ny, 0])
        ax.yaxis.tick_right()
        ax.set_aspect(1)

    axd["C"].axvline(rr.pix_error.mean(), color="y")
    sns.kdeplot(rr.pix_error, ax=axd["C"], clip=(0, b * 100))
    *_, maxy = axd["C"].axis()
    axd["C"].axis([0, b, 0, maxy])
    fig.tight_layout()
    return rr


def review_sample(sample, figsize=(8, 6), s=50):
    import cartopy.crs as ccrs
    from matplotlib import pyplot as plt

    ny, nx = sample["shape"]

    fig, axd = plt.subplot_mosaic(
        [
            ["A", "A", "A", "B", "B"],
            ["A", "A", "A", "B", "B"],
            ["A", "A", "A", "B", "B"],
            ["M", "M", "M", "B", "B"],
        ],
        figsize=figsize,
        per_subplot_kw={
            "M": {"projection": ccrs.PlateCarree()},
            "A": {"projection": ccrs.PlateCarree()},
        },
    )

    ax = axd["M"]
    _style_ax(ax, mode="basic")
    ax.set_extent([-180, 180, -55, 70])
    ax.scatter(sample["x"], sample["y"], c="r")

    ax = axd["A"]
    _style_ax(ax)
    _ = ax.scatter(sample["x"], sample["y"], c=sample["z"], s=s, alpha=0.5)

    ax = axd["B"]
    ax.scatter(sample["col"], sample["row"], c=sample["z"], s=s, alpha=1)
    ax.axis([0, nx, ny, 0])
    ax.axis("equal")
    ax.yaxis.tick_right()

    fig.tight_layout()
    return fig, axd
