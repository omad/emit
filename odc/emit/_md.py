import json
from base64 import b64decode, b64encode
from copy import deepcopy
from typing import Any, Iterable, Iterator, Literal

import fsspec
import zarr.convenience as zc
from odc.geo import MaybeCRS, geom, wh_, xy_
from odc.geo.gcp import GCPGeoBox, GCPMapping
from odc.geo.xr import xr_coords
from toolz import get_in

from .vendor.eosdis_store.dmrpp import to_zarr

__all__ = ["cmr_to_stac"]

SomeDoc = dict[str, Any]


ZarrSpecMode = Literal["default"] | Literal["raw"]

BBOX_KEYS = (
    "westernmost_longitude",
    "southernmost_latitude",
    "easternmost_longitude",
    "northernmost_latitude",
)


class Cfg:
    """
    Configuration namespace.
    """

    keep_keys = {"ORBIT", "ORBIT_SEGMENT", "SCENE", "SOLAR_ZENITH", "SOLAR_AZIMUTH"}
    transforms = {
        "ORBIT": int,
        "ORBIT_SEGMENT": int,
        "SCENE": int,
        "SOLAR_ZENITH": float,
        "SOLAR_AZIMUTH": float,
    }
    renames = {
        "SOLAR_AZIMUTH": "view:sun_azimuth",
    }
    gsd = 60
    default_pt_idx = (2, 3, 0, 1)


def emit_id(url: str, postfix: str = "") -> str:
    *_, _dir, _ = url.split("/")
    return _dir + postfix


def shape_from_spec(doc: SomeDoc) -> tuple[int, int]:
    if "refs" in doc:
        doc = doc["refs"]
    return tuple(json.loads(doc["lon/.zarray"])["shape"])


def pack_json(d: str | bytes) -> str:
    return json.dumps(json.loads(d), separators=(",", ":"))


def is_chunk_key(k: str) -> bool:
    *_, leaf = k.rsplit("/", 1)
    return leaf[0].isdigit()


def _do_edits(refs):
    dims = {"downtrack": "y", "crosstrack": "x", "bands": "band"}
    coords = {
        ("y", "x"): "lon lat",
        ("band",): "wavelengths",
        ("y", "x", "band"): "lon lat wavelengths",
    }
    drop_vars = ("location/glt_x", "location/glt_y", "build_dmrpp_metadata")
    flatten_groups = ("location", "sensor_band_parameters")
    skip_zgroups = set(f"{group}/.zgroup" for group in flatten_groups)
    drop_ds_attrs = ("history", "geotransform")

    def _keep(k):
        p, *_ = k.rsplit("/", 1)
        if p in drop_vars:
            return False
        if p in skip_zgroups:
            return False
        return True

    def patch_zattrs(doc, dims, coords):
        A_DIMS = "_ARRAY_DIMENSIONS"
        assert isinstance(doc, dict)

        if A_DIMS not in doc:
            return doc

        doc[A_DIMS] = [dims.get(dim, dim) for dim in doc[A_DIMS]]
        if (coords := coords.get(tuple(doc[A_DIMS]), None)) is not None:
            doc["coordinates"] = coords

        return doc

    def edit_one(k, doc):
        if k.endswith("/.zattrs"):
            doc = patch_zattrs(doc, dims, coords)
        if k == ".zattrs":
            doc = {k: v for k, v in doc.items() if k not in drop_ds_attrs}

        group, *_ = k.rsplit("/", 2)
        if group in flatten_groups:
            k = k[len(group) + 1 :]

        return (k, doc)

    return dict(edit_one(k, doc) for k, doc in refs.items() if _keep(k))


def _walk_dirs_up(p: str) -> Iterator[str]:
    while True:
        *d, _ = p.rsplit("/", 1)
        if not d:
            break
        (p,) = d
        yield p


def _find_groups(all_paths: Iterable[str]) -> list[str]:
    # group is any directory without .zarray in it
    all_bands = set()
    all_dirs = set()

    for p in all_paths:
        if p.endswith(".zarray"):
            all_bands.add(p.rsplit("/", 1)[0])

        for _dir in _walk_dirs_up(p):
            if _dir in all_dirs:
                break
            all_dirs.add(_dir)

    return list(all_dirs - all_bands)


def to_zarr_spec(
    dmrpp_doc: str | bytes,
    url: str | None = None,
    *,
    mode: ZarrSpecMode = "default",
    footprint: geom.Geometry | None = None,
) -> tuple[dict[str, Any], GCPGeoBox | None]:
    def to_docs(zz: dict[str, Any]) -> Iterator[tuple[str, str | tuple[str | None, int, int]]]:
        # sorted keys are needed to work around problem in fsspec directory listing 1430
        for k in sorted(zz, key=lambda p: (p.count("/"), p)):
            doc = zz[k]
            if k.endswith("/.zchunkstore"):
                prefix, _ = k.rsplit("/", 1)
                for chunk_key, info in doc.items():
                    yield f"{prefix}/{chunk_key}", (url, info["offset"], info["size"])
            else:
                yield k, json.dumps(doc, separators=(",", ":"))

    zz = to_zarr(dmrpp_doc)
    if mode == "raw":
        zz.update({f"{group}/.zgroup": {"zarr_format": 2} for group in _find_groups(zz)})
    else:
        zz = _do_edits(zz)

    spec = dict(to_docs(zz))
    if footprint is None or mode == "raw":
        return spec, None

    ny, nx, *_ = shape_from_spec(spec)
    pix = [xy_(0, 0), xy_(0, ny), xy_(nx, ny), xy_(nx, 0)]
    wld = [xy_(x, y) for x, y in footprint.exterior.points[:4]]

    gbox = GCPGeoBox(wh_(nx, ny), GCPMapping(pix, wld, footprint.crs))

    return spec, gbox


def _asset_name_from_url(u):
    return u.rsplit("/", 1)[-1].split("_")[2]


def _footprint(cmr, pts_index):
    pts = get_in(
        [
            "SpatialExtent",
            "HorizontalSpatialDomain",
            "Geometry",
            "GPolygons",
            0,
            "Boundary",
            "Points",
        ],
        cmr,
        no_default=True,
    )
    if pts_index is not None:
        pts = [pts[i] for i in pts_index]

    return geom.polygon([(p["Longitude"], p["Latitude"]) for p in pts], 4326)


def _json_safe_chunk(v):
    if isinstance(v, bytes):
        return b64encode(v).decode("ascii")
    return v


def _unjson_chunk(v):
    if isinstance(v, str):
        return b64decode(v.encode("ascii"))
    return v


def cmr_to_stac(
    cmr: SomeDoc | str | bytes,
    dmrpp_doc: str | bytes | None = None,
    gcp_crs: MaybeCRS = None,
    pts_idx: tuple[int, int, int, int] = Cfg.default_pt_idx,
) -> SomeDoc:
    # pylint: disable=too-many-locals
    if isinstance(cmr, (str, bytes)):
        cmr = json.loads(cmr)

    uu = [x["URL"] for x in cmr["RelatedUrls"] if x["Type"] in {"GET DATA VIA DIRECT ACCESS"}]

    visual_url, *_ = [
        x["URL"] for x in cmr["RelatedUrls"] if x["URL"].startswith("https:") and x["URL"].endswith(".png")
    ]

    assets = {
        _asset_name_from_url(u): {
            "href": u,
            "title": _asset_name_from_url(u).lower(),
            "type": "application/x-hdf5",
            "roles": ["data"],
        }
        for u in uu
    }
    assets.update(
        {
            "visual": {
                "href": visual_url,
                "title": "Visual Preview",
                "type": "image/png",
                "roles": ["overview"],
            }
        }
    )

    dt_range = cmr["TemporalExtent"]["RangeDateTime"]
    footprint = _footprint(cmr, pts_idx)

    gg = {
        "id": cmr["GranuleUR"],
        "bbox": footprint.boundingbox.bbox,
        **footprint.geojson(),
        "assets": assets,
        "collection": f"{cmr['CollectionReference']['ShortName']}.{cmr['CollectionReference']['Version']}",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/view/v1.0.0/schema.json",
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
        ],
    }

    proj_props: dict[str, Any] = {}
    if dmrpp_doc is not None:
        if gcp_crs is None:
            spec, gbox = to_zarr_spec(dmrpp_doc, footprint=footprint)
        else:
            spec, gbox = to_zarr_spec(dmrpp_doc, footprint=footprint.to_crs(gcp_crs))

        assert gbox is not None

        proj_props = {
            "proj:epsg": gbox.crs.epsg,
            "proj:shape": gbox.shape.shape,
            "proj:transform": gbox.approx.affine[:6],
        }
        zc.consolidate_metadata(spec)
        md = json.loads(spec[".zmetadata"])["metadata"]
        md["spatial_ref/.zarray"] = {
            "chunks": [],
            "compressor": None,
            "dtype": "<i4",
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": [],
            "zarr_format": 2,
        }
        spatial_ref = xr_coords(gbox)["spatial_ref"]
        md["spatial_ref/.zattrs"] = {
            "_ARRAY_DIMENSIONS": [],
            **spatial_ref.attrs,
        }
        md[".zattrs"]["coordinates"] = " ".join(["spatial_ref", "wavelengths", "lon", "lat"])

        chunks = {k: _json_safe_chunk(v) for k, v in spec.items() if is_chunk_key(k)}
        chunks["spatial_ref/0"] = _json_safe_chunk(spatial_ref.data.astype("<i4").tobytes())
        assets["RFL"].update(
            {
                "zarr:metadata": md,
                "zarr:chunks": chunks,
            }
        )

    attrs = {
        Cfg.renames.get(n, n): Cfg.transforms.get(n, lambda x: x)(v[0])
        for n, v in ((ee["Name"], ee["Values"]) for ee in cmr["AdditionalAttributes"])
        if n in Cfg.keep_keys
    }
    # TODO: check math for zenith -> elevation conversion
    attrs["view:sun_elevation"] = 90 - attrs.pop("SOLAR_ZENITH")

    gg["properties"].update(
        {
            "datetime": dt_range["BeginningDateTime"],
            "start_datetime": dt_range["BeginningDateTime"],
            "end_datetime": dt_range["EndingDateTime"],
            "created": cmr["DataGranule"]["ProductionDateTime"],
            "eo:cloud_cover": cmr["CloudCover"],
            "platform": cmr["Platforms"][0]["ShortName"],
            "instruments": [p["ShortName"] for p in cmr["Platforms"][0]["Instruments"]],
            "gsd": Cfg.gsd,
            **proj_props,
            **attrs,
        }
    )

    return gg


def subchunk_consolidated(
    variable: str,
    metadata: dict[str, Any],
    chunk_store: dict[str, Any],
    *,
    factor: int | None = None,
    rows_per_chunk: int | None = None,
):
    """
    Subchunk consolidated metadata.

    :param variable: variable name
    :param metadata: parsed consolidated metadata dictionary
    :param chunk_store: chunk store
    :param factor: shrink factor
    :param rows_per_chunk: alternative to ``factor``
    :raises ValueError: when factor does not divide data into integer parts
    :return: ``(metadata, chunk_store)`` modified in-place
    """
    # pylint: disable=too-many-locals
    meta = metadata[f"{variable}/.zarray"]
    chunks_orig = meta["chunks"]
    if factor is None:
        assert rows_per_chunk is not None
        factor = chunks_orig[0] // rows_per_chunk

    if chunks_orig[0] % factor == 0:
        chunk_new = [chunks_orig[0] // factor] + chunks_orig[1:]
    else:
        raise ValueError("Must subchunk by exact integer factor")

    meta["chunks"] = chunk_new
    prefix = f"{variable}/"
    chunk_keys = [k for k in chunk_store if k.startswith(prefix) and is_chunk_key(k)]

    for k in chunk_keys:
        url, offset, size = chunk_store[k]
        kpart = k[len(prefix) :]
        sep = "." if "." in kpart else "/"
        idx = tuple(map(int, kpart.split(sep)))
        new_chunk_sz = size // factor

        ii = idx[0]
        for _ in range(factor):
            new_index = (ii, *idx[1:])
            chunk_path = sep.join(str(idx) for idx in new_index)
            chunk_store[f"{variable}/{chunk_path}"] = (url, offset, new_chunk_sz)
            ii += 1
            offset += new_chunk_sz

    return metadata, chunk_store


def fs_from_stac_doc(doc, fs, *, factor=None, rows_per_chunk=None, asset="RFL"):
    src = doc["assets"][asset]

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

    return fsspec.filesystem("reference", fo=md_store, fs=fs, target=src["href"])
