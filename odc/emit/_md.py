from typing import Any, Iterator, Iterable, Literal
from odc.geo import geom, xy_
from odc.geo.gcp import GCPGeoBox, GCPMapping
from .vendor.eosdis_store.dmrpp import to_zarr
from toolz import get_in
import json

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


def emit_id(url: str, postfix: str = "") -> str:
    *_, _dir, _ = url.split("/")
    return _dir + postfix


def shape_from_spec(doc: SomeDoc) -> tuple[int, int]:
    if "refs" in doc:
        doc = doc["refs"]
    return tuple(json.loads(doc["lon/.zarray"])["shape"])


def _do_edits(refs):
    dims = {"downtrack": "y", "crosstrack": "x"}
    coords = {
        ("y", "x"): "lon lat",
        ("bands",): "wavelengths",
        ("y", "x", "bands"): "lon lat wavelengths",
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
    mode: ZarrSpecMode = "default",
) -> dict[str, Any]:
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

    refs = dict(to_docs(zz))
    return refs


def _asset_name_from_url(u):
    return u.rsplit("/", 1)[-1].split("_")[2]


def _footprint(cmr):
    return geom.polygon(
        [
            (p["Longitude"], p["Latitude"])
            for p in get_in(
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
        ],
        4326,
    )


def cmr_to_stac(
    cmr: SomeDoc | str | bytes,
    dmrpp_doc: str | bytes | None = None,
) -> SomeDoc:
    # pylint: disable=too-many-locals
    shape: tuple[int, int] | None = None
    zz: SomeDoc | None = None
    if dmrpp_doc is not None:
        zz = to_zarr_spec(dmrpp_doc)

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

    if zz is not None:
        assets["RFL"].update({"zarr:spec": zz})
        shape = shape_from_spec(zz)

    dt_range = cmr["TemporalExtent"]["RangeDateTime"]
    footprint = _footprint(cmr)

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
    if shape is not None:
        h, w = shape[:2]
        pix = [xy_(0, 0), xy_(0, h), xy_(w, h), xy_(w, 0)]
        wld = [xy_(x, y) for x, y in footprint.exterior.points[:4]]

        gbox = GCPGeoBox((h, w), GCPMapping(pix, wld, 4326))
        proj_props = {
            "proj:epsg": gbox.crs.epsg,
            "proj:shape": gbox.shape.shape,
            "proj:transform": gbox.approx.affine[:6],
        }

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
