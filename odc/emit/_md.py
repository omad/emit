from typing import Any
from odc.geo import geom, xy_
from odc.geo.gcp import GCPGeoBox, GCPMapping
from toolz import get_in

__all__ = ["cmr_to_stac"]

SomeDoc = dict[str, Any]


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
    cmr: SomeDoc,
    shape: tuple[int, int] | dict[str, tuple[int, int]] | None = None,
):
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

    if isinstance(shape, dict):
        shape = shape.get(gg["id"], None)

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
