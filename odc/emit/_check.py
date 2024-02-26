import importlib
import sys
from pathlib import Path

PKGS = [
    "odc.emit",
    "odc.geo",
    "odc.stac",
    "datacube",
    "sqlalchemy",
    "s3fs",
    "pystac",
    "dask",
    "distributed",
]


def dump_py_env(*extras):
    print("-" * 100)
    print("LIBS:")
    for pkg in PKGS + list(extras):
        try:
            m = importlib.import_module(pkg)
            pth = str(Path(m.__file__).parent).replace(str(Path.home()), "~")
            v = getattr(m, "__version__", "??")
            pkg = f"{pkg}=={v}"
        except ModuleNotFoundError:
            pkg = f"{pkg} NOT FOUND"
            pth = "MISSING"

        print(f"  {pkg:35}  # {pth}")

    print()
    print("-" * 100)
    print("PYTHONPATH:")
    print("  " + "\n  ".join(sys.path))
