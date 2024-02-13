import json
from pathlib import Path

import zict
from zstandard import ZstdCompressor, ZstdDecompressor


class _Zstd:
    def __init__(self, level=6):
        self._level = level
        self._comp = None
        self._decomp = None

    def __reduce__(self):
        return _Zstd, (self._level,)

    def compress(self, x):
        if self._comp is None:
            self._comp = ZstdCompressor(self._level)
        return self._comp.compress(x)

    def decompress(self, z):
        if self._decomp is None:
            self._decomp = ZstdDecompressor()
        return self._decomp.decompress(z)


def open_zict(fname, mode="a", level=6):
    _zstd = _Zstd(level)
    store = zict.Zip(fname, mode=mode)
    return zict.Func(_zstd.compress, _zstd.decompress, store)


def _to_json(doc):
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def open_zict_json(src, mode="a", level=6):
    if isinstance(src, (str, Path)):
        zstd = open_zict(src, mode=mode, level=level)
    else:
        zstd = src
    return zict.Func(
        _to_json,
        json.loads,
        zstd,
    )


def _stac_store():
    # pylint: disable=protected-access
    fname = "/tmp/emit.zip"
    if (_cache := getattr(_stac_store, "_cache", None)) is not None:
        return _cache

    _stac_store._cache = open_zict_json(fname, "r")
    return _stac_store._cache
