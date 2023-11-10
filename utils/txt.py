"""
Utilities for working with text files.
"""
import contextlib
import gzip
import json
import lzma
import sys
from itertools import islice, takewhile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

PathLike = Union[Path, str]

T = TypeVar("T")


def txt_lines(fname: PathLike) -> Iterator[str]:
    _open = _get_open(fname)
    with _open(fname, "rt") as src:
        yield from src


def slurp_lines(fname: PathLike) -> List[str]:
    lines = map(lambda s: s.strip(), txt_lines(fname))
    return [line for line in lines if len(line)]


def slurp(fname: PathLike) -> str:
    _open = _get_open(fname)
    with _open(fname, "rt") as src:
        return src.read()


def slurp_binary(fname: PathLike) -> bytes:
    _open = _get_open(fname)
    with _open(fname, "rb") as src:
        return src.read()


def paste(txt: Union[Iterable[str], str, bytes], fname: PathLike):
    _open = _get_open(fname)
    mode = "wb" if isinstance(txt, bytes) else "wt"
    with _open(fname, mode) as dst:
        if isinstance(txt, (str, bytes)):
            dst.write(txt)
        elif isinstance(txt, (list, tuple)):
            dst.write("\n".join(txt))
        else:
            # Assume a lazy sequence of lines without a \n
            for line in txt:
                dst.write(line)
                dst.write("\n")


def chunks(xx: Iterable[T], /, chunk: int) -> Iterator[List[T]]:
    # ensure it's an iterator
    xx = iter(xx)
    while True:
        bunch = list(islice(xx, chunk))
        if len(bunch) == 0:
            return
        yield bunch


def _get_open(fname: PathLike) -> Callable:
    ops: Dict[str, Callable] = {
        "gz": gzip.open,
        "xz": lzma.open,
    }

    @contextlib.contextmanager
    def _open_std(path, mode, *args, **kw):
        # pylint: disable=unused-argument
        if "w" in mode:
            yield sys.stdout
        yield sys.stdin

    if str(fname) == "-":
        return _open_std

    suffix = str(fname).rsplit(".", maxsplit=1)[-1].lower()
    return ops.get(suffix, open)


def with_njson_sink(
    docs: Iterable[Dict[str, Any]], fname: PathLike
) -> Iterator[Dict[str, Any]]:
    _open = _get_open(fname)
    with _open(fname, "wt") as dst:
        for doc in docs:
            txt = json.dumps(doc, separators=(",", ":"))
            dst.write(txt + "\n")
            yield doc


def to_njson(docs: Iterable[Dict[str, Any]], fname: PathLike) -> None:
    for _ in with_njson_sink(docs, fname):
        pass


def from_njson(fname: PathLike) -> Iterator[Dict[str, Any]]:
    _open = _get_open(fname)
    with _open(fname, "rt") as src:
        for doc in src:
            yield json.loads(doc)


def save_cookies(fname: PathLike, cookies: List[Tuple[str, str]]):
    """
    Dump cookies to a text file.

    Cookie: {name}={value}

    Can then use it with ``curl -H @filename ...``
    """
    with open(fname, "wt", encoding="utf-8") as dst:
        for name, value in cookies:
            dst.write(f"Cookie: {name}={value}\n")


def find_common_prefix(urls: Iterable[str], *, delimiter: str = "/") -> str:
    parts: Optional[List[Set[str]]] = None
    for u in urls:
        pp = u.split(delimiter)
        if parts is None:
            parts = [set([p]) for p in pp]
            continue

        for s, v in zip(parts, pp):
            s.add(v)

        parts = list(takewhile(lambda s: len(s) == 1, parts))

    if parts is None or len(parts) == 0:
        return ""

    return delimiter.join([p.pop() for p in parts]) + delimiter


def dump_html_table(df, output_file: PathLike):
    table_css = """
table.dataframe {
  text-align: right;
  border-collapse: collapse;
  border: none;
}
table.dataframe td, table.dataframe th {
  padding: 2px 6px;
  border: none;
}
table.dataframe tbody td {
  font-size: 13px;
}
table.dataframe tbody tr:nth-child(even) {
  background: #EBEBEB;
}
table.dataframe thead {
  background: #FFFFFF;
  border-bottom: 2px solid #333333;
}
table.dataframe thead th {
  font-size: 15px;
  font-weight: bold;
  color: #333333;
}
table.dataframe thead th:first-child {
  border-left: none;
  text-align: right;
}
table.dataframe tbody th {
   text-align: right;
   font-weight: 600;
}
"""
    paste(
        f"""<!doctype html><html><head><style>{table_css}</style></head>
 <body>{df.to_html()}</body></html>""",
        output_file,
    )
