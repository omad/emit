import sys

from ._check import dump_py_env

if __name__ == "__main__":
    if sys.argv[1:] == ["info"]:
        dump_py_env()
