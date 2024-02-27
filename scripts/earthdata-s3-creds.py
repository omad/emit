#!/env/bin/python
import os

import requests
from pathlib import Path
import sys
import json

dst_file = "/tmp/earthdata-s3.env"
dst_file_json = dst_file.replace(".env", ".json")
tk_file = Path.home() / ".safe" / "earth-data.tk"

if (tk := os.environ.get("EARTHDATA_TOKEN", None)) is None:
    if tk_file.exists():
        print(f"Reading from: {tk_file}", file=sys.stderr)
        tk = open(tk_file, "rt", encoding="utf8").read().strip()

if tk is None:
    print(f"please set EARTHDATA_TOKEN= or create: {tk_file}", file=sys.stderr)
    sys.exit(1)

print("Calling /s3credentials endpoint", file=sys.stderr)
CREDENTIALS = requests.get(
    "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials",
    headers={"Authorization": f"Bearer {tk}"},
).json()


json.dump(CREDENTIALS, open(dst_file_json, "wt", encoding="utf8"))

with open(dst_file, "wt", encoding="utf8") as f:
    print(
        """export AWS_ACCESS_KEY_ID={accessKeyId}
export AWS_SECRET_ACCESS_KEY={secretAccessKey}
export AWS_SESSION_TOKEN={sessionToken}
export AWS_DEFAULT_REGION=us-west-2
""".format(
            **CREDENTIALS
        ),
        file=f,
    )

print(f"Saved to: {dst_file} {dst_file_json}")
