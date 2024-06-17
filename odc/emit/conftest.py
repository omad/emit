import json
from pathlib import Path

import pytest

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
