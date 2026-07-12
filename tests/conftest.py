import json
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_documents.json"


@pytest.fixture(scope="session")
def synthetic_documents():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))
