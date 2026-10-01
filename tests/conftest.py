"""Gemeinsame Test-Bausteine: ein kleines Modell, das einmal pro Testlauf trainiert wird."""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hotel_ranker.api import app
from hotel_ranker.features import DATA_DIR
from hotel_ranker.train import save_artifact, train

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    """Trainiert auf 600 Suchen (ca. 10 % der Daten), das dauert nur wenige Sekunden."""

    df = pd.read_csv(DATA_DIR / "impressions.csv")
    search_ids = df.search_id.drop_duplicates().sample(600, random_state=0)
    model, metadata = train(df[df.search_id.isin(search_ids)])

    out = tmp_path_factory.mktemp("artifacts")
    save_artifact(model, metadata, out)
    return out / "current"


@pytest.fixture(scope="session")
def client(artifact_dir):
    """API mit diesem Test-Modell. 'with' ist noetig, sonst laeuft der lifespan nicht."""

    patch = pytest.MonkeyPatch()
    patch.setenv("MODEL_DIR", str(artifact_dir))
    with TestClient(app) as c:
        yield c
    patch.undo()


@pytest.fixture
def example_request():
    return json.loads((EXAMPLES / "request_example.json").read_text())
