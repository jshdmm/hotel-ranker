"""Tests fuer das Training und das gespeicherte Artefakt."""

import json

import joblib
import pandas as pd

from hotel_ranker.train import ndcg_at_5


def test_ndcg_perfekte_reihenfolge_ist_1():
    y = pd.Series([1, 0, 0, 0])
    gruppe = pd.Series([1, 1, 1, 1])
    assert ndcg_at_5(y, [0.9, 0.5, 0.3, 0.1], gruppe) == 1.0


def test_ndcg_schlechte_reihenfolge_ist_kleiner():
    y = pd.Series([1, 0, 0, 0])
    gruppe = pd.Series([1, 1, 1, 1])
    assert ndcg_at_5(y, [0.1, 0.5, 0.3, 0.9], gruppe) < 1.0


def test_artefakt_enthaelt_modell_und_metadaten(artifact_dir):
    meta = json.loads((artifact_dir / "metadata.json").read_text())
    for key in ("version", "git_commit", "sklearn_version", "params", "metrics", "split_date"):
        assert key in meta
    modell = joblib.load(artifact_dir / "model.joblib")
    assert modell.n_features_in_ == len(meta["params"]["feature_columns"])
