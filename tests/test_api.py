"""Tests fuer die API: Antwortformat, Sortierung und Validierung."""

import copy
import json

import pytest


def test_health_zeigt_status_und_modellversion(client, artifact_dir):
    r = client.get("/health")
    assert r.status_code == 200
    version = json.loads((artifact_dir / "metadata.json").read_text())["version"]
    assert r.json() == {"status": "ok", "model_version": version}


def test_rank_antwortformat_und_sortierung(client, example_request):
    r = client.post("/rank", json=example_request)
    assert r.status_code == 200
    body = r.json()
    ranking = body["ranking"]

    assert body["model_version"]
    assert [x["rank"] for x in ranking] == [1, 2, 3]
    scores = [x["score"] for x in ranking]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)
    assert {x["hotel_id"] for x in ranking} == {h["hotel_id"] for h in example_request["hotels"]}


def test_rank_akzeptiert_hotel_ohne_rating(client, example_request):
    assert example_request["hotels"][1]["rating"] is None      # H1377 ist ein neues Hotel
    assert client.post("/rank", json=example_request).status_code == 200


def aendern(req, wert_setzen):
    req = copy.deepcopy(req)
    wert_setzen(req)
    return req


UNGUELTIG = {
    "stars = 0": lambda r: r["hotels"][0].update(stars=0),
    "stars = 6": lambda r: r["hotels"][0].update(stars=6),
    "Preis = 0": lambda r: r["hotels"][0].update(price_per_night=0),
    "device unbekannt": lambda r: r["search"].update(device="tv"),
    "keine Hotels": lambda r: r.update(hotels=[]),
    "Suche fehlt": lambda r: r.pop("search"),
}


@pytest.mark.parametrize("fall", UNGUELTIG)
def test_ungueltige_anfragen_ergeben_422(client, example_request, fall):
    req = aendern(example_request, UNGUELTIG[fall])
    assert client.post("/rank", json=req).status_code == 422
