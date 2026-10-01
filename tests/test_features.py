"""Tests fuer die Feature-Logik: sie wird von Training UND Serving benutzt."""

import numpy as np
import pandas as pd

from hotel_ranker.features import build_features

PARAMS = {"rating_fill": 7.5, "feature_columns": None}


def make_df(**overrides):
    row = {
        "hotel_id": "H1", "city": "Palma", "stars": 4, "rating": 8.0, "review_count": 100,
        "price_per_night": 120.0, "distance_to_center_km": 1.0, "has_pool": 1,
        "breakfast_included": 0, "free_cancellation": 1, "days_until_checkin": 45,
        "stay_length": 7, "device": "mobile", "user_segment": "family",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_fehlendes_rating_wird_mit_dem_wert_aus_params_gefuellt():
    X = build_features(make_df(rating=None), PARAMS)
    assert X.loc[0, "rating"] == 7.5


def test_abgeleitete_features():
    X = build_features(make_df(), PARAMS)
    assert X.loc[0, "price_per_star"] == 30.0               # 120 / 4 Sterne
    assert X.loc[0, "log_reviews"] == np.log1p(100)
    assert X.loc[0, "early_booking"] == 1                   # 45 Tage > 30


def test_early_booking_grenze():
    assert build_features(make_df(days_until_checkin=30), PARAMS).loc[0, "early_booking"] == 0
    assert build_features(make_df(days_until_checkin=31), PARAMS).loc[0, "early_booking"] == 1


def test_position_ziel_und_id_sind_keine_features():
    df = make_df().assign(position=1, booked=0)
    X = build_features(df, PARAMS)
    for spalte in ("position", "booked", "hotel_id"):
        assert spalte not in X.columns


def test_serving_liefert_genau_die_trainingsspalten():
    spalten = build_features(make_df(), PARAMS).columns.tolist()
    params = {"rating_fill": 7.5, "feature_columns": spalten}
    X = build_features(make_df(city="Wien"), params)       # Stadt aus dem Training unbekannt
    assert X.columns.tolist() == spalten
    assert X.filter(like="city_").sum().sum() == 0         # keine Stadt-Spalte ist gesetzt


def test_eingabe_wird_nicht_veraendert():
    df = make_df(rating=None)
    build_features(df, PARAMS)
    assert pd.isna(df.loc[0, "rating"])
