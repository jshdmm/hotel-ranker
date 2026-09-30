"""Erzeugt synthetische Such-Impressions für den Hotelvergleich.

Simuliert einen Export aus dem Tracking: Jede Suche zeigt 10 Hotels,
pro Suche wird höchstens ein Hotel gebucht (oder keins).

Aufruf:  python src/make_data.py        -> schreibt <repo>/data/impressions.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "impressions.csv"

RNG = np.random.default_rng(24)
N_HOTELS = 400
N_SEARCHES = 6000
HOTELS_PER_SEARCH = 10

CITIES = {  # Stadt: (Referenzpreis, Anteil)
    "Berlin": (110, 0.28),
    "München": (140, 0.22),
    "Hamburg": (120, 0.18),
    "Köln": (95, 0.14),
    "Palma": (130, 0.18),
}


def make_hotels() -> pd.DataFrame:
    city_names = list(CITIES)
    city = RNG.choice(city_names, size=N_HOTELS, p=[CITIES[c][1] for c in city_names])
    stars = RNG.choice([1, 2, 3, 4, 5], size=N_HOTELS, p=[0.05, 0.15, 0.4, 0.3, 0.1])
    ref = np.array([CITIES[c][0] for c in city])
    price = np.round(ref * (0.45 + 0.28 * stars) * RNG.lognormal(0, 0.18, N_HOTELS), 2)
    review_count = RNG.negative_binomial(2, 0.004, N_HOTELS)
    review_count[RNG.random(N_HOTELS) < 0.06] = 0  # neue Hotels ohne Bewertungen
    rating = np.clip(np.round(RNG.normal(6.4 + 0.45 * stars, 0.6), 1), 3.0, 10.0)
    rating = np.where(review_count == 0, np.nan, rating)
    return pd.DataFrame(
        {
            "hotel_id": [f"H{1000 + i}" for i in range(N_HOTELS)],
            "city": city,
            "stars": stars,
            "rating": rating,
            "review_count": review_count,
            "price_per_night": price,
            "distance_to_center_km": np.round(RNG.gamma(1.6, 1.8, N_HOTELS), 2),
            "has_pool": (RNG.random(N_HOTELS) < np.where(city == "Palma", 0.7, 0.12)).astype(int),
            "breakfast_included": (RNG.random(N_HOTELS) < 0.45).astype(int),
            "free_cancellation": (RNG.random(N_HOTELS) < 0.6).astype(int),
        }
    )


def utility(df: pd.DataFrame) -> np.ndarray:
    ref = df["city"].map({c: v[0] for c, v in CITIES.items()})
    rating = df["rating"].fillna(7.0)
    seg = df["user_segment"]
    u = (
        0.45 * (rating - 7.5)
        + 0.20 * (df["stars"] - 3)
        - 0.011 * (df["price_per_night"] - ref * 1.3)
        + 0.12 * np.log1p(df["review_count"])
        - 0.10 * df["distance_to_center_km"]
        - 0.14 * (df["position"] - 1)
    )
    u += np.where(seg == "business", -0.25 * df["distance_to_center_km"] + 0.006 * (df["price_per_night"] - ref), 0)
    u += np.where(seg == "family", 0.9 * df["has_pool"] + 0.5 * df["breakfast_included"], 0)
    u += np.where(seg == "leisure", 0.35 * df["has_pool"] + 0.15 * (rating - 7.5), 0)
    u += np.where(df["days_until_checkin"] > 30, 0.4 * df["free_cancellation"], 0)
    u += np.where(df["device"] == "mobile", -0.08 * (df["position"] - 1), 0)
    return u.to_numpy() + RNG.gumbel(0, 0.6, len(df))


def main() -> None:
    hotels = make_hotels()
    rows = []
    start = pd.Timestamp("2026-06-01")
    for s in range(N_SEARCHES):
        city = RNG.choice(list(CITIES), p=[v[1] for v in CITIES.values()])
        pool = hotels[hotels["city"] == city]
        shown = pool.sample(HOTELS_PER_SEARCH, random_state=int(RNG.integers(1e9))).copy()
        shown["search_id"] = f"S{100000 + s}"
        shown["search_date"] = (start + pd.Timedelta(days=int(RNG.integers(0, 91)))).date().isoformat()
        shown["days_until_checkin"] = int(RNG.gamma(1.4, 18)) + 1
        shown["stay_length"] = int(RNG.choice([1, 2, 3, 4, 5, 7, 10, 14], p=[0.2, 0.25, 0.2, 0.1, 0.07, 0.1, 0.05, 0.03]))
        shown["device"] = RNG.choice(["mobile", "desktop", "tablet"], p=[0.58, 0.36, 0.06])
        shown["user_segment"] = RNG.choice(["leisure", "business", "family"], p=[0.5, 0.25, 0.25])
        shown["position"] = RNG.permutation(HOTELS_PER_SEARCH) + 1
        rows.append(shown)
    df = pd.concat(rows, ignore_index=True)

    u = utility(df)
    df["booked"] = 0
    no_booking_utility = 2.2
    for _, idx in df.groupby("search_id").groups.items():
        uu = np.append(u[idx], no_booking_utility)
        p = np.exp(uu - uu.max())
        choice = RNG.choice(len(uu), p=p / p.sum())
        if choice < len(idx):
            df.loc[idx[choice], "booked"] = 1

    cols = [
        "search_id", "search_date", "days_until_checkin", "stay_length", "device", "user_segment",
        "hotel_id", "city", "stars", "rating", "review_count", "price_per_night",
        "distance_to_center_km", "has_pool", "breakfast_included", "free_cancellation",
        "position", "booked",
    ]
    df[cols].sort_values(["search_date", "search_id", "position"]).to_csv(OUT_PATH, index=False)
    print(f"{len(df)} Impressions, {df['search_id'].nunique()} Suchen, Buchungsrate {df['booked'].mean():.3%}")


if __name__ == "__main__":
    main()
