# Packages
from pathlib import Path
import pandas as pd
import numpy as np


# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"




# LOAD DATA
def load_impressions(excel_path: Path) -> pd.DataFrame:

    """
    Function for loading in the impressions data and return them to a pandas dataframe.
    """

    # get data
    df_impressions = pd.read_csv(excel_path)
    print('fehlende Ratings:', df_impressions.rating.isna().sum())

    return df_impressions



# BUILD FEATURES
def build_features(df_impressions: pd.DataFrame, params: dict) -> pd.DataFrame:

    """
    Baut die Feature-Matrix. Wird von Training UND Serving benutzt.
    params kommt aus dem Training (metadata.json): rating_fill, feature_columns.
    """

    df_impressions = df_impressions.copy()

    # Add fehlende Ratings | Fuellwert kommt aus dem Training, nicht aus diesen Daten
    df_impressions['rating'] = df_impressions['rating'].fillna(params['rating_fill'])

    # ADD features
    df_impressions['price_per_star'] = df_impressions.price_per_night / df_impressions.stars
    df_impressions['log_reviews'] = np.log1p(df_impressions.review_count)
    df_impressions['early_booking'] = (df_impressions.days_until_checkin > 30).astype(int)

    features = ['days_until_checkin', 'stay_length', 'stars', 'rating', 'log_reviews',
                'price_per_night', 'price_per_star', 'distance_to_center_km', 'has_pool',
                'breakfast_included', 'free_cancellation', 'early_booking',
                'device', 'user_segment', 'city']

    X = pd.get_dummies(df_impressions[features], columns=['device', 'user_segment', 'city'])

    # Beim Serving: genau die Spalten wie im Training
    if params.get('feature_columns') is not None:
        X = X.reindex(columns=params['feature_columns'], fill_value=0)

    return X



if __name__ == '__main__':
    df_impressions = load_impressions(DATA_DIR / 'impressions.csv')

    rating_mean = df_impressions.rating.mean()
    print('Mittelwert:', round(rating_mean, 2))

    X = build_features(df_impressions, {'rating_fill': rating_mean, 'feature_columns': None})
    print(f'Form der Feature Matrix: {X.shape}')
    print(f"Die Spalten sehen so aus: {X.columns}")
