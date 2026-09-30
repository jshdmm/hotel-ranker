# Packages
import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import ndcg_score, roc_auc_score
from hotel_ranker.features import build_features, load_impressions


# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

# Defaults
SPLIT_DATE = "2026-08-01"
LABEL = "booked"


def get_git_commit() -> str:
    """Kurzer Git-Hash, damit das Artefakt weiss, aus welchem Code es entstanden ist."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def ndcg_at_5(y_true, y_score, groups) -> float:
    """NDCG@5 pro Suche, gemittelt. Nur Suchen mit Buchung und mehr als einem Hotel."""

    ev = pd.DataFrame({"y": y_true.values, "p": y_score, "g": groups.values})

    scores = [
        ndcg_score([g.y.values], [g.p.values], k=5)
        for _, g in ev.groupby("g")
        if g.y.sum() > 0 and len(g) > 1
    ]

    return float(np.mean(scores))


def train(df_impressions: pd.DataFrame, split_date: str = SPLIT_DATE) -> tuple:

    """Function to train the GBM. Gibt Modell und Metadaten zurueck."""

    df_impressions["search_date"] = pd.to_datetime(df_impressions.search_date)

    # split into train / test based on search timestamp
    train_df = df_impressions[df_impressions.search_date <  split_date]
    test_df  = df_impressions[df_impressions.search_date >= split_date]

    # params werden NUR auf dem Train-Split gelernt, sonst leaken Testdaten ins Training
    params = {"rating_fill": float(train_df.rating.mean()), "feature_columns": None}

    # build training features and get training target
    X_train = build_features(train_df, params)
    y_train = train_df.loc[X_train.index, LABEL]

    # jetzt sind die Spalten bekannt -> params vervollstaendigen
    params["feature_columns"] = X_train.columns.tolist()

    # test features and test target
    X_test = build_features(test_df, params)
    y_test = test_df.loc[X_test.index, LABEL]

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation auf dem Test-Split
    p_test = model.predict_proba(X_test)[:, 1]
    groups = test_df.loc[X_test.index, "search_id"]

    metrics = {
        "auc": float(roc_auc_score(y_test, p_test)),
        "ndcg_at_5": ndcg_at_5(y_test, p_test, groups),
        # Baseline: nach Bewertung sortieren. Schlaegt das Modell das nicht, lohnt es sich nicht.
        "ndcg_at_5_baseline_rating": ndcg_at_5(y_test, X_test.rating.values, groups),
    }

    # Score-Verteilung als Zahlen statt als Plot -> maschinell vergleichbar
    score_quantiles = {
        "min": float(p_test.min()),
        "p50": float(np.quantile(p_test, 0.50)),
        "p90": float(np.quantile(p_test, 0.90)),
        "p99": float(np.quantile(p_test, 0.99)),
        "max": float(p_test.max()),
    }

    now = datetime.now(timezone.utc)

    # Modell Metadaten
    metadata_dict = { "version": f"{now.strftime('%Y-%m-%dT%H-%M-%S')}_{get_git_commit()}",
                    "trained_at": now.isoformat(),
                    "git_commit": get_git_commit(),
                    "sklearn_version": sklearn.__version__,
                    "python_version": platform.python_version(),
                    "params": params,
                    "metrics": metrics,
                    "score_quantiles": score_quantiles,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "split_date": split_date,
                    "model_params": model.get_params()
    }

    return model, metadata_dict


def save_artifact(model, metadata_dict: dict, out_dir: Path) -> Path:
    """Modell und Metadaten versioniert ablegen, 'current' auf die neue Version zeigen lassen."""

    version_dir = out_dir / metadata_dict["version"]
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, version_dir / "model.joblib")

    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=2)

    # 'current' zeigt auf die aktive Version -> Rollback ist ein Symlink-Wechsel
    current = out_dir / "current"
    if current.is_symlink() or current.exists():
        current.unlink()
    current.symlink_to(version_dir.name)

    return version_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trainiert das Beliebtheits-Modell.")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "impressions.csv")
    parser.add_argument("--out",  type=Path, default=ROOT_DIR / "artifacts")
    parser.add_argument("--split-date", default=SPLIT_DATE)
    args = parser.parse_args()

    df_impressions = load_impressions(args.data)
    model, metadata_dict = train(df_impressions, args.split_date)
    version_dir = save_artifact(model, metadata_dict, args.out)

    m = metadata_dict["metrics"]
    print(f"Version:  {metadata_dict['version']}")
    print(f"AUC:      {m['auc']:.4f}")
    print(f"NDCG@5:   {m['ndcg_at_5']:.4f}  (Baseline rating: {m['ndcg_at_5_baseline_rating']:.4f})")
    print(f"Artefakt: {version_dir}")
