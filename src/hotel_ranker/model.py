# Packages
import json
import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from hotel_ranker.features import build_features

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]

# Welches Artefakt geladen wird: per Umgebungsvariable MODEL_DIR (Docker, Tests),
# sonst der 'current'-Symlink. Die Variable wird erst in load() gelesen.
DEFAULT_MODEL_DIR = ROOT_DIR / "artifacts" / "current"


class Ranker:
    """Haelt Modell + Metadaten im Speicher und sortiert Hotels fuer eine Suche."""

    def __init__(self, model, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.params = metadata["params"]
        self.version = metadata["version"]

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "Ranker":
        """Artefakt einmal laden -- beim Start des Service, nicht pro Request."""

        model_dir = Path(model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
        model = joblib.load(model_dir / "model.joblib")

        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Pickle-Kompatibilitaet: mit anderer sklearn-Version trainiert?
        if metadata["sklearn_version"] != sklearn.__version__:
            warnings.warn(
                f"Modell mit scikit-learn {metadata['sklearn_version']} trainiert, "
                f"geladen mit {sklearn.__version__}"
            )

        return cls(model, metadata)

    def rank(self, search: dict, hotels: list[dict]) -> list[dict]:
        """Scores berechnen und Hotels absteigend sortieren."""

        # eine Zeile pro Hotel, Suchkontext an jede Zeile haengen (wie im Training)
        df_hotels = pd.DataFrame(hotels).assign(**search)

        X = build_features(df_hotels, self.params)
        scores = self.model.predict_proba(X)[:, 1]

        ranking = sorted(
            ({"hotel_id": h, "score": round(float(s), 4)} for h, s in zip(df_hotels.hotel_id, scores)),
            key=lambda r: r["score"],
            reverse=True,
        )
        for i, r in enumerate(ranking, start=1):
            r["rank"] = i

        return ranking


if __name__ == "__main__":
    # kleiner Rauchtest mit dem abgestimmten Beispiel-Request
    ranker = Ranker.load()
    with open(ROOT_DIR / "examples" / "request_example.json") as f:
        request = json.load(f)

    result = {"model_version": ranker.version,
              "ranking": ranker.rank(request["search"], request["hotels"])}
    print(json.dumps(result, indent=2))
