"""Der Webservice: GET /health und POST /rank."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hotel_ranker.model import Ranker
from hotel_ranker.schemas import RankRequest, RankResponse

logger = logging.getLogger("uvicorn.error")   # erscheint im Log von Gunicorn/Docker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Beim Start: Modell einmal laden. Danach liegt es im Speicher."""

    app.state.ranker = Ranker.load()
    logger.info("Modell geladen: %s", app.state.ranker.version)
    yield


app = FastAPI(title="Hotel-Ranker", lifespan=lifespan)


@app.get("/health")
def health():
    """Lebt der Service, und welche Modellversion laeuft?"""

    return {"status": "ok", "model_version": app.state.ranker.version}


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    """Hotels einer Suche nach Buchungswahrscheinlichkeit sortieren."""

    ranker = app.state.ranker
    ranking = ranker.rank(req.search.model_dump(), [h.model_dump() for h in req.hotels])
    return {"model_version": ranker.version, "ranking": ranking}
