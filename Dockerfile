# Gleiche Minor-Version wie beim Training (das Modell ist ein Pickle)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models/current

WORKDIR /app

# Nur der Code kommt ins Image. Das Modell kommt zur Laufzeit als Volume nach /models.
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Nicht als root laufen. Die UID muss zur UID des Service-Users auf dem Server passen.
RUN useradd --system --uid 10001 --no-create-home app
USER app

EXPOSE 8000

# slim hat kein curl, deshalb pruefen wir mit Python
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", "hotel_ranker.api:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", "-b", "0.0.0.0:8000", \
     "--timeout", "30", "--access-logfile", "-"]
