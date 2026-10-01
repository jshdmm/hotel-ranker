# Hotel-Ranker

Ein kleiner Webservice, der die Hotels einer Suche nach **Buchungswahrscheinlichkeit** sortiert
(„Beliebtheit"-Sortierung). Ein trainiertes Modell (scikit-learn) steckt hinter einer REST-API.
Der Service läuft im Docker-Container auf einer Ubuntu-VM und wird mit Ansible bereitgestellt.

## Schnellstart

```bash
make venv        # Python-Umgebung anlegen und installieren
make train       # Modell trainieren, Ergebnis liegt in artifacts/
make check       # Lint und Tests
make deploy      # Image bauen und auf die VM bringen (IP in deploy/ansible/inventory.ini)
```

Die Schritte unten erklären, was dabei jeweils passiert.

## 1. Überblick: wie alles zusammenhängt

```
data/impressions.csv
        │  python -m hotel_ranker.train
        ▼
artifacts/<Version>/model.joblib + metadata.json      artifacts/current  →  aktive Version
        │  Ansible kopiert die Version auf den Server
        ▼
Container (Docker, läuft nicht als root)
   gunicorn → uvicorn → api.py → model.py → features.py
        ▲
        │  POST /rank    GET /health
     Client
```

| Datei | Aufgabe |
|---|---|
| `src/hotel_ranker/features.py` | Die **eine** Feature-Funktion. Training und Service benutzen sie beide. |
| `src/hotel_ranker/train.py` | Trainiert, bewertet, schreibt Modell und Metadaten nach `artifacts/<Version>/`. |
| `src/hotel_ranker/model.py` | Lädt ein Artefakt und sortiert Hotels (`Ranker`). |
| `src/hotel_ranker/schemas.py` | Prüft Anfrage und Antwort der API (Pydantic). |
| `src/hotel_ranker/api.py` | Der Webservice mit `/health` und `/rank`. |
| `tests/` | pytest für Features, Training und API. |
| `Dockerfile` | Baut das Image. Das Modell steckt **nicht** drin. |
| `deploy/ansible/` | Playbook, das einen frischen Server einrichtet. |
| `Makefile` | Kurzbefehle für alles Obige. |
| `pyproject.toml` | Abhängigkeiten mit exakten Versionen. |

So sieht das Repo aus:

```
hotel-ranker/
├── src/hotel_ranker/        der Code (features, train, model, schemas, api)
├── tests/                   pytest
├── data/impressions.csv     Trainingsdaten (Export von Lena)
├── examples/                abgestimmtes API-Format (Anfrage und Antwort)
├── deploy/ansible/          Playbook, Inventar, benötigte Module
├── artifacts/               trainierte Modelle (nicht im Git, entsteht durch make train)
├── beliebtheit_v2.ipynb     Lenas Original-Notebook
├── Dockerfile, .dockerignore
├── Makefile                 Kurzbefehle
├── pyproject.toml           Abhängigkeiten und Werkzeug-Einstellungen
└── TASK.md                  die Aufgabenstellung
```

## 2. Setup (lokal)

Voraussetzungen: Python 3.10 (wie beim Training), Docker, eine Ubuntu-VM mit SSH-Zugang (hier Multipass).

```bash
python3 -m venv .venv            # eigene Python-Umgebung für dieses Projekt
source .venv/bin/activate        # aktivieren, der Prompt zeigt (.venv)
pip install -e ".[dev,deploy]"   # Projekt + Test-Werkzeuge + Ansible installieren
```

Kurzform: `make venv`. Mit `make help` siehst du alle Kurzbefehle.

Die `requirements.txt` ist Lenas Übergabestand. Maßgeblich ist die `pyproject.toml`.

## 3. Training

```bash
python -m hotel_ranker.train        # oder: make train
```

Erwartete Ausgabe:

```
Version:  2026-10-01T19-08-44_cc237ab
AUC:      0.6438
NDCG@5:   0.4367  (Baseline rating: 0.3759)
Artefakt: .../artifacts/2026-10-01T19-08-44_cc237ab
```

Was passiert:

1. Die Daten werden **nach Datum** geteilt: Juni und Juli zum Trainieren (40 160 Zeilen), August zum Testen (19 840 Zeilen).
2. Der Füllwert für fehlende Ratings wird **nur aus den Trainingsdaten** berechnet.
3. `features.py` baut die Feature-Matrix, ein Gradient-Boosting-Modell wird trainiert.
4. Auf dem Test-Monat werden gemessen: **AUC** (Trennschärfe) und **NDCG@5** (wie weit oben stehen die gebuchten Hotels). Als Vergleich dient eine **Baseline**, die einfach nach Bewertung sortiert.
5. Es entsteht ein Ordner `artifacts/<Version>/` mit `model.joblib` und `metadata.json`. Der Link `artifacts/current` zeigt danach auf diese Version.

Jeder Lauf erzeugt eine neue Version, alte Versionen bleiben liegen.

`artifacts/` liegt **nicht im Git** (steht in der `.gitignore`). Wer das Repo klont, erzeugt das Modell selbst mit `make train`. Die Trainingsdaten `data/impressions.csv` liegen dagegen im Repo.

Die `metadata.json` enthält: Version, Zeitpunkt, Git-Commit, scikit-learn- und Python-Version, Metriken,
Score-Verteilung, Datum des Splits, Modell-Parameter und `params` (Füllwert und Spaltenliste).
Der Service braucht `params`, damit er genauso rechnet wie das Training.

So sieht eine `metadata.json` aus (gekürzt):

```json
{
  "version": "2026-10-01T19-08-44_cc237ab",
  "trained_at": "2026-10-01T19:08:44.052736+00:00",
  "git_commit": "cc237ab",
  "sklearn_version": "1.7.2",
  "python_version": "3.10.11",
  "params": {
    "rating_fill": 7.905263853637902,
    "feature_columns": ["... 23 Spalten, in genau der Reihenfolge des Trainings ..."]
  },
  "metrics": {
    "auc": 0.6437586105702946,
    "ndcg_at_5": 0.43669322507270597,
    "ndcg_at_5_baseline_rating": 0.3758806649700512
  },
  "score_quantiles": {"min": 0.019, "p50": 0.061, "p90": 0.112, "p99": 0.188, "max": 0.432},
  "train_rows": 40160,
  "test_rows": 19840,
  "split_date": "2026-08-01",
  "model_params": {"... alle Einstellungen des Modells ..."}
}
```

Die Version besteht aus Zeitpunkt und Git-Commit. Das ist das Format aus `examples/response_example.json`.

## 4. Service lokal starten

```bash
uvicorn hotel_ranker.api:app --reload      # Entwicklung, oder: make run
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/rank -H 'Content-Type: application/json' \
     -d @examples/request_example.json | python -m json.tool
```

| Endpunkt | Zweck |
|---|---|
| `GET /health` | Lebt der Service, und **welche Modellversion läuft?** Antwort: `{"status":"ok","model_version":"..."}` |
| `POST /rank` | Sortiert die Hotels einer Suche. Format wie in `examples/`. |
| `GET /docs` | Automatische Swagger-Oberfläche zum Ausprobieren. |

**Beispiel.** Die Anfrage besteht aus dem Suchkontext (`search`) und den Hotels (`hotels`):

```json
{
  "search": {"days_until_checkin": 45, "stay_length": 7, "device": "mobile", "user_segment": "family"},
  "hotels": [
    {"hotel_id": "H1234", "city": "Palma", "stars": 4, "rating": 8.7, "review_count": 812,
     "price_per_night": 164.0, "distance_to_center_km": 3.1, "has_pool": 1,
     "breakfast_included": 1, "free_cancellation": 1},
    {"hotel_id": "H1377", "city": "Palma", "stars": 3, "rating": null, "...": "..."}
  ]
}
```

Die Antwort nennt die Modellversion und die Hotels in der neuen Reihenfolge:

```json
{
  "model_version": "2026-10-01T19-08-44_cc237ab",
  "ranking": [
    {"hotel_id": "H1234", "score": 0.2116, "rank": 1},
    {"hotel_id": "H1012", "score": 0.0748, "rank": 2},
    {"hotel_id": "H1377", "score": 0.0471, "rank": 3}
  ]
}
```

Der `score` ist die geschätzte Wahrscheinlichkeit, dass das Hotel gebucht wird (0 bis 1). In den Daten wird etwa jedes 14. Hotel gebucht (6,9 %),
deshalb sind Werte zwischen 2 und 20 % normal. `rank 1` steht ganz oben. Ein Hotel ohne Rating (`null`) ist ein neues Hotel,
es bekommt den Füllwert aus dem Training.

Bei einer ungültigen Anfrage (hier `stars: 0`) sagt die Antwort genau, welches Feld das Problem ist:

```json
{"detail": [{"type": "greater_than_equal", "loc": ["body", "hotels", 0, "stars"],
             "msg": "Input should be greater than or equal to 1", "input": 0}]}
```

Falsche Eingaben (`stars` gleich 0, unbekanntes Gerät, keine Hotels) beantwortet der Service mit **422**.
Das Modell selbst würde sie ohne Fehler durchrechnen, deshalb prüft `schemas.py` sie vorher.

Der Container startet den Service mit Gunicorn und Uvicorn-Workern (so verlangt es die Aufgabe):

```bash
gunicorn hotel_ranker.api:app -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000
```

## 5. Tests und Lint

```bash
make check          # = ruff check . und pytest
```

Erwartet: `All checks passed!` und `18 passed`.

| Test | Prüft |
|---|---|
| `test_features.py` | Füllwert kommt aus `params`, abgeleitete Features stimmen, `position`/Ziel/ID sind keine Features, Serving liefert genau die Trainingsspalten, Eingabe bleibt unverändert |
| `test_train.py` | NDCG-Berechnung, Artefakt enthält Modell und Metadaten |
| `test_api.py` | `/health`, Antwortformat und Sortierung, Hotel ohne Rating, ungültige Eingaben ergeben 422 |

Die API-Tests trainieren ein kleines Modell auf 600 Suchen. Das dauert nur wenige Sekunden.
Das `make check` ersetzt hier die CI: Es läuft lokal vor jedem Commit.

## 6. Docker

```bash
docker build -t hotel-ranker:0.1.0 .        # oder: make docker
docker run -d --name hr -p 8000:8000 -v "$PWD/artifacts:/models:ro" hotel-ranker:0.1.0
docker ps                                   # Status wird nach ca. 10 Sekunden "healthy"
docker exec hr id                           # uid=10001(app), also nicht root
```

- **Schlank:** Basis `python:3.10-slim`, nur der Code kommt hinein (Image ca. 675 MB, davon der größte Teil scikit-learn, pandas, numpy).
- **Nicht root:** `USER app` mit UID 10001.
- **Healthcheck:** Docker fragt alle 30 Sekunden `/health`.
- **Modell als Volume:** Es liegt neben dem Image und wird beim Start gelesen. So braucht ein neues Modell **keinen neuen Image-Bau**.
- **Gleiche Python-Version wie beim Training:** Das Modell ist ein Pickle und hängt an den Bibliotheksversionen.

Gemessen im Container (lokal, mit HTTP): 200 Hotels in **etwa 8 ms** im Median, höchstens 23 ms. Vorgabe war unter 50 ms.

## 7. Deployment auf die VM (Ansible)

Voraussetzung: Die VM läuft, du erreichst sie per SSH, und ihre IP steht in `deploy/ansible/inventory.ini`
(sie steht in der Ausgabe von `multipass list`). Außerdem muss einmal trainiert worden sein (`make train`), sonst gibt es kein Modell zum Kopieren.

```bash
make deploy
```

`make deploy` macht drei Dinge: es baut das Image und speichert es als Archiv
(`deploy/ansible/files/hotel-ranker-0.1.0.tar.gz`, nur wenn es fehlt), installiert die Ansible-Module
und führt das Playbook aus. Ohne Make:

```bash
cd deploy/ansible
../../.venv/bin/ansible-galaxy collection install -r requirements.yml -p collections
../../.venv/bin/ansible-playbook site.yml
```

Was `site.yml` auf dem Server erledigt, in dieser Reihenfolge:

1. **Docker installieren** (`apt`) und dafür sorgen, dass es läuft und **beim Booten startet**.
2. **Service-User** `hranker` mit UID 10001 anlegen, ohne Login. Die UID ist dieselbe wie im Container.
3. **Ordner** `/opt/hotel-ranker/artifacts` anlegen und die **aktive Modellversion** hineinkopieren.
4. **`current`-Link** auf diese Version setzen. Ändert er sich, wird der Container neu gestartet.
5. **Image-Archiv** kopieren und in Docker **laden** (ohne Registry).
6. **Firewall:** SSH (22) und den Service (8000) erlauben, danach aktivieren. Alles andere Eingehende ist gesperrt. SSH kommt zuerst, sonst sperrt man sich aus.
7. **Container starten** mit Neustart-Regel `unless-stopped`, Port 8000 und dem Modell-Ordner als Volume.

**Zweiter Durchlauf ändert nichts.** Ergebnis der Läufe:

```
Erster Lauf:    ok=13  changed=11  failed=0
Zweiter Lauf:   ok=12  changed=0   failed=0
```

Alle Schritte nutzen Ansible-Module (`apt`, `user`, `file`, `copy`, `docker_container`, `ufw`), keine `shell`-Befehle.
Deshalb prüft Ansible zuerst, ob der Zustand schon stimmt. (Beim allerersten Lauf startet der Container einmal doppelt, weil der Link neu gesetzt wird. Das ist harmlos.)

So sieht es danach auf dem Server aus:

```
/opt/hotel-ranker/artifacts/
├── 2026-10-01T19-08-44_cc237ab/        Modellversion mit model.joblib und metadata.json
├── <weitere Versionen, falls zurückgerollt wurde>/
└── current -> 2026-10-01T19-08-44_cc237ab      die aktive Version
/tmp/hotel-ranker-0.1.0.tar.gz          Image-Archiv (wird beim Neustart des Servers geleert)

Container:   hotel-ranker  (Image hotel-ranker:0.1.0, Port 8000)
Volume:      /opt/hotel-ranker/artifacts  →  /models  (nur lesbar)
Service-User: hranker (UID 10001)
```

Die wichtigsten Einstellungen stehen oben in `deploy/ansible/site.yml` unter `vars`:

| Variable | Bedeutung |
|---|---|
| `app_dir` | Ordner auf dem Server (`/opt/hotel-ranker`) |
| `service_user`, `service_uid` | Service-User und UID, die UID muss zu `USER app` im Dockerfile passen |
| `image_name`, `image_tag` | Name und Version des Images, der Tag kommt aus `VERSION` im Makefile |
| `container_name` | Name des Containers |
| `model_version` | Welche Modellversion aktiv ist. Standard: das Ziel von `artifacts/current`. Überschreibbar mit `-e model_version=...` |

Prüfen, von deinem Rechner aus:

```bash
curl -s http://192.168.252.3:8000/health
curl -s -X POST http://192.168.252.3:8000/rank -H 'Content-Type: application/json' -d @examples/request_example.json
```

Zusätzlich geprüft: Nach `sudo reboot` der VM ist der Service **nach wenigen Sekunden von selbst wieder da**
(Docker startet beim Booten, der Container hat `unless-stopped`, die Firewall bleibt aktiv).

## 8. Betrieb

| Frage | Antwort |
|---|---|
| Läuft der Service? Welches Modell? | `curl http://<IP>:8000/health` zeigt Status **und** `model_version`. |
| Docker-Status | `ssh ubuntu@<IP> 'sudo docker ps'` zeigt auch `healthy` oder `unhealthy`. |
| Logs | `ssh ubuntu@<IP> 'sudo docker logs --tail 50 hotel-ranker'` (enthält „Modell geladen: ..." und jede Anfrage) |
| Neues Modell einspielen | `make train`, danach `make deploy`. Kopiert nur die neue Version, biegt `current` um, startet neu. |
| **Rollback** | `cd deploy/ansible && ../../.venv/bin/ansible-playbook site.yml -e model_version=<ältere Version>`. Die Namen stehen in `artifacts/`. |
| Neuer Code | `VERSION` im `Makefile` erhöhen und `make deploy`. Ohne neue Version würde Docker das alte Image behalten. |

Modellwechsel und Rollback sind ausprobiert: Version A → B → A, jeweils in wenigen Sekunden, `/health` zeigt die jeweilige Version.

**Retraining (nur skizziert, nicht gebaut):** Ein Timer (cron oder systemd) auf einem Rechner mit den neuen Daten führt
`train` aus, prüft die neue Version mit dem Beispiel-Request und deployt dann wie oben.
Ist das neue Modell schlechter, ist der Rückweg der Rollback-Befehl. Das alte Modell bleibt ja auf dem Server liegen.

## 9. Entscheidungen und Begründung

| Entscheidung | Warum |
|---|---|
| `src/`-Layout, Paket mit `pyproject.toml` | Importieren geht nur nach der Installation. So testet man dasselbe wie im Container. |
| Exakte Versionen (`==`) | Das Modell ist ein Pickle und hängt an den Bibliotheksversionen. |
| **Eine** Feature-Funktion für Training und Service | Sonst rechnen beide irgendwann verschieden. |
| `params` in der `metadata.json` | Der Service rechnet mit demselben Füllwert und denselben Spalten wie das Training. |
| Zeitsplit statt Zufallssplit | Die Hotels einer Suche landen nicht in beiden Mengen. So wird auch die Praxis nachgestellt: trainiert auf der Vergangenheit, angewendet auf Neues. |
| Versionsordner und `current`-Link | Rollback heißt: Link umbiegen und neu starten. |
| Modell als Volume, nicht im Image | Ein Image für alle Modellversionen. Modellwechsel ohne Neubau. |
| Modell einmal beim Start laden | Beim Laden von der Platte wäre das 50-ms-Budget sofort weg. |
| `def` statt `async def` in `/rank` | Das Modell rechnet auf der CPU. Bei `async def` würde es die Event-Loop blockieren. |
| Gunicorn mit Uvicorn-Workern | Vorgabe der Aufgabe. Gunicorn verwaltet die Prozesse, Uvicorn bedient die Anfragen. |
| Container ohne root, UID wie auf dem Server | Weniger Schaden bei einem Ausbruch, und Dateirechte auf dem Volume passen. |
| Image als Archiv statt Registry | Lokal gibt es keine Registry. |
| Ansible-Module statt `shell` | Das macht den zweiten Lauf ruhig (idempotent). |
| `make check` statt CI | Wir arbeiten lokal. Das ist dieselbe Prüfung ohne Server. |

## 10. Was ich Lena zurückmelden würde (Probleme im Notebook)

1. **`position` als Feature.** Die Buchungsrate fällt von 12,8 % auf Platz 1 auf 2,9 % auf Platz 10. Beim Scoring wurde `position = 1` gesetzt, also bekam jedes Hotel den Bonus von Platz 1. Das Training kennt nur die Plätze 1 bis 10, produktiv kommen 50 bis 200 Hotels. Entfernt. Deshalb sind AUC und Scores niedriger und ehrlicher.
2. **Zufallssplit.** Die 10 Zeilen einer Suche landeten in beiden Mengen, und NDCG lief über etwa 2 Zeilen je Suche. Jetzt Zeitsplit.
3. **Rating-Füllwert** wurde über alle Daten berechnet (7,9101, also inklusive Testdaten) und beim Scoring als `7.91` fest eingetragen. Jetzt nur aus den Trainingsdaten (7,9053) und in der `metadata.json`.
4. **Spaltenliste** hing an Notebook-Variablen. Jetzt `feature_columns` im Artefakt.
5. **`stars = 0`** ergibt in `price_per_star` unendlich und das Modell rechnet trotzdem. Jetzt Validierung.
6. **Pickle ohne Versionen**, Lena nutzt Python 3.12, das Projekt 3.10. Jetzt gepinnt, Versionen im Artefakt, Warnung beim Laden.
7. **Kein Seed** am Modell. Jetzt `random_state=42`, damit ein Lauf wiederholbar ist.

## 11. Bekannte Grenzen und nächste Schritte

Grenzen:

- **Positionseffekt** nur entfernt, nicht modelliert.
- **10 gegen 200 Hotels:** Jede Trainingssuche hat 10 Hotels, produktiv sind es 50 bis 200. „Top 5 von 10" ist leichter, die NDCG-Zahl ist also optimistisch. Echte Qualität zeigt erst ein A/B-Test.
- **Unbekannte Städte** rechnet das Modell still als „keine Stadt". Es kennt fünf.
- **Keine Authentifizierung, kein TLS, kein Reverse Proxy.** Port 8000 ist direkt offen.
- **Docker umgeht ufw:** Veröffentlichte Ports sind auch erreichbar, wenn ufw sie sperrt. Die Regel für 8000 hält nur die Absicht fest. Mit einem Proxy würde man `127.0.0.1:8000` verwenden.
- **Eine VM**, keine Ausfallsicherheit.
- **Kein Monitoring** außer `/health` und Logs. Die Score-Verteilung wird nur beim Training gemessen.
- **Modell wird nur beim Start geladen.** Ein Wechsel braucht den Neustart.
- **Retraining** ist nicht automatisiert.
- **`git_commit` im Artefakt** sagt nicht, ob der Code beim Training uncommittete Änderungen hatte.

Als Nächstes würde ich: `/metrics` für Prometheus (Latenz, Score-Verteilung gegen die Werte in der `metadata.json`), nginx auf einer zweiten VM,
einen Timer für das Retraining und eine CI-Datei mit Lint, Tests und Image-Build.

## 12. Konzeptfragen (Kurzantworten)

**Wie merke ich, dass das Modell in Produktion schlechter wird?**
Buchungen kommen verzögert, deshalb zuerst Stellvertreter: Score-Verteilung gegen die `score_quantiles` in der `metadata.json`, Verteilung der Eingaben (z. B. PSI), Anteil fehlender Ratings, Fehlerrate und Latenz. Später NDCG und Buchungsrate je Position aus den echten Logs, immer im Vergleich zur Baseline.

**Wie kommt ein neues Modell sicher in den Service und wieder zurück?**
Neue Version daneben ablegen, mit dem Beispiel-Request prüfen, `current` umbiegen, neu starten, `model_version` in `/health` kontrollieren. Zurück: Link zurück, neu starten. Das ist als `-e model_version=...` im Playbook gebaut.

**Wie würde ich einen LLM-Service (z. B. „bestes Hotelbild zuerst") anders überwachen?**
Es gibt kein einzelnes Label. Qualität misst man indirekt (Klicks und Buchungen, ein Prüfset mit menschlichen Urteilen, ein LLM als Prüfer). Dazu Kosten, Tokens und Latenz je Anfrage, Prompt- und Modellversion mitloggen und Stichproben lesen. Die Ausgabe ist nicht deterministisch, der Anbieter kann das Modell ändern, und Prompt-Injection ist ein Sicherheitsthema.

## 13. Wenn etwas nicht klappt

| Meldung oder Beobachtung | Ursache und Lösung |
|---|---|
| `ModuleNotFoundError: No module named 'hotel_ranker'` | Die Umgebung ist nicht aktiv, `python3` ist dann das System-Python. `source .venv/bin/activate` oder `.venv/bin/python -m ...` benutzen. |
| `make deploy` findet kein Modell | Das Playbook liest `artifacts/current`. Gibt es das noch nicht, erst `make train`. |
| Container beendet sich sofort, im Log `FileNotFoundError: /models/current/model.joblib` und `Worker failed to boot` | Das Volume fehlt oder `current` zeigt ins Leere. Beim `docker run` `-v "$PWD/artifacts:/models:ro"` mitgeben. |
| Ansible: `Failed to connect to the host via ssh` | VM aus? IP in `deploy/ansible/inventory.ini` mit `multipass list` vergleichen. |
| SSH: `REMOTE HOST IDENTIFICATION HAS CHANGED` | Die VM wurde neu angelegt. Den alten Eintrag löschen: `ssh-keygen -R <IP>`. |
| Neuer Code läuft auf dem Server nicht | Die Image-Version ist gleich geblieben. `VERSION` im Makefile erhöhen, dann `make deploy`. |
| `/rank` antwortet mit 422 | Im Feld `detail` stehen `loc` (welches Feld) und `msg` (was falsch ist). |
| `Address already in use` | Port 8000 ist belegt. `lsof -i :8000` zeigt, von wem. |

## 14. Kleines Glossar

| Begriff | Bedeutung |
|---|---|
| **AUC** | Wie gut das Modell gebuchte von nicht gebuchten Hotels trennt. 0,5 ist Zufall, 1 ist perfekt. |
| **NDCG@5** | Wie weit oben die gebuchten Hotels in den ersten fünf Plätzen stehen. 1 ist die perfekte Reihenfolge. |
| **Baseline** | Eine einfache Vergleichsregel (hier: nach Bewertung sortieren). Das Modell muss sie schlagen. |
| **Pickle / joblib** | Eine Datei, in der ein Python-Objekt gespeichert ist. Sie lässt sich nur mit passenden Bibliotheksversionen sicher laden. |
| **Leakage** | Das Modell sieht beim Training etwas, das es bei der Vorhersage nicht wüsste (hier: `position`, und der Mittelwert über die Testdaten). |
| **Image / Container / Volume** | Image: der Bauplan. Container: die laufende Instanz. Volume: ein Ordner vom Server, der in den Container eingeblendet wird. |
| **Healthcheck** | Docker fragt regelmäßig `/health` und merkt sich, ob der Service antwortet. |
| **Idempotent** | Ein zweiter Lauf ändert nichts mehr, weil der Zustand schon stimmt. |
| **Symlink** | Ein Verweis auf einen Ordner. `current` zeigt auf die aktive Modellversion. |
| **Rollback** | Zurück auf eine ältere Version. |
