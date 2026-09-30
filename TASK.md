# Probeaufgabe (Mock): Beliebtheitssortierung in Produktion bringen

*Übungsaufgabe im Stil des Kennenlerntags · Zeitrahmen: ~5 Stunden · KI-Tools ausdrücklich erlaubt*

## Kontext

Lena aus dem Data-Science-Team Hotel hat ein Modell entwickelt, das für eine Hotelsuche die
Buchungswahrscheinlichkeit pro Hotel schätzt. Damit soll die „Beliebtheit"-Sortierung in der
Ergebnisliste verbessert werden. Sie übergibt dir:

- `beliebtheit_v2.ipynb` – ihr Notebook (Exploration, Training, Beispiel-Scoring)
- `data/impressions.csv` – Trainingsdaten (Export der letzten 3 Monate)
- `requirements.txt`
- `examples/request_example.json` / `response_example.json` – mit dem Produktmanagement abgestimmtes API-Format

Data Scientists arbeiten bei uns lokal und haben keinen Zugriff auf die Server. Alles ab hier liegt
bei dir als DevOps Data Science.

## Anforderungen aus Produkt und Betrieb

- Pro Suche kommen 50–200 Hotels. Die Antwortzeit sollte **< 50 ms** liegen.
- Neue Daten kommen täglich. Das Modell soll regelmäßig neu trainiert werden können.
- Der Service läuft auf unseren **Linux-Servern (Ubuntu) im eigenen Rechenzentrum**, dockerisiert.
- Server werden **nicht manuell** konfiguriert, sondern über Configuration Management (bei uns: SaltStack → Ansible).
- Wir müssen jederzeit sehen können, **welche Modellversion** läuft und ob der Service gesund ist.

## Aufgabe

### Must-have
1. **Refactoring:** Überführe den Notebook-Code in ein wartbares Python-Projekt. Training und
   Vorhersage sind getrennt, die Feature-Logik existiert nur **einmal**.
2. **Training als Skript:** Ein Befehl trainiert das Modell und schreibt ein versioniertes
   Modell-Artefakt plus Metadaten (Version, Datum, Metriken).
3. **REST-Service:** FastAPI mit mindestens
   - `GET /health` – Liveness
   - `POST /rank` – Request/Response wie in `examples/`
   - Betrieb mit **Gunicorn** (Uvicorn-Worker)
4. **Tests:** pytest für Feature-Logik und API.
5. **Docker:** schlankes Image, läuft nicht als root, Healthcheck.
6. **Deployment:** Der Service läuft auf einem **frischen Ubuntu-Server** (VM), provisioniert per
   **Ansible**: Docker installieren, Service-User, Artefakte, Container starten, Firewall.
   Ein zweiter Durchlauf des Playbooks ändert nichts.
7. **README:** Setup, Deployment, Betrieb, deine wichtigsten Entscheidungen und bekannte Grenzen.

### Nice-to-have (wähle selbst, was sinnvoll ist)
8. **CI-Pipeline** (`bitbucket-pipelines.yml`): Lint, Tests, Image-Build.
9. **Monitoring:** `/metrics` (Prometheus), strukturierte Logs, Latenz, Score-Verteilung.
10. **Retraining:** geplanter Job (cron oder systemd-Timer), der neu trainiert. Wie kommt ein neues
    Modell sicher in den laufenden Service – und wie wieder zurück?
11. **Netzwerk:** Reverse Proxy (nginx) auf einer zweiten VM. Der Service ist nur über den Proxy erreichbar.
12. **Cloud (Skizze reicht):** Terraform für einen S3-Bucket, in dem Modell-Artefakte versioniert liegen.

### Konzeptfragen für die Abschlussbesprechung
- Was hättest du Lena zurückgemeldet? Welche Probleme im Notebook würden in Produktion Ärger machen?
- Wie merkst du, dass das Modell in Produktion schlechter wird?
- Wie würdest du einen **LLM-basierten** Service (z. B. „bestes Hotelbild zuerst") anders überwachen
  als dieses Modell?

## Abgabe
Git-Repository (Commits mit sinnvollen Messages) plus eine ca. 15-minütige Vorstellung:
was läuft, warum so, was als Nächstes.
