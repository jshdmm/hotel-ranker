VENV    := .venv
PY      := $(VENV)/bin/python
VERSION := 0.1.0
IMAGE   := hotel-ranker:$(VERSION)

.PHONY: help venv train test lint check run docker image collections deploy

help:
	@echo "make venv         Umgebung anlegen und installieren"
	@echo "make train        Modell trainieren (neue Version in artifacts/)"
	@echo "make check        Lint und Tests (unsere lokale CI)"
	@echo "make run          Service lokal starten (Entwicklung)"
	@echo "make docker       Image bauen"
	@echo "make deploy       Image bauen und per Ansible auf die VM bringen"

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install -e ".[dev,deploy]"

train:
	$(PY) -m hotel_ranker.train

lint:
	$(VENV)/bin/ruff check .

test:
	$(PY) -m pytest -q

check: lint test

run:
	$(VENV)/bin/uvicorn hotel_ranker.api:app --reload

docker:
	docker build -t $(IMAGE) .

# Das Archiv wird nur gebaut, wenn es fuer diese VERSION noch fehlt.
# Neuer Code: VERSION erhoehen, dann entsteht ein neues Image und ein neues Archiv.
ARCHIV := deploy/ansible/files/hotel-ranker-$(VERSION).tar.gz

$(ARCHIV):
	docker build -t $(IMAGE) .
	mkdir -p deploy/ansible/files
	docker save $(IMAGE) | gzip > $(ARCHIV)

image: $(ARCHIV)

collections:
	cd deploy/ansible && ../../$(VENV)/bin/ansible-galaxy collection install -r requirements.yml -p collections

deploy: image collections
	cd deploy/ansible && ../../$(VENV)/bin/ansible-playbook site.yml -e image_tag=$(VERSION)
