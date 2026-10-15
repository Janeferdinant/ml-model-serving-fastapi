PY ?= python

.PHONY: install train promote test lint typecheck audit serve drift-demo loadtest docs docker smoke all

install:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

train:
	$(PY) -m mlserving train --out models

promote:
	$(PY) -m mlserving promote v1 --model-dir models

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

typecheck:
	$(PY) -m mypy

audit:
	$(PY) -m pip_audit -r requirements.txt

serve:
	$(PY) -m mlserving serve --port 8000

drift-demo:
	$(PY) scripts/make_drift_demo.py

loadtest:
	$(PY) loadtest/run_loadtest.py --workers 1 4 --out docs/runs

docs:
	$(PY) scripts/build_explorer.py

docker:
	docker build -t mlserving:local .

smoke:
	sh scripts/docker_smoke.sh mlserving:local

all: lint typecheck test
