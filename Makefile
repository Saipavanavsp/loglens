# Run from WSL/Linux/macOS. On Windows PowerShell use ./tasks.ps1 <target> instead.
PY ?= python
N ?= 50

.PHONY: install lint format test check data data-download data-parse data-sample label \
	golden review split stats eval up down

install:
	$(PY) -m pip install -e ".[data,gateway,eval,dashboard,dev]"
	pre-commit install

lint:
	ruff check src tests dashboard training evals
	black --check src tests dashboard training

format:
	ruff check --fix src tests dashboard training
	black src tests dashboard training

test:
	pytest

check: lint test

# --- Phase 1: data ---
data: data-download data-parse data-sample

data-download:
	$(PY) -m loglens.data.download

data-parse:
	$(PY) -m loglens.data.parse

data-sample:
	$(PY) -m loglens.data.sample --n 3000

label:  # dry run by default; add ARGS="--no-dry-run" to spend
	$(PY) -m loglens.data.label --limit $(N) $(ARGS)

golden:
	$(PY) -m loglens.data.review select --n 200

review:
	streamlit run dashboard/review_app.py

split:
	$(PY) -m loglens.data.split

stats:
	$(PY) -m loglens.data.stats

# --- Phase 3: evals ---
eval:
	$(PY) -m loglens.evals.run $(ARGS)

# --- Phase 6: stack ---
up:
	docker compose -f infra/docker-compose.yml up -d --build

down:
	docker compose -f infra/docker-compose.yml down
