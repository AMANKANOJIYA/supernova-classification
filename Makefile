.PHONY: setup data train test reproduce clean

PYTHON ?= python3.12
VENV   := .venv
PIP    := $(VENV)/bin/pip

# Creates a 3.12 venv and installs the project + dev deps.
# NOTE: `pip freeze` after this is the reproducible lockfile referenced in
# pyproject.toml — regenerate requirements-lock.txt if you touch dependencies.
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) freeze > requirements-lock.txt

# Step 4 (src/readers/) does not exist yet as of Day 1 — see CLAUDE.md §6.
# This target is a stub until then.
data:
	@echo "data: no readers implemented yet (CLAUDE.md §6 step 4). Nothing to do."

# Step 6+ (src/train/) does not exist yet as of Day 1.
train:
	@echo "train: no trainer implemented yet (CLAUDE.md §6 step 6). Nothing to do."

test:
	$(VENV)/bin/pytest

# Full pipeline: data -> train -> eval report. Stubbed until steps 4/6 land.
reproduce: data train
	@echo "reproduce: pipeline stages above are stubs until CLAUDE.md §6 steps 4 and 6 are built."

clean:
	rm -rf $(VENV) requirements-lock.txt .pytest_cache
	find . -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} +
