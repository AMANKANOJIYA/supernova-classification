.PHONY: setup data data-elasticc2 data-elasticc2-verify train test reproduce clean infra-up replay worker api frontend

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

# ELAsTiCC2 training-sample model directories (CLAUDE.md §5, §6 step 4): the
# 3-class SN Ia/Ib-c/II set only -- SNIa-91bg and SNIax are Day-11 exotics,
# deliberately not fetched here. `-c` resumes a truncated download, so
# re-running this target is always safe and is the correct fix if
# make data-elasticc2-verify below fails for any directory.
ELASTICC2_BASE_URL := https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2
ELASTICC2_MODELS := SNIa-SALT3 SNIIb+HostXT_V19 \
	SNIb+HostXT_V19 SNIb-Templates SNIc+HostXT_V19 SNIc-Templates SNIcBL+HostXT_V19 \
	SNII+HostXT_V19 SNII-NMF SNII-Templates SNIIn+HostXT_V19 SNIIn-MOSFIT

data-elasticc2:
	mkdir -p data/raw/elasticc2
	cd data/raw/elasticc2 && for d in $(ELASTICC2_MODELS); do \
		wget -r -np -nH --cut-dirs=5 -R "index.html*" -c \
			"$(ELASTICC2_BASE_URL)/ELASTICC2_TRAIN_02_$${d}/"; \
	done

# Fails loudly (CLAUDE.md §5.1) if any model directory does not have exactly
# 40 HEAD.FITS and 40 PHOT.FITS files.
data-elasticc2-verify:
	$(VENV)/bin/python -c "\
from src import taxonomy; \
from src.readers import elasticc2; \
import glob, os; \
gentype_to_classid = taxonomy.load_gentype_of_classid('data/raw/elasticc2/elasticc2_gentypeofclassid.csv'); \
dirs = sorted(glob.glob('data/raw/elasticc2/ELASTICC2_TRAIN_02_*')); \
[elasticc2.assert_directory_complete(d, elasticc2.read_model_directory(d, gentype_to_classid)[1]) for d in dirs if os.path.isdir(d)]; \
print(f'{len(dirs)} director(y/ies) verified complete')"

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

# CLAUDE.md §6 step 12/13: Kafka + Redis + Postgres, local dev containers.
infra-up:
	docker compose -f infra/docker-compose.yml up -d

# Replay-mode producer (archived Tier B parquet, CLAUDE.md §6 step 12). Needs
# `make infra-up` and a running `make worker` to actually score anything --
# this target only produces onto Kafka. SPLIT/SPEED are overridable, e.g.
# `make replay SPLIT=test SPEED=500`.
replay:
	$(VENV)/bin/python -m src.serve.replay --split $${SPLIT:-test} --speed-days-per-second $${SPEED:-0}

# The one consumer (CLAUDE.md §6: "same consumer code" for replay and live).
worker:
	$(VENV)/bin/python -m src.serve.inference_worker

# FastAPI serving layer (src/serve/api.py) -- the frontend's API base URL
# (frontend/.env.local) points here. Needs `make infra-up` first.
api:
	$(VENV)/bin/uvicorn src.serve.api:app --host 0.0.0.0 --port 8010

# Next.js dashboard (frontend/, CLAUDE.md-instructed rebuild). Talks to
# `make api` over HTTP -- run both together, e.g. in separate terminals:
#   make api
#   make frontend
frontend:
	npm --prefix frontend run dev
