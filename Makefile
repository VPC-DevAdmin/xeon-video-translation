# polyglot-demo — command interface.
# Run `make help` for the menu.
#
# Config can live in a `.env` file at the repo root. Any CLI variable you'd
# normally pass (FIXTURE, TARGET, LIPSYNC, API_BASE, ...) is also read from
# there if present. Override per-invocation with `make run TARGET=ja`.

-include .env
export

# --- Defaults (override via .env or CLI) --------------------------------------
API_BASE   ?= http://localhost:8088
MUSETALK_HEALTH_URL ?= http://localhost:8089/health
TARGET     ?= es
LIPSYNC    ?= none
FIXTURE    ?= artifacts/inputs/IMG_7228.MOV
OUT_ROOT   ?= ./artifacts/jobs

.DEFAULT_GOAL := help

# Pretty-printed help, derived from `## comments` on each target.
.PHONY: help
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "; printf "\nUsage: make \033[36m<target>\033[0m [VAR=value ...]\n\nTargets:\n"} \
	/^[a-zA-Z0-9_-]+:.*?## / { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nCurrent config (override via .env or CLI):\n"
	@printf "  API_BASE = %s\n" "$(API_BASE)"
	@printf "  TARGET   = %s\n" "$(TARGET)"
	@printf "  LIPSYNC  = %s\n" "$(LIPSYNC)"
	@printf "  FIXTURE  = %s\n" "$(FIXTURE)"
	@printf "  OUT_ROOT = %s\n" "$(OUT_ROOT)"
	@printf "\nExamples:\n"
	@printf "  make run-musetalk FIXTURE=artifacts/inputs/mine.mov TARGET=ja\n"
	@printf "  make fetch JOB=b0965\n\n"

# --- Stack lifecycle ----------------------------------------------------------
.PHONY: up down restart rebuild

up:  ## Build and start all services (docker compose up -d --build)
	docker compose up -d --build

down:  ## Stop all services
	docker compose down

restart:  ## Restart backend + lipsync-musetalk (picks up bind-mounted code)
	docker compose restart backend lipsync-musetalk

rebuild:  ## Rebuild and restart both app containers
	docker compose up -d --build backend lipsync-musetalk

# --- Observability ------------------------------------------------------------
.PHONY: logs logs-musetalk logs-frontend health status

logs:  ## Tail backend logs
	docker compose logs -f backend

logs-musetalk:  ## Tail MuseTalk service logs
	docker compose logs -f lipsync-musetalk

logs-frontend:  ## Tail frontend logs
	docker compose logs -f frontend

status: health  ## Alias for health

health:  ## Hit /health on both services
	@echo '--- backend ---'
	@curl -sS $(API_BASE)/health | jq . 2>/dev/null || echo '  (unreachable at $(API_BASE))'
	@echo ''
	@echo '--- lipsync-musetalk ---'
	@curl -sS $(MUSETALK_HEALTH_URL) | jq . 2>/dev/null || echo '  (unreachable at $(MUSETALK_HEALTH_URL))'

# --- Model downloads ----------------------------------------------------------
.PHONY: models models-wav2lip models-musetalk models-all

models:  ## Download backend models (Whisper + NLLB + XTTS)
	docker compose exec backend bash /app/scripts/download_models.sh

models-wav2lip:  ## Backend models + Wav2Lip checkpoint
	docker compose exec -e LIPSYNC_BACKEND=wav2lip backend bash /app/scripts/download_models.sh

models-musetalk:  ## MuseTalk service weights (~1.4 GB)
	docker compose exec lipsync-musetalk bash /app/scripts/download_models.sh

models-all: models-wav2lip models-musetalk  ## All models including Wav2Lip and MuseTalk

# --- Running jobs -------------------------------------------------------------
.PHONY: run run-none run-wav2lip run-musetalk

run:  ## Run smoke test with current $(LIPSYNC) / $(TARGET) / $(FIXTURE)
	@test -f "$(FIXTURE)" || { echo "FIXTURE not found: $(FIXTURE)"; echo "Set FIXTURE=... or put a clip under artifacts/inputs/."; exit 1; }
	API_BASE=$(API_BASE) FIXTURE=$(FIXTURE) TARGET=$(TARGET) LIPSYNC=$(LIPSYNC) ./scripts/smoke_test.sh

run-none: LIPSYNC=none
run-none: run  ## Shortcut: run with LIPSYNC=none (dub-over, fastest)

run-wav2lip: LIPSYNC=wav2lip
run-wav2lip: run  ## Shortcut: run with LIPSYNC=wav2lip

run-musetalk: LIPSYNC=musetalk
run-musetalk: run  ## Shortcut: run with LIPSYNC=musetalk (slow, ~20 min/clip)

# --- Artifacts ----------------------------------------------------------------
.PHONY: list fetch progress progress-report watch inputs clean-jobs

list:  ## List recent jobs on the backend
	@API_BASE=$(API_BASE) ./scripts/fetch.sh --list

fetch:  ## Download latest job artifacts (override with JOB=prefix)
	@API_BASE=$(API_BASE) OUT_ROOT=$(OUT_ROOT) ./scripts/fetch.sh $(JOB)

progress:  ## One-shot status report on latest (or JOB=prefix) job
	@API_BASE=$(API_BASE) ./scripts/progress.sh $(JOB)

progress-report: progress  ## Alias for progress

watch:  ## Poll a running job until it finishes (INTERVAL=secs to tune)
	@API_BASE=$(API_BASE) INTERVAL=$(or $(INTERVAL),5) ./scripts/progress.sh --watch $(JOB)

inputs:  ## List available input fixtures under artifacts/inputs/
	@if [ -d artifacts/inputs ] && [ "$$(ls -A artifacts/inputs 2>/dev/null | grep -v '^\.gitkeep$$')" ]; then \
		ls -lh artifacts/inputs/ | grep -v '\.gitkeep$$'; \
	else \
		echo '(artifacts/inputs/ is empty; drop .mov/.mp4/.webm files there)'; \
	fi

clean-jobs:  ## Remove all downloaded job artifacts
	rm -rf $(OUT_ROOT) && mkdir -p $(OUT_ROOT) && echo "cleared $(OUT_ROOT)"
