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
LATENTSYNC_HEALTH_URL ?= http://localhost:8090/health
TARGET     ?= es
LIPSYNC    ?= none
# TTS backend: xtts (default, 16 langs) or f5tts (EN/ZH base, cleaner prosody).
# See docs/models.md for the language support matrix before switching.
TTS        ?= xtts
FIXTURE    ?= artifacts/inputs/IMG_7228.MOV
OUT_ROOT   ?= ./artifacts/jobs

# --- QUALITY ladder ----------------------------------------------------------
# Progressive numeric dial for MuseTalk output quality. Higher = more
# features, more wall-clock time. Passed as per-request overrides to the
# lipsync service — no container restart needed between invocations.
# Numeric today; will likely migrate to `low|medium|high` once the set of
# features stabilizes.
QUALITY ?= 3

ifeq ($(QUALITY),1)
  _QUALITY_LABEL         := minimum (raw blend, no face restore)
  _Q_BLEND_MODE          := raw
  _Q_BLEND_FEATHER       := 0.06
  _Q_FACE_RESTORE        := none
  _Q_FACE_RESTORE_FIDEL  :=
  _Q_FACE_RESTORE_BLEND  :=
else ifeq ($(QUALITY),2)
  _QUALITY_LABEL         := balanced (jaw blend, no face restore)
  _Q_BLEND_MODE          := jaw
  _Q_BLEND_FEATHER       := 0.04
  _Q_FACE_RESTORE        := none
  _Q_FACE_RESTORE_FIDEL  :=
  _Q_FACE_RESTORE_BLEND  :=
else ifeq ($(QUALITY),3)
  _QUALITY_LABEL         := full (jaw blend + CodeFormer at 0.7/0.6)
  _Q_BLEND_MODE          := jaw
  _Q_BLEND_FEATHER       := 0.04
  _Q_FACE_RESTORE        := codeformer
  _Q_FACE_RESTORE_FIDEL  := 0.7
  _Q_FACE_RESTORE_BLEND  := 0.6
else
  $(error QUALITY must be 1, 2, or 3 (got $(QUALITY)). Room for 4+ when more \
features land (LatentSync, F5-TTS). See docs/lipsync.md for what each level \
ships.)
endif

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
	@printf "  TTS      = %s\n" "$(TTS)"
	@printf "  FIXTURE  = %s\n" "$(FIXTURE)"
	@printf "  OUT_ROOT = %s\n" "$(OUT_ROOT)"
	@printf "  QUALITY  = %s — %s\n" "$(QUALITY)" "$(_QUALITY_LABEL)"
	@printf "\nExamples:\n"
	@printf "  make run-musetalk FIXTURE=artifacts/inputs/mine.mov TARGET=ja\n"
	@printf "  make run-musetalk QUALITY=2      # skip face restore for speed\n"
	@printf "  make run-f5tts TARGET=en          # F5-TTS base (EN/ZH only)\n"
	@printf "  make run-latentsync              # scaffold service returns structured 501 (PR-LS-1a)\n"
	@printf "  make fetch JOB=b0965\n\n"

# --- Stack lifecycle ----------------------------------------------------------
.PHONY: up down restart rebuild

up:  ## Build and start all services (docker compose up -d --build)
	docker compose up -d --build

down:  ## Stop all services
	docker compose down

restart:  ## Restart backend + lipsync services (picks up bind-mounted code)
	docker compose restart backend lipsync-musetalk lipsync-latentsync

rebuild:  ## Rebuild and restart all app containers
	docker compose up -d --build backend lipsync-musetalk lipsync-latentsync

# --- Observability ------------------------------------------------------------
.PHONY: logs logs-musetalk logs-latentsync logs-frontend health status

logs:  ## Tail backend logs
	docker compose logs -f backend

logs-musetalk:  ## Tail MuseTalk service logs
	docker compose logs -f lipsync-musetalk

logs-latentsync:  ## Tail LatentSync service logs
	docker compose logs -f lipsync-latentsync

logs-frontend:  ## Tail frontend logs
	docker compose logs -f frontend

status: health  ## Alias for health

health:  ## Hit /health on all services
	@echo '--- backend ---'
	@curl -sS $(API_BASE)/health | jq . 2>/dev/null || echo '  (unreachable at $(API_BASE))'
	@echo ''
	@echo '--- lipsync-musetalk ---'
	@curl -sS $(MUSETALK_HEALTH_URL) | jq . 2>/dev/null || echo '  (unreachable at $(MUSETALK_HEALTH_URL))'
	@echo ''
	@echo '--- lipsync-latentsync ---'
	@curl -sS $(LATENTSYNC_HEALTH_URL) | jq . 2>/dev/null || echo '  (unreachable at $(LATENTSYNC_HEALTH_URL))'

# --- Model downloads ----------------------------------------------------------
.PHONY: models models-wav2lip models-musetalk models-all

models:  ## Download backend models (Whisper + NLLB + XTTS)
	docker compose exec backend bash /app/scripts/download_models.sh

models-wav2lip:  ## Backend models + Wav2Lip checkpoint
	docker compose exec -e LIPSYNC_BACKEND=wav2lip backend bash /app/scripts/download_models.sh

models-musetalk:  ## MuseTalk service weights (~1.4 GB)
	docker compose exec lipsync-musetalk bash /app/scripts/download_models.sh

models-f5tts:  ## F5-TTS base checkpoint (~1.3 GB, EN/ZH)
	docker compose exec -e TTS_BACKEND=f5tts backend bash /app/scripts/download_models.sh

models-latentsync:  ## LatentSync weights (PR-LS-1a: placeholder, prints next-step note)
	docker compose exec lipsync-latentsync bash /app/scripts/download_models.sh

models-all: models-wav2lip models-musetalk models-f5tts  ## All models (Wav2Lip + MuseTalk + F5-TTS). Add `models-latentsync` once PR-LS-1b lands.

# --- Running jobs -------------------------------------------------------------
.PHONY: run run-none run-wav2lip run-musetalk run-f5tts run-latentsync

run:  ## Run smoke test with current $(LIPSYNC) / $(TARGET) / $(TTS) / $(FIXTURE) / $(QUALITY)
	@test -f "$(FIXTURE)" || { echo "FIXTURE not found: $(FIXTURE)"; echo "Set FIXTURE=... or put a clip under artifacts/inputs/."; exit 1; }
	@echo "Backend:  lipsync=$(LIPSYNC)  tts=$(TTS)  target=$(TARGET)"
	$(if $(filter musetalk,$(LIPSYNC)),@echo "MuseTalk quality:  QUALITY=$(QUALITY) — $(_QUALITY_LABEL)",)
	API_BASE=$(API_BASE) \
	  FIXTURE=$(FIXTURE) \
	  TARGET=$(TARGET) \
	  LIPSYNC=$(LIPSYNC) \
	  TTS_BACKEND=$(TTS) \
	  $(if $(filter musetalk,$(LIPSYNC)),MUSETALK_BLEND_MODE=$(_Q_BLEND_MODE) MUSETALK_BLEND_FEATHER=$(_Q_BLEND_FEATHER) MUSETALK_FACE_RESTORE=$(_Q_FACE_RESTORE) MUSETALK_FACE_RESTORE_FIDELITY=$(_Q_FACE_RESTORE_FIDEL) MUSETALK_FACE_RESTORE_BLEND=$(_Q_FACE_RESTORE_BLEND) ,) \
	  ./scripts/smoke_test.sh

run-none: LIPSYNC=none
run-none: run  ## Shortcut: run with LIPSYNC=none (dub-over, fastest)

run-wav2lip: LIPSYNC=wav2lip
run-wav2lip: run  ## Shortcut: run with LIPSYNC=wav2lip

run-musetalk: LIPSYNC=musetalk
run-musetalk: run  ## Shortcut: run with LIPSYNC=musetalk (slow, ~20 min/clip)

run-f5tts: TTS=f5tts
run-f5tts: run  ## Shortcut: run with TTS=f5tts (EN/ZH only on base checkpoint)

# LatentSync — SD 1.5 latent-diffusion lipsync. ~25-60 min per clip on CPU
# depending on length. Resume cache lives under /models/cache/latentsync_denoise/;
# re-running the same inputs skips the expensive half.
run-latentsync: LIPSYNC=latentsync
run-latentsync: run  ## Shortcut: run with LIPSYNC=latentsync (SD 1.5 diffusion, slow)

# Adaptive variant: same run, but on OOM (detected via container restart)
# it bumps memory via `docker update` and retries — the resume cache
# makes the retry cheap (~3 min for just the restore step). Use this
# for clips you don't know the memory footprint of yet.
run-latentsync-adaptive:  ## LatentSync with auto memory escalation on OOM (48g -> 64/96/128g)
	@test -f "$(FIXTURE)" || { echo "FIXTURE not found: $(FIXTURE)"; exit 1; }
	@echo "QUALITY=$(QUALITY) — $(_QUALITY_LABEL)"
	API_BASE=$(API_BASE) \
	  FIXTURE=$(FIXTURE) \
	  TARGET=$(TARGET) \
	  TTS_BACKEND=$(TTS) \
	  CONTAINER=polyglot-lipsync-latentsync \
	  ./scripts/run_latentsync_adaptive.sh

# --- Job control --------------------------------------------------------------
.PHONY: cancel reconnect

cancel:  ## Cancel a running job (JOB=<id-or-prefix>). Frees the queue slot.
	@test -n "$(JOB)" || { echo "Usage: make cancel JOB=<job-id-or-prefix>"; exit 2; }
	@full_id=$$(curl -sS "$(API_BASE)/jobs?limit=200" \
	    | jq -r --arg p "$(JOB)" '.jobs[] | select(.job_id | startswith($$p)) | .job_id' \
	    | head -1); \
	if [ -z "$$full_id" ]; then echo "no job matches prefix '$(JOB)'"; exit 1; fi; \
	echo "cancelling $$full_id"; \
	curl -sS -X POST "$(API_BASE)/jobs/$$full_id/cancel" | jq .

reconnect:  ## Reconnect to a running job without submitting a new one (JOB=<id-or-prefix>)
	@test -n "$(JOB)" || { echo "Usage: make reconnect JOB=<job-id-or-prefix>"; exit 2; }
	@full_id=$$(curl -sS "$(API_BASE)/jobs?limit=200" \
	    | jq -r --arg p "$(JOB)" '.jobs[] | select(.job_id | startswith($$p)) | .job_id' \
	    | head -1); \
	if [ -z "$$full_id" ]; then echo "no job matches prefix '$(JOB)'"; exit 1; fi; \
	echo "reconnecting to $$full_id"; \
	API_BASE=$(API_BASE) ./scripts/smoke_test.sh --reconnect $$full_id

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
