# LatentSync lipsync microservice — CPU-only, dep-isolated.
#
# Layered deliberately (largest-cached → most-frequently-edited):
#   1. apt deps (ffmpeg, build tools for a couple of source wheels)
#   2. torch 2.5.1 CPU wheels (heaviest single layer; cached across PRs)
#   3. project deps from pyproject.toml (diffusers, mediapipe, ...)
#   4. application code (cheapest to rebuild)
#
# Staging:
#   1a (shipped):  FastAPI scaffold only; no torch; /lipsync returns 501.
#   1b (this PR):  Full ML stack + real download_models.sh. /ready turns
#                  green after `make rebuild`; /weights turns green after
#                  `make models-latentsync`. /lipsync still returns 501.
#   1c (next):     Vendor inference, flip INFERENCE_IMPLEMENTED = True.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
      curl \
      git \
      # build-essential: several wheels on this stack (python_speech_features,
      # face-alignment, sometimes decord) compile small C extensions from
      # source when no pre-built wheel matches Python 3.11 + CPU.
      build-essential \
      # mediapipe links against libGL for some visualization paths; keep
      # the minimal GL stubs even though we never open a display.
      libgl1 \
      libglib2.0-0 \
      # Performance stack — mirrors services/musetalk/Dockerfile. TCMalloc
      # reduces allocator contention under heavy ML load; libiomp5 comes
      # from the intel-openmp pip package; we just need google-perftools
      # on the system for the LD_PRELOAD below.
      libgoogle-perftools4 \
      # Linux 5.8+ refuses to load shared objects that declare an executable
      # stack (PT_GNU_STACK with RWE). Intel Extension for PyTorch 2.5 on
      # some releases ships libintel-ext-pt-cpu.so with that flag set;
      # without clearing it, every `import intel_extension_for_pytorch`
      # would abort with
      #   ImportError: cannot enable executable stack as shared object requires
      # We patch the flag off with patchelf post-install.
      patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision were originally pinned to LatentSync upstream's
# requirements.txt (`torch==2.5.1` / `torchvision==0.20.1`). We bumped
# to the 2.6 line in this patch because Intel Extension for PyTorch
# 2.5.0 was yanked from PyPI — the lowest un-yanked IPEX is 2.6.0,
# and IPEX 2.6.x tracks torch 2.6.x exactly. Running a torch 2.5.1 /
# IPEX 2.6.0 combo would segfault at import on most systems.
#
# This aligns the whole repo's torch stack at 2.6 (backend and
# lipsync-musetalk were already there), at the cost of diverging from
# LatentSync's upstream-tested version by one minor release. Risk is
# low: the vendored inference code doesn't depend on torch API
# surfaces that changed between 2.5 and 2.6. If quality regresses
# after this bump, that's the first thing to blame.
#
# setuptools 81 dropped pkg_resources; insightface + onnx + various HF
# paths still import it. Pin to <81 like the other services.
RUN pip install --upgrade pip wheel \
    && pip install "setuptools>=68,<81" \
    && pip install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       "torch>=2.6,<2.7" \
       "torchvision>=0.21,<0.22"

COPY pyproject.toml ./
RUN pip install -e .

# Clear the executable-stack marker from IPEX's native libs (see apt
# patchelf note above). Targeted only at intel_extension_for_pytorch
# and intel-openmp in case either ever ships the same flag; patchelf
# failures are tolerated so a future clean wheel won't break the build.
RUN find /usr/local/lib/python3.11/site-packages/intel_extension_for_pytorch \
         /usr/local/lib/python3.11/site-packages/intel_openmp \
         -name "*.so*" 2>/dev/null \
    | xargs -r -n1 -I{} sh -c 'patchelf --clear-execstack "{}" 2>/dev/null || true'

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

# Preload TCMalloc + Intel OpenMP before uvicorn starts. libiomp5 comes
# from the `intel-openmp` pip wheel (site-packages/intel_openmp/lib) —
# we resolve the real path at runtime because the wheel's layout can
# vary across versions. TCMalloc is the Debian libgoogle-perftools
# package. Both failures are non-fatal; uvicorn starts either way.
# Mirrors services/musetalk/Dockerfile's entrypoint.
RUN echo '#!/bin/bash\n\
IOMP_LIB="$(python -c "import intel_openmp, os; print(os.path.join(os.path.dirname(intel_openmp.__file__), \"lib\", \"libiomp5.so\"))" 2>/dev/null)"\n\
TCMALLOC_LIB="$(find /usr/lib -name "libtcmalloc.so*" 2>/dev/null | head -1)"\n\
PRELOAD=""\n\
[ -f "$IOMP_LIB" ] && PRELOAD="$IOMP_LIB"\n\
[ -n "$TCMALLOC_LIB" ] && PRELOAD="${PRELOAD:+$PRELOAD:}$TCMALLOC_LIB"\n\
[ -n "$PRELOAD" ] && export LD_PRELOAD="$PRELOAD"\n\
echo "LD_PRELOAD=$LD_PRELOAD" >&2\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"' > /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

CMD ["/usr/local/bin/entrypoint.sh"]
