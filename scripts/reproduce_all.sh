#!/usr/bin/env bash
# Regenerate every headline table from data/raw. Requires `make install` and the RE1 data.
# PUBLISHED=1 also reruns the published baselines, which takes about two hours.
# ROBUSTNESS=1 also runs the perturbation sweep and the probes, which takes about three hours.
# FROZEN=1 drops --exploratory, once a config is frozen with `cascadelens freeze`.
set -euo pipefail

UV="${UV:-uv}"
cascadelens() {
    "$UV" run --frozen cascadelens "$@"
}

if [ ! -d data/raw/RE1-OB ]; then
    echo "data/raw/RE1-OB is missing. Place the RE1 suites in data/raw (see README.md)." >&2
    exit 1
fi

cascadelens data build --config configs/data.yaml
cascadelens data audit --config configs/data.yaml
cascadelens splits build --config configs/splits.yaml
cascadelens splits check --config configs/splits.yaml
cascadelens features build --config configs/features.yaml --workers 4
cascadelens features check --config configs/features.yaml
cascadelens experiments leakage-demo
cascadelens experiments baselines-clean
cascadelens eval power --config configs/eval.yaml

status=(--exploratory)
if [ "${FROZEN:-0}" = "1" ]; then
    status=()
fi
cascadelens experiments ranker-clean ${status[@]+"${status[@]}"}

if [ "${PUBLISHED:-0}" = "1" ]; then
    published() {
        "$UV" run --frozen --extra published cascadelens published "$@"
    }
    published reproduce --name baro
    published reproduce --name lingam_randomwalk
    published evaluate
fi

if [ "${ROBUSTNESS:-0}" = "1" ]; then
    cascadelens perturb summarize --workers 4
    cascadelens experiments robustness --stage all ${status[@]+"${status[@]}"}
    for probe in identity graph-only mask-only window; do
        cascadelens probes "$probe" ${status[@]+"${status[@]}"}
    done
fi
