#!/usr/bin/env bash
# Regenerate every headline table from data/raw. Requires `make install` and the RE1 data.
# The ranker is not implemented yet, so its tables are not part of this run.
# PUBLISHED=1 also reruns the published baselines, which takes about two hours.
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

if [ "${PUBLISHED:-0}" = "1" ]; then
    published() {
        "$UV" run --frozen --extra published cascadelens published "$@"
    }
    published reproduce --name baro
    published reproduce --name lingam_randomwalk
    published evaluate
fi
