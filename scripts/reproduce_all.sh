#!/usr/bin/env bash
# Check the environment and the data, then run every stage with `make all`.
# Nothing is downloaded unless you pass --download, because the dataset's license is yours to accept.
# FROZEN=1 drops --exploratory after a freeze. SKIP_PUBLISHED=1 leaves out the published baselines.
set -euo pipefail

UV="${UV:-uv}"
download=0
for argument in "$@"; do
    case "$argument" in
        --download) download=1 ;;
        *)
            echo "Unknown option $argument. The only option is --download." >&2
            exit 2
            ;;
    esac
done

"$UV" run --frozen python scripts/check_env.py --machine-only

if [ "$download" = "1" ]; then
    echo "Downloading RE1. By passing --download you accept the dataset's license."
    "$UV" run --frozen --extra published python scripts/download_data.py
fi

"$UV" run --frozen python scripts/check_env.py
make all
