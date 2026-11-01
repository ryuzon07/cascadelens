#!/usr/bin/env bash
# Clone this repository into a temporary folder, install it fresh, and check that it works.
#
# By default this runs `make smoke` on the committed fixture and verifies the generated headline
# table and provenance. That is not S6, and the log records it as `smoke`.
#
# S6 needs the manual full run, on the real data:
#   DATA_DIR=/path/to/raw bash scripts/clean_clone_check.sh --full
# DATA_DIR must hold RE1-OB, RE1-SS and RE1-TT. It copies them to the clone, runs
# `bash scripts/reproduce_all.sh` there (about 6 hours), fails if any regenerated table differs from
# the committed one, and logs a `full` row. Only such a row can change the S6 verdict.
# Set FROZEN=1 after a freeze; a frozen run replaces exploratory numbers, so run its comparison
# against a commit that holds frozen results.
#
# WORK_DIR=<folder> keeps the clone there, so a stopped full run resumes from its last stage.
set -euo pipefail

repository="$(git rev-parse --show-toplevel)"
log="$repository/docs/results/reproduction_log.md"
UV="${UV:-uv}"
kind=smoke
if [ "${1:-}" = "--full" ]; then
    kind=full
    if [ ! -d "${DATA_DIR:-}/RE1-OB" ]; then
        echo "Set DATA_DIR to the folder holding RE1-OB, RE1-SS and RE1-TT." >&2
        exit 2
    fi
fi

folder="${WORK_DIR:-$(mktemp -d)}"
started="$(date +%s)"
outcome=fail

finish() {
    runtime=$(($(date +%s) - started))
    commit="$(git -C "$repository" rev-parse --short=12 HEAD)"
    row="| $(date -u +%Y-%m-%d) | $kind | \`$commit\` | $(uname -sm) | $runtime | $outcome |"
    echo "$row" >>"$log"
    echo "$row"
    if [ -z "${WORK_DIR:-}" ]; then
        rm -rf "$folder"
    fi
}
trap finish EXIT

if [ ! -d "$folder/repo" ]; then
    mkdir -p "$folder"
    git clone --quiet "$repository" "$folder/repo"
fi
cd "$folder/repo"
"$UV" sync --frozen --extra dev --extra api --quiet

if [ "$kind" = "full" ]; then
    if [ ! -d data/raw ]; then
        mkdir -p data
        cp -r "$DATA_DIR" data/raw
    fi
    "$UV" sync --frozen --all-extras --quiet
    bash scripts/reproduce_all.sh
    # The documents above were rebuilt from the new results, so a sync check alone proves nothing.
    # The committed tables are the claim being reproduced.
    git diff --exit-code --stat -- docs/results/tables docs/results/headline_table.md
else
    make smoke
fi
"$UV" run --frozen python -m cascadelens.report verify --root .
"$UV" run --frozen python -m cascadelens.report check --root .
outcome=pass
