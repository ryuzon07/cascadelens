#!/usr/bin/env bash
# Clone this repository into a temporary folder, install it fresh, and check that it works.
#
# By default this runs `make smoke` on the committed fixture and verifies the generated headline
# table and provenance. That is not S6, and the log records it as `smoke`.
#
# S6 needs the manual full run, on the real data:
#   DATA_DIR=/path/to/raw bash scripts/clean_clone_check.sh --full
# DATA_DIR must hold RE1-OB, RE1-SS and RE1-TT. It copies them to the clone, runs
# `bash scripts/reproduce_all.sh` there (about 6 hours), checks that the result is in sync, and logs a
# `full` row. Only such a row can change the S6 verdict. Set FROZEN=1 after a freeze.
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

folder="$(mktemp -d)"
trap 'rm -rf "$folder"' EXIT
started="$(date +%s)"
outcome=fail

record() {
    runtime=$(($(date +%s) - started))
    commit="$(git -C "$repository" rev-parse --short=12 HEAD)"
    row="| $(date -u +%Y-%m-%d) | $kind | \`$commit\` | $(uname -sm) | $runtime | $outcome |"
    echo "$row" >>"$log"
    echo "$row"
}
trap 'record; rm -rf "$folder"' EXIT

git clone --quiet "$repository" "$folder/repo"
cd "$folder/repo"
"$UV" sync --frozen --extra dev --extra api --quiet

if [ "$kind" = "full" ]; then
    mkdir -p data
    cp -r "$DATA_DIR" data/raw
    "$UV" sync --frozen --all-extras --quiet
    bash scripts/reproduce_all.sh
else
    make smoke
fi
"$UV" run --frozen python -m cascadelens.report verify --root .
"$UV" run --frozen python -m cascadelens.report check --root .
outcome=pass
