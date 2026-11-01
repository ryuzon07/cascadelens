# Reproduction log

`scripts/clean_clone_check.sh` appends one row per run. S6 counts only a `full` row that passed.

| Date | Kind | Commit | Machine | Runtime (s) | Outcome |
| --- | --- | --- | --- | --- | --- |
| 2026-11-01 | smoke | `1cce6d538462` | MINGW64_NT-10.0-26200 x86_64 | 117 | pass |
| 2026-11-01 | full | `68ebb4507a77` | MINGW64_NT-10.0-26200 x86_64 | 1 | fail |
| 2026-11-01 | full | `68ebb4507a77` | MINGW64_NT-10.0-26200 x86_64 | 36323 | pass |
