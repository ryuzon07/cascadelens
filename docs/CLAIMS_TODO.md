# Claims to verify before publishing

Nothing below has been confirmed from a primary source. Do not publish a claim that is still open.
The checklist is PRD section 14, and the table is regenerated from the documents.

## Checklist

- [ ] Uptime Institute outage-cost percentages and their scope
- [ ] Google Cloud incident of June 12, 2025: details from the official post-incident report
- [ ] Cloudflare incident duration and cause from its official blog
- [ ] FSE 2026 benchmark: 1,430 cases, 21% average and 37% best Top-1, number of models evaluated
- [ ] The September 2026 retrieval-versus-reranking paper: title, authors, findings
- [ ] AegisLab and "CascadeLens" name overlap (GitHub, arXiv, PyPI)
- [ ] RCAEval case counts, fault types, systems, baseline implementations, against primary sources
- [ ] Dataset licenses, download formats and sizes
- [ ] BARO published numbers and the case subset they use

## Also open

- [ ] S6: a completed full run from a fresh clone on the real data, logged in
  `docs/results/reproduction_log.md`
- [ ] Freeze the config, rerun without `--exploratory`, and regenerate every document
- [ ] Author and repository URL in `CITATION.cff`
- [ ] Export `docs/REPORT.pdf`: pandoc is not installed here

## Markers in the documents

<!-- generated:markers; regenerate with: python -m cascadelens.report build -->

| File | Marker | Count |
| --- | --- | --- |
| `MODEL_CARD.md` | `[VERIFY: RCAEval dataset license terms]` | 1 |
| `README.md` | `[VERIFY: RCAEval dataset license terms]` | 1 |
| `docs/CascadeLens Full Project Documentation (PRD, TRD, Protocol, Plan).md` | `[VERIFY exact suites, case counts, fault types, file formats, download sizes, license\]` | 1 |
| `docs/CascadeLens Full Project Documentation (PRD, TRD, Protocol, Plan).md` | `[VERIFY existence, license, schema\]` | 2 |
| `docs/CascadeLens Full Project Documentation (PRD, TRD, Protocol, Plan).md` | `[VERIFY sizes\]` | 1 |
| `docs/CascadeLens Full Project Documentation (PRD, TRD, Protocol, Plan).md` | `[VERIFY suites, case counts, fault types\]` | 1 |
| `docs/CascadeLens Full Project Documentation (PRD, TRD, Protocol, Plan).md` | `[VERIFY\]` | 5 |
| `docs/DEMO_SCRIPT.md` | `[VERIFY: repository URL]` | 1 |
| `docs/PRD_TRD.md` | `[VERIFY exact suites, case counts, fault types, file formats, download sizes, license]` | 1 |
| `docs/PRD_TRD.md` | `[VERIFY existence, license, schema]` | 2 |
| `docs/PRD_TRD.md` | `[VERIFY sizes]` | 1 |
| `docs/PRD_TRD.md` | `[VERIFY suites, case counts, fault types]` | 1 |
| `docs/PRD_TRD.md` | `[VERIFY]` | 5 |
| `docs/REPORT.md` | `[VERIFY: RCAEval dataset license terms]` | 1 |
| `docs/REPORT.md` | `[VERIFY: benchmark-critique citations]` | 1 |
| `docs/REPORT.md` | `[VERIFY: survey citations for each family are not in this repository]` | 1 |

<!-- /generated -->
