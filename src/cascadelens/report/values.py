"""Single numbers for prose. Templates hold ``{{name}}``, so no result is typed into a document."""

from dataclasses import dataclass

from cascadelens.report.sources import REPRODUCTION_LOG, Store, ci, fraction, pct
from cascadelens.report.tables import (
    CLEAN,
    PLAIN,
    RANKER,
    S3_POINT,
    SHIFT_POINT,
    _grid,
    full_run,
)


@dataclass(frozen=True)
class Value:
    """One number or phrase, and the source it came from."""

    text: str
    source: str


def all_values(store: Store) -> dict[str, Value]:
    """Return every value a template may use."""
    robustness = store("robustness").metrics
    hypotheses = robustness["hypotheses"]
    ranker = hypotheses["ranker"]
    calibration = store("calibration").metrics
    results = calibration["results"]
    clean = results[RANKER][CLEAN]
    p2 = results[RANKER][S3_POINT]
    shift = results[RANKER][SHIFT_POINT]
    grid = _grid(store)
    leak = store("leakage").metrics
    ablations = {row["variant"]: row for row in robustness["ablation"]}
    ranker_clean = store("ranker").metrics
    window = store("probe_window").metrics["injection_position"]
    mask = store("probe_mask").metrics
    lingam = store("lingam").metrics["comparison"]["RE1-SS"]

    def top1(method: str, point: str) -> str:
        return ci(grid[(method, point)]["Top-1"])

    by_source: dict[str, dict[str, str]] = {}
    by_source["leakage"] = {
        "leak_flat": ci(leak["paired_differences"]["lightgbm_flat"]["Top-1"]),
    }
    by_source["robustness"] = {
        "status": str(robustness["status"]),
        "s1": ci(ranker["S1"]),
        "s2": ci(ranker["S2"]),
        "s2_unshifted": ci(ranker["difference_from_best"]),
        "plain_unshifted": ci(hypotheses["plain_ranker"]["difference_from_best"]),
        "best_baseline": str(hypotheses["s2_comparator"]["method"]),
        "s3_flat": ci(hypotheses["S3"]["lightgbm_flat"]),
        "s3_masking": ci(hypotheses["S3"]["lightgbm_flat_masking"]),
        "s3_frequency": ci(hypotheses["S3"]["frequency"]),
        "s3_baro": ci(hypotheses["S3"]["baro"]),
        "ranker_clean": top1(RANKER, CLEAN),
        "plain_clean": top1(PLAIN, CLEAN),
        "flat_clean": top1("lightgbm_flat", CLEAN),
        "masking_clean": top1("lightgbm_flat_masking", CLEAN),
        "ranker_p2": top1(RANKER, S3_POINT),
        "plain_p2": top1(PLAIN, S3_POINT),
        "flat_p2": top1("lightgbm_flat", S3_POINT),
        "masking_p2": top1("lightgbm_flat_masking", S3_POINT),
        "mask_gain": ci(hypotheses["mask_gain_at_s3_point"]),
        "ablate_signal": ci(ablations["ablate_signal"]["clean"]),
        "ablate_graph": ci(ablations["ablate_graph"]["clean"]),
        "ablate_temporal": ci(ablations["ablate_temporal"]["clean"]),
        "single_stage_clean": ci(ablations["single_stage_augmented"]["clean"]),
        "single_stage_p2": ci(ablations["single_stage_augmented"]["p2_30"]),
    }
    by_source["calibration"] = {
        "s4": ci(clean["selective_gain"]),
        "s5": fraction(clean["ece_difference"]),
        "ece_before": fraction(clean["ece_before"]),
        "ece_after": fraction(clean["ece_after"]),
        "ece_p2": fraction(p2["ece_after"]),
        "ece_shift": fraction(shift["ece_after"]),
        "brier_before": fraction(clean["brier_before"]),
        "brier_after": fraction(clean["brier_after"]),
        "top1_full": ci(clean["top1"]),
        "top1_50": ci(clean["selective"]["0.5"]),
        "aurc": f"{clean['aurc']:.3f}",
        "aurc_random": f"{clean['aurc_random']:.3f}",
        "aurc_oracle": f"{clean['aurc_oracle']:.3f}",
        "coverage_clean": pct(clean["realised_coverage"]),
        "coverage_p2": pct(p2["realised_coverage"]),
        "coverage_shift": pct(shift["realised_coverage"]),
    }
    by_source["ranker"] = {
        "n_cases": str(ranker_clean["group_run"][0]["cases"]),
        "retrieval_group_run": pct(ranker_clean["retrieval_recall_at_chosen_k"]["group_run"]),
        "retrieval_loso": pct(ranker_clean["retrieval_recall_at_chosen_k"]["loso"]),
    }
    values = {
        name: Value(text, source)
        for source, entries in by_source.items()
        for name, text in entries.items()
    }
    values["window_share"] = Value(pct(window["share_within_2pct_of_middle"]), "probe_window")
    values["mask_only"] = Value(ci(mask["top1"]), "probe_mask")
    values["mask_floor"] = Value(pct(mask["uniform_floor"]), "probe_mask")
    values["lingam_gap"] = Value(f"{lingam['difference']:+.3f}", "lingam")
    return values | s6_values(store)


def s6_values(store: Store) -> dict[str, Value]:
    """Return the S6 phrases, which follow the reproduction log like the headline table does."""
    run = full_run(store.root)
    log = "`docs/results/reproduction_log.md`"
    if run is None:
        phrases = {
            "s6_limitation": "**S6 is not demonstrated** on the real data. See the reproduction "
            "statement.",
            "s6_summary": "S6 has not been demonstrated on the real data.",
            "s6_todo": f"- [ ] S6: a completed full run from a fresh clone on the real data, "
            f"logged in {log}",
        }
    else:
        phrases = {
            "s6_limitation": "**S6 covers exploratory results only.** The logged full run "
            "regenerated the committed tables, but no frozen config has been run.",
            "s6_summary": "S6 passed from a fresh clone on the real data, on exploratory results.",
            "s6_todo": f"- [x] S6: a completed full run from a fresh clone on the real data, "
            f"logged in {log}",
        }
    return {name: Value(text, REPRODUCTION_LOG) for name, text in phrases.items()}
