"""Saved models: versioned directories with a sidecar JSON that ties them to their inputs.

Layout under ``experiments/ranker_clean/cache/models/``, which is not committed::

    <method>/<split>/seed_<seed>/fold_<fold>/v<config hash prefix>/
        stage1.txt     absent for the heuristic scorer and the single-stage model
        stage2.txt
        sidecar.json
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb

from cascadelens.features.config import FeaturesConfig
from cascadelens.models.data import RankerError
from cascadelens.models.ranker import FittedFold
from cascadelens.models.retrieval import BoosterScorer, HeuristicScorer, Scorer
from cascadelens.models.tuning import Choice
from cascadelens.results import git_commit

MODELS_DIR = Path("experiments/ranker_clean/cache/models")
SIDECAR = "sidecar.json"
STAGE1 = "stage1.txt"
STAGE2 = "stage2.txt"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Provenance:
    """Where a saved model came from."""

    method: str
    split: str
    seed: int
    fold: int
    config_hash: str
    feature_config_hash: str
    data_checksum: str
    features: dict[str, Any]


def artifact_dir(provenance: Provenance, root: Path = Path()) -> Path:
    """Return the versioned directory of a model."""
    return (
        root
        / MODELS_DIR
        / provenance.method
        / provenance.split
        / f"seed_{provenance.seed}"
        / f"fold_{provenance.fold}"
        / f"v{provenance.config_hash[:12]}"
    )


def _scorer_kind(scorer: Scorer | None) -> str:
    if scorer is None:
        return "none"
    return "heuristic" if isinstance(scorer, HeuristicScorer) else "lightgbm"


def save_artifact(fold: FittedFold, provenance: Provenance, root: Path = Path()) -> Path:
    """Write the models, and the sidecar last so a partial save has none."""
    directory = artifact_dir(provenance, root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SIDECAR).unlink(missing_ok=True)
    if isinstance(fold.scorer, BoosterScorer):
        fold.scorer.booster.save_model(str(directory / STAGE1))
    fold.booster.save_model(str(directory / STAGE2))
    sidecar = {
        **asdict(provenance),
        "choice": fold.choice.to_dict(),
        "columns": list(fold.columns),
        "format_version": FORMAT_VERSION,
        "git_commit": git_commit(),
        "scorer": _scorer_kind(fold.scorer),
        "val_top1": fold.val_top1,
    }
    with (directory / SIDECAR).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return directory


def read_sidecar(directory: Path) -> dict[str, Any]:
    """Read a sidecar, refusing an incomplete or unknown artifact."""
    path = directory / SIDECAR
    if not path.is_file():
        raise RankerError(f"No complete model in {directory}")
    sidecar: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if sidecar["format_version"] != FORMAT_VERSION:
        raise RankerError(f"Model format {sidecar['format_version']} is not supported")
    return sidecar


def load_artifact(directory: Path) -> tuple[FittedFold, FeaturesConfig]:
    """Load a saved model and the feature settings it was trained on."""
    sidecar = read_sidecar(directory)
    columns = tuple(sidecar["columns"])
    scorer: Scorer | None = None
    if sidecar["scorer"] == "lightgbm":
        scorer = BoosterScorer(lgb.Booster(model_file=str(directory / STAGE1)))
    elif sidecar["scorer"] == "heuristic":
        scorer = HeuristicScorer(columns)
    booster = lgb.Booster(model_file=str(directory / STAGE2))
    fold = FittedFold(scorer, booster, Choice(**sidecar["choice"]), columns, sidecar["val_top1"])
    return fold, FeaturesConfig.model_validate(sidecar["features"])


def latest_artifact(root: Path = Path()) -> Path:
    """Return the most recently saved model."""
    sidecars = list((root / MODELS_DIR).rglob(SIDECAR))
    if not sidecars:
        raise RankerError(f"No saved model under {root / MODELS_DIR}; run `cascadelens train`")
    return max(sidecars, key=lambda path: (path.stat().st_mtime_ns, str(path))).parent
