"""Download RE1 through RCAEval's own helpers. Runs only from ``reproduce_all.sh --download``.

Passing --download means you have read and accepted the dataset's license. It needs the published
extra: ``uv sync --extra published``.
"""

from pathlib import Path

from RCAEval.utility import (
    download_re1ob_dataset,
    download_re1ss_dataset,
    download_re1tt_dataset,
)

RAW = Path("data/raw")


def main() -> None:
    """Download each suite into data/raw."""
    RAW.mkdir(parents=True, exist_ok=True)
    for download in (download_re1ob_dataset, download_re1ss_dataset, download_re1tt_dataset):
        download(local_path=str(RAW))


if __name__ == "__main__":
    main()
