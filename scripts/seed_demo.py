"""Load the synthetic samples into the incident database. Needs the api extra.

Run ``python scripts/seed_demo.py``. Every incident is marked as demo data, and running it again
adds nothing. Benchmark data is never read.
"""

import argparse
from pathlib import Path

from cascadelens.api.incidents import IncidentService
from cascadelens.api.seed import seed_demo
from cascadelens.api.store import Store, open_engine
from cascadelens.inference import DEFAULT_TOLERANCE_SECONDS, load_model
from cascadelens.models.artifacts import latest_artifact


def main() -> None:
    """Parse the arguments and load the samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/app.db"))
    parser.add_argument("--model", type=Path, help="Saved model; the latest one by default.")
    arguments = parser.parse_args()
    model = load_model(arguments.model or latest_artifact())
    store = Store(open_engine(arguments.database))
    added = seed_demo(IncidentService(store, model, DEFAULT_TOLERANCE_SECONDS))
    print(f"Added {added} demo incident(s) to {arguments.database}.")


if __name__ == "__main__":
    main()
