"""Write the API schema to docs/openapi.json, which CI compares with the running app.

Run ``python scripts/export_openapi.py``. Needs the api extra and a saved model, which the schema
does not depend on.
"""

import json
from pathlib import Path

from cascadelens.api.main import ApiSettings, create_app
from cascadelens.models.artifacts import latest_artifact

SNAPSHOT = Path("docs/openapi.json")


def main() -> None:
    """Build the app in memory and save its schema."""
    schema = create_app(ApiSettings(model_dir=latest_artifact())).openapi()
    text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    SNAPSHOT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {SNAPSHOT}.")


if __name__ == "__main__":
    main()
