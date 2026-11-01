import sys
from pathlib import Path

for folder in ("baselines", "models", "perturb"):
    sys.path.insert(0, str(Path(__file__).parent / folder))
