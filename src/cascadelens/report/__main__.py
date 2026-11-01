"""``python -m cascadelens.report build`` writes all; ``check`` fails if anything is stale."""

import argparse
import sys
from pathlib import Path

from cascadelens.report.build import COMMAND, build, check, verify


def main() -> int:
    """Run one command."""
    parser = argparse.ArgumentParser(prog="python -m cascadelens.report")
    parser.add_argument("command", choices=("build", "check", "verify"))
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.command == "build":
        print(f"Wrote {len(build(arguments.root, arguments.out))} files.")
        return 0
    if arguments.command == "verify":
        problems = verify(arguments.out if arguments.out else arguments.root)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    stale = check(arguments.root)
    for name in stale:
        print(f"Out of date: {name}", file=sys.stderr)
    if stale:
        print(f"Run `{COMMAND}` and commit the result.", file=sys.stderr)
        return 1
    print("Generated files match the committed result JSONs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
