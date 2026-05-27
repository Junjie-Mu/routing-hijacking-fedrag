"""Print a compact inventory of result summary files."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize available paper result files")
    parser.add_argument("--result-dir", default="paper_results")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise SystemExit(f"Result directory not found: {result_dir}")

    files = sorted(p for p in result_dir.iterdir() if p.is_file())
    print(f"{len(files)} summary files in {result_dir}:")
    for path in files:
        print(f"  - {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
