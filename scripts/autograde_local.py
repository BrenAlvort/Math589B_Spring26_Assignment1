from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elastic_rod.autograde_api import run_suite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["accuracy", "speed"], default="accuracy")
    args = ap.parse_args()
    print(json.dumps(run_suite(args.mode), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
