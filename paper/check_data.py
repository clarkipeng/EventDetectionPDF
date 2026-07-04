"""Validate expected dataset files before launching experiment matrices."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = {
    "sleep": ["train_series.parquet", "train_events.csv"],
    "seizure": ["seizure_events.csv", "seizure_256Hz_dataset"],
    "bowshock": ["martian_bow_shock_dataset.pkl", "martian_bow_shock_events.csv"],
    "fraud": ["credit_card_fraud_dataset.csv", "credit_card_fraud_events.csv"],
}


def check_dataset(dataset, datadir):
    datadir = Path(datadir)
    missing = []
    for name in REQUIRED[dataset]:
        path = datadir / name
        if not path.exists():
            missing.append(str(path))
    return missing


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-dir", default="data/sleep")
    parser.add_argument("--seizure-dir", default="data/seizure")
    parser.add_argument("--bowshock-dir", default="data")
    parser.add_argument("--fraud-dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=["sleep", "seizure"])
    return parser.parse_args()


def main():
    args = parse_args()
    datadirs = {
        "sleep": args.sleep_dir,
        "seizure": args.seizure_dir,
        "bowshock": args.bowshock_dir,
        "fraud": args.fraud_dir,
    }
    failed = False
    for dataset in args.datasets:
        missing = check_dataset(dataset, datadirs[dataset])
        if missing:
            failed = True
            print(f"{dataset}: missing")
            for path in missing:
                print(f"  - {path}")
        else:
            print(f"{dataset}: ok ({datadirs[dataset]})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
