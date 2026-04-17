import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a raw ACAC progress.csv into the author's Timesteps/Eval-Return CSV format."
    )
    parser.add_argument(
        "--progress_csv",
        type=Path,
        required=True,
        help="Path to a raw training progress.csv file.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark directory name, for example ovcA7 or bp6.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="acac_user",
        help="Method directory name to write under icml25_result/log/<benchmark>/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed index to write as <seed>.csv.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "icml25_result" / "log",
        help="Root output directory. Defaults to repo icml25_result/log.",
    )
    return parser.parse_args()


def export_curve(progress_csv, output_csv):
    with progress_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {progress_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Timesteps", "Eval/Return"])
        for row in rows:
            writer.writerow([row["Timesteps"], row["Eval/ReturnAverage"]])


def main():
    args = parse_args()
    progress_csv = args.progress_csv.resolve()
    output_root = args.output_root.resolve()
    output_csv = output_root / args.benchmark / args.method / f"{args.seed}.csv"
    export_curve(progress_csv, output_csv)
    print(output_csv)


if __name__ == "__main__":
    main()
