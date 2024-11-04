#!/usr/bin/env python3
from pathlib import Path
from typing import List
import pandas as pd
import argparse, os, shutil


def main():
    args = argparse.ArgumentParser(description="Filtering instances using total time")
    args.add_argument(
        "-c", "--csv", type=Path, required=True, help="Path to the csv file"
    )
    args.add_argument(
        "-l", "--lower-time-bound", type=int, required=True, help="Lower time threshold"
    )
    args.add_argument(
        "-u", "--upper-time-bound", type=int, required=True, help="Upper time threshold"
    )
    args.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    args.add_argument(
        "-p", "--params", type=Path, required=True, help="Path to the params files"
    )

    args = args.parse_args()

    base_model_results = pd.read_csv(args.csv)
    # Check if all instances are Satisfiable or Timeout in the csv file
    assert all(
        base_model_results["Satisfiable"] | base_model_results["TimeOut"]
    ), "Some instances are not Satisfiable or Timeout"

    # Filter instances based on total time
    base_model_results = base_model_results[
        base_model_results["TotalTime"] >= args.lower_time_bound
    ]
    base_model_results = base_model_results[
        base_model_results["TotalTime"] <= args.upper_time_bound
    ]

    # Save the filtered instances
    params_folder: Path = args.params
    output_folder: Path = args.output
    instance_names: List[Path] = [
        params_folder / instance for instance in base_model_results["Instance"].unique()
    ]
    os.makedirs(output_folder, exist_ok=True)
    for instance in instance_names:
        shutil.copy(instance, output_folder / instance.name)


if __name__ == "__main__":
    main()
