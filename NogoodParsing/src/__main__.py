"""
Script to process instance directories, parse variable representations, increment bins based on learnt clauses,
and compute top indices for variables across multiple seeds and instances.

This script reads data from instance directories, processes variables, and computes heatmaps
to find the most significant variable indices based on learnt clauses from SAT solvers.

Usage:
    python script_name.py -i <instance_folder> -s <shared_variables>

Arguments:
    -i, --instance-folder: Path to the folder containing instance directories.
    -s, --shared-variables: Path to the shared_variables file.
    -b, --number-of-bins: Number of bins to use for the binning. Default is 10.
"""

import argparse
import json
import sys
import os
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, Future, as_completed, wait
import matplotlib.pyplot as plt

from binning import (
    parse_instance,
    bin_instance,
    find_top_indices,
    CustomJSONEncoder,
)

np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Process instance directories and compute top indices."
    )
    arg_parser.add_argument(
        "-i",
        "--instance-folder",
        type=str,
        required=True,
        help="Path to the folder containing instance directories.",
    )
    arg_parser.add_argument(
        "-s",
        "--shared-variables",
        type=str,
        required=True,
        help="Path to the shared_variables file.",
    )
    arg_parser.add_argument(
        "-b",
        "--number-of-bins",
        type=int,
        default=10,
        help="Number of bins to use for the binning. Default is 10.",
    )
    args = arg_parser.parse_args()

    number_of_bins = args.number_of_bins
    instance_folder_path = Path(args.instance_folder)
    shared_variables_path = Path(args.shared_variables)

    if not instance_folder_path.is_dir():
        logger.error(
            f"Instance folder '{instance_folder_path}' does not exist or is not a directory."
        )
        sys.exit(1)

    if not shared_variables_path.exists():
        logger.error(f"Shared variables file not found at '{shared_variables_path}'")
        sys.exit(1)

    try:
        with shared_variables_path.open() as f_shared:
            shared_variables = json.load(f_shared)
    except Exception as e:
        logger.error(f"Failed to load shared variables file: {e}")
        sys.exit(1)

    instance_dirs: List[str] = next(os.walk(instance_folder_path))[1]
    if len(instance_dirs) == 0:
        logger.error(f"No instance directories found in '{instance_folder_path}'")
        sys.exit(1)

    seeds: List[str] = next(os.walk(instance_folder_path / instance_dirs[0]))[1]

    # Check if all seeds of an instance contain the {instance}.find_bins file
    # If so, do not add it to the list
    instances_to_process: List[str] = []
    for instance in instance_dirs:
        all_seeds_have_find_bins = True
        for seed in seeds:
            find_bins_path = (
                instance_folder_path / instance / seed / f"{instance}.find_bins"
            )
            if not find_bins_path.exists():
                all_seeds_have_find_bins = False
                break
        if not all_seeds_have_find_bins:
            instances_to_process.append(instance)
        else:
            logger.info(
                f"All seeds of instance '{instance}' already have find_bins. Skipping processing."
            )

    aux_paths: List[Path] = [
        instance_folder_path / instance / seeds[0] / f"{instance}.aux.gz"
        for instance in instances_to_process
    ]

    find_json: List[Path] = [
        instance_folder_path / instance / seeds[0] / f"{instance}.finds"
        for instance in instances_to_process
    ]

    # Check if the paths exist. They should :)
    assert all(aux_path.exists() for aux_path in aux_paths)
    assert all(find_path.exists() for find_path in find_json)

    # Check if the sizes of all training data are the same
    assert len(aux_paths) == len(find_json) == len(instances_to_process)
    parse_representation_object_args: List[Tuple[Path, Path]] = list(
        zip(aux_paths, find_json)
    )

    # Parse one aux file from each instance to get the variable representations as the same representation is shared across all seeds.
    with ProcessPoolExecutor(max_workers=1) as executor:
        parsing_futures: List[Future] = []
        for i, (aux_path, find_path) in enumerate(parse_representation_object_args):
            for seed in seeds:
                future = executor.submit(
                    parse_instance,
                    instance_folder_path,
                    aux_path,
                    find_path,
                    instances_to_process[i],
                    seed,
                )
                parsing_futures.append(future)

    wait(parsing_futures)

    for instance in instance_dirs:
        for seed in seeds:
            find_bins_path = (
                instance_folder_path / instance / seed / f"{instance}.find_bins"
            )
            if not find_bins_path.exists():
                logger.error(
                    f"This should never happen. Find bins not found at '{find_bins_path}'"
                )
                sys.exit(1)

    combined_bins: Dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=1) as executor:
        binning_futures: List[Future] = []
        for instance in instance_dirs:
            for seed in seeds:
                find_bins_path = (
                    instance_folder_path / instance / seed / f"{instance}.find_bins"
                )
                future = executor.submit(
                    bin_instance, find_bins_path, shared_variables, number_of_bins
                )
                binning_futures.append(future)

        for future in as_completed(binning_futures):
            try:
                result = future.result()

                # Combine the results
                for variable_name, binned_arr in result.items():
                    if variable_name in combined_bins:
                        combined_bins[variable_name] += binned_arr
                    else:
                        combined_bins[variable_name] = binned_arr
            except Exception as exc:
                logger.error(f"Generated an exception: {exc}")

    top_indices_json: Dict[str, Union[int, List[Dict[str, Any]]]] = {
        "bin_size": number_of_bins
    }

    for variable_name, arr in combined_bins.items():
        arr = arr / np.sum(arr) * 100
        top_indices_json[variable_name] = find_top_indices(arr, n=5)

    try:
        output_path = instance_folder_path / "top_indices.json"
        with output_path.open("w") as f_output:
            json.dump(
                top_indices_json,
                f_output,
                indent=4,
                cls=CustomJSONEncoder,
            )
    except Exception as e:
        logger.error(f"Failed to write top indices to '{output_path}': {e}")

    for variable_name, arr in combined_bins.items():
        arr = arr / np.sum(arr) * 100
        # Plot the heatmap
        plt.figure(figsize=(10, 10))
        plt.imshow(arr, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Heatmap for {variable_name}")
        plt.savefig(instance_folder_path / f"{variable_name}.png")


if __name__ == "__main__":
    main()
