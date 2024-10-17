#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import subprocess
import time
import textwrap
from typing import Dict, List


def find_essence_file(problem_directory: Path) -> Path:
    # Find the first *.essence file in the problem directory
    for file in problem_directory.iterdir():
        if file.suffix == ".essence":
            return file
    raise FileNotFoundError("Essence file not found in the specified problem directory")


def get_param_files(problem_directory: Path) -> List[Path]:
    # Filter the directory to only contain *.param files
    param_dir = problem_directory.joinpath("params")
    return [f for f in param_dir.iterdir() if f.suffix == ".param"]


def generate_slurm_file(
    problem: str,
    essence_file: Path,
    param: Path,
    seed: int,
    time_limit: int,
    solver="cadical",
) -> Path:
    results_dir = Path("problems", problem, "Results")
    eprime_src = Path(
        "problems", problem, "model000001.eprime"
    )  # Default name generated by conjure
    param_stem = param.stem

    target_dir = results_dir.joinpath(param_stem, str(seed))
    learnt_file = target_dir.joinpath(f"{param_stem}.learnt")
    finds_file = target_dir.joinpath(f"{param_stem}.finds")
    aux_file = target_dir.joinpath(f"{param_stem}.aux")
    os.makedirs(target_dir, exist_ok=True)

    savilerow_options = f"-timelimit {time_limit} -O2 -finds-to-json -out-finds /shared/{finds_file} -out-aux /shared/{aux_file}"
    solver_options = f"-t {time_limit} --seed={seed} --output-learnts --learnt-file /shared/{learnt_file}"

    # Create directories
    slurm_sh = Path("slurm", "sh", problem)
    slurm_stderr = Path("slurm", "stderr", problem)
    slurm_stdout = Path("slurm", "stdout", problem)

    os.makedirs(slurm_sh, exist_ok=True)
    os.makedirs(slurm_stderr, exist_ok=True)
    os.makedirs(slurm_stdout, exist_ok=True)

    current_dir = Path(os.getcwd())
    slurm_file_base = f"{param_stem}_{seed}"
    slurm_file = slurm_sh.joinpath(f"{slurm_file_base}.sh")
    error_file = current_dir.joinpath(slurm_stderr, f"{slurm_file_base}.error")
    out_file = current_dir.joinpath(slurm_stdout, f"{slurm_file_base}.out")

    slurm_script = textwrap.dedent(
        f"""#!/bin/bash
        #SBATCH --job-name={problem}_{slurm_file_base}
        #SBATCH -e {error_file}
        #SBATCH -o {out_file}
        #SBATCH --cpus-per-task=2
        #SBATCH --mem=16GB
        #SBATCH --time=03:00:00
        
        timeout {time_limit + 30 * 60} docker run --rm \\
            -v {current_dir}:/shared:z \\
            --platform=linux/amd64 \\
            conjure-dump-nogoods \\
            conjure solve --use-existing-models=/shared/{eprime_src} /shared/{essence_file} /shared/{param} -o /shared/{target_dir} \\
                --copy-solutions=off \\
                --log-level LogNone \\
                --savilerow-options "{savilerow_options}" \\
                --solver {solver} \\
                --solver-options "{solver_options}"

        gzip {learnt_file}
        gzip {aux_file}
    """
    )

    slurm_file.touch(mode=0o755)
    with open(slurm_file, "w") as f:
        f.write(slurm_script)

    return slurm_file


def get_queue_size() -> int:
    """Get the current number of jobs in the Slurm queue."""
    user = os.getenv("USER")
    if user:
        result = subprocess.run(
            ["squeue", "-h", "-u", user], capture_output=True, text=True
        )
        return len(result.stdout.split("\n")) - 1
    else:
        raise EnvironmentError


def submit_job(script_path: Path) -> None:
    """Submit a job to Slurm."""
    subprocess.run(["sbatch", script_path], capture_output=True, text=True)


def model_essence_file(problem: str, essence_file: Path) -> None:
    """Generate the eprime model file from the essence file."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{os.getcwd()}:/shared:z",
        "--platform=linux/amd64",
        "conjure-dump-nogoods",
        "conjure",
        "model",
        "-q",
        "af",
        "-a",
        "af",
        "-o",
        f"/shared/problems/{problem}",
        f"/shared/{essence_file}",
    ]
    
    subprocess.run(
        cmd, capture_output=True, text=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run an Essence problem with a given parameter file and seed"
    )
    parser.add_argument(
        "-p",
        "--problem",
        nargs="+",
        required=True,
        type=Path,
        help="The name of the problem directory",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        nargs="+",
        required=False,
        default=[42],
        type=int,
        help="The seed to run the problem with",
    )
    parser.add_argument(
        "-q",
        "--max-queue",
        type=int,
        required=False,
        default=5000,
        help="Maximum number of queued jobs. Default is 5000.",
    )
    parser.add_argument(
        "-b",
        "--batch-files",
        type=bool,
        required=False,
        default=False,
        help="Batch the slurm files",
    )
    parser.add_argument(
        "-t",
        "--time-limit",
        type=int,
        required=False,
        default=7200,
        help="Time limit for each job in seconds. Default is 7200.",
    )
    args = parser.parse_args()

    problem_classes: List[Path] = args.problem
    seeds: List[int] = args.seeds

    slurm_files: Dict[str, List[Path]] = {}
    for problem in problem_classes:
        print("Generating slurm files for problem {problem}")
        essence_file = find_essence_file(problem)
        param_files = get_param_files(problem)
        problem_stem = problem.stem
        slurm_files[problem_stem] = []
        for param in param_files:
            for seed in seeds:
                slurm_files[problem.stem].append(
                    generate_slurm_file(
                        problem_stem, essence_file, param, seed, args.time_limit
                    )
                )
        model_essence_file(problem_stem, essence_file)

    if args.batch_files:
        print("Batching slurm files...")
        for problem, files in slurm_files.items():
            for file in files:
                while get_queue_size() >= args.max_queue:
                    print(f"Queue full ({args.max_queue} jobs). Waiting...")
                    time.sleep(60)

                submit_job(file)
                print(f"Submitted job: {file}")
            
            # Potentially Bin the files here after submitting all the jobs for a problem



if __name__ == "__main__":
    main()
