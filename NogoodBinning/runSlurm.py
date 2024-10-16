#!/usr/bin/env python
import os
import subprocess
import time
import argparse
from typing import List

def get_queue_size() -> int:
    """Get the current number of jobs in the Slurm queue."""
    user = os.getenv("USER")
    if user:
        result = subprocess.run(["squeue", "-h", "-u", user], capture_output=True, text=True)
        return len(result.stdout.split("\n")) - 1
    else:
        raise EnvironmentError

def submit_job(script_path) -> None:
    """Submit a job to Slurm."""
    subprocess.run(["sbatch", script_path])

def main():
    arg_parser = argparse.ArgumentParser(description="Submit all Slurm jobs in a directory.")
    arg_parser.add_argument("-s", "--slurm-files", type=str, required=True, help="Directory containing the slurm files.")
    arg_parser.add_argument("-q", "--max-queue", type=int, required=False, default=5000, help="Maximum number of queued jobs. Default is 5000.")
    args = arg_parser.parse_args()

    # Get list of all bash scripts
    scripts: List[str] = [f for f in os.listdir(args.slurm_files) if f.endswith(".sh")]
    
    for script in scripts:
        script_path = os.path.join(args.slurm_files, script)
        
        # Wait if queue is full
        while get_queue_size() >= args.max_queue:
            print(f"Queue full ({args.max_queue} jobs). Waiting...")
            time.sleep(60)  # Wait for 1 minute before checking again
        
        # Submit the job
        submit_job(script_path)
        print(f"Submitted job: {script}")

if __name__ == "__main__":
    main()
