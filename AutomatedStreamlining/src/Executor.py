import logging
import subprocess
import time
from typing import List, Optional


def callable(command: List[str]) -> Optional[tuple[bytes, bytes, float]]:
    try:
        logging.info(f"Command {command}")
        time_before = time.time()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        time_taken = time.time() - time_before

        logging.info(f"Time taken for command: {time_taken}")
        return out, err, time_taken
    except subprocess.CalledProcessError as e:
        if e.stdout:
            logging.error(f"STDOUT: Error during invocation: {e.stdout}")
        if e.stderr:
            logging.error(f"STDERR: Error during invocation: {e.stderr}")


def default_error_callback(error: str):
    logging.error(error)
