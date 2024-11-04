import re
import logging
from typing import Dict, List
from pathlib import Path

"""
# WCTIME: wall clock time in seconds
WCTIME=0.023697
# CPUTIME: CPU time in seconds (USERTIME+SYSTEMTIME)
CPUTIME=0.021245
# USERTIME: CPU time spent in user mode in seconds
USERTIME=0.015175
# SYSTEMTIME: CPU time spent in system mode in seconds
SYSTEMTIME=0.00607
# CPUUSAGE: CPUTIME/WCTIME in percent
CPUUSAGE=89.6528
# MAXVM: maximum virtual memory used in KiB
MAXVM=0
# TIMEOUT: did the solver exceed the time limit?
TIMEOUT=false
# MEMOUT: did the solver exceed the memory limit?
MEMOUT=false
"""

patterns = [
    ("WallClockTime", re.compile(r"WCTIME=([0-9]*.[0-9]*|0|[0-9]*)")),
    ("CPUTime", re.compile(r"CPUTIME=([0-9]*.[0-9]*|0|[0-9]*)")),
    ("UserTime", re.compile(r"USERTIME=([0-9]*.[0-9]*|0|[0-9]*)")),
    ("SystemTime", re.compile(r"SYSTEMTIME=([0-9]*.[0-9]*|0|[0-9]*)")),
    ("CPUUsage", re.compile(r"CPUUSAGE=([0-9]*.[0-9]*|0|[0-9]*)")),
    ("Timeout", re.compile(r"TIMEOUT=(false|true)")),
]


class RunSolverStats:
    def __init__(
        self,
        time_out: bool,
        real_time: float,
        cpu_time: float,
        cpu_user_time: float,
        cpu_system_time: float,
        cpu_usage: float,
    ):
        self._time_out = time_out
        self._real_time = real_time
        self._cpu_time = cpu_time
        self._cpu_user_time = cpu_user_time
        self._cpu_system_time = cpu_system_time
        self._cpu_usage = cpu_usage

    def time_out(self):
        return self._time_out

    def get_real_time(self):
        return self._real_time

    def get_cpu_time(self):
        return self._cpu_time

    def __str__(self):
        return f"""
            'RealTime' : {self._real_time},
            'CPUTime' : {self._cpu_time},
            'CPUUserTime' : {self._cpu_user_time},
            'CPUSystemTime' : {self._cpu_system_time},
            'CPUUsage' : {self._cpu_usage},
            'Timeout' : {self._time_out}
        """

    def keys(self):
        return {
            "RealTime": self._real_time,
            "CPUTime": self._cpu_time,
            "CPUUserTime": self._cpu_user_time,
            "CPUSystemTime": self._cpu_system_time,
            "CPUUsage": self._cpu_usage,
            "Timeout": self._time_out,
        }


class RunSolver:
    def __init__(self, thread_id: int, stage_name: str) -> None:
        self.output_file: Path = Path(f"{stage_name}_{thread_id}.txt")

    def _output_file(self) -> str:
        with open(self.output_file, "r") as output:
            run_solver_stats = output.read()

        self.output_file.unlink()
        return run_solver_stats

    def grab_runsolver_stats(self) -> RunSolverStats:
        output = self._output_file()
        matches = {}
        for line in output.splitlines():
            for matcher in patterns:
                if matcher[1].match(line):
                    match = matcher[1].match(line)
                    matches[matcher[0]] = match.group(1)  # type: ignore
                    break
        try:
            to_return = RunSolverStats(
                matches["Timeout"] == "true",
                float(matches["WallClockTime"]),
                float(matches["CPUTime"]),
                float(matches["UserTime"]),
                float(matches["SystemTime"]),
                float(matches["CPUUsage"]),
            )

            return to_return
        except Exception as e:
            logging.info(f"RunSolver Output: {output}")
            logging.info(f"RunSolver Matches: {matches}")
            logging.info(e)
            raise Exception()

    # -w /dev/null redirects the watcher output to /dev/null to prevent it filling up memory buffers
    def generate_runsolver_command(self, command, total_time) -> List[str]:
        return [
            "runsolver",
            "-v",
            self.output_file,
            "-d 0",
            f"-W {total_time}",
            "-w /dev/null",
        ] + command


def translate_to_runsolver_stats(result: Dict[str, str], prefix: str) -> RunSolverStats:
    return RunSolverStats(
        bool(result[f"{prefix}_Timeout"]),
        float(result[f"{prefix}_RealTime"]),
        float(result[f"{prefix}_CPUTime"]),
        float(result[f"{prefix}_CPUUserTime"]),
        float(result[f"{prefix}_CPUSystemTime"]),
        float(result[f"{prefix}_CPUUsage"]),
    )
