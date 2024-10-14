import logging
import pathlib
from typing import Dict, List
import re, os
from Toolchain.InstanceStats import InstanceStats
from Toolchain.Solver import Solver

patterns: List[tuple[str, re.Pattern[str]]] = [
    ("chronological", re.compile(r"chronological:\s+([0-9]*)")),
    ("conflicts", re.compile(r"conflicts:\s+([0-9]*)")),
    ("decisions", re.compile(r"decisions:\s+([0-9]*)")),
    ("learned", re.compile(r"learned:\s+([0-9]*)")),
    ("learned_lits", re.compile(r"learned_lits:\s+([0-9]*)")),
    ("minimized", re.compile(r"minimized:\s+([0-9]*)")),
    ("shrunken", re.compile(r"shrunken:\s+([0-9]*)")),
    ("minishrunken", re.compile(r"minishrunken:\s+([0-9]*)")),
    ("ofts", re.compile(r"otfs:\s+([0-9]*)")),
    ("propagations", re.compile(r"propagations:\s+([0-9]*)")),
    ("reduced", re.compile(r"reduced:\s+([0-9]*)")),
    ("rephased", re.compile(r"rephased:\s+([0-9]*)")),
    ("restarts", re.compile(r"restarts:\s+([0-9]*)")),
    ("stabilizing", re.compile(r"stabilizing:\s+([0-9]*)")),
    ("subsumed", re.compile(r"subsumed:\s+([0-9]*)")),
    ("strengthened", re.compile(r"strengthened:\s+([0-9]*)")),
    ("trail_reuses", re.compile(r"trail reuses:\s+([0-9]*)")),
    (
        "time",
        re.compile(r"total process time since initialization:\s+([0-9]*.[0-9]*|0|0.0)"),
    ),
]


class Cadical(Solver):
    def __init__(self):
        Solver.__init__(self, "cadical", "sat", "-out-sat")

    def get_savilerow_output_file(self, eprime_model: str, raw_instance: str) -> str:
        raw_eprime_model = os.path.basename(eprime_model).split(".")[0]
        return f"{raw_eprime_model}-{raw_instance}.dimacs"

    def execute(self, output_file: str) -> List[str]:
        random_seed = 42
        return ["cadical", "-v", f"--seed={random_seed}", output_file]

    def parse_std_out(
        self,  output_file: str, out: bytes, instance_stats: InstanceStats
    ) -> Dict[str, str | bool]:
        output = out.decode("ascii")
        stats: Dict[str, str | bool] = {}
        stats["satisfiable"] = False
        for line in output.splitlines():
            for matcher in patterns:
                if match := matcher[1].search(line):
                    if match:
                        if "total process time since" in line:
                            stats["time"] = match[1]
                        elif "trail reuses" in line:
                            stats["trail_reuses"] = match[1]
                        else:
                            stats[matcher[0]] = match[1]
                        break

            if "s SATISFIABLE" in line:
                stats["satisfiable"] = True

        instance_stats.add_solver_output(stats)
        instance_stats.set_solver_name(self.get_solver_name())
        instance_stats.set_satisfiable(bool(stats["satisfiable"]))
        
        logging.debug(f"Removing {output_file}")
        pathlib.Path(os.path.join("/", output_file)).unlink(missing_ok=True)
        
        return stats

    def parse_std_err(self, out: bytes, instance_stats: InstanceStats):
        return

    def get_stat_names(self) -> List[str]:
        return [
            "solver_satisfiable",
            *[f"solver_{p[0]}" for p in patterns],
            "solver_randomSeed",
            "solver_time",
        ]


"""
s SATISFIABLE
c --- [ statistics ] ---------------------------------------------------------
c 
c chronological:              1164        40.12 %  of conflicts
c conflicts:                  2901      6685.92    per second
c decisions:                 19664     45319.51    per second
c learned:                    2671        92.07 %  per conflict
c learned_lits:             582374       100.00 %  learned literals
c minimized:                     0         0.00 %  learned literals
c shrunken:                 326276        56.03 %  learned literals
c minishrunken:              15729         2.70 %  learned literals
c otfs:                         12         0.41 %  of conflict
c propagations:            1097474         2.53 M  per second
c reduced:                     128         4.41 %  per conflict
c rephased:                      1      2901.00    interval
c restarts:                      5       580.20    interval
c stabilizing:                   1        65.53 %  of conflicts
c subsumed:                    508         0.04 %  of all clauses
c strengthened:                 12         0.00 %  of all clauses
c trail reuses:                  0         0.00 %  of incremental calls

"""
