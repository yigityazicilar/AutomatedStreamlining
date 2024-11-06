import logging
from Toolchain.Solver import Solver
from typing import List


class SavileRow:
    def formulate(
        self, eprime_model: str, eprime_param: str, solver: Solver, sr_output_file: str
    ) -> List[str]:
        solver_flag = solver.get_savilerow_flag()
        command = [
            "savilerow",
            "-in-param",
            eprime_param,
            "-in-eprime",
            eprime_model,
            "-num-solutions",
            "1",
            f"-{solver_flag}",
            f"{solver.get_savilerow_output_flag()}",
            f"{sr_output_file}",
            "-preprocess",
            "None",
        ]
        return command

    def parse_std_out(self, out, instance_stats):
        return

    def parse_std_err(self, out, instance_stats):
        return
