import logging
from pathlib import Path
from Toolchain.Conjure import Conjure
from Toolchain.RunSolver import RunSolver
from Toolchain.SavileRow import SavileRow
from Toolchain.InstanceStats import InstanceStats
from Toolchain.StageTimeout import StageTimeout
import threading
import subprocess
from subprocess import TimeoutExpired
import os, math
from typing import Callable, List, Optional
from Toolchain.Solver import Solver
from functools import partial


class Stage:

    def __init__(
        self,
        name: str,
        stage_callable: Callable,
        parse_std_out_callable: Callable,
        parse_std_err_callable: Callable,
        args: tuple,
    ):
        logging.info(f"Stage callable {stage_callable}")
        self.name: str = name
        self.stage_callable: Callable = stage_callable
        self.args: tuple = args
        self.parse_std_out: Callable = parse_std_out_callable
        self.parse_std_err: Callable = parse_std_err_callable

    def get_name(self):
        return self.name


class Pipeline:
    def __init__(
        self,
        eprime_model: Path,
        working_directory: Path,
        instance_dir: Path,
        essence_param_file: str,
        solver: Solver,
        event: threading.Event,
        total_time: float,
        streamliners: Optional[str] = None,
    ) -> None:
        self.eprime_model = eprime_model
        self.working_directory = working_directory
        self.instance_dir = instance_dir
        raw_eprime_model = os.path.basename(eprime_model).split(".")[0]
        self.total_time = total_time
        self.essence_param_file = essence_param_file
        self.raw_instance = os.path.basename(essence_param_file).split(".")[0]
        if streamliners:
            self.output_eprime_param = (
                f"{raw_eprime_model}-{self.raw_instance}-{streamliners}.eprime-param"
            )
        else:
            self.output_eprime_param = (
                f"{raw_eprime_model}-{self.raw_instance}.eprime-param"
            )
        self.savilerow_output = solver.get_savilerow_output_file(
            self.eprime_model, self.raw_instance, streamliners
        )

        # logging.info(self.savilerow_output)

        self.solver = solver
        self.event = event
        conjure: Conjure = Conjure()
        savilerow: SavileRow = SavileRow()
        self.conjure_stage = Stage(
            "conjure",
            conjure.translate_essence_param,
            conjure.parse_std_out,
            conjure.parse_std_err,
            (
                self.instance_dir,
                eprime_model,
                essence_param_file,
                self.output_eprime_param,
            ),
        )
        self.savilerow_stage = Stage(
            "savilerow",
            savilerow.formulate,
            savilerow.parse_std_out,
            savilerow.parse_std_err,
            (
                self.eprime_model,
                self.output_eprime_param,
                solver,
                self.savilerow_output,
            ),
        )
        self.solver_stage = Stage(
            solver.get_solver_name(),
            solver.execute,
            partial(solver.parse_std_out, self.savilerow_output),
            solver.parse_std_err,
            (self.savilerow_output,),
        )

    def _run_stage(
        self, stage: Stage, runsolver_command: List[str], instance_stats
    ) -> tuple[bytes, bytes]:
        logging.debug(f"Running stage {stage.get_name()}")
        process = subprocess.Popen(
            runsolver_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        while True:
            try:
                outs, errs = process.communicate(timeout=5)

                if outs:
                    stage.parse_std_out(outs, instance_stats)
                if errs:
                    stage.parse_std_err(errs, instance_stats)
                    logging.debug(errs)

                return outs, errs
            except TimeoutExpired:
                logging.debug("Process still running.")

            if self.event.is_set():
                logging.debug(
                    f"Flag is set, killing current command {runsolver_command}"
                )
                instance_stats.set_killed()
                # Kill Process
                process.kill()
                # Clear out any last buffers
                outs, errs = process.communicate()
                return outs, errs

    def _call(self, stage, instance_stats):
        if self.event.is_set():
            logging.debug("Event is set, skipping stage")
            return
        command = stage.stage_callable(*stage.args)
        runsolver: RunSolver = RunSolver(threading.get_ident(), stage.get_name())
        runsolver_command = runsolver.generate_runsolver_command(
            command, max(int(math.ceil(self.total_time)), 1)
        )
        logging.debug(
            f"Executing {runsolver_command} on thread {threading.get_ident()}"
        )

        outs, errs = self._run_stage(stage, runsolver_command, instance_stats)
        logging.debug("Ran the command")

        # Grab runsolver related stats
        runsolver_stats = runsolver.grab_runsolver_stats()
        instance_stats.add_stage_stats(stage.get_name(), runsolver_stats)

        self.total_time -= runsolver_stats.get_real_time()

        if runsolver_stats.time_out():
            logging.debug("Stage timed out")
            instance_stats.set_timeout()
            raise StageTimeout(f"{stage.get_name()} ran out of time")

        logging.debug(f"Total time left {self.total_time}")
        return outs, errs

    def execute(self):
        logging.debug(
            f"Executing pipeline {self.raw_instance} with model {self.eprime_model}"
        )

        instance_stats = InstanceStats()

        try:
            # Translate Parameter file
            self._call(self.conjure_stage, instance_stats)
            # Savilerow
            self._call(self.savilerow_stage, instance_stats)
            # Solver
            self._call(self.solver_stage, instance_stats)

        except StageTimeout as e:
            instance_stats.set_satisfiable(False)
        return instance_stats
