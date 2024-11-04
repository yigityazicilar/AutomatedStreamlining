import logging
from pathlib import Path
import os, glob, sys
import threading
from matplotlib.pylab import f
import pandas as pd
from Toolchain.InstanceStats import InstanceStats
from SingleModelStreamlinerEvaluation import SingleModelStreamlinerEvaluation
from Toolchain.Conjure import Conjure
from Toolchain.Solver import Solver
from typing import Any, Dict, List
import concurrent.futures


class BaseModelStats:
    def __init__(
        self,
        base_stats_file: Path,
        working_dir: Path,
        training_instance_dir: Path,
        solver: Solver,
    ) -> None:
        self.base_stats_file: Path = base_stats_file
        self.training_instance_paths: List[Path] = [
            Path(instance)
            for instance in glob.glob(str(training_instance_dir / "*.param"))
        ]
        self.training_instances: List[str] = [
            instance.stem for instance in self.training_instance_paths
        ]

        self.training_df: pd.DataFrame = self._load_base_stats(base_stats_file, solver)
        self.working_dir: Path = working_dir
        self.instance_dir: Path = training_instance_dir
        self.solver: Solver = solver
        self.conjure: Conjure = Conjure()
        self.event = threading.Event()

    def _callback(self, instance: Path, result: InstanceStats) -> None:
        logging.debug(f"Callback for {instance} run. Stage {result.get_stages()}.")
        instance_stages = result.get_stages()
        combined_keys: Dict[str, str | bool | int | float] = {
            "Instance": instance.stem,
            "TotalTime": result.total_time(),
            "Satisfiable": result.satisfiable(),
            "Killed": result.killed(),
            "TimeOut": result.timeout(),
            "Solver": self.solver.get_solver_name(),
        }
        for stage_name, stage in instance_stages.items():
            for key, value in stage.keys().items():
                combined_keys[f"{stage_name}_{key}"] = value

        for key, value in result.solver_stats().items():
            combined_keys[f"solver_{key}"] = value
        combined_keys_df = pd.DataFrame(combined_keys, index=[0])
        if self.training_df.empty:
            self.training_df = combined_keys_df
        else:
            self.training_df = pd.concat(
                [self.training_df, combined_keys_df], ignore_index=True
            )
        # logging.info("Callback:", combined_keys)
        self.training_df.to_csv(self.base_stats_file, index=False)

    def _load_base_stats(self, output_file: Path, solver: Solver) -> pd.DataFrame:
        if os.path.exists(output_file):
            self.training_df = pd.read_csv(output_file)
            self.training_df = self.training_df[
                self.training_df["Instance"].isin(self.training_instances)
            ]
            return self.training_df
        else:
            return pd.DataFrame(
                columns=[
                    "Instance",
                    "TotalTime",
                    "Satisfiable",
                    "Killed",
                    "TimeOut",
                    "Solver",
                    "conjure_RealTime",
                    "conjure_CPUTime",
                    "conjure_CPUUserTime",
                    "conjure_CPUSystemTime",
                    "conjure_CPUUsage",
                    "conjure_Timeout",
                    "savilerow_RealTime",
                    "savilerow_CPUTime",
                    "savilerow_CPUUserTime",
                    "savilerow_CPUSystemTime",
                    "savilerow_CPUUsage",
                    "savilerow_Timeout",
                ]
                + [
                    f"{solver.get_solver_name()}_{x}"
                    for x in [
                        "RealTime",
                        "CPUTime",
                        "CPUUserTime",
                        "CPUSystemTime",
                        "CPUUsage",
                        "Timeout",
                    ]
                ]
                + solver.get_stat_names()
            )

    def evaluate_training_instances(self, essence_spec: Path, conf: Dict[str, Any]):
        # Evaluate the base specification across the training instances
        base_combination = None
        instances_to_eval_str = set(self.training_instances) - set(
            self.training_df["Instance"]
        )
        instances_to_eval = set()
        for instance in instances_to_eval_str:
            instances_to_eval.add(
                next(
                    instance_path
                    for instance_path in self.training_instance_paths
                    if instance in instance_path.stem
                )
            )

        if not instances_to_eval:
            return self.training_df

        generated_models = self.conjure.generate_streamlined_models(
            essence_spec,
            base_combination,
            output_dir=self.working_dir / "conjure-output",
        )

        streamlinerEval = SingleModelStreamlinerEvaluation(
            generated_models[0],
            instances_to_eval,
            pd.DataFrame(),
            self.solver,
            concurrent.futures.ThreadPoolExecutor(
                max_workers=conf["executor"]["num_cores"]
            ),
            3600 * 1.5,
            lambda x: x,
            self.event,
        )
        streamlinerEval.execute(self._callback, lambda _, err: logging.exception(err))

        if len(self.training_df["Instance"].unique()) != len(self.training_instances):
            logging.error(
                "Length of base training directory does not match number of training instances"
            )
            sys.exit(1)

        return self.training_df

    def results(self) -> pd.DataFrame:
        return self.training_df
