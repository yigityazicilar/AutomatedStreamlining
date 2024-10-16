import logging
import os, glob, sys
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
        base_stats_file: str,
        working_dir: str,
        training_instance_dir: str,
        solver: Solver,
    ) -> None:
        self.base_stats_file: str = base_stats_file
        self.training_instances: List[str] = [
            instance.split("/")[-1]
            for instance in glob.glob(f"{training_instance_dir}/*.param")
        ]
        self.training_df: pd.DataFrame = self._load_base_stats(base_stats_file, solver)
        self.working_dir: str = working_dir
        self.instance_dir: str = training_instance_dir
        self.solver: Solver = solver
        self.conjure: Conjure = Conjure()

    def _callback(self, instance: str, result: InstanceStats) -> None:
        logging.debug(f"Callback for {instance} run. Stage {result.get_stages()}.")
        instance_stages = result.get_stages()
        combined_keys: Dict[str, str | bool | int | float] = {
            "Instance": instance.split("/")[-1],
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

    def _load_base_stats(self, output_file: str, solver: Solver) -> pd.DataFrame:
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

    def evaluate_training_instances(self, essence_spec: str, conf: Dict[str, Any]):
        # Evaluate the base specification across the training instances
        base_combination = None
        instances_to_eval = set(self.training_instances) - set(
            self.training_df["Instance"]
        )
        if not instances_to_eval:
            return self.training_df

        generated_models = self.conjure.generate_streamlined_models(
            essence_spec, base_combination, output_dir=os.path.join(self.working_dir, "conjure-output")
        )
        
        streamlinerEval = SingleModelStreamlinerEvaluation(
            generated_models[0],
            self.working_dir,
            self.instance_dir,
            instances_to_eval,
            pd.DataFrame(),
            self.solver,
            concurrent.futures.ThreadPoolExecutor(max_workers=conf["executor"]["num_cores"]),
            3600 * 1.5,
            lambda x: x
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
