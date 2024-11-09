import logging
from pathlib import Path
import threading

import pandas as pd
from Toolchain.Pipeline import Pipeline
from Toolchain.InstanceStats import InstanceStats
from Toolchain.Solver import Solver
import concurrent.futures
from threading import Event
from typing import Callable, Dict, Optional, Set


def _default_callback(instance: str, data: str):
    logging.debug(instance, data)


def _default_err_callback(instance: str, err: str):
    logging.exception(err)


class SingleModelStreamlinerEvaluation:
    def __init__(
        self,
        model: Path,
        instances: Set[Path],
        stats: pd.DataFrame,
        solver: Solver,
        executor: concurrent.futures.ThreadPoolExecutor,
        event: threading.Event,
        time: float | Callable[[float], float],
        streamliner: Optional[str] = None,
    ) -> None:
        self.model = model
        self.instances = instances
        self.stats = stats
        self.solver = solver
        self.executor = executor
        self.time = time
        self.event = event
        self.streamliner = streamliner

    def generate_pipeline(self, training_instance: Path) -> Pipeline:
        logging.debug(
            f"Generating pipeline for {training_instance} and model {self.model}"
        )

        if callable(self.time):
            total_time: float = self.time(
                list(
                    self.stats[self.stats["Instance"] == training_instance.name][
                        "TotalTime"
                    ]
                )[0]
            )
        else:
            total_time: float = self.time

        return Pipeline(
            self.model,
            training_instance,
            self.solver,
            self.event,
            total_time,
            self.streamliner,
        )

    def execute(
        self,
        callback: Callable = _default_callback,
        error_callback: Callable = _default_err_callback,
    ) -> Dict[str, InstanceStats]:
        if len(self.instances) == 0:
            return {}

        mappings: Dict[str, Pipeline] = {}
        for instance in self.instances:
            mappings[instance.name] = self.generate_pipeline(instance)

        results: Dict[str, InstanceStats] = {}

        futures: Dict[concurrent.futures.Future[InstanceStats], str] = {
            self.executor.submit(pipeline.execute): instance
            for instance, pipeline in mappings.items()
        }

        for future in concurrent.futures.as_completed(futures):
            if self.event.is_set():
                break

            instance = futures[future]
            try: 
                data: InstanceStats = future.result()
                results[instance] = data
                # Call the callback method passed in
                callback(instance, data)
            except Exception as exc:
                error_callback(instance, exc)

        return results
