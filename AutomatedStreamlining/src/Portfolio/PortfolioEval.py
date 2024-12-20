import copy
from functools import partial
import glob
import logging
from pathlib import Path
import threading
from typing import Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from Toolchain.SolverFactory import get_solver
from SingleModelStreamlinerEvaluation import SingleModelStreamlinerEvaluation
from Toolchain.Conjure import Conjure
from Search.StreamlinerModelStats import StreamlinerModelStats
from Search.BaseModelStats import BaseModelStats
from Util import unwrap


#! Think about how we should calculate time taken for each instance.
#! Are we running the streamliner and the base model in a portfolio setting?
#! If so, the whichever is minimum should be doubled as both are run in parallel.
class PortfolioEval:
    def __init__(
        self,
        essence_spec: Path,
        base_model_stats: BaseModelStats,
        streamliner_stats: StreamlinerModelStats,
        conf: Dict[str, Any],
    ) -> None:
        self.essence_spec = essence_spec
        self.base_model_stats = copy.deepcopy(base_model_stats.results())
        self.streamliner_stats = streamliner_stats
        self.portfolio: Dict[str, Any] = unwrap(conf.get("portfolio"))
        self.conf = conf
        self.conjure: Conjure = Conjure()
        self.working_directory = unwrap(conf.get("working_directory"))
        self.instance_directory = unwrap(conf.get("instance_directory"))
        self.executor = ThreadPoolExecutor(max_workers=conf["executor"]["num_cores"])
        self.event = threading.Event()

    def evaluate(self):
        instances_to_eval: Set[Path] = set(
            [
                Path(param)
                for param in glob.glob(str(self.instance_directory / "*.param"))
            ]
        )

        for streamliner in self.portfolio.keys():
            if streamliner in set(
                self.streamliner_stats.results()["Streamliner"].unique()
            ):
                continue

            generated_models = self.conjure.generate_streamlined_models(
                self.essence_spec,
                streamliner,
                output_dir=self.working_directory / "conjure-output" / streamliner,
            )
            if len(generated_models) == 1:
                logging.info(generated_models)
                streamlinerEval = SingleModelStreamlinerEvaluation(
                    model=generated_models[0],
                    instances=instances_to_eval,
                    stats=self.base_model_stats,
                    solver=get_solver(unwrap(self.conf.get("solver"))),
                    executor=self.executor,
                    event=self.event,
                    time=lambda x: x * 1.05,
                    streamliner=streamliner,
                )

                callback = partial(self.streamliner_stats.callback, streamliner)
                # We now need to parse these results into some format that we can use as a reference point
                _ = streamlinerEval.execute(callback=callback)

        redFirst = sorted(
            self.portfolio.keys(),
            reverse=True,
            key=lambda x: float(self.portfolio[x]["OverallSolvingTimeReduction"]),
        )
        appFirst = sorted(
            self.portfolio.keys(),
            reverse=True,
            key=lambda x: float(self.portfolio[x]["AvgApplic"]),
        )
        streamliner_run_results = self.streamliner_stats.results()
        stats = {"RedFirst": [], "AppFirst": [], "Oracle": []}
        for instance in instances_to_eval:
            instance_base_time = list(
                self.base_model_stats[self.base_model_stats["Instance"] == instance][
                    "TotalTime"
                ]
            )[0]
            instance_runs_streamlined = streamliner_run_results[
                streamliner_run_results["Instance"] == instance
            ]

            # Reduction First
            total_time = 0
            satisfiable = False
            for streamliner in redFirst:
                streamlined_run = instance_runs_streamlined[
                    instance_runs_streamlined["Streamliner"] == streamliner
                ]
                satisfiable_with_streamliner = list(streamlined_run["Satisfiable"])[0]

                if satisfiable_with_streamliner:
                    total_time += list(streamlined_run["TotalTime"])[0]
                    satisfiable = True
                    break

                total_time += list(streamlined_run["TotalTime"])[0]

            if satisfiable:
                total_time = min(total_time, instance_base_time)
            # elif total_time < instance_base_time:
            #     total_time = (total_time * 2) + (instance_base_time - total_time)
            else:
                total_time = instance_base_time

            stats["RedFirst"].append((instance, instance_base_time / total_time))

            # Applicability First
            total_time = 0
            satisfiable = False
            for streamliner in appFirst:
                streamlined_run = instance_runs_streamlined[
                    instance_runs_streamlined["Streamliner"] == streamliner
                ]
                satisfiable_with_streamliner = list(streamlined_run["Satisfiable"])[0]

                if satisfiable_with_streamliner:
                    total_time += list(streamlined_run["TotalTime"])[0]
                    satisfiable = True
                    break

                total_time += list(streamlined_run["TotalTime"])[0]

            if satisfiable:
                total_time = min(total_time, instance_base_time)
            # elif total_time < instance_base_time:
            #     total_time = (total_time * 2) + (instance_base_time - total_time)
            else:
                total_time = instance_base_time

            stats["AppFirst"].append((instance, instance_base_time / total_time))

            # Oracle
            try:
                min_streamlined_time = min(
                    instance_runs_streamlined[instance_runs_streamlined["Satisfiable"]][
                        "TotalTime"
                    ]
                )

                stats["Oracle"].append(
                    (
                        instance,
                        instance_base_time
                        / (min(min_streamlined_time, instance_base_time)),
                    )
                )
            except:
                # total_time = sum(instance_runs_streamlined["TotalTime"])
                # if total_time < instance_base_time:
                #     total_time = (total_time * 2) + (instance_base_time - total_time)
                # else:
                total_time = instance_base_time

                stats["Oracle"].append((instance, instance_base_time / total_time))

        for portfolio_method in stats.keys():
            print(stats[portfolio_method])
            print(
                f"{portfolio_method}: {np.average([speedup for _, speedup in stats[portfolio_method]])}"
            )
