import json
import logging
from pathlib import Path
from typing import Any, List, Dict, Optional, Set
import os, json

import numpy as np
import pandas as pd
from Toolchain.InstanceStats import InstanceStats
from Util import unwrap

AVG_APPLIC_KEY = "AvgApplic"
OVERALL_SOLVING_TIME_REDUCTION_KEY = "OverallSolvingTimeReduction"
MEAN_REDUCTION_KEY = "MeanReduction"
MEDIAN_REDUCTION_KEY = "MedianReduction"
STD_DEV_KEY = "StandardDeviation"
QUANTILES_KEY = "Quantiles"


class HydraEval:
    """
    With the inputted portfolio we need to know how well each streamliner did on each instance as with Hydra
    you take the performance of the portfolio and you union this with the
    """

    def __init__(
        self,
        overall_portfolio: Dict[
            str, Dict[str, Dict[str, float | np.floating[Any] | list[np.floating[Any]]]]
        ],
        best_instance_results: Dict[str, InstanceStats],
    ):
        self.__overall_portfolio = overall_portfolio
        self.__best_instance_results = best_instance_results
        self.__cur_portfolio: Dict[
            str, Dict[str, float | np.floating[Any] | list[np.floating[Any]]]
        ] = {}

    def _test(
        self,
        current_result: InstanceStats,
        best_result: Optional[InstanceStats],
    ) -> bool:
        """
        Test whether our current result is superior to any of the streamliners in our portfolio on this given instance
        :param current_result: The result attained from StreamlinerEval
        :param best_result: The best result from our current portfolio
        :return: True if our current result dominates, False otherwise
        """
        # If there is no streamliner in the portfolio that is satisfiable on this instance, the best result will be None
        if best_result == None:
            return True

        if (
            current_result.satisfiable()
            and best_result.satisfiable()
            and current_result.total_time() <= best_result.total_time()
        ):
            return True
        elif current_result.satisfiable() and not best_result.satisfiable():
            return True
        else:
            return False

    def _dominated(self, x: Dict, y: Dict) -> int:
        """
        Test whether x dominates y on Average Applicability and Mean Reduction
        :param x: Dict of objective values
        :param y: Dict of objective values
        :return: 1 if x dominates, 0 otherwise
        """
        return int(
            (
                x[AVG_APPLIC_KEY] > y[AVG_APPLIC_KEY]
                and x[MEAN_REDUCTION_KEY] >= y[MEAN_REDUCTION_KEY]
            )
            or (
                x[AVG_APPLIC_KEY] >= y[AVG_APPLIC_KEY]
                and x[MEAN_REDUCTION_KEY] > y[MEAN_REDUCTION_KEY]
            )
        )

    def _non_dominated(self, objective_values: Dict, cur_portfolio: Dict) -> bool:
        """
        Test whether the objective values of the current streamliner are dominated by the portfolio
        :param objective_values: Objective values of current streamliner evaluated
        :param cur_portfolio: Current non-dominated portfolio of streamliners
        :return: sum of the number of streamliners in the portfolio that dominate our current streamliner
        """
        if not cur_portfolio:
            return True
        else:
            return (
                sum(
                    map(
                        lambda x: self._dominated(x, objective_values),
                        cur_portfolio.values(),
                    )
                )
                == 0
            )

    def _remove_dominated_combinations(
        self, objective_values: Dict, cur_portfolio: Dict
    ) -> Dict:
        return dict(
            filter(
                lambda x: not self._dominated(objective_values, x[1]),
                cur_portfolio.items(),
            )
        )

    def combine_results(self, results: Dict[str, InstanceStats]):
        combined_results: Dict[str, InstanceStats] = {}
        for x in results:
            if self._test(results[x], self.__best_instance_results.get(x)):
                combined_results[x] = results[x]
            else:
                combined_results[x] = unwrap(self.__best_instance_results.get(x))

        return combined_results

    def eval_streamliner(
        self,
        streamliner_combo: str,
        results: Dict[str, InstanceStats],
        training_results: pd.DataFrame,
    ) -> int:
        logging.info("Hydra: Evaluating Streamliner")
        if len(results) == 0:
            return 0

        # Combine the results of the current portfolio with the new streamliner
        combined_results = self.combine_results(results)
        objective_values = self._objective_values(combined_results, training_results)

        logging.info(f"{streamliner_combo} has results: {objective_values}")
        if not self.exists_in_portfolio(streamliner_combo) and self._non_dominated(
            objective_values, self.__cur_portfolio
        ):
            logging.info(
                f"Streamliner {streamliner_combo} is non-dominated in the portfolio so adding"
            )
            self.__cur_portfolio[streamliner_combo] = objective_values
            self.__cur_portfolio = self._remove_dominated_combinations(
                objective_values, self.__cur_portfolio
            )
            return 1
        else:
            logging.info(
                f"Streamliner {streamliner_combo} is dominated. Not adding to portfolio"
            )
            return 0

    def exists_in_portfolio(self, streamliner_combo: str) -> bool:
        for round in self.__overall_portfolio:
            if streamliner_combo in self.__overall_portfolio[round].keys():
                return True
        return False

    def save_portfolio_name(self, filename: Path) -> None:
        with open(filename, "w") as portfolio:
            portfolio.write(json.dumps(self.__cur_portfolio))

    def portfolio(
        self,
    ) -> Dict[str, Dict[str, float | np.floating[Any] | list[np.floating[Any]]]]:
        return self.__cur_portfolio

    def _objective_values(
        self,
        results: Dict[str, InstanceStats],
        training_results: pd.DataFrame,
    ) -> Dict[str, float | np.floating[Any] | list[np.floating[Any]]]:
        total_number_of_instances: int = len(training_results["Instance"].unique())

        applicability: float = (
            sum([int(results[x].satisfiable()) for x in results])
            / total_number_of_instances
        )

        total_solving_time: float = sum(
            [results[x].solver_time() for x in results if results[x].satisfiable()]
        )

        reductions: List[float] = [
            (
                float(
                    training_results[training_results["Instance"] == x][
                        "solver_time"
                    ].values[0]
                )
                - results[x].solver_time()
            )
            / float(
                training_results[training_results["Instance"] == x][
                    "solver_time"
                ].values[0]
            )
            for x in results
            if results[x].satisfiable()
        ]

        satisfiable_instances: Set[str] = set(
            [x for x in results if results[x].satisfiable()]
        )
        base_total_time: float = training_results[
            training_results["Instance"].isin(satisfiable_instances)
        ]["solver_time"].sum()

        # This is calculating an overall reduction in cumulative time on all sat instances based upon total time of the pipeline (Conjure + SR + Solver)
        if applicability > 0:
            overall_solving_time_reduction = (
                float(base_total_time) - float(total_solving_time)
            ) / float(base_total_time)
            mean_reduction = np.mean(reductions)
            median_reduction = np.median(reductions)
            std_dev = np.std(reductions)
            quantiles = np.quantile(reductions, [0.25, 0.5, 0.75]).tolist()

        else:
            mean_reduction = 0
            median_reduction = 0
            std_dev = 0
            overall_solving_time_reduction = 0
            quantiles = []

        return {
            AVG_APPLIC_KEY: applicability,
            OVERALL_SOLVING_TIME_REDUCTION_KEY: overall_solving_time_reduction,
            MEAN_REDUCTION_KEY: mean_reduction,
            MEDIAN_REDUCTION_KEY: median_reduction,
            STD_DEV_KEY: std_dev,
            QUANTILES_KEY: quantiles,
        }
