import numpy as np
from Portfolio.HydraEval import HydraEval
from Search.MOMCTS import MOMCTS
from typing import List, Dict, Optional, Set, Any
from Search.BaseModelStats import BaseModelStats
from Search.StreamlinerModelStats import StreamlinerModelStats
from Toolchain.InstanceStats import InstanceStats, translate_to_instance_stats
import logging
import pandas as pd
from Util import unwrap


class HydraPortfolio:

    def __init__(
        self,
        essence_spec: str,
        base_model_stats: BaseModelStats,
        streamliner_model_stats: StreamlinerModelStats,
        conf: Dict[str, Any],
        validation_instances: Optional[List[str]] = None,
    ):
        self._essence_spec = essence_spec
        self._training_results = base_model_stats.results()
        self._streamliner_model_stats = streamliner_model_stats
        self._validation_instances = validation_instances
        self.conf = conf
        self.num_rounds: int = unwrap(conf.get("hydra")).get("num_rounds")

    def build_portfolio(self, portfolio_name=None) -> Dict[str, Dict[str, Dict[str, float | np.floating[Any] | list[np.floating[Any]]]]]:
        cur_round: int = 0
        best_instance_results: Dict[str, InstanceStats] = {}
        overall_portfolio: Dict[str, Dict[str, Dict[str, float | np.floating[Any] | list[np.floating[Any]]]]] = {}
        while True:
            eval = HydraEval(overall_portfolio, best_instance_results, self.conf["working_directory"])
            search = MOMCTS(
                self._essence_spec,
                self._training_results,
                eval,
                self.conf,
                self._streamliner_model_stats,
            )

            search.search(portfolio_name)
            cur_portfolio = eval.portfolio()
            overall_portfolio[str(cur_round)] = cur_portfolio
            cur_round += 1
            # logging.info(overall_portfolio)
            best_instance_results = self.generate_best_instance_stats(overall_portfolio)

            if cur_round == self.num_rounds:
                logging.info(overall_portfolio)
                return overall_portfolio

    # Generate the best stats for the instances
    def generate_best_instance_stats(
        self, overall_portfolio: Dict[str, Dict[str, Dict[str, float | np.floating[Any] | list[np.floating[Any]]]]]
    ) -> Dict[str, InstanceStats]:
        best_instance_stats: Dict[str, InstanceStats] = {}
        df: pd.DataFrame = self._streamliner_model_stats.results()
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        streamliners_in_portfolio: Set[str] = set()
        for round in overall_portfolio.keys():
            [
                streamliners_in_portfolio.add(streamliner)
                for streamliner in overall_portfolio[round].keys()
            ]

        # Grab only the satisfiable runs
        df_sat = df[df["Satisfiable"]]
        # Grab only the results on our current portfolio
        df_sat = df_sat[df_sat["Streamliner"].isin(streamliners_in_portfolio)]

        # logging.info(df_sat.groupby('Streamliner')['Satisfiable'].sum())
        # logging.info(df_sat.groupby('Instance')['Satisfiable'].sum())

        instance_stats_df = df_sat.groupby("Instance").apply(
            lambda x: df_sat.loc[pd.to_numeric(x["solver_time"]).idxmin()]
        )
        instance_stats_df = instance_stats_df.reset_index(drop=True)

        for _, row in instance_stats_df.iterrows():
            instance = row["Instance"]
            best_instance_stats[instance] = translate_to_instance_stats(row)

        return best_instance_stats
