import json
import logging, argparse, yaml
import random
from typing import Any, Dict
from pathlib import Path
import shutil
import os
from StreamlinerSelection import StreamlinerSelection
from Portfolio.PortfolioEval import PortfolioEval
from Portfolio.HydraPortfolio import HydraPortfolio
from Toolchain.Solver import Solver
from Search.StreamlinerModelStats import StreamlinerModelStats
from Search.BaseModelStats import BaseModelStats
import Toolchain.SolverFactory as SolverFactory
from sklearn.model_selection import KFold
from enum import Enum

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s:\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Define the Mode enum
class Mode(Enum):
    TEST = "test"
    TRAIN = "train"
    EVAL = "eval"

    def __str__(self):
        return self.value


def main():
    parser = argparse.ArgumentParser(description="Streamlining an Essence Spec")

    parser.add_argument(
        "-m",
        "--mode",
        type=Mode,
        choices=list(Mode),
        required=True,
        help="Mode of operation: train, test, eval",
    )

    initial_args = parser.parse_known_args()[0]

    parser.add_argument(
        "-w",
        "--working_dir",
        type=Path,
        help="Working directory",
        required=True,
    )

    parser.add_argument(
        "-i",
        "--instance_dir",
        type=Path,
        help="Directory containing instances for streamliner evaluation",
        required=True,
    )

    if initial_args.mode == Mode.TEST:
        parser.add_argument(
            "-p",
            "--portfolio",
            type=Path,
            required=True,
            help="The portfolio that will be used for testing",
        )

    if initial_args.mode == Mode.TRAIN:
        parser.add_argument(
            "-s",
            "--streamliners_to_use",
            type=StreamlinerSelection,
            choices=list(StreamlinerSelection),
            required=True,
            help="Streamliners that should be used",
        )

        parser.add_argument(
            "-f",
            "--fold_num",
            type=int,
            help="Which fold to run for training",
            required=True,
        )

    # Parse all arguments
    args = parser.parse_args()
    working_directory: Path = args.working_dir
    instance_dir: Path = args.instance_dir
    # Find the only essence file in the working directory
    essence_spec = list(working_directory.glob("*.essence"))[0]

    with open(f"{working_directory}/conf.yaml", "r") as conf_file:
        conf = yaml.safe_load(conf_file)
        conf["working_directory"] = working_directory
        conf["instance_directory"] = instance_dir
        if args.mode == Mode.TRAIN:
            conf["streamliners_to_use"] = args.streamliners_to_use

    solver: Solver = SolverFactory.get_solver(conf.get("solver"))

    match args.mode:
        case Mode.TEST:
            test(
                args.portfolio,
                working_directory,
                instance_dir,
                essence_spec,
                conf,
                solver,
            )
        case Mode.TRAIN:
            train(
                args.fold_num,
                working_directory,
                instance_dir,
                essence_spec,
                conf,
                solver,
            )
        case Mode.EVAL:
            eval(working_directory, instance_dir, essence_spec, conf, solver)


def train(
    fold_num: int,
    working_directory: Path,
    instance_dir: Path,
    essence_spec: Path,
    conf: Dict[str, Any],
    solver: Solver,
):
    _, _, instances = next(os.walk(instance_dir))
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    train_test_iterator = k_fold.split(instances)

    train_set, test_set = list(train_test_iterator)[fold_num]
    train_dir = instance_dir / "Train"
    test_dir = instance_dir / "Test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    random.seed(42)

    for j in train_set:
        shutil.copyfile(
            instance_dir / instances[j],
            train_dir / instances[j],
        )

    for j in test_set:
        shutil.copyfile(
            instance_dir / instances[j],
            test_dir / instances[j],
        )

    conf["instance_directory"] = train_dir

    baseModelStats = BaseModelStats(
        working_directory / "BaseModelResults.csv",
        working_directory,
        train_dir,
        solver,
    )
    baseModelStats.evaluate_training_instances(essence_spec, conf)

    streamlinerModelStats = StreamlinerModelStats(
        working_directory / f"StreamlinerModelStatsFold{fold_num}.csv",
        solver,
    )

    portfolio_builder = HydraPortfolio(
        essence_spec, baseModelStats, streamlinerModelStats, conf
    )

    os.makedirs(working_directory / "portfolios", exist_ok=True)
    portfolio_builder.build_portfolio(
        working_directory / "portfolios" / f"PortfolioFold{fold_num}.json"
    )


def test(
    portfolio_loc: Path,
    working_directory: Path,
    instance_dir: Path,
    essence_spec: Path,
    conf: Dict[str, Any],
    solver: Solver,
):
    with open(portfolio_loc, "r") as portfolio_f:
        portfolio = json.load(portfolio_f)
        conf["portfolio"] = portfolio

    baseModelStats = BaseModelStats(
        working_directory / "BaseModelResultsTest.csv",
        working_directory,
        instance_dir,
        solver,
    )

    baseModelStats.evaluate_training_instances(essence_spec, conf)

    streamlinerModelStats = StreamlinerModelStats(
        working_directory / "StreamlinerModelStatsTest.csv",
        solver,
    )

    evaluator = PortfolioEval(essence_spec, baseModelStats, streamlinerModelStats, conf)
    evaluator.evaluate()


def eval(working_directory, instance_dir, essence_spec, conf, solver):
    baseModelStats = BaseModelStats(
        working_directory / "BaseModelResults.csv",
        working_directory,
        instance_dir,
        solver,
    )
    baseModelStats.evaluate_training_instances(essence_spec, conf)


if __name__ == "__main__":
    main()
