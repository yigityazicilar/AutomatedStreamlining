import json
import logging, argparse, yaml
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


# Define the Mode enum
class Mode(Enum):
    TRAIN = "train"
    TEST = "test"
    FOLD = "fold"

    def __str__(self):
        return self.value

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s:\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = argparse.ArgumentParser(description="Streamlining an Essence Spec")

parser.add_argument(
    "-m",
    "--mode",
    type=Mode,
    choices=list(Mode),
    required=True,
    help="Mode of operation: train, test, fold",
)

initial_args = parser.parse_known_args()[0]

if initial_args.mode != Mode.TEST:
    parser.add_argument(
        "-s",
        "--streamliners_to_use",
        type=StreamlinerSelection,
        choices=list(StreamlinerSelection),
        required=True,
        help="Streamliners that should be used",
    )

parser.add_argument(
    "-w",
    "--working_dir",
    type=str,
    help="Working directory",
    required=True,
)

parser.add_argument(
    "-i",
    "--instance_dir",
    type=str,
    help="Directory containing instances for streamliner evaluation",
    required=True,
)

if initial_args.mode == Mode.TEST:
    parser.add_argument(
        "-p",
        "--portfolio",
        type=str,
        required=True,
        help="The portfolio that will be used for testing",
    )

if initial_args.mode == Mode.FOLD:
    parser.add_argument(
        "-f",
        "--fold_num",
        type=int,
        help="Which fold to run for training",
        required=False,
    )

# Parse all arguments
args = parser.parse_args()
working_directory = args.working_dir
instance_dir = args.instance_dir
essence_spec = f"{working_directory}/model.essence"

with open(f"{working_directory}/conf.yaml", "r") as conf_file:
    conf = yaml.safe_load(conf_file)
    conf["working_directory"] = working_directory
    conf["instance_directory"] = instance_dir
    if args.mode != Mode.TEST:
        conf["streamliners_to_use"] = args.streamliners_to_use

solver: Solver = SolverFactory.get_solver(conf.get("solver"))

if args.mode == Mode.TRAIN:
    baseModelStats = BaseModelStats(
        f"{working_directory}/BaseModelResults.csv",
        working_directory,
        instance_dir,
        solver,
    )
    baseModelStats.evaluate_training_instances(
        f"{working_directory}/model.essence", conf
    )

    streamlinerModelStats = StreamlinerModelStats(
        f"{working_directory}/StreamlinerModelStats.csv",
        solver,
    )

    portfolio_builder = HydraPortfolio(
        essence_spec, baseModelStats, streamlinerModelStats, conf
    )

    portfolio_builder.build_portfolio()

if args.mode == Mode.TEST:
    with open(args.portfolio, "r") as portfolio_f:
        portfolio = json.load(portfolio_f)
        conf["portfolio"] = portfolio

    baseModelStats = BaseModelStats(
        f"{working_directory}/BaseModelResultsTestFold0.csv",
        working_directory,
        instance_dir,
        solver,
    )

    baseModelStats.evaluate_training_instances(
        f"{working_directory}/model.essence", conf
    )

    streamlinerModelStats = StreamlinerModelStats(
        f"{working_directory}/StreamlinerModelStatsTestFold0.csv",
        solver,
    )

    evaluator = PortfolioEval(essence_spec, baseModelStats, streamlinerModelStats, conf)
    evaluator.evaluate()

if args.mode == Mode.FOLD:
    _, _, instances = list(os.walk(instance_dir))[0]

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    train_test_iterator = k_fold.split(instances)

    if args.fold_num == None:
        for i, (train, test) in enumerate(train_test_iterator):
            train_dir = os.path.join(instance_dir, "Train")
            test_dir = os.path.join(instance_dir, "Test")
            os.mkdir(train_dir)
            os.mkdir(test_dir)

            for j in train:
                shutil.copyfile(
                    os.path.join(instance_dir, instances[j]),
                    os.path.join(train_dir, instances[j]),
                )

            for j in test:
                shutil.copyfile(
                    os.path.join(instance_dir, instances[j]),
                    os.path.join(test_dir, instances[j]),
                )

            conf["instance_directory"] = train_dir

            baseModelStats = BaseModelStats(
                f"{working_directory}/BaseModelResults.csv",
                working_directory,
                train_dir,
                solver,
            )
            baseModelStats.evaluate_training_instances(
                f"{working_directory}/model.essence", conf
            )

            streamlinerModelStats = StreamlinerModelStats(
                f"{working_directory}/StreamlinerModelStatsFold{i}.csv",
                solver,
            )

            portfolio_builder = HydraPortfolio(
                essence_spec, baseModelStats, streamlinerModelStats, conf
            )

            portfolio_round = portfolio_builder.build_portfolio(f"PortfolioFold{i}.json")
    else:
        train, test = list(train_test_iterator)[args.fold_num]
        train_dir = os.path.join(instance_dir, "Train")
        test_dir = os.path.join(instance_dir, "Test")
        os.mkdir(train_dir)
        os.mkdir(test_dir)

        for j in train:
            shutil.copyfile(
                os.path.join(instance_dir, instances[j]),
                os.path.join(train_dir, instances[j]),
            )

        for j in test:
            shutil.copyfile(
                os.path.join(instance_dir, instances[j]),
                os.path.join(test_dir, instances[j]),
            )

        conf["instance_directory"] = train_dir

        baseModelStats = BaseModelStats(
            f"{working_directory}/BaseModelResults.csv",
            working_directory,
            train_dir,
            solver,
        )
        baseModelStats.evaluate_training_instances(
            f"{working_directory}/model.essence", conf
        )

        streamlinerModelStats = StreamlinerModelStats(
            f"{working_directory}/StreamlinerModelStatsFold{args.fold_num}.csv",
            solver,
        )

        portfolio_builder = HydraPortfolio(
            essence_spec, baseModelStats, streamlinerModelStats, conf
        )

        portfolio_round = portfolio_builder.build_portfolio(f"portfolios/PortfolioFold{args.fold_num}.json")
