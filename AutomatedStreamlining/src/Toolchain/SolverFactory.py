from Toolchain.Solver import Solver
from Toolchain.Chuffed import Chuffed
from Toolchain.Cadical import Cadical


def get_solver(solver: str) -> Solver:
    if solver == "chuffed":
        return Chuffed()
    elif solver == "cadical":
        return Cadical()
    else:
        raise Exception("Unsupported Solver")
