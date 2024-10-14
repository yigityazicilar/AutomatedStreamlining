from typing import Dict
from Toolchain.RunSolver import RunSolverStats, translate_to_runsolver_stats


class InstanceStats:
    def __init__(self):
        self._time_out = False
        self._stages: Dict[str, RunSolverStats] = {}
        self._killed = False
        self._solver_stats = {}
        self._satisfiable = False

    def get_stages(self) -> Dict[str, RunSolverStats]:
        return self._stages

    def add_stage_stats(self, name: str, stats: RunSolverStats):
        self._stages[name] = stats

    def satisfiable(self):
        return self._satisfiable

    def set_killed(self):
        self._killed = True

    def killed(self):
        return self._killed

    def set_timeout(self):
        self._time_out = True
        self._satisfiable = False

    def timeout(self):
        return self._time_out

    def set_satisfiable(self, satisfiable: bool):
        self._satisfiable = satisfiable

    def add_solver_output(self, solver_stats: Dict[str, str | bool]):
        self._solver_stats = solver_stats

    def solver_stats(self) -> Dict[str, str | bool]:
        return self._solver_stats

    def set_solver_name(self, solver_name: str):
        self.solver_name = solver_name

    def solver_time(self):
        return self._stages[self.solver_name].get_real_time()

    def total_time(self):
        if self._stages:
            return sum([result.get_real_time() for _, result in self._stages.items()])
        # Shouldn't be called if Stages are Null
        raise Exception()

    def __str__(self):
        return f"""
              'TimeOut' : {self._time_out},
              'Satisfiable' : {self._satisfiable},
              'Killed' : {self._killed},
              'Stages': {self._stages},
              'SolverStats' : {self._solver_stats}
          """


def translate_to_instance_stats(result):
    solver = result["Solver"]
    conjure_stats = translate_to_runsolver_stats(
        dict(filter(lambda item: "conjure" in item[0], result.items())), "conjure"
    )
    savilerow_stats = translate_to_runsolver_stats(
        dict(filter(lambda item: "savilerow" in item[0], result.items())), "savilerow"
    )
    solver_stats = translate_to_runsolver_stats(
        dict(filter(lambda item: solver in item[0], result.items())), solver
    )
    killed = result["Killed"]
    satisfiable = result["Satisfiable"]
    timeout = result["TimeOut"]
    solver_output_stats = dict(filter(lambda item: "solver" in item[0], result.items()))

    instanceStats = InstanceStats()
    instanceStats.set_satisfiable(satisfiable)
    if killed:
        instanceStats.set_killed()
    if timeout:
        instanceStats.set_timeout()
    instanceStats.add_solver_output(solver_output_stats)
    instanceStats.add_stage_stats("conjure", conjure_stats)
    instanceStats.add_stage_stats("savilerow", savilerow_stats)
    instanceStats.add_stage_stats(solver, solver_stats)
    instanceStats.set_solver_name(solver)
    return instanceStats
