from abc import abstractmethod
from typing import Dict, List, Optional

from Toolchain.InstanceStats import InstanceStats


class Solver:
    def __init__(self, solver_name, savilerow_flag, savilerow_output_flag):
        self.solver_name = solver_name
        self.savilerow_flag = savilerow_flag
        self.savilerow_output_flag = savilerow_output_flag

    def get_solver_name(self) -> str:
        return self.solver_name

    def get_savilerow_flag(self) -> str:
        return self.savilerow_flag

    def get_savilerow_output_flag(self) -> str:
        return self.savilerow_output_flag

    @abstractmethod
    def get_savilerow_output_file(self, eprime_model, raw_instance, streamliner: Optional[str] = None ) -> str:
        pass

    @abstractmethod
    def execute(self, output_file: str) -> List[str]:
        pass

    @abstractmethod
    def parse_std_out(self, output_file: str, out: bytes, instance_stats: InstanceStats) -> Dict[str, str]:
        pass

    @abstractmethod
    def parse_std_err(self, out, instance_stats):
        pass

    @abstractmethod
    def get_stat_names(self) -> List[str]:
        pass
