import logging
import glob
from pathlib import Path
import Executor
from typing import Optional

portfolio_size = 1


class Conjure:
    def generate_streamliners(self, essence_spec: Path) -> Optional[tuple[bytes, bytes]]:
        # return ['conjure', 'streamlining', essence_spec, f'--portfolio-size={self.portfolio_size}', f'-o {output_dir}']
        logging.debug(f"Generating candidate streamliners")
        command = ["conjure", "streamlining", essence_spec]
        if maybe_res := Executor.callable(command):
            stdout, stderr, _ = maybe_res
            return stdout, stderr
        else:
            logging.error(f"No candidate streamliners were generated")
            return None

    def generate_streamlined_models(
        self,
        essence_spec: Path,
        streamliner_combination: Optional[str],
        output_dir: str,
    ) -> list[Path]:
        # If this is a streamliner combination (- in the combo), translate to ','
        if not streamliner_combination:
            streamliner_combination = ""
        if "-" in streamliner_combination:
            streamliner_combination = ",".join(streamliner_combination.split("-"))
        command = [
            "conjure",
            "modelling",
            essence_spec,
            "--generate-streamliners",
            streamliner_combination,
            f"--portfolio={portfolio_size}",
            "-o",
            output_dir,
        ]
        logging.debug(
            f"Building streamlined models for {streamliner_combination}: {command}"
        )
        if maybe_res := Executor.callable(command):
            _, _, _ = maybe_res
            
        return [Path(eprime) for eprime in glob.glob(f"{output_dir}/*.eprime")]

    def translate_essence_param(
        self,
        instance_dir: str,
        eprime_model: str,
        essence_param: str,
        output_eprime_param: str,
    ):
        return [
            "conjure",
            "translate-parameter",
            f"--eprime={eprime_model}",
            f"--essence-param={instance_dir}/{essence_param}",
            f"--eprime-param={output_eprime_param}",
        ]

    def parse_std_out(self, out, instance_stats):
        return

    def parse_std_err(self, out, instance_stats):
        return
