from typing import Dict, Any
import numpy as np

type Round = int
type Streamliner = str

type StreamlinerPerformanceMetrics = Dict[
    str, float | np.floating[Any] | list[np.floating[Any]]
]
type Portfolio = Dict[Streamliner, StreamlinerPerformanceMetrics]
