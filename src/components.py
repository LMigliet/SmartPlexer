"""
This code provides a structured way to 
represent complex data related to scientific experiments. 

By organizing the data hierarchically where:
- Experiment contain Panels, 
- Panels contain Wells
- Wells can be inlier or outlier nad they contain:
    - Param (fitted parameters)
    - intensities of amplification curve
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Param:
    name: str
    value: float


@dataclass
class Well:
    id: int
    type: str  # 'inlier' or 'outlier'
    intensities: List[float]
    params: List[Param]


@dataclass
class Panel:
    id: int
    primermix: str
    target: str
    assay: str
    wells: List[Well]
    target_concentration: Optional[float] = None


@dataclass
class Experiment:
    id: int
    panels: List[Panel]
