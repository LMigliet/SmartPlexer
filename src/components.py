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

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Param:
    name: str
    value: float


@dataclass
class Well:
    id: int
    type: str  # 'inlier' or 'outlier'
    intensities: List[float] = field(repr=False)
    params: List[Param] = field(repr=False)

    def get_param_value(self, param_name: str) -> Optional[float]:
        """
        Get the value of a specific parameter from this well.
        :param param_name: The name of the parameter to retrieve.
        :return: The value of the parameter if found, otherwise None.
        """
        for param in self.params:
            if param.name == param_name:
                return param.value
        return None


@dataclass
class Panel:
    id: int
    primermix: str
    target: str
    assay: str
    wells: List[Well] = field(repr=False)
    target_concentration: Optional[float] = None

    def get_parameter_values(self, parameter_name: str) -> List[Optional[float]]:
        """
        Get the values of a specific parameter across all wells in this panel.
        :param parameter_name: The name of the parameter to retrieve.
        :return: A list of parameter values, with None for wells where the parameter is not found.
        """
        return [well.get_param_value(parameter_name) for well in self.wells]


@dataclass
class Experiment:
    id: int
    panels: List[Panel]
