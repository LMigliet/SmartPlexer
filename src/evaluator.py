from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.components import Experiment, Panel


class AssayEvaluator:

    def __init__(
        self,
        experiments: Dict[str, Experiment],
        assay_combos_dict: Dict[str, Dict[str, str]],
    ):
        """
        Initializes the AssayEvaluator with experiment data and assay combinations.

        :param experiments: Dictionary of experiment objects.
        :param assay_combos_dict: Dictionary with primer mixes and their corresponding assay combinations.
        """
        self.experiments = experiments
        self.assay_combos_dict = assay_combos_dict

    def extract_parameter_values(
        self,
        primer_mix: str,
        parameter_name: str,
    ) -> Dict[str, Dict[str, List[Optional[float]]]]:
        """
        Extracts parameter values for each assay in a given primer mix.

        :param primer_mix: The primer mix identifier (e.g., "PM3.01").
        :param parameter_name: The name of the parameter to extract values for.
        :return: A dictionary with assay names as keys and dictionaries of parameter values as values.
        """
        result = {}

        # Get the assays associated with the specified primer mix
        assays = self.assay_combos_dict.get(primer_mix, {})

        # Loop through each assay in the primer mix
        for assay_name in assays.values():
            # Initialize a list to store the parameter values for the assay
            parameter_values = []

            # Loop through all experiments and panels to find matching panels
            for experiment in self.experiments.values():
                for panel in experiment.panels:
                    if panel.assay == assay_name and panel.primermix != "singleplex":
                        # Collect the specific parameter values using the method from Panel class
                        parameter_values.extend(
                            panel.get_parameter_values(parameter_name)
                        )

            # Store the list of parameter values in the result dictionary
            result[assay_name] = {parameter_name: parameter_values}

        return result

    def evaluate_median_distances(
        self,
        primer_mix: str,
        parameter_name: str,
    ) -> Dict[str, float]:
        """
        Evaluates a primer mix by calculating the median distances between assays.

        :param primer_mix: The primer mix identifier (e.g., "PM3.01").
        :param parameter_name: The name of the parameter to evaluate.
        :return: A dictionary with 'average_distance' and 'minimum_distance' for the primer mix.
        """
        # Step 1: Extract parameter values
        assay_parameter_values = self.extract_parameter_values(
            primer_mix, parameter_name
        )

        # Step 2: Calculate medians
        median_values = {}
        for assay_name, parameters in assay_parameter_values.items():
            for param_name, values in parameters.items():
                median_values[assay_name] = np.median(
                    [v for v in values if v is not None]
                )

        # Step 3: Calculate distances
        assay_names = list(median_values.keys())
        distances = []
        for i in range(len(assay_names)):
            for j in range(i + 1, len(assay_names)):
                distance = abs(
                    median_values[assay_names[i]] - median_values[assay_names[j]]
                )
                distances.append(distance)

        # Step 4: Calculate average and minimum distances
        if distances:
            average_distance = np.mean(distances)
            minimum_distance = np.min(distances)
        else:
            average_distance = 0.0
            minimum_distance = 0.0

        return {
            "ADS": average_distance,  # average_distance
            "MDS": minimum_distance,  # minimum_distance
        }

    @staticmethod
    def rank_primer_mixes(ads_mds_allpm: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Ranks primer mixes based on their average distance and minimum distance.

        :param ads_mds_allpm: Dictionary with primer mixes as keys and a dictionary of 'average_distance' and 'minimum_distance' as values.
        :return: DataFrame with ranked primer mixes.
        """
        df = pd.DataFrame.from_dict(ads_mds_allpm, orient="index")
        df["rank_MDS"] = df["MDS"].rank(ascending=False)
        df["rank_ADS"] = df["ADS"].rank(ascending=False)
        df["final_rank"] = df["rank_ADS"] * df["rank_MDS"]  # GEO: rank combo
        return df.sort_values("final_rank")
