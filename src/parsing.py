import os
from typing import List, Optional

import pandas as pd

from src.components import Experiment, Panel, Well
from src.retrive import parse_wells


def load_experiment_data(experiment_id: str, data_path: str) -> dict:
    """
    Load experiment data from CSV files for a given experiment ID.

    Args:
        experiment_id (str): The ID of the experiment to load data for.
        data_path (str): The root directory path where the experiment data is stored.

    Returns:
        dict: A dictionary containing DataFrames for inliers' and outliers'
              amplification curves and parameters.
    """
    experiment_data = {}
    experiment_id_path = os.path.join(data_path, experiment_id)

    experiment_data["inliers_params_df"] = pd.read_csv(
        f"{experiment_id_path}/processed/inliers_params.csv"
    )
    experiment_data["inliers_ac_df"] = pd.read_csv(
        f"{experiment_id_path}/processed/inliers_ac.csv"
    )
    experiment_data["outliers_ac_df"] = pd.read_csv(
        f"{experiment_id_path}/processed/outliers_ac.csv"
    )
    experiment_data["outliers_params_df"] = pd.read_csv(
        f"{experiment_id_path}/processed/outliers_params.csv"
    )

    return experiment_data


def group_wells_by_panel(
    inlier_wells: List[Well],
    inliers_ac_df: pd.DataFrame,
    outlier_wells: Optional[List[Well]] = None,
) -> dict:
    """
    Group wells by their panel and create Panel objects.

    Args:
        inlier_wells (List[Well]): A list of Well objects marked as inliers.
        inliers_ac_df (pd.DataFrame): DataFrame containing inlier amplification curve data.
        outlier_wells (Optional[List[Well]]): A list of Well objects marked as outliers, if provided.

    Returns:
        dict: A dictionary where keys are panel IDs and values are Panel objects
              containing wells grouped by their respective panels.
    """
    panels = {}

    # Combine inlier and outlier wells if outlier_wells is provided
    all_wells = inlier_wells + (outlier_wells if outlier_wells else [])

    for well in all_wells:
        panel_id = inliers_ac_df.at[well.id, "Panel"]
        primermix = inliers_ac_df.at[well.id, "PrimerMix"]
        target = inliers_ac_df.at[well.id, "Target"]
        assay = inliers_ac_df.at[well.id, "Assay"]
        target_concentration = inliers_ac_df.at[well.id, "Conc"]

        if panel_id not in panels:
            panels[panel_id] = Panel(
                id=panel_id,
                primermix=primermix,
                target=target,
                assay=assay,
                wells=[],
                target_concentration=target_concentration,
            )

        panels[panel_id].wells.append(well)

    return panels


def create_experiment_objects(
    experiment_ids: List[str],
    data_path: str,
    meta_len: int,
    outlier_wells: Optional[bool] = None,
) -> dict:
    """
    Create Experiment objects from a list of experiment IDs and organize data into Panels.

    Args:
        experiment_ids (List[str]): List of experiment IDs to process.
        data_path (str): The root directory path where the experiment data is stored.
        meta_len (int): Number of metadata columns before the actual data starts in the CSV.
        outlier_wells (Optional[bool]): Flag indicating whether to include outlier wells.

    Returns:
        dict: A dictionary where keys are experiment IDs and values are Experiment objects
              containing grouped Panels.
    """

    print("Creating experiments...")
    experiments = {}

    for experiment_id in experiment_ids:
        print(f"Processing experiment: {experiment_id}")

        # Load experiment data
        experiment_data = load_experiment_data(experiment_id, data_path)

        # Parse Inlier Wells
        inlier_wells = parse_wells(
            experiment_data["inliers_ac_df"],
            experiment_data["inliers_params_df"],
            "inlier",
            nmeta=meta_len,
        )

        # Parse Outlier Wells if outlier_wells is True
        outlier_wells_list = None
        if outlier_wells:
            outlier_wells_list = parse_wells(
                experiment_data["outliers_ac_df"],
                experiment_data["outliers_params_df"],
                "outlier",
                nmeta=meta_len,
            )

        # Group wells by their panel and create Panel objects
        panels = group_wells_by_panel(
            inlier_wells,
            experiment_data["inliers_ac_df"],
            outlier_wells_list,
        )

        # Create Experiment object
        experiments[experiment_id] = Experiment(
            id=experiment_id, panels=list(panels.values())
        )

    return experiments
