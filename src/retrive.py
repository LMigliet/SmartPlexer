import pandas as pd

from src.components import Param, Well


def parse_params(params_df: pd.DataFrame, well_id: str, nmeta: int):
    """
    Parse parameters from a DataFrame for a specific well.

    Args:
        params_df (pd.DataFrame): DataFrame containing parameters with wells as index.
        well_id (str): Identifier of the well for which parameters are to be parsed.
        nmeta (int): Number of metadata columns before the actual parameter columns.

    Returns:
        List[Param]: A list of Param objects, each representing a parameter with a name and value.
    """
    params = []
    df = params_df.iloc[:, nmeta:].copy()
    for col in df.columns:
        if well_id in df.index:
            value = df.at[well_id, col]
            params.append(Param(name=col, value=value))
    return params


def parse_wells(
    ac_df: pd.DataFrame, params_df: pd.DataFrame, well_type: str, nmeta: int = 5
):
    """
    Parse wells and their associated data from given DataFrames.

    Args:
        ac_df (pd.DataFrame): DataFrame containing well data with intensities and other metrics.
        params_df (pd.DataFrame): DataFrame containing parameters associated with each well.
        well_type (str): Type of the wells being parsed (e.g., "control", "sample").
        nmeta (int, optional): Number of metadata columns before the actual data columns. Defaults to 5.

    Returns:
        List[Well]: A list of Well objects, each representing a well with its id, type, intensities, and parameters.
    """
    wells = []
    for well_id, row in ac_df.iterrows():
        intensities = row[nmeta:].tolist()
        params = parse_params(params_df, well_id, nmeta)
        wells.append(
            Well(id=well_id, type=well_type, intensities=intensities, params=params)
        )
    return wells
