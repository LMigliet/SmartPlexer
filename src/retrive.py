import pandas as pd

from src.components import Param, Well


# Helper function to parse Params
def parse_params(params_df, well_id, nmeta):
    params = []
    df = params_df.iloc[:, nmeta:].copy()
    for col in df.columns:
        if well_id in df.index:
            value = df.at[well_id, col]
            params.append(Param(name=col, value=value))
    return params


# Helper function to parse Wells
def parse_wells(ac_df, params_df, well_type, nmeta=5):
    wells = []
    for well_id, row in ac_df.iterrows():
        intensities = row[nmeta:].tolist()
        params = parse_params(params_df, well_id, nmeta)
        wells.append(
            Well(id=well_id, type=well_type, intensities=intensities, params=params)
        )
    return wells
