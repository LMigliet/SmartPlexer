
import pandas as pd 
import numpy as np

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import to_hex


def colourblind():
    # COLOURBLIND LIBRARY FOR PLOTTING
	mpl.rcParams['axes.prop_cycle'] = cycler(color=[to_hex(i) for i in [
		(0,0.45,0.70), 
		(0.9, 0.6, 0.0), 
		(0.0, 0.60, 0.50), 
		(0.8, 0.4, 0), 
		(0.35, 0.7, 0.9), 
		(0.8, 0.6, 0.7), 
		(0,0,0), 
		(0.5, 0.5, 0.5), 
		(0.95, 0.9, 0.25)]])


def list_from_key(dic, key):
    """
    Take a dictionary with ID and EXP_ID, to return a list of EXP_ID.

    params
    - dic: input dictionary.
    - key: provided list of keys.

    return:
        List of specified values
    """
    return [dic[x] for x in key]


def order_columns(df, NMETA):
    """
    sorting the column indexes (This is valid for both AC or MC data).
    This fix the issue of column of dataframes, 
    making number (i.e. cycles or Temp) as FLOATs instead of str.

    params:
    - df: input dataframe on which to reorder column indexes
    - NMETA: number of metadata columns

    return:
    - df: dataframe with sorted columns
    """

    unordered_idxs = [float(i) for i in df.columns[NMETA:]]
    df.columns = list(df.columns[:NMETA]) + unordered_idxs
    df = df[list(df.columns[:NMETA]) + sorted(unordered_idxs)]
    return df


def data_downselection(df, by_column, keys, keep=True):
    """
    The function input is a df and a column
    
    - df: dataframe for the downselection
    - by_column: the column of the dataframe for the down-selection
    - keys: list of values you want to keep or remove 
        (i.e. if i want to remove NTC, keys=['NTC'], keep=False)
    - keep: true = keep data / false = remove data
    
    return: down-selected dataframe
    """
    if keep == True:
        return df[df[by_column].isin(keys)].reset_index(drop=True)
    else:
        return df[~df[by_column].isin(keys)].reset_index(drop=True)
