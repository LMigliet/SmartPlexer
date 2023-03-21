import pandas as pd
import numpy as np

def remove_background(df, NMETA, ct_start, ct_skip):
    """
    background fluorescence removal to set the start of the curve to ZERO.
    
    params:
    - df: DataFrame with raw curves data and metadata
    - NMETA: data with well information
    - ct_start: the initial Cycle for the removal avarage calculation
    - ct_skip: number of cycle to fit for the removal avarage calculation
    
    return:
    - df_rb_all: DataFrame with removed background for each cycles.
    """
    
    ct_end = ct_start + ct_skip
    
    rb_value = df.iloc[:, NMETA+ct_start : NMETA+ct_end].mean(axis = 1)
    df_rb = df.iloc[:,NMETA:].apply(lambda x: x - rb_value)
    df_rb_all = pd.concat([df.iloc[:, :NMETA],df_rb], axis=1)
    
    return df_rb_all


def remove_delay_ac(df, NMETA, proportion_thresh, cycle_thresh):
    """
    remove curves with cycle threshold larger than cyc_thresh
    
    param:
    - df: DataFrame with the curves data and metadata
    - NMETA: data with well information
    - proportion_thresh: set the proportion threshold (the increasing of the curve, i.e. 0.1)
    - cycle_thresh: set the threshold value for cycle
    
    return:
    - df: DataFrame with delayed curves removal
    """
    
    curve = df.iloc[:, NMETA:]
    curve_max = curve.max(axis = 1)
    curve_min = curve.min(axis = 1)
    
    mask = curve.sub(curve_min, axis = 0).div(curve_max - curve_min, axis = 0) > proportion_thresh
    cycle_thresh_df = mask.ne(False, axis = 1).idxmax(axis = 1).astype('float64')
    
    return df[cycle_thresh_df < cycle_thresh]


def remove_noisy_data(df, NMETA, THRESH=13, panel_remove=False, plot=False):
    """
    remove noisy curve due to instrumentation or experimental errors.
    The df_temp will keep adding 1 to the Zero-Crossing (or 'ZC') columns, 
    which indicates how many times the forward_cycles has less fluo of the 
    previous one.
    If the number in ZC column is higher than the THRESHOLD, the curves is noisy.
    
    params:
    - df: DataFrame with raw curves data and metadata
    - NMETA: data with well information
    - thresh: threshold value (zero crossing) to determine noisy data
    - panel_remove: remove all the curves in the panel that has noisy data
    - plot: plot the curve
    
    return:
    - df_filtered: DataFrame with the filtered curves
    """
    
    df_temp = df.iloc[:, :NMETA]
    
    # We calculate the difference of cycle2 - cycle1.
    # if cycle2 fluorescence is less than cycle1 fluorescence
    # the 'ZC' will plus 1 the values.
    
    for cycle in range(df.iloc[:,NMETA:-1].shape[1]):
        df_temp[cycle] = df.iloc[:, NMETA+cycle+1] - df.iloc[:, NMETA+cycle]
    
    df_temp['ZC'] = (df_temp.iloc[:,NMETA:] < 0).astype(int).sum(axis = 1)

    df_outlier = df_temp[df_temp['ZC'] > THRESH]
    
    # if outliers are not present, then we return the same dataframe
    if df_outlier.shape[0] == 0:
        return df
    
    # if outliers are present, then we return the filtered dataframe
    else:
        if panel_remove:
            outlier_channels = df_outlier['Channel'].unique()
            df_filtered = df[~df['Channel'].isin(outlier_channels)]
        else:
            df_filtered = df[df_temp['ZC'] < THRESH]
        return df_filtered
    

def concat_norm_curve_on_FFI(df, NMETA):
    """
    calculate the normalization and concat at the end
    
    param:
    - df: DataFrame with the curves data and metadata
    - NMETA: number of metadata columns
    
    return:
    - df_return: DataFrame concated normalized curves
    """
    
    curves = df.iloc[:, NMETA:]
    metadata = df.iloc[:, :NMETA]

    row_min = np.array(curves.values.min(axis=1), ndmin=2).T
    row_max = np.array(curves.values.max(axis=1), ndmin=2).T

    curves_norm = (curves - row_min) / (row_max - row_min)
    
    df_return = pd.concat([metadata, curves_norm], axis=1)

    return df_return
