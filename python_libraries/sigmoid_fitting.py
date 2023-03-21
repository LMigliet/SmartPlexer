import pandas as pd
import numpy as np
import scipy.optimize as opt
from multiprocessing import Pool
import time
import itertools


def Richard_Curve(x, a, b, c, d, e, f):
    return -b / (1. + c * np.exp(d * (x - e)))**(1 / f) + a


def Five_Parameter_Sigmoid(x, a, b, c, d, e):
    return a / (1. + np.exp(-c * (x - d)))**e + b


def fitting_function(data, func, p0, bounds, maxfev):
    """
    group the data for model training
    
    params:
    - df_singleplex: DataFrame of the singleplex data
    - df_multiplex: DataFrame of the multiplex data
    - df_combination: DataFrame of combination information
    
    return:
    - Sigmoid_Parameters: fitted parameters for the curve
    """
    
    x = data.index.values
    y = data.values
    sigmoid_parameters, _ = opt.curve_fit(func, x, y, p0 = p0, maxfev = maxfev, bounds = bounds)
    return sigmoid_parameters


def sigmoid_fitting_parallel(df, NMETA, func, p0, bounds, maxfev, core_number):
    """
    fit the raw curve with indicated function in parallel
    
    params:
    - df: DataFrame with the raw curves and metadata
    - NMETA: data with well information
    - cycle_number: the number of fluoresence cycles
    - func: indicated function
    - p0: ground choice of the parameters
    - bounds: bounds of the fitted parameters
    - maxfev: maximum number of the training
    - core_number: the number of the cores used for parallel
    
    return:
    - df_integrated_data: DataFrame of the integrated data
                          (metadata, raw curves, sigmoid parameters, sigmoid curves, MSE)
    - range_list: the columns index of each slice of the integrated data
    """
    
    if func == Five_Parameter_Sigmoid:
        NPARAM = 5
        df_cols = ['a', 'b', 'c', 'd', 'e']
    elif func == Richard_Curve:
        NPARAM = 6
        df_cols = ['a', 'b', 'c', 'd', 'e', 'f']
    
    data = df.iloc[:, NMETA:]
    metadata = df.iloc[:, :NMETA]
    
    df_split = np.array_split(data, core_number, axis = 0)
    pool = Pool(core_number)
    sigmoid_params = pd.concat(pool.map(warp_sigmoid_fitting, 
                                                 zip(df_split, 
                                                     itertools.repeat(func), 
                                                     itertools.repeat(p0),
                                                     itertools.repeat(bounds),
                                                     itertools.repeat(maxfev),
                                                    )), axis = 0)
    pool.close()
    pool.join()
    sigmoid_curves = sigmoid_params.apply(lambda p: func(data.columns.astype('float64'), *p))
    sigmoid_curves.index = data.index
    
    sigmoid_params_df = pd.DataFrame(np.vstack(sigmoid_params), columns = df_cols)
    
    sigmoid_curves_df = pd.DataFrame(np.vstack(sigmoid_curves), columns = data.columns)
    
    df_MSE = pd.DataFrame(((sigmoid_curves_df.values - data.values) ** 2).mean(axis = 1), columns = ['MSE'])
    
    df_fitted_param = pd.concat([metadata, sigmoid_params_df, df_MSE], axis = 1)
    
    df_fitted_curves = pd.concat([metadata, sigmoid_curves_df], axis = 1)
    
    return df_fitted_param, df_fitted_curves


def warp_sigmoid_fitting(params):
    """
    wrap up the parameters used for fitting
    
    params:
    - params: wrap-up parameters
    
    return:
    fitting sigmoid parameters
    """
    
    df, func, p0, bounds, maxfev = params
    return df.apply(lambda x: fitting_function(x, func, p0, bounds, maxfev), axis = 1)


def pivot_fitting(df, func, NMETA, bounds, maxfev, core_number, thresh_MSE, seed=None):
    """
    Initial the parameters for the fitting, i.e. p0
    (remove the fitting parameters whose mean squared errors are larger than threshold value)
    
    - df: DataFrame with filtered curves data and metadata
    - func: indicated function
    - NMETA: data with well information
    - cycle_number: number of fluoresence cycles
    - bounds: bounds of the fitted parameters
    - maxfev: maximum number of the training
    - core_number: the number of the cores used for parallel
    - thresh_MSE: threshold value of the mean squared error
    
    return: 
    Initial value of p0
    """
    
    initial_p0 = (0, 0, 0, 0, 0)
    df_fitted_initial, _ = sigmoid_fitting_parallel(df.sample(100, random_state=seed), NMETA,
                                                    func, initial_p0, 
                                                    bounds, maxfev, core_number)
    
    df_filtered_fitted_initial = df_fitted_initial[df_fitted_initial['MSE'] <= thresh_MSE]
    
    return np.mean(df_filtered_fitted_initial.iloc[:, NMETA : -1].values, axis = 0)
