import pandas as pd
import numpy as np
import itertools

import scipy.optimize as opt
from scipy.special import comb
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean


def compute_score(df, NMETA, pm_data):
    """
    compute score according to processed data with different curves types (e.g. norm, fitted parameters).
    4 types of score to compute (mean - ADS, min - MDS, mod_mean - modified ADS, mod_min: modifited MDS - modified min)

    params: 
    - df: dataframe contains processed data on which to compute score
    - NMETA: number of metadata columns
    - pm_data: a dictionary containing primer information for each possible primermix

    return:
    - feature_df: tabular result of computed score (including score between each pair of targets)
    """
    
    n_target = len(df['Target'].unique())

    combination_distances = []

    for pm in pm_data:
        
        sample_dict = pm_data[pm]
        assay_name_list = []
        temp_dict = {}

        for assay, assay_df in sample_dict.groupby('Assay'):
            assay_name_list.append(assay)
            temp_dict[assay] = {
                "median": assay_df.iloc[:,NMETA:].median().values,  # mean value for each cycle (it's a curve)
                "std": assay_df.iloc[:,NMETA:].std().mean()  # std value for all curves (as 1 value)
                }

        single_combo_res = assay_name_list.copy()
        single_combo_res.insert(0, pm)

        for combination in list(itertools.combinations(assay_name_list, 2)):

            eucledian_dist = euclidean(temp_dict[combination[0]]['median'],
                                       temp_dict[combination[1]]['median'])
            std_mean = (temp_dict[combination[0]]['std'] + temp_dict[combination[1]]['std']) / 2

            single_combo_res.append(eucledian_dist)
            single_combo_res.append(eucledian_dist / std_mean)

        combination_distances.append(single_combo_res)
        
    cols = ['PrimerMix']  # define the first column and appanding others

    for i in range(0, n_target):
        cols.append('Assay' + str(i + 1))

    for i in range(0, int(comb(n_target, 2))):
        cols.append('Distance' + str(i + 1))
        cols.append('MDistance' + str(i + 1))

    feature_df_temp = pd.DataFrame(combination_distances, columns=cols)
    
    feature_df = feature_df_temp.copy()

    feature_df['mean'] = feature_df_temp.iloc[:, 1+n_target::2].mean(axis = 1)
    feature_df['min'] = feature_df_temp.iloc[:, 1+n_target::2].min(axis = 1)

    feature_df['mod_mean'] = feature_df_temp.iloc[:, 1+n_target+1::2].mean(axis = 1)
    feature_df['mod_min'] = feature_df_temp.iloc[:, 1+n_target+1::2].min(axis = 1)

    return feature_df


def generate_correlation_df_S2M(df_x, df_y):
    """
    compute dataframe containing Pearson coefficient given two series of data.

    params:
    - df_x: x-axis data (e.g. simulated score)
    - df_y: y-axis data (e.g. empirical score)

    return:
    - df_correlation: dataframe with Pearson coefficient based on df_x and df_y
    """
    name_list = df_x.columns
        
    coef_list = []
    
    x_data = df_x
    y_data = df_y
    for col in x_data.columns:
        coef_list.append(np.corrcoef(x_data.loc[:, col].values,
                                          y_data.loc[:, col].values)[0, 1])
    
    df_correlation = pd.DataFrame([name_list, coef_list], index = ['Metric', 'Correlation coefficient'])
    
    return df_correlation


##########################################################################################
# COMBINATION FUNCTION TO GET THE TOP AND BOTTOM MULTIPLEXES #############################
##########################################################################################


def get_n_largest_score(df, n):
    """
    get the first n combinations with the largest score.
    4 types of score = 
    - mean ("ADS")
    - min ("MDS")
    - mod_mean
    - mod_min

    params:
    - df: dataframe containing scores (ADS and MDS)
    - n: first n elements

    return:
    - df_largest_score: integrated dataframe of n primermix with the largest scores (all 4 types)
    """

    cols = ['mean', 'min', 'mod_mean', 'mod_min']
    
    pm_mean = df.nlargest(n, 'mean')['PrimerMix'].reset_index(drop = True)
    pm_min = df.nlargest(n, 'min')['PrimerMix'].reset_index(drop = True)
    
    pm_mod_mean = df.nlargest(n, 'mod_mean')['PrimerMix'].reset_index(drop = True)
    pm_mod_min = df.nlargest(n, 'mod_min')['PrimerMix'].reset_index(drop = True)
    
    df_largest_score = pd.concat([pm_mean, pm_min, pm_mod_mean, pm_mod_min], axis = 1)
    df_largest_score.columns = cols
    
    return df_largest_score


def get_n_least_score(df, n):
    """
    get the first n combinations with the least score.
    4 types of score = 
    - mean ("ADS")
    - min ("MDS")
    - mod_mean
    - mod_min

    params:
    - df: dataframe containing scores (ADS and MDS)
    - n: last n elements

    return:
    - df_least_score: integrated dataframe of n primermix with the least scores (all 4 types)
    """

    cols = ['mean', 'min', 'mod_mean', 'mod_min']
    
    pm_mean = df.nsmallest(n, 'mean')['PrimerMix'].reset_index(drop = True)
    pm_min = df.nsmallest(n, 'min')['PrimerMix'].reset_index(drop = True)
    
    pm_mod_mean = df.nsmallest(n, 'mod_mean')['PrimerMix'].reset_index(drop = True)
    pm_mod_min = df.nsmallest(n, 'mod_min')['PrimerMix'].reset_index(drop = True)
    
    df_least_score = pd.concat([pm_mean, pm_min, pm_mod_mean, pm_mod_min], axis = 1)
    df_least_score.columns = cols
    
    return df_least_score


def get_combination(df, df_combination, number_combination, top=True):
    """
    This function get the combination from a dataframe with distances values.

    attributes:
    - df_distances: df with distances and mean/min scores
    - df_combination: all the possible combinations in a df
    - number_combination: selected number of combinations to down-select
    - top: the order to get the combo 
           (i.e. if you want the top 3, then top=true, if you want the bottom 3 top=False)

    return: DF with the assays and the combo selected
    """
    for num in range(df_combination.shape[0]):
        
        if top == True:
            df_temp = get_n_largest_score(df, num)
        else:
            df_temp = get_n_least_score(df, num)
        
        combination_list = []
            
        for primer in df_temp.loc[:, 'mean']:
            if (df_temp.loc[:, 'min'] == primer).any() == True:
                combination_list.append(primer)
        
        if len(combination_list) >= number_combination:
            if top == True:
                print('Search ' + str(num) + ' to get top ' + str(number_combination) + ' combinations.')
                break
            else:
                print('Search ' + str(num) + ' to get bottom ' + str(number_combination) + ' combinations.')
                break
    
    df_return = df_combination[df_combination.index.isin(combination_list)]
    
    return df_return
