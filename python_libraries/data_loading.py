import pandas as pd
import numpy as np

import glob


def load_data(file_string, data_type):
    """
    Takes the string which is a path folder where the dPCR data are stored.
    Those data should be in a pandas dataframe format (CSV)
    
    params:
    - file_string: path to the data folder (where the CVS file will be)
    - data_type: type of data you have (i.e. singleplex or multiplex)
    
    return:
    - df_return: dataframe of raw data from dPCR
    """

    file_list = glob.glob(f'{file_string}/raw_data/{data_type}/*{data_type}*.csv')
    
    # if more than one file is present, we concanate the files
    if len(file_list) > 1:
        df_list = []
        
        for file in file_list:
            df_list.append(pd.read_csv(file, index_col=0))
        
        df_return = pd.concat(df_list, axis=0).reset_index(drop=True)
        return df_return
    
    # if only one file, just read..
    else:
        df_return = pd.read_csv(file_list[0], index_col=0).reset_index(drop=True)
        return df_return
    

def load_raw_data_by_expid(data_path, id_list, data_type):
    """
    load raw data according to experiment ID.

    params: 
    - data_path: file path of the raw data
    - id_list: list of specified experiment ID (by which data will be loaded)
    - data_type: types of raw data ("SINGLEPLEX" or "MULITPLEX")

    return:
        Dataframe containing raw data specified by input list of experiment ID
    """

    data_list = []
    for exp_id in id_list:
        data_list.append(pd.read_csv(f'{data_path}/raw_data/{data_type}/{exp_id}.csv', index_col=0))
    
    return pd.concat(data_list, axis=0).reset_index(drop=True)


def load_processed_data_by_expid(data_path, id_list, curve_type):
    """
    load processed data according to experiment ID.

    params: 
    - data_path: file path of the processed data
    - id_list: list of specified experiment ID (by which data will be loaded)
    - data_type: types of processed data ("SINGLEPLEX" or "MULITPLEX")

    return:
        Dataframe containing processed data specified by input list of experiment ID
    """

    data_list = []
    for exp_id in id_list:
        data_list.append(pd.read_csv(f'{data_path}/processed_data/{exp_id}_{curve_type}.csv', index_col=0))
    
    return pd.concat(data_list, axis=0).reset_index(drop=True)


def store_by_id(df, data_path, data_type='processed_data', curve_type='', id_list=[]):
    """
    store data according to experiment ID.
    
    NB. For the time being the type of data we have are the following:
    data_type =
    - accuracy_data
    - processed_data (THIS SHOULD BE USED HERE)
    - raw_data
    - score_data
    - selected_combinations

    params:
    - df: dataframe to store
    - data_path: path to store data
    - data_type: types of data ("SINGLEPLEX" or "MULTIPLEX")
    - curve_type: types of curves (e.g. raw curves, norm curves)
    - id_list: list of experiment ID that will be saved
    
    return: a csv file saved in the specific folder
    """
    
    if id_list == []:
        id_list = df['Exp_ID'].unique()
    
    for exp_id in id_list:
        df_temp = df[df['Exp_ID'] == exp_id]
        
        if curve_type == '':
            df_temp.reset_index(drop=True).to_csv(f'{data_path}/{data_type}/{exp_id}.csv')
        else:
            df_temp.reset_index(drop=True).to_csv(f'{data_path}/{data_type}/{exp_id}_{curve_type}.csv')


def get_pm_accuracy_based_on_rank(folder_path, combo_dict, df_accuracy):
    """
    Takes PrimerMixes list and accuracies to return only 
    the selected combo by using the combo_dict.

    param:
    - folder_path = path where data are stored.
    - combo_dict = dictionary with the position of the PrimerMix
                   based on scores (i.e. top_six, bot_six, etc...).
    - df_accuracy = a dataframe with all the accuracies 
                    for all the combination created or tested.
    
    return: dataframe with selected primer mixes and 
            KNN accuracies (both raw and norm) plus position.
    """

    combo_list = []
    type_list = []
    
    for pm_position in combo_dict.keys():
        combo_list.append(pd.read_csv(f'{folder_path}/selected_combinations/assay_combinations_{combo_dict[pm_position]}.csv', index_col=0))
    
    for index, row in df_accuracy.iterrows():
        for combo_index, df_combo in enumerate(combo_list):
            if row.name in df_combo.index:
                type_list.append(combo_dict[combo_index])
                
    df_box = pd.concat([df_accuracy.reset_index(), pd.Series(type_list, name='Position')], axis=1)
    
    return df_box.sort_values(by='Position')




 #################




def load_processed_data_by_name(folder_path, curve_type, data_type, NMETA):
    """
    Loading the DataFrames for scoring computing.

    - folder_path: 
    - curve_type:
    - data_type:
    - NMETA:

    return
    """
    
    if curve_type == 'c_param':
        df_to_compute_temp = pd.read_csv(f'{folder_path}/processed_data/fitted_param_{data_type}.csv', index_col = 0)
        return get_columns_by_name(df_to_compute_temp, NMETA, ['c'])
    
    if curve_type == 'norm_c_param':
        df_to_compute_temp = pd.read_csv(f'{folder_path}/processed_data/norm_fitted_param_{data_type}.csv', index_col = 0)
        return get_columns_by_name(df_to_compute_temp, NMETA, ['c'])
    
    else:
        df_to_compute_temp = pd.read_csv(f'{folder_path}/processed_data/{curve_type}_{data_type}.csv', index_col = 0)
        return get_columns_by_name(df_to_compute_temp, NMETA)
        

def load_processed_data(folder_path, id_list, curve_type, NMETA):
    """
    Loading the DataFrames for scoring computing.

    - folder_path: 
    - curve_type:
    - data_type:
    - NMETA:

    return
    """
    
    if curve_type == 'c_param':
        df_to_compute_temp = load_processed_data_by_expid(folder_path, id_list, 'fitted_param')
        return get_columns_by_name(df_to_compute_temp, NMETA, ['c'])
    
    if curve_type == 'norm_c_param':
        df_to_compute_temp = load_processed_data_by_expid(folder_path, id_list, 'norm_fitted_param')
        return get_columns_by_name(df_to_compute_temp, NMETA, ['c'])
    
    else:
        df_to_compute_temp = load_processed_data_by_expid(folder_path, id_list, curve_type)
        return get_columns_by_name(df_to_compute_temp, NMETA)


def load_data_for_training(folder_path, exp_id_train, exp_id_test, CURVE_TYPE):
    """

    """
    if (CURVE_TYPE == 'raw_rb') | (CURVE_TYPE == 'norm_curve') | (CURVE_TYPE == 'fitted_param'):
        df_train = load_processed_data_by_expid(folder_path, exp_id_train, CURVE_TYPE)
        df_test = load_processed_data_by_expid(folder_path, exp_id_test, CURVE_TYPE)
    
    if (CURVE_TYPE == 'fitted_param_without_d') | (CURVE_TYPE == 'c_param'):
        df_train = load_processed_data_by_expid(folder_path, exp_id_train, 'fitted_param')
        df_test = load_processed_data_by_expid(folder_path, exp_id_test, 'fitted_param')
        
    return df_train, df_test


def get_columns_by_name(df, NMETA, column_list = []):
    """
    
    """
    if column_list == []:
        return df
    else:
        return pd.concat([df.iloc[:, :NMETA], df.loc[:, column_list]], axis = 1)

    


def load_selected_combo(data_path, id_list, combo_dict):
    """
    load data for selected combinations.
    (e.g. "all", "selected", "top_six")

    params:
    - data_path: file path where data is stored
    - id_list: list of id to specify combinations
    - combo_dict: dictionary of combo options

    return:
        integrated dataframe containing data specified by input id list
    """

    data_list = []
    for exp_id in id_list:
        data_list.append(pd.read_csv(f'{data_path}/selected_combinations/assay_combinations_{combo_dict[exp_id]}.csv'))
    
    return pd.concat(data_list, axis=0).set_index('Label')
    