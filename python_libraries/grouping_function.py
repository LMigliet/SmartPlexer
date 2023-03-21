
def get_primermix_dataframes(df_data, df_combination, data_type):
    """
    Groupping all the information of each assays (or singleplex assays) 
    into a PrimerMix (or multiplex) and returning a dictionary with 
    keys: PrimerMix
    values = dataframe with all the assays information (NMETA and dPCR raw data)
    
    params:
    - df: DataFrame to group
    - df_combination: DataFrame of combination information
    - data_type: could be singleplex or multiplex

    return:
    - pm_dict: Dictionary containing all the information
                   (key: PrimerMix, values: dataframe of that primermix)
    """

    ASSAY_CHECK = 'Assay'
    GROUP_PM = 'PrimerMix'

    pm_dict = {}
    
    if data_type == 'singleplex':
        assays = df_combination.copy()
        for pm, df in assays.groupby(assays.index):
            pm_dict[pm] = df_data[df_data[ASSAY_CHECK].isin(df.values[0])]
            
    if data_type == 'multiplex':
        for pm, df in df_data.groupby(GROUP_PM):
            pm_dict[pm] = df
            
    return pm_dict


def group_single_and_multi_dfs(df_singleplex, df_multiplex, df_combination):
    """
    group the data for model training. Example:
    
    pm_dict = 
                {
                    'PM3.01':{
                        'singleplex': df_single
                        'multiplex': df_multi
                    }
                }

    params:
    - df_singleplex: DataFrame of the singleplex data
    - df_multiplex: DataFrame of the multiplex data
    - df_combination: DataFrame of combination information
    
    return:
    - pm_dict: Dictionary containing another dictionary of the pm with all the information
    """

    ASSAY_CHECK = 'Assay'
    GROUP_PM = 'PrimerMix'
    REMOVE_ON = 'Target'

    pm_dict = {}
    
    # singleplex
    assays = df_combination.copy()
    for pm, df__ in assays.groupby(assays.index):
        pm_dict[pm] = {"singleplex" : df_singleplex[df_singleplex[ASSAY_CHECK].isin(df__.values[0])]}

    # multiplex
    for pm, df_ in df_multiplex.groupby(GROUP_PM):
        pm_dict[pm]["multiplex"] = df_

    # remove the PrimerMix without complete data in both Single and Multiplex
    for pm, pm_df in pm_dict.copy().items():
        if len(pm_df['singleplex'][REMOVE_ON].unique()) != len(pm_df['multiplex'][REMOVE_ON].unique()):
            del pm_dict[pm]
            
    return pm_dict


def group_train_and_test_dfs(df_train, df_test, df_combination):
    """
    group the data for model training. Example:
    
    pm_dict = 
                {
                    'PM3.01':{
                        'singleplex': df_single
                        'multiplex': df_multi
                    }
                }

    params:
    - df_singleplex: DataFrame of the singleplex data
    - df_multiplex: DataFrame of the multiplex data
    - df_combination: DataFrame of combination information
    
    return:
    - pm_dict: Dictionary containing another dictionary of the pm with all the information
    """

    ASSAY_CHECK = 'Assay'
    GROUP_PM = 'PrimerMix'
    REMOVE_ON = 'Target'

    pm_dict = {}
    
    # singleplex
    for pm, df_ in df_train.groupby(GROUP_PM):
        pm_dict[pm] = {"training": df_}

    # multiplex
    for pm, df_ in df_test.groupby(GROUP_PM):
        pm_dict[pm]["testing"] = df_

    # remove the PrimerMix without complete data in both Single and Multiplex
    for pm, pm_df in pm_dict.copy().items():
        if len(pm_df['training'][REMOVE_ON].unique()) != len(pm_df['testing'][REMOVE_ON].unique()):
            del pm_dict[pm]
            
    return pm_dict
    