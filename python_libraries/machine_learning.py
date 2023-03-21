import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import python_libraries.data_loading as loadfunc
import python_libraries.grouping_function as groupfunc

def get_curves_type(pm_dict, NMETA, curve_type = 'raw_rb'): 
    """
    helper function to extract specified data for training.

    params:
    - pm_dict: dictionary containing all primermix information
    - NMETA: number of metadata information
    - curve_type: types of curves (default: "raw_rb")

    return:
    - x_train: training data based on selected curve type
    - x_test: testing data based on selected curve type
    """
    
    if (curve_type == 'raw_rb') | (curve_type == 'norm_curve'):
        x_train = pm_dict['training'].iloc[:, NMETA:]
        x_test = pm_dict['testing'].iloc[:, NMETA:]
        
    if curve_type == 'fitted_param':
        x_train = pm_dict['training'].loc[:, ['a', 'b', 'c', 'd', 'e']]
        x_test = pm_dict['testing'].loc[:, ['a', 'b', 'c', 'd', 'e']]    
    
    if curve_type == 'fitted_param_without_d':
        x_train = pm_dict['training'].loc[:, ['a', 'b', 'c', 'e']]
        x_test = pm_dict['testing'].loc[:, ['a', 'b', 'c', 'e']]
        
    if curve_type == 'c_param':
        x_train = pm_dict['training'].loc[:, ['c']]
        x_test = pm_dict['testing'].loc[:, ['c']]
        
    return x_train, x_test


def model_training(folder_path, exp_id_train, exp_id_test, df_combination, 
                                        curve_type, NMETA, model, model_name):
    '''
    Train the model and get the accuracies.

    - df_singleplex: DataFrame with singleplex information
    - df_multiplex: DataFrame with multiplex information
    - training_object: A string indicates which section of the data is used
    - model: An array containing different classification methods
    - model_name: An array containing names of different classification methods
    - df_index: An array containing names of different combinations
    
    return:
    - df_accuracy_array: DataFrame with accuracy
    - df_array_multiplex: Arrays containing dataframe for each multiplex combination
    - label_array: Arrays containing primer information for each combination
    '''
    
    df_train, df_test = loadfunc.load_data_for_training(
                            folder_path, exp_id_train, exp_id_test, curve_type)
    
    pm_dict = groupfunc.group_train_and_test_dfs(
                            df_train, df_test, df_combination)
    
    accuracy_array = []
    confusion_matrix_array = []
    pms = []

    df_column = [x + curve_type for x in model_name]
    
    print(f'\nM2M accuracy computing: {curve_type.upper()}')
    
    for pm in pm_dict:
        print(pm)
        pms.append(pm)

        sample_dict = pm_dict[pm]
        
        x_train, x_test = get_curves_type(sample_dict, NMETA, curve_type=curve_type)
        
        y_train = sample_dict['training'].Target.values
        y_test = sample_dict['testing'].Target.values

        assay_accuracy = []
        assay_confusion_matrix = []
        label = np.unique(y_train)
        label_number = len(label)

        for m in model:
            m.fit(x_train, y_train)
            assay_accuracy.append(m.score(x_test, y_test))

            y_pred = m.predict(x_test)
            confusionmatrix = np.zeros((label_number, label_number))
            confusionmatrix = np.add(confusionmatrix, confusion_matrix(y_test, y_pred, labels=label))
            df_confusion_matrix = pd.DataFrame(confusionmatrix, columns=label, index=label)
            assay_confusion_matrix.append(df_confusion_matrix)
    
        accuracy_array.append(assay_accuracy)
        confusion_matrix_array.append(assay_confusion_matrix)
        
    df_accuracy_array = pd.DataFrame(accuracy_array,
                                     columns=df_column,
                                     index=pm_dict.keys())
    
    return pms, df_accuracy_array, confusion_matrix_array


def FitAndEvaluateModel(X, Y, model, n_kfolds=10):
    """
    helper function to evaluate accuracy using StratifiedKFold cross-validations.
    call in function "model_training_M2M_kfold()".

    params:
    - X: training data
    - Y: label data
    - model: classification model (e.g. KNNClassifier)
    - n_kfolds: number of folds to evaluate (default: 10)

    return:
    - conf_matrix: accumulated confusion matrix from each fold
    """

    # Use StratifiedKFold cross-validations
    skf = StratifiedKFold(n_splits=n_kfolds, shuffle=True, random_state=0)
    
    labels = np.unique(Y)
    num_labels = len(labels)
    conf_matrix = np.zeros( (num_labels, num_labels) )
    
    for train_index, test_index in skf.split(X, Y):

        # Divide the k-fold dataset
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = Y[train_index], Y[test_index]

        # Fit the model provided and predict the y-values
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Obtain Confusion Matrix for that fold and add it to the previous one
        conf_matrix = np.add(conf_matrix, confusion_matrix(y_test, y_pred, labels=labels))
        
    return conf_matrix


def model_training_M2M_kfold(df_multi, df_combination, NMETA, models, model_name, CURVE_TYPE, PLEX_TYPE="3_plex"):
    """
    classification accracy computation for each combination using K-fold.

    params:
    - df_multi: dataframe of multiplex data (including META and fluorescence value at each cycle)
    - df_combination: dataframe containing primer information for interesting combinations
    - NMETA: number of metadata columns
    - models: classification model (e.g. KNNClassifier)
    - model_name: name of classification model (e.g. "KNN")
    - CURVE_TYPE: type of curve used for accuracy evaluation ("raw", "norm", "fitted_param")
    - PLEX_TYPE: number of plex of the evaluation (e.g. "3_plex", "7_plex")

    return:
    - df_accuracy: dataframe containing classification accuracy for all combinations
    - df_cm_list: dataframe containing confusion matrix for each combination evaluation
    """
    
    df_cm_list = []

    accuracy_list = []
    
    df_accuracy = df_combination.copy()
    
    NTARGET = len(df_multi['Target'].unique())
    
    for index, model in enumerate(models):
        
        print(f'Curve type: {CURVE_TYPE.upper()}, Plex type: {PLEX_TYPE.upper()}, Model name: {model}')
        
        for combination in df_combination.index:
            df_temp = df_multi[df_multi['PrimerMix'] == combination]

            X = df_temp.iloc[:, NMETA:]
            y = df_temp.loc[:, 'Assay'].values

            cf_matrix = FitAndEvaluateModel(X, y, model)

            df_cm_temp = pd.DataFrame(cf_matrix, columns=np.unique(y), index = np.unique(y))

            df_cm_list.append(df_cm_temp)

            accuracy_list.append(np.sum(df_cm_temp.values * np.eye(NTARGET))/np.sum(df_cm_temp.values) * 100)
            
        df_accuracy[f'{model_name[index]}'] = accuracy_list
    
    return df_accuracy, df_cm_list
