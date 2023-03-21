import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from sklearn.linear_model import LinearRegression

import python_libraries.score_processing as scorefunc
import python_libraries.data_loading as loadfunc
import python_libraries.utilities as utils

utils.colourblind()

def curve_plot_by(df, NMETA, by_column='Assay', col_number=3, dpi=100, size=(10, 10), 
                  color='C00', title_font=20, label_font=10, ylim=True, grid=True, alpha=0.3):
    """
    Plot any curves or group of assays and group them by a column decided by the user.

    param:
    - df: input dataframe (tabular data for amplification curves)
    - NMETA: number of metadata columns
    - by_column: Selected column by which to group (default: "Assay")
    - col_number: number of subplots on each row (default: 3)
    - dpi: dpi of the figure (default: 100)
    - size: size of the figure (default: (10, 7))
    - color: plotting color of curves (default: "C00")
    - title_font: size of figure title (default: 20)
    - label_font: size of label (default: 10)
    - same_range: same scale of y-axis for all the subplots (default: False)
    - ylim: range of y-axis (default: [0, 3])
    - grid: if showing grid in the plot (default: True)
    - alpha: transparency of grid (default: 0.3)

    return plots
    """
    
    n_unique_id = len(df[by_column].unique())
    row_number = math.ceil(n_unique_id / col_number)

    fig, ax = plt.subplots(row_number, col_number, figsize=size, dpi=dpi)
    ax = ax.flatten()

    for i, (assay, df_) in enumerate(df.groupby(by_column)):

        ax[i].plot(df_.iloc[:, NMETA:].T, color=color)
        ax[i].set_title(assay, size=title_font)
        ax[i].set_xlabel('Cycle', size=label_font)
        ax[i].set_ylabel('Fluoresence', size=label_font)

        if ylim == True:
            ylim_max = df.iloc[:, NMETA:NMETA+45].iloc[:,-1].max()
            ax[i].set_ylim(-0.1, ylim_max+0.2)
        
        if grid == True:
            ax[i].grid(alpha=alpha)

    plt.tight_layout()
    plt.show()


def plot_comparison_S2M(df_single, df_multi, NMETA, selected_assay=False, groupby=['PrimerMix','Assay']):
    """
    Plotting side by side the singleplex and multiplex curves for each assay
    
    param:
    - df_single: dataframe of singleplex data
    - df_multi: dataframe of multiplex data
    - NMETA: number of metadata columns
    - selected_assay: default is False, but you can chose to plot only one assay (i.e. 'CHK_N_02')
    - groupby: grouped criterion

    return plots
    """
    
    if selected_assay:
        df_multi = df_multi.loc[df_multi['Assay']==selected_assay]

    for i, ((pm, assay), df_) in enumerate(df_multi.groupby(groupby)):

        fig, ax = plt.subplots(1,2, figsize=(6,3), dpi=100)
        fig.suptitle(f"{pm} -- Multiplex panel {df_['Channel'].unique()}", size=20)
        
        ax[0].plot(df_single[df_single['Assay']==assay].iloc[:, NMETA:].T, color="C00")
        ax[0].set_title(f'Singleplex: {assay}')
        ax[0].set_xticklabels([])
        ax[0].grid(alpha=0.3)
        
        ax[1].plot(df_.iloc[:,NMETA:].T, color="C01")
        ax[1].set_title(f'Multiplex: {assay}')
        ax[1].set_xticklabels([])
        ax[1].grid(alpha=0.3)
        
        # setting the ylim so they look the same as is side-by-side
        ylim_max_single = ax[0].get_ylim()[1]
        ylim_max_multi = ax[1].get_ylim()[1]
        
        if ylim_max_single >= ylim_max_multi: 
            ax[0].set_ylim(-0.1, ylim_max_single)
            ax[1].set_ylim(-0.1, ylim_max_single)
        else:
            ax[0].set_ylim(-0.1, ylim_max_multi)
            ax[1].set_ylim(-0.1, ylim_max_multi)

        plt.tight_layout()
        plt.show()

def combination_plot(folder_path, exp_id, df_combination, NMETA, x_label='', y_label='', size=(10, 7), dpi=100,
                     col_number=3, title_font=20, label_font=10, legend=False, alpha=0.3):
    """
    Plot median curves for all targets in a combination.
    This is used for visualization of selected combination, and will give an intuition how "good" a combination is.
    "Good" means the combo may performance well in ACA classifier development.

    params:
    - folder_path: file path for processed data
    - exp_id: list of experiment ID to load data from
    - df_combination: dataframe containing all combination information (primers)
    - NMETA: number of metadata columns
    - x_label: label for x-axis
    - y_label: label for y-axis
    - size: size of the figure (default: (10, 7))
    - dpi: dpi of the figure (default: 100)
    - col_number: number of subplots on each row (default: 3)
    - title_font: size of figure title (default: 20)
    - label_font: size of label (default: 10)
    - legend: if to show legend or not (default: False)
    - alpha: transparency of grid (default: 0.3)

    return plots
    """
    row_number = math.ceil(df_combination.shape[0] / col_number)
    
    fig, axs = plt.subplots(row_number, col_number, figsize=size, dpi=dpi)

    x_interval = 4
    
    df_curve = loadfunc.load_processed_data_by_expid(folder_path, exp_id, 'raw_rb')
    
    colormap = [f'C0{i}' for i in range(len(df_curve['Target'].unique()))]
    
    for row in range(row_number):
        for col in range(col_number):
            if row * col_number + col < df_combination.shape[0]:
                df_temp = df_curve[df_curve['Assay'].isin(df_combination.iloc[row * col_number + col, :].values)]
                target_counter = 0
                for assay, assay_df in df_temp.groupby('Assay'):
                    axs[row, col].plot(assay_df.iloc[:, NMETA:].T.median(axis = 1),
                                       color = colormap[target_counter], label = assay)
                    target_counter = target_counter + 1
                axs[row, col].set_xlabel(x_label, size = label_font)
                axs[row, col].set_ylabel(y_label, size = label_font)
                axs[row, col].set_title(df_combination.index.tolist()[row * col_number + col], size = title_font)
                axs[row, col].set_xticks(np.linspace(1, 45, x_interval, dtype=int))
                axs[row, col].grid(alpha = alpha)

                if legend == True:
                    axs[row, col].legend()

    plt.tight_layout()
    plt.show()


def scatter_plot_with_correlation_line(df_x, df_y, col_number=2, x_label='', y_label='', dpi=100, 
                                       size=(10, 7), title_font=15, label_font=10, alpha=0.3,
                                       primer='', legend=False, legend_size=6, dot_size=10, line_size=1):
    """
    plot the correlation for 4 types of scores with linear regression line.

    params:
    - df_x: score dataframe for singleplex data
    - df_y: score dataframe for multiplex data
    - col_number: number of subplots on each row
    - x_label: label for x-axis
    - y_label: label for y-axis
    - dpi: dpi of the figure (default: 100)
    - size: size of the figure (default: (10, 7))
    - title_font: size of figure title (default: 20)
    - label_font: size of label (default: 10)
    - alpha: transparency of grid (default: 0.3)
    - primer: list of primermix information
    - legend: if to show legend or not (default: False)
    - legend_size: size of legend
    - dot_size: size of each marker in scatter plot
    - line_size: width of correlation line

    return:
    - fig: figure with correlation result for different types of score
    """
    correlation_matrix = scorefunc.generate_correlation_df_S2M(df_x, df_y)
        
    row_number = math.ceil(df_x.shape[1] / col_number)

    fig, axs = plt.subplots(row_number, col_number, figsize=size, dpi=dpi)

    colors = cm.Set1(np.linspace(0, 1, df_x.shape[0]))
    for row in range(row_number):
        for col in range(col_number):
            if row * col_number + col < df_x.shape[1]:
                x = df_x.iloc[:, row * col_number + col]
                y = df_y.iloc[:, row * col_number + col]
                
                plot_correlation_line(axs, row, col, x, y, line_size=line_size)
                
                for x_temp, y_temp, color_temp, primer_temp in zip(x, y, colors, primer):
                    axs[row, col].scatter(x_temp, y_temp, color = color_temp, label = primer_temp, s = dot_size)
                axs[row, col].set_title(df_x.columns[row * col_number + col] + '(coeff = ' + str(round(correlation_matrix.iloc[1, row * col_number + col], 2)) + ')',
                                        size = title_font)
                axs[row, col].set_xlabel(x_label, size = label_font)
                axs[row, col].set_ylabel(y_label, size = label_font)
                axs[row, col].grid(alpha = alpha)
                if legend == True:
                    axs[row, col].legend(prop={'size': legend_size})
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_correlation_line(axs, row, col, x, y, color = cm.Set1(1), line_size = 1):
    """
    plot regression line based on the multiplex-singleplex score dot point.

    params:
    - axs: figure instance to plot correlation line
    - row: row index to specify the subplot
    - col: col index to specify the subplot
    - x: training data for linear regression model
    - y: label data for linear regression model
    - color: color of regression line
    - line_size: width of the regression line
    """

    X = x.values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    
    y_pred = np.sort(X, axis = 0).dot(reg.coef_) + reg.intercept_
    axs[row, col].plot(np.sort(x.values), y_pred, color = color, linewidth = line_size)
