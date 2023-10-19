import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dic


# configure pandas table display
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def sns_styleset():
    """Configure parameters for plotting"""
    sns.set_theme(context='paper',
                  style='whitegrid',
                  # palette='deep',
                  palette=['#c44e52',
                           '#8c8c8c',
                           '#937860',
                           '#ccb974',
                           '#4c72b0',
                           '#dd8452'],
                  font='Arial')
    mpl.rcParams['figure.dpi']        = 300
    mpl.rcParams['axes.linewidth']    = 1
    mpl.rcParams['grid.color']        = '.8'
    mpl.rcParams['axes.edgecolor']    = '.15'
    mpl.rcParams['axes.spines.right']        = False
    mpl.rcParams['axes.spines.top']          = False
    mpl.rcParams['xtick.bottom']      = True
    mpl.rcParams['ytick.left']        = True
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['xtick.color']       = '.15'
    mpl.rcParams['ytick.color']       = '.15'
    mpl.rcParams['xtick.major.size']  = 3
    mpl.rcParams['ytick.major.size']  = 3
    mpl.rcParams['font.size']         = 14
    mpl.rcParams['axes.titlesize']    = 14
    mpl.rcParams['axes.labelsize']    = 13
    mpl.rcParams['legend.fontsize']   = 14
    mpl.rcParams['legend.frameon']    = False
    mpl.rcParams['xtick.labelsize']   = 13
    mpl.rcParams['ytick.labelsize']   = 13


def split_by_factor(
    transf: np.array,
    accuracies: np.array,
    transf_factors: Dict[str, float],
    factor: str) -> np.array:

    if factor not in transf_factors:
        raise ValueError(f'Invalid transformation factor provided. Valid factors are {list(transf_factors.keys())}')

    col = list(transf_factors.keys()).index(factor)
    
    transf = np.vstack(transf)
    sorted_indexes = np.argsort(transf[:, col])
    sorted_transf = transf[sorted_indexes]
    sorted_acc = accuracies[sorted_indexes]
    # join epsilon arrays and accuracies
    sorted_acc = sorted_acc[:, np.newaxis]
    data = np.hstack((sorted_transf, sorted_acc))
    
    unique_factor_vals = np.unique(sorted_transf[:, col])
    
    split_eps = [data[data[:, col] == contrast] for contrast in unique_factor_vals]

    return split_eps



