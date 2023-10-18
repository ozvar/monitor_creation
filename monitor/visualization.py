import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


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
    matplotlib.rcParams['figure.dpi']        = 300
    matplotlib.rcParams['axes.linewidth']    = 1
    matplotlib.rcParams['grid.color']        = '.8'
    matplotlib.rcParams['axes.edgecolor']    = '.15'
    mpl.rcParams['axes.spines.right']        = False
    mpl.rcParams['axes.spines.top']          = False
    matplotlib.rcParams['xtick.bottom']      = True
    matplotlib.rcParams['ytick.left']        = True
    matplotlib.rcParams['xtick.major.width'] = 1
    matplotlib.rcParams['ytick.major.width'] = 1
    matplotlib.rcParams['xtick.color']       = '.15'
    matplotlib.rcParams['ytick.color']       = '.15'
    matplotlib.rcParams['xtick.major.size']  = 3
    matplotlib.rcParams['ytick.major.size']  = 3
    matplotlib.rcParams['font.size']         = 14
    matplotlib.rcParams['axes.titlesize']    = 14
    matplotlib.rcParams['axes.labelsize']    = 13
    matplotlib.rcParams['legend.fontsize']   = 14
    matplotlib.rcParams['legend.frameon']    = False
    matplotlib.rcParams['xtick.labelsize']   = 13
    matplotlib.rcParams['ytick.labelsize']   = 13
