import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from itertools import cycle
from typing import Dict, List
from datetime import datetime


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


def plot_metrics(history, run_id, k_fold, fig_dir, metric='loss'):
    sns_styleset()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend()
    plot_path = fig_dir / f'{timestamp}_run_{run_id+1}_kfold_{k_fold}_{metric}.png'
    plt.savefig(plot_path)
    plt.close()


def accuracy_heatmaps(
    transf: list,
    accuracies: np.array,
    transf_factors: Dict[str, float],
    epsilons: list,
    factor: str,
    fig_dir: str):

    split_eps = split_by_factor(
        transf=transf,
        accuracies=accuracies,
        transf_factors=transf_factors,
        factor=factor
    )
    # define column/fig labels
    labels = list(transf_factors.keys())
    labels.append('accuracy')
    # generate dataframe and plot for each unique epsilon value
    for i in range(len(epsilons)):
        df = pd.DataFrame(split_eps[i], columns=labels)
        df = df.round(2)
        factor_val = df[factor][0].round(1)
        df.drop(columns=[factor], inplace=True)
        # produce heatmap
        heatmap_df = df.pivot(df.columns[0], df.columns[1], df.columns[2])
        ax = sns.heatmap(heatmap_df, annot=False, cmap='flare', vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
        ax.invert_yaxis()
        plt.title(f'{factor.capitalize()} = {factor_val}')
        plt.xlabel(df.columns[1].capitalize())
        plt.ylabel(df.columns[0].capitalize())
        # save and close
        fig_name = f'epsilons_accuracy_heatmap_{factor}_constant_at_{factor_val}.png'
        plt.savefig(os.path.join(fig_dir, fig_name), bbox_inches='tight')
        plt.close()


def two_factor_accuracy_heatmap(
    transf: List[List[float]],
    accuracies: np.array,
    transf_factors: Dict[str, float],
    epsilons: list,
    fig_dir: str):
    data = []
    for i, eps_combination in enumerate(transf):
        # Each row in the data to include factor names as keys and their corresponding epsilon as values, plus accuracy
        row = {factor: eps for factor, eps in zip(transf_factors.keys(), eps_combination)}
        row['accuracy'] = accuracies[i]
        data.append(row)
    # Convert list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    df = df.round(2)
    # Assuming the first two keys in transf_factors are the ones to be used for axes
    factor_x, factor_y = list(transf_factors.keys())[:2]
    # Pivot the DataFrame to get a matrix where rows are one factor, columns are another, and cells are accuracies
    heatmap_data = df.pivot(index=factor_x, columns=factor_y, values='accuracy')
    # Plot the heatmap
    #plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=False, fmt=".2f", vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
    plt.xlabel(factor_y.capitalize())
    plt.ylabel(factor_x.capitalize())
    plt.gca().invert_yaxis()
    # Save the figure
    fig_name = 'model_accuracy_heatmap.png'
    plt.savefig(os.path.join(fig_dir, fig_name), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    true_labels: np.array,
    predicted_labels: np.array,
    run_id: int,
    kth_fold: int,
    fig_dir: Path):
    cm = confusion_matrix(true_labels, predicted_labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig_name = f'{timestamp}_confusion_matrix_run_{run_id+1}_kfold_{kth_fold}.png'
    plt.savefig(os.path.join(fig_dir, fig_name), bbox_inches='tight')


def plot_roc_curve(
        true_labels: np.array,
        predicted_labels: np.array,
        n_classes: int,
        run_id: int,
        kth_fold: int,
        fig_dir: Path):
    sns_styleset()
    # compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predicted_labels[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), predicted_labels.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # cggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # plot all ROC curves
    plt.figure(figsize=(8, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average OvR (area = {roc_auc["micro"]:0.2f})',
             linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average OvR (area = {roc_auc["macro"]:0.2f})',
             linestyle=':', linewidth=4)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'Class {i+1} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig_name = f'{timestamp}_roc_curve_run_{run_id+1}_fold_{kth_fold+1}.png'
    plt.savefig(fig_dir / fig_name, bbox_inches='tight')
    plt.close()
