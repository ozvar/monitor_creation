import tensorflow as tf
import numpy as np
import os, glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_gtsrb, load_cifar 
from utils import setup_logger
from combineTransform import combinatorial_transf
from parameters import *


def _sortfunc(j: str) -> int:
    el = 0
    # Extract the number portion from the string
    num_str = j.split('data')[-1].split('.npy')[0]
    if num_str.isdigit():
        el = int(num_str)
    return el


def computePerformance(
        model: tf.keras.models,
        data_dir: str,
        labels: pd.DataFrame
        ) -> np.array:
    list_files = [f for f in glob.glob(os.path.join(data_dir, '*.npy')) 
                  if os.path.basename(f).startswith('data')]
    accuracy = np.zeros(len(list_files))
    list_files.sort(key=_sortfunc)
    for i, filename in enumerate(list_files):
        print(filename)
        data = np.load(filename)
        per = model.evaluate(data, labels)
        accuracy[i] = per[1]
    
    return accuracy


def label_dataset(
        accuracy: np.array,
        labels: np.array
        ) -> list:
    labData=[]
    for i in range(len(accuracy)):
        for j in range(len(labels)):
            if (j == 0 and accuracy[i] >= labels[j]):
                labData.append((accuracy[i], j))
                break
            elif (j != 0 and labels[j - 1] > accuracy[i] >= labels[j]):
                labData.append((accuracy[i], j))
            elif (j == len(labels)-1 and accuracy[i]< labels[j]):
                labData.append((accuracy[i], j+1))
    return labData


def table_of_class_against_epsilons(
        transf_factors: dict,
        transf: np.array,
        labels: np.array,
        acc_bounds: np.array
        ):
    # Initialize a dictionary to hold epsilon ranges for each class
    class_epsilon_ranges = {i: {factor: [] for factor in transf_factors.keys()} for i in range(len(acc_bounds) + 1)}
    # Iterate over labels to fill class_epsilon_ranges with epsilon values
    for i, label_info in enumerate(labels):
        class_label = int(label_info[1])
        for j, factor in enumerate(transf_factors.keys()):
            class_epsilon_ranges[class_label][factor].append(transf[i][j])
    # Convert epsilon lists to ranges (min-max)
    for class_label, factors in class_epsilon_ranges.items():
        for factor, epsilons in factors.items():
            min_eps, max_eps = min(epsilons), max(epsilons)
            # Update the dictionary with ranges
            class_epsilon_ranges[class_label][factor] = f"{factor} {min_eps:.2f}-{max_eps:.2f}"
    # Create DataFrame from class_epsilon_ranges
    df_rows = []
    for class_label, factors in class_epsilon_ranges.items():
        row = [f"{v}" for v in factors.values()]
        df_rows.append(["Class " + str(class_label)] + row)
    # Adding column names
    column_names = ['Class'] + list(transf_factors.keys())
    # Create and return the DataFrame
    df = pd.DataFrame(df_rows, columns=column_names)
    
    return df


def log_accuracy_across_classes(
        transf_factors: dict,
        epsilons: np.array,
        transf_dir: Path,
        data_dir: Path,
        acc_bounds: np.array,
        log_label: str='labels'
        ):
    logger= setup_logger(data_dir, 0, log_label)
    # create table of epsilon values in each class
    comb_prod = combinatorial_transf(
            transf_factors=transf_factors,
            eps=epsilons,
            )
    labData = np.load(transf_dir / 'labDatasets.npy')
    df = table_of_class_against_epsilons(
            transf_factors=transf_factors,
            transf=comb_prod,
            labels=labData,
            acc_bounds=acc_bounds
            )
    logger.info('Table of epsilon values in each class')
    logger.info(df)


def compute_and_save_accuracies(
        model_dir: Path,
        dataset: str,
        model_name: str,
        data_dir: Path,
        acc_bounds: list):
    # load model
    model_dir, data_dir = Path(model_dir), Path(data_dir)
    model = tf.keras.models.load_model(model_dir / dataset / model_name)
    # load data based on the dataset name
    if dataset == 'gtsrb':
        _, _, _, y_test, labels = load_gtsrb()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    # compute performance
    acc = computePerformance(
            model=model,
            data_dir=data_dir,
            labels=y_test)
    np.save(data_dir / 'accuracy', acc)
    # label datasets 
    labels = np.array(acc_bounds)
    labData = label_dataset(
            accuracy=acc,
            labels=labels)
    np.save(data_dir / 'labDatasets', labData)
