import tensorflow as tf
import numpy as np
import os, glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_gtsrb, load_cifar 
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
