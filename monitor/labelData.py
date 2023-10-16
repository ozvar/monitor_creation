import tensorflow as tf
import numpy as np
import os, glob
import pandas as pd
from datasets import load_gtsrb, load_cifar 
from parameters import *


def _sortfunc(j: int) -> int:
    el = 0
    if len(j) == 22:
        el = int(j[17])
    if len(j) == 23:
        el = int(j[17:19])
    if len(j) == 24:
        el = int(j[17:20])
    return el


def computePerformance (model: tf.keras.models, data_dir: str, labels: pd.DataFrame ) -> np.array :
    i = 0
    list_files = glob.glob(os.path.join(data_dir, '*.npy'))
    accuracy = np.zeros(len(list_files))
    list_files.sort(key=_sortfunc)
    for filename in list_files:
        data = np.load(filename)
        per = model.evaluate(data, labels)
        accuracy[i] = per[1]
        print(filename)
        i += 1
    return accuracy


def labelDataset (accuracy : np.array, label : np.array) -> list:

    labData=[]
    for i in range(len(accuracy)):
        for j in range(len(label)):
            if (j == 0 and accuracy[i] >= label[j]):
                labData.append((accuracy[i], j))
                break
            elif (j != 0 and label[j - 1] > accuracy[i] >= label[j]):
                labData.append((accuracy[i], j))
            elif (j == len(label)-1 and accuracy[i]< label[j]):
                labData.append((accuracy[i], j+1))
    return labData


if __name__ == "__main__":
    model= tf.keras.models.load_model(os.path.join(MODEL_DIR, DATASET, MODEL))
    # preparing for option to train on CIFAR too
    if dataset == 'gtsrb': 
        [X_train, y_train, X_test, y_test, labels] = load_gtsrb()
    print(y_test)
    acc = computePerformance(model, data_dir, y_test)
    np.save(os.path.join(DATA_DIR, 'accuracy'), acc)
    acc = np.load(os.path.join(DATA_DIR, 'accuracy.npy'))
    labels = np.array(ACC_BOUNDS)
    labData = labelDataset(acc, labels)
    print(labData[0][1])
    print(acc)
    np.save(os.path.join(DATA_DIR, 'labDatasets'), labData)
    print(labData)