import tensorflow as tf
import numpy as np
import os, glob
import pandas as pd
from datasets import load_gtsrb


def _sortfunc(j: int) -> int:
    el = 0
    if len(j) == 22:
        el = int(j[17])
    if len(j) == 23:
        el = int(j[17:19])
    if len(j) == 24:
        el = int(j[17:20])
    return el

def computePerformance (model: tf.keras.models, dir: str, labels: pd.DataFrame ) -> np.array :
    i = 0
    list_files = glob.glob(os.path.join(dir, "*.npy"))
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

model= tf.keras.models.load_model('models/gtsrb/model3b.h5')
[X_train, y_train, X_test, y_test, labels] = load_gtsrb()
print(y_test)
dir = "modifieddata"
#acc = computePerformance(model, dir, y_test)
#np.save(os.path.join("modifieddata", "accuracy"), acc)
acc = np.load("modifieddata/accuracy.npy")
labels = np.array([0.70, 0.40])
labData = labelDataset(acc, labels)
print(labData[0][1])
print(acc)
np.save(os.path.join("modifieddata", "labDatasets"),labData)
print(labData)
