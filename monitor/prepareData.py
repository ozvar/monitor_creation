import numpy as np
import os
import typing
import random
import pickle
from pathlib import Path
from tensorflow.keras.utils import to_categorical


def count_samples_classes(labData: list, num_classes: int) -> np.array:
    count_classes = np.zeros(num_classes)
    for el in labData:
        i = el[1]
        count_classes[int(i)] += 1
    return count_classes


def find_indexes(labData: list, cl: int, num_classes: int)-> np.array:
    num_samples = count_samples_classes(labData, num_classes)
    ind = np.zeros(int(num_samples[cl]))
    j = 0
    for i in range(len(labData)):
        if labData[i][1] == cl:
            ind[j] = i
            j += 1
    return ind


def find_samples (ind: np.array, num_ind_tr: int, num_ind_test: int) -> typing.Tuple[np.array, np.array]:
    indexes_tr = np.full(num_ind_tr, -1)
    indexes_test = np.full(num_ind_test, -1)
    for i in range(num_ind_tr):
        j = random.randint(0, len(ind) - 1)
        while np.any(indexes_tr == ind[j]):
            j = random.randint(0, len(ind) - 1)
        indexes_tr[i] = ind[j]
    for i in range(num_ind_test):
        j = random.randint(0, len(ind) - 1)
        while np.any(indexes_tr == ind[j]) or np.any(indexes_test == ind[j]):
            j = random.randint(0, len(ind) - 1)
        indexes_test[i] = ind[j]
    return indexes_tr, indexes_test


def conc_samples(data_dir: str, ind: np.array) -> np.array:
    for i, idx in enumerate(ind):
        file_path = data_dir / f"data{int(idx)}.npy"
        loaded_data = np.load(file_path)
        if i == 0:
            im = loaded_data
        else:
            im = np.concatenate((im, loaded_data))
            
    return im


def shuffle_arrays(arrays: list, set_seed=-1):
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)


def process_class(class_idx, data_dir, labData, ntrainind, ntestind, num_classes):
    indexes_train, indexes_test = find_samples(find_indexes(labData, class_idx, num_classes), ntrainind, ntestind)
    images_train = conc_samples(data_dir, indexes_train)
    labels_train = np.full(len(images_train), class_idx)
    images_test = conc_samples(data_dir, indexes_test)
    labels_test = np.full(len(images_test), class_idx)

    return (images_train, labels_train, images_test, labels_test, indexes_train, indexes_test)


def prepare_and_save_data(data_dir, ntrainind, ntestind, acc_bounds):
    # load dataset labels
    labData = np.load(data_dir / 'labDatasets')
    # determine the number of classes based on acc_bounds 
    num_classes = len(acc_bounds) + 1

    # process data for each class
    all_trainX, all_trainY, all_testX, all_testY, all_trainind, all_testind = [], [], [], [], [], []

    for class_idx in range(num_classes):
        imtr, labeltr, imtest, labeltest, indexestr, indexestest = process_class(class_idx, data_dir, labData, ntrainind, ntestind, num_classes)
        all_trainX.append(imtr)
        all_trainY.append(labeltr)
        all_testX.append(imtest)
        all_testY.append(labeltest)
        all_trainind.append(indexestr)
        all_testind.append(indexestest)

    # concatenate all data
    trainX = np.concatenate(all_trainX)
    trainY = to_categorical(np.concatenate(all_trainY))
    testX = np.concatenate(all_testX)
    testY = to_categorical(np.concatenate(all_testY))
    trainind = np.concatenate(all_trainind)
    testind = np.concatenate(all_testind)

    shuffle_arrays([trainX, trainY])
    shuffle_arrays([testX, testY])

    data = [trainX, trainY, testX, testY]
    indexes = [trainind, testind]

    with open(data_dir / 'data.pickle', 'wb') as f:
        pickle.dump(data, f)
    with open(data_dir / 'indexes.pickle', 'wb') as f:
        pickle.dump(indexes, f)
