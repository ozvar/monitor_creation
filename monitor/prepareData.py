import numpy as np
import os
import gc
import typing
import random
import pickle
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from utils import setup_logger


def count_samples_classes(
        labData: list,
        num_classes: int
        ) -> np.array:
    count_classes = np.zeros(num_classes)
    for el in labData:
        i = el[1]
        count_classes[int(i)] += 1
    return count_classes


def find_indexes(
        labData: list,
        cl: int,
        num_classes: int
        )-> np.array:
    num_samples = count_samples_classes(labData, num_classes)
    ind = np.zeros(int(num_samples[cl]))
    j = 0
    for i in range(len(labData)):
        if labData[i][1] == cl:
            ind[j] = i
            j += 1
    return ind


def find_samples (
        ind: np.array,
        num_ind_tr: int,
        num_ind_test: int
        ) -> typing.Tuple[np.array, np.array]:
    indexes_tr = np.full(num_ind_tr, -1)
    indexes_test = np.full(num_ind_test, -1)
    pbar = tqdm(range(num_ind_tr), position=1, leave=True)
    for i in pbar:
        pbar.set_description(f"Finding training sample {i+1}/{num_ind_tr}")
        j = random.randint(0, len(ind) - 1)
        while np.any(indexes_tr == ind[j]):
            j = random.randint(0, len(ind) - 1)
        indexes_tr[i] = ind[j]
    pbar = tqdm(range(num_ind_test), position=1, leave=True)
    for i in pbar:
        pbar.set_description(f"Finding test sample {i+1}/{num_ind_test}")
        j = random.randint(0, len(ind) - 1)
        while np.any(indexes_tr == ind[j]) or np.any(indexes_test == ind[j]):
            j = random.randint(0, len(ind) - 1)
        indexes_test[i] = ind[j]
    return indexes_tr, indexes_test


def conc_samples(
        transf_dir: Path,
        ind: np.array
        ) -> np.array:
    for i, idx in enumerate(ind):
        file_path = transf_dir / f"data{int(idx)}.npy"
        loaded_data = np.load(file_path)
        if i == 0:
            im = loaded_data
        else:
            im = np.concatenate((im, loaded_data))
            
    return im


def shuffle_arrays(
        arrays: list,
        set_seed=-1):
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


def process_class(
        class_idx: int,
        transf_dir: Path,
        labData: np.array,
        ntrainind: int,
        ntestind: int,
        num_classes: int):
    indexes_train, indexes_test = find_samples(find_indexes(labData, class_idx, num_classes), ntrainind, ntestind)
    images_train = conc_samples(transf_dir, indexes_train)
    labels_train = np.full(len(images_train), class_idx)
    images_test = conc_samples(transf_dir, indexes_test)
    labels_test = np.full(len(images_test), class_idx)

    return (images_train, labels_train, images_test, labels_test, indexes_train, indexes_test)


def prepare_and_save_data(
        out_dir: Path,
        transf_dir: Path,
        ntrainind: int,
        ntestind: int,
        acc_bounds: list,
        run_id: int):
    # setup logger
    logger = setup_logger(out_dir, run_id)
    # load variables
    labData = np.load(transf_dir/ 'labDatasets.npy')
    num_classes = len(acc_bounds) + 1
    params = {
        "class_idx": None,
        "transf_dir": transf_dir,
        "labData": labData,
        "ntrainind": ntrainind,
        "ntestind": ntestind,
        "num_classes": num_classes
    }
    # randomly sample datasets from each class    
    pbar = tqdm(range(num_classes), position=0)
    results = []
    for i in pbar:
        pbar.set_description(f"Processing class {i+1}/{num_classes}")
        results.append(process_class(**{**params, "class_idx": i}))
    all_trainX, all_trainY, all_testX, all_testY, all_trainind, all_testind = zip(*results)
    trainX, trainY = np.concatenate(all_trainX), to_categorical(np.concatenate(all_trainY))
    testX, testY = np.concatenate(all_testX), to_categorical(np.concatenate(all_testY))
    trainind, testind = np.concatenate(all_trainind), np.concatenate(all_testind)
    shuffle_arrays([trainX, trainY])
    shuffle_arrays([testX, testY])
    with open(out_dir / 'data.pickle', 'wb') as f:
        pickle.dump([trainX, trainY, testX, testY], f)
    with open(out_dir / 'indexes.pickle', 'wb') as f:
        pickle.dump([trainind, testind], f)
    # describe class boundaries and number of samples per class
    counts = count_samples_classes(labData, num_classes)
    logger.info(f"Class boundaries: {acc_bounds}")
    logger.info(f"Number of samples per class: {counts}")
    # close the logger
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    # free memory
    del all_trainX, all_trainY, all_testX, all_testY, all_trainind, all_testind
    del trainX, trainY, testX, testY, trainind, testind
    del results
    gc.collect()
