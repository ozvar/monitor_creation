import numpy as np
import gc
import typing
import os
import pickle
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from utils import setup_logger
from utils import generate_images_from_dataset


def count_samples_classes(
        lab_data: list,
        num_classes: int
        ) -> np.array:
    count_classes = np.zeros(num_classes)
    for el in lab_data:
        i = el[1]
        count_classes[int(i)] += 1
    return count_classes


def find_indexes(
        lab_data: list,
        cl: int,
        num_classes: int
        )-> np.array:
    num_samples = count_samples_classes(lab_data, num_classes)
    ind = np.zeros(int(num_samples[cl]))
    j = 0
    for i in range(len(lab_data)):
        if lab_data[i][1] == cl:
            ind[j] = i
            j += 1
    return ind


def find_samples(
        ind: np.array,
        num_ind_tr: int,
        num_ind_test: int
        ) -> typing.Tuple[np.array, np.array]:

    # Ensure the condition is met
    if num_ind_tr + num_ind_test > len(ind):
        raise ValueError(f"The sum of num_ind_tr ({num_ind_tr}) and num_ind_test ({num_ind_test}) exceeds the length of ind ({len(ind)}).")

    # Randomly shuffle the indices
    shuffled_indices = np.copy(ind)
    np.random.shuffle(shuffled_indices)

    # Split into training and test indices
    indexes_tr = shuffled_indices[:num_ind_tr]
    indexes_test = shuffled_indices[num_ind_tr:num_ind_tr + num_ind_test]

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
        lab_data: np.array,
        ntrainind: int,
        ntestind: int,
        num_classes: int):
    indexes_train, indexes_test = find_samples(find_indexes(lab_data, class_idx, num_classes), ntrainind, ntestind)
    images_train = conc_samples(transf_dir, indexes_train)
    labels_train = np.full(len(images_train), class_idx)
    images_test = conc_samples(transf_dir, indexes_test)
    labels_test = np.full(len(images_test), class_idx)

    return (images_train, labels_train, images_test, labels_test, indexes_train, indexes_test)


def prepare_class(
        class_idx: int,
        train_prop: float,
        transf_dir: Path,
        lab_data: np.array):

    samples_list = []
    index_list = []

    for i, (perf, label) in enumerate(lab_data):
        if label == class_idx:
            file_path = transf_dir / f"data{int(i)}.npy"
            loaded_data = np.load(file_path)
            samples_list.append(loaded_data)
            index_list.append(np.full(len(loaded_data), i))

    # Concatenate all samples and indices
    samples = np.concatenate(samples_list, axis=0)
    train_ind = np.concatenate(index_list, axis=0)

    # Shuffle and split the dataset
    indices = np.arange(len(samples))
    np.random.shuffle(indices)
    samples, train_ind = samples[indices], train_ind[indices]

    split_idx = int(train_prop * len(samples))
    train_samples, test_samples = samples[:split_idx], samples[split_idx:]
    train_ind, test_ind = train_ind[:split_idx], train_ind[split_idx:]
    labels_train = np.full(len(train_samples), class_idx)
    labels_test = np.full(len(test_samples), class_idx)

    return train_samples, labels_train, test_samples, labels_test, train_ind, test_ind


def prepare_and_save_data(
        out_dir: Path,
        image_dir: Path,
        transf_dir: Path,
        train_prop: float,
        acc_bounds: list,
        imageind: list,
        run_id: int,
        log_label: str='data'):
    # setup logger
    logger = setup_logger(out_dir, run_id, log_label)
    # load variables
    lab_data = np.load(transf_dir/ 'labDatasets.npy')
    acc = np.load(transf_dir / 'accuracy.npy')
    num_classes = len(acc_bounds) + 1
    params = {
        "class_idx": None,
        "transf_dir": transf_dir,
        "lab_data": lab_data,
        "train_prop": train_prop
    }
    # randomly sample datasets from each class    
    pbar = tqdm(range(num_classes), position=0)
    results = []
    for i in pbar:
        pbar.set_description(f"Processing class {i+1}/{num_classes}")
        results.append(prepare_class(**{**params, "class_idx": i}))
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
    counts = count_samples_classes(lab_data, num_classes)
    logger.info(f"Class boundaries: {acc_bounds}")
    logger.info(f"Number of samples per class: {counts}")
    # generate images from training and test datasets
    for i, inds in enumerate(all_trainind):
        run_image_dir = image_dir / f'run_{run_id+1}_train'
        os.makedirs(run_image_dir, exist_ok=True)
        generate_images_from_dataset(
                transf_dir=transf_dir,
                data_set=f'data{int(inds[0])}.npy', 
                acc_class=f'class_{i+1}',
                indexes=imageind,
                image_size=(128, 128),
                out_dir=run_image_dir
                )
    for i, inds in enumerate(all_testind):
        run_image_dir = image_dir / f'run_{run_id+1}_test'
        os.makedirs(run_image_dir, exist_ok=True)
        generate_images_from_dataset(
                transf_dir=transf_dir,
                data_set=f'data{int(inds[0])}.npy', 
                acc_class=f'class_{i+1}',
                indexes=imageind,
                image_size=(128, 128),
                out_dir=run_image_dir
                )
    # close the logger
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    # free memory
    del all_trainX, all_trainY, all_testX, all_testY, all_trainind, all_testind
    del trainX, trainY, testX, testY, trainind, testind
    del results
    gc.collect()
