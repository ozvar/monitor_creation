import numpy as np
import os
from PIL import Image
from datasets import load_gtsrb, load_cifar 
from transforms import haze, increase_contrast, gaussianblureps
from parameters import TRANSF, DATA_DIR


def combinatorial_transf(eps: list) -> list:
    for i in range(len(eps)):
        if i == 0:
            haze = eps[i]
        elif i == 1:
            blur = eps[i]
        elif i == 2:
            contrast = eps[i]
    combprod = []
    for h in haze:
        for b in blur:
            for c in contrast:
                combprod.append([h, b, c])
    return combprod


def apply_combined_transf(image: np.array, transf: list) -> np.array:
    image = haze(image, transf[0])
    image = gaussianblureps(image, transf[1])
    image = increase_contrast(image, transf[2])
    return image


def create_dataset(data: np.array, transf: list) -> np.array:
    newdata = np.empty(data.shape)
    for i in range(data.shape[0]):
        newdata[i] = apply_combined_transf(data[i], transf)
    return newdata


def create_all_datasets(data: np.array, allcombtransf: list, out_dir: str):
    for i in range(len(allcombtransf)):
        newdata = create_dataset(data, allcombtransf[i])
        print(allcombtransf[i])
        print(i)
        np.save(os.path.join(out_dir, f'data{i}.npy', newdata))


def gen_datasets_from_transforms(transf: list, dataset: str, out_dir: str):
    # hard coding number of influencing factors for now
    alltransf = [transf]*3
    combprod = combinatorial_transf(alltransf)
    print(combprod)
    # preparing for option to train on CIFAR too
    if dataset == 'gtsrb': 
        [X_train, y_train, X_test, y_test, labels] = load_gtsrb()
    create_all_datasets(X_test, combprod, out_dir)
