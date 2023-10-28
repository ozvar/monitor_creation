import numpy as np
import os
from PIL import Image
from datasets import load_gtsrb, load_cifar 
from transforms import haze, increase_contrast, gaussianblureps
from itertools import product
from typing import Dict


def combinatorial_transf(transf_factors: Dict[str, float], eps: list) -> list:
    # Convert eps to numpy array for easy scaling
    eps = np.array(eps)
    # Scale each epsilon list by the transformation scalar
    scaled_eps_lists = []
    for transf, scalar in transf_factors.items():
        scaled_eps_lists.append(eps * scalar)
    # Generate all combinations of the scaled epsilon values
    comb_prod = [list(combination) for combination in product(*scaled_eps_lists)]
    
    return comb_prod


def apply_combined_transf(transf_factors: Dict[str, float], image: np.array, transf: list) -> np.array:
    keys = list(transf_factors.keys())
    for i in range(len(keys)):
        if keys[i] == 'haze':
            image = haze(image, epsilon=transf[i])
        elif keys[i] == 'contrast':
            image = increase_contrast(image, epsilon=transf[i])
        elif keys[i] == 'blur':
            image = gaussianblureps(image, epsilon=transf[i])

    return image


def create_dataset(data: np.array, transf: list) -> np.array:
    newdata = np.empty(data.shape)
    for i in range(data.shape[0]):
        newdata[i] = apply_combined_transf(data[i], transf)
    return newdata


def create_all_datasets(data: np.array, allcombtransf: list, out_dir: str):
    for i in range(len(allcombtransf)):
        newdata = create_dataset(data, allcombtransf[i])
        np.save(os.path.join(out_dir, f'data{i}.npy'), newdata)


def gen_datasets_from_transforms(
        transf_factors: Dict[str, float],
        epsilons: list,
        dataset: str,
        out_dir: str
        ):
    comb_prod = combinatorial_transf(transf_factors=transf_factors, eps=epsilons)
    # preparing for option to train on CIFAR too
    if dataset == 'gtsrb': 
        [X_train, y_train, X_test, y_test, labels] = load_gtsrb()
    create_all_datasets(X_test, comb_prod, out_dir)
