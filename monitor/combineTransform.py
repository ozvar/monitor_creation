import numpy as np
import os
from PIL import Image
from datasets import load_gtsrb, load_cifar 
from transforms import haze, increase_contrast, gaussianblureps
from itertools import product
from typing import Dict
from tqdm import tqdm


def combinatorial_transf(
        transf_factors: Dict[str, float],
        eps: list
        ) -> list:
    # Convert eps to numpy array for easy scaling
    eps = np.array(eps)
    # Scale each epsilon list by the transformation scalar
    scaled_eps_lists = []
    for transf, scalar in transf_factors.items():
        scaled_eps_lists.append(eps * scalar)
    # Generate all combinations of the scaled epsilon values
    comb_prod = [list(combination) for combination in product(*scaled_eps_lists)]
    
    return comb_prod


def apply_combined_transf(
        transf_factors: Dict[str, float],
        image: np.array,
        transf: list
        ) -> np.array:
    keys = list(transf_factors.keys())
    for i in range(len(keys)):
        if keys[i] == 'haze':
            image = haze(image, epsilon=transf[i])
        elif keys[i] == 'contrast':
            image = increase_contrast(image, epsilon=transf[i])
        elif keys[i] == 'blur':
            image = gaussianblureps(image, epsilon=transf[i])

    return image


def create_dataset(
        transf_factors: Dict[str, float],
        data: np.array,
        transf: list,
        out_dir: str
        ) -> np.array:
    newdata = np.empty(data.shape)
    pbar = tqdm(range(data.shape[0]), position=1, leave=False)
    for i in pbar:
        pbar.set_description(f'Degrading image #{i+1}')
        newdata[i] = apply_combined_transf(
                transf_factors=transf_factors,
                image=data[i],
                transf=transf)
    return newdata


def create_all_datasets(
        transf_factors: Dict[str, float], 
        data: np.array,
        allcombtransf: list,
        out_dir: str
        ):
    pbar = tqdm(range(len(allcombtransf)), position=0)
    for i in pbar:
        pbar.set_description(f'Creating dataset #{i+1}')
        newdata = create_dataset(
                transf_factors=transf_factors,
                data=data,
                transf=allcombtransf[i],
                out_dir=out_dir)
        np.save(os.path.join(out_dir, f'data{i}.npy'), newdata)


def gen_datasets_from_transforms(
        transf_factors: Dict[str, float],
        epsilons: list,
        dataset: str,
        out_dir: str
        ):
    comb_prod = combinatorial_transf(
            transf_factors=transf_factors,
            eps=epsilons)
    # preparing for option to train on CIFAR too
    if dataset == 'gtsrb': 
        [X_train, y_train, X_test, y_test, labels] = load_gtsrb()
    create_all_datasets(
            transf_factors=transf_factors,
            data=X_test, 
            allcombtransf=comb_prod,
            out_dir=out_dir)
