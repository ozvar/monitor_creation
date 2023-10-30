import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), 'monitor'))

from monitor import *
from pprint import pprint

if __name__ == "__main__":

    # Initialize wandb
    # wandb.init(project='my_project')

    for exp in experiments:

        pprint(exp)

        RESULS_DIR, FIG_DIR, TRANSF_DIR = create_out_dirs(
                transf_factors=exp['TRANSF_FACTORS']
                )

        gen_datasets_from_transforms(
                transf_factors=exp['TRANSF_FACTORS'],
                epsilons=exp['ESPILONS'],
                dataset=exp['DATASET'],
                out_dir=TRANSF_DIR
                )

        compute_and_save_accuracies(
                model_dir=exp['MODEL_DIR'],
                dataset=exp['DATASET'],
                data_dir=TRANSF_DIR,
                acc_bounds=exp['ACC_BOUNDS']
                )

        prepare_and_save_data(
                data_dir=exp['RESULTS_DIR'],
                n_trainind=exp['N_TRAININD'],
                n_testind=exp['N_TESTIND'],
                acc_bounds=exp['ACC_BOUNDS']
                )




