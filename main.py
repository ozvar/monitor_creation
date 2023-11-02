import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import multiprocessing

sys.path.append(os.path.join(os.getcwd(), 'monitor'))

from monitor import *
from pprint import pprint

if __name__ == "__main__":

    # Initialize wandb
    # wandb.init(project='my_project')

    multiprocessing.set_start_method('spawn')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", gpu_devices)
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    for exp in experiments:
        
        pprint(exp, sort_dicts=False)

        RESULTS_DIR, FIG_DIR, TRANSF_DIR, MONITOR_DIR = create_out_dirs(
                transf_factors=exp['TRANSF_FACTORS']
                )

        # only generate datasets on the first run
        # gen_datasets_from_transforms(
        #         transf_factors=exp['TRANSF_FACTORS'],
        #         epsilons=exp['EPSILONS'],
        #         dataset=exp['DATASET'],
        #         out_dir=TRANSF_DIR
        #         )

        compute_and_save_accuracies(
                model_dir=exp['MODEL_DIR'],
                dataset=exp['DATASET'],
                model_name=exp['MODEL'],
                data_dir=TRANSF_DIR,
                acc_bounds=exp['ACC_BOUNDS']
                )

        for run in range(exp['RUNS']):


            prepare_and_save_data(
                    out_dir=RESULTS_DIR,
                    transf_dir=TRANSF_DIR,
                    ntrainind=exp['N_TRAININD'],
                    ntestind=exp['N_TESTIND'],
                    acc_bounds=exp['ACC_BOUNDS'],
                    run_id=run
                    )

            p = multiprocessing.Process(
                    target=train_monitor,
                    args=(
                        MONITOR_DIR,    
                        RESULTS_DIR,
                        exp['K_FOLDS'],
                        True,
                        run
                        )
                    )

            p.start()
            p.join()
            # shut down Process
            p.terminate()

            # train_monitor(
            #     model_dir=MONITOR_DIR,
            #     data_dir=RESULTS_DIR,
            #     k_folds=exp['K_FOLDS'],
            #     train_model=True,
            #     run_id=run
            #     )

        # only run first experiment for now
        break 
