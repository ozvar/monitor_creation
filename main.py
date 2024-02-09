import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import multiprocessing
import wandb

sys.path.append(os.path.join(os.getcwd(), 'monitor'))

from monitor import *
from pprint import pprint
from pprint import pformat

if __name__ == "__main__":

    # Initialize wandb
    #wandb.login()

    multiprocessing.set_start_method('spawn')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    sns_styleset()

    for exp in experiments:
        
        # run = wandb.init(
        #         project='ML_monitors',
        #         config=exp
        #         )

        pprint(exp, sort_dicts=False)

        RESULTS_DIR, FIG_DIR, TRANSF_DIR, MONITOR_DIR, IMAGE_DIR = create_out_dirs(
                transf_factors=exp['TRANSF_FACTORS']
                )

        if exp['SEED'] > 0:
            tf.random.set_seed(exp['SEED'])
            np.random.seed(exp['SEED'])

        logger = setup_logger(RESULTS_DIR, run_id=-1, log_label='params')
        logger.info(pformat(exp))

        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        # only generate datasets on the first run
        # gen_datasets_from_transforms(
        #         transf_factors=exp['TRANSF_FACTORS'],
        #         epsilons=exp['EPSILONS'],
        #         dataset=exp['DATASET'],
        #         out_dir=TRANSF_DIR
        #         )

        # generate_all_images_for_transformations(
        #         transf_dir=exp['TRANSF_DIR'],
        #         image_size=(128, 128),
        #         out_dir=exp['IMAGE_DIR'] / 'all_images',
        #         )

        # compute_and_save_accuracies(
        #         model_dir=exp['MODEL_DIR'],
        #         dataset=exp['DATASET'],
        #         model_name=exp['MODEL'],
        #         data_dir=TRANSF_DIR,
        #         acc_bounds=exp['ACC_BOUNDS']
        #         )

        log_accuracy_across_classes(
                transf_factors=exp['TRANSF_FACTORS'],
                epsilons=exp['EPSILONS'],
                transf_dir=TRANSF_DIR,
                data_dir=RESULTS_DIR,
                acc_bounds=exp['ACC_BOUNDS'],
                log_label='labels'
                )
        

        for run in range(exp['RUNS']):
            prepare_and_save_data(
                    out_dir=RESULTS_DIR,
                    image_dir=IMAGE_DIR,
                    transf_dir=TRANSF_DIR,
                    train_prop=exp['TRAIN_PROP'],
                    imageind=exp['IMAGE_IND'],
                    acc_bounds=exp['ACC_BOUNDS'],
                    run_id=run,
                    seed=exp['SEED']
                    )

            # p = multiprocessing.Process(
            #          target=train_monitor,
            #          args=(
            #              MONITOR_DIR,    
            #              RESULTS_DIR,
            #              FIG_DIR,
            #              exp['K_FOLDS'],
            #              exp['BATCH_SIZE'],
            #              exp['EPOCHS'], 
            #              exp['SEED'],
            #              run
            #              )
            #          )
            # # manage memory in multiprocessing process
            # p.start()
            # p.join()
            # p.terminate()

            p = multiprocessing.Process(
                     target=test_monitor,
                     args=(
                         MONITOR_DIR,    
                         RESULTS_DIR,
                         exp['K_FOLDS'],
                         exp['SEED'],
                         run,
                         FIG_DIR
                         )
                     )
            # manage memory in multiprocessing process
            p.start()
            p.join()
            p.terminate()
        # only run first experiment for now
        break 
