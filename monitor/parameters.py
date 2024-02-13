experiments = [
    # {
    #     "TRANSF_FACTORS": {
    #         "haze": 0.8,
    #         "blur": 0.6,
    #         "contrast": 1.0
    #     },
    #     "EPSILONS": [0, 0.2, 0.5, 0.8, 1],
    #     "ACC_BOUNDS": [0.70, 0.40],
    #     "TRAIN_PROP": 0.8,
    #     "K_FOLDS": 5,
    #     "BATCH_SIZE": 64,
    #     "RUNS": 5,
    #     "MODEL_DIR": "models",
    #     "DATASET": "gtsrb",
    #     "MODEL": "model3b.h5"
    # },
    {
        "TRANSF_FACTORS": {
            "haze": 0.8,
            "blur": 0.6
        },
        "EPSILONS": [0, 0.2, 0.5, 0.8, 1],
        "ACC_BOUNDS": [0.70, 0.40],
        "TRAIN_PROP": 0.8,
        "IMAGE_IND": [0, 1000, 2000, 3000, 4000],
        "K_FOLDS": 5,
        "BATCH_SIZE": 128,
        "EPOCHS": 20,
        "RUNS": 1,
        "SEED": 42,
        "MODEL_DIR": "models",
        "DATASET": "gtsrb",
        "MODEL": "model3b.h5"
    },
    # {
    #     "TRANSF_FACTORS": {
    #         "blur": 0.6,
    #         "haze": 0.8
    #     },
    #     "EPSILONS": [0, 0.2, 0.5, 0.8, 1],
    #     "ACC_BOUNDS": [0.70, 0.40],
    #     "TRAIN_PROP": 0.8,
    #     "IMAGE_IND": [0, 1000, 2000, 3000, 4000],
    #     "K_FOLDS": 5,
    #     "BATCH_SIZE": 64,
    #     "RUNS": 5,
    #     "MODEL_DIR": "models",
    #     "DATASET": "gtsrb",
    #     "MODEL": "model3b.h5"
    # }
]
