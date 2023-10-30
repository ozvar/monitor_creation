experiments = [
    {
        "TRANSF_FACTORS": {
            "haze": 0.8,
            "blur": 0.6
        },
        "EPSILONS": [0, 0.2, 0.5, 0.8, 1],
        "ACC_BOUNDS": [0.70, 0.40],
        "K_FOLDS": 5,
        "RUNS": 5,
        "DATA_DIR": "transformations",
        "MODEL_DIR": "models",
        "FIG_DIR": "results/figures",
        "DATASET": "gtsrb",
        "MODEL": "model3b.h5"
    },
    {
        "TRANSF_FACTORS": {
            "blur": 0.6,
            "haze": 0.8
        },
        "EPSILONS": [0, 0.2, 0.5, 0.8, 1],
        "ACC_BOUNDS": [0.70, 0.40],
        "K_FOLDS": 5,
        "RUNS": 5,
        "DATA_DIR": "transformations",
        "MODEL_DIR": "models",
        "FIG_DIR": "results/figures",
        "DATASET": "gtsrb",
        "MODEL": "model3b.h5"
    }
]
