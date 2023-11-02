import numpy as np
import os
import gc
import pickle
import typing
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from utils import setup_logger


def loadData(data_dir: Path) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    with open(data_dir / 'data.pickle', 'rb') as f:
        [trainX, trainY, testX, testY] = pickle.load(f)
    return trainX, trainY, testX, testY


def create_model():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.Conv2D(256, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation='softmax')
        ]
    )


def train_monitor(model_dir: str, data_dir: Path, k_folds: int, train_model: bool, run_id: int):
    # initialize logging and paths
    logger = setup_logger(data_dir, run_id)
    model_dir = Path(model_dir)
    # load data
    [trainX, trainY, testX, testY] = loadData(data_dir)
    if train_model:
        logger.info('Training model...')
        for k, (train_idx, test_idx) in enumerate(KFold(n_splits=k_folds,
                                    shuffle=True).split(trainX, trainY)):
            # clear the session
            tf.keras.backend.clear_session()
            # implement early stopping
            # early_stopping = tf.keras.callbacks.EarlyStopping(
            #     monitor='val_loss',
            #     patience=2,
            #     restore_best_weights=True
            # )

            # calling the model and compile it
            seq_model = create_model()
            seq_model.compile(
                loss  = tf.keras.losses.CategoricalCrossentropy(),
                metrics  = tf.keras.metrics.CategoricalAccuracy(),
                optimizer = tf.keras.optimizers.Adam())

            log_message = (f'K-Fold: {k+1}\n'
                           f'Train Set: {trainX[train_idx].shape}, {trainY[train_idx].shape}\n'
                           f'Test Set: {trainX[test_idx].shape}, {trainY[test_idx].shape}')
            logger.info(log_message)

            # run the model
            seq_model.fit(trainX[train_idx], trainY[train_idx],
                    batch_size=64,
                    epochs=2,
                    validation_data=(trainX[test_idx], trainY[test_idx]))
            seq_model.save_weights(model_dir / f'wg_{k+1}_2.h5')

            # log model training accuracy
            train_acc = seq_model.evaluate(trainX[train_idx], trainY[train_idx])
            logger.info(f'Train Accuracy: {train_acc}')
            # log model validation accuracy
            val_acc = seq_model.evaluate(trainX[test_idx], trainY[test_idx])
            logger.info(f'Validation Accuracy: {val_acc}')

    logger.info('Evaluating model...')
    for k in range(k_folds):
        model = create_model()
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=tf.keras.metrics.CategoricalAccuracy(),
            optimizer=tf.keras.optimizers.Adam())
        # log shape of test set
        logger.info(f'Test Set: {testX.shape}, {testY.shape}')
        # load model weights
        model.load_weights(model_dir / f'wg_{k+1}_2.h5')
        # log model test accuracy
        test_acc = model.evaluate(testX, testY)
        logger.info(f'Test Accuracy for k={k+1}: {test_acc}')
    
    logger.info('Done!')
    # close the logger
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # clear memory
    del [trainX, trainY, testX, testY]
    del model
    del seq_model
    tf.keras.backend.clear_session()
    gc.collect()

