import numpy as np
import os
import pickle
import typing
import tensorflow as tf
from sklearn.model_selection import KFold
from parameters import MODEL_DIR

PICKLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.pickle')

def loadData() -> typing.Tuple[np.array, np.array, np.array, np.array]:
    with open(PICKLE_PATH, 'rb') as f:
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


def trainMonitor(model_dir: str, k_fold: int, train: bool):
    [trainX, trainY, testX, testY] = loadData()
    if train:
        for kfold, (train, test) in enumerate(KFold(n_splits=k_fold,
                                    shuffle=True).split(trainX, trainY)):
            # clear the session
            tf.keras.backend.clear_session()

            # calling the model and compile it
            seq_model = create_model()
            seq_model.compile(
                loss  = tf.keras.losses.CategoricalCrossentropy(),
                metrics  = tf.keras.metrics.CategoricalAccuracy(),
                optimizer = tf.keras.optimizers.Adam())

            print('Train Set')
            print(trainX[train].shape)
            print(trainY[train].shape)

            print('Test Set')
            print(trainX[test].shape)
            print(trainY[test].shape)

            # run the model
            seq_model.fit(trainX[train], trainY[train],
                    batch_size=128, epochs=2, validation_data=(trainX[test], trainY[test]))
            seq_model.save_weights(os.path.join(model_dir, 'monitor', f'wg_{kfold}_2.h5'))
    else:
        model = create_model()
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=tf.keras.metrics.CategoricalAccuracy(),
            optimizer=tf.keras.optimizers.Adam())

        print('Test Set')
        print(testX.shape)
        print(testY.shape)
        model.load_weights(os.path.join(model_dir, 'monitor', f'wg_0_2.h5'))
        print(model.evaluate(testX, testY))
        model.load_weights(os.path.join(model_dir, 'monitor', f'wg_1_2.h5'))
        print(model.evaluate(testX, testY))
        model.load_weights(os.path.join(model_dir, 'monitor', f'wg_2_2.h5'))
        print(model.evaluate(testX, testY))
