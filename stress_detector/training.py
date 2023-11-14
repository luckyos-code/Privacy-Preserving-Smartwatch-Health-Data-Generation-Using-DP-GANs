# from stress_detector.data.datatype import DataType
from random import shuffle
from typing import Tuple, Optional, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

import pandas as pd

from stress_detector import constants
from stress_detector.data.datatype import DataType
from stress_detector.model import build_cnn, build_transformer, build_cnn_lstm

import time
from datetime import timedelta

from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise as tfp_computer_noise
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdamOptimizer

"""
Calculate Delta for given training dataset size n
"""
def compute_delta(n):
    # delta should be one magnitude lower than inverse of training set size: 1/n
    # e.g. 1e-5 for n=60.000
    # take 1e-x, were x is the magnitude of training set size
    delta = np.power(10, - float(len(str(n)))) # remove all trailing decimals
    return delta

"""
Calculate noise for given training hyperparameters
"""
def compute_noise(n, batch_size, target_epsilon, epochs, delta, min_noise=1e-5):
    return tfp_computer_noise(n, batch_size, target_epsilon, epochs, delta, min_noise)

# compute_dp_sgd_privacy.compute_dp_sgd_privacy(18900, #210*6*15 = 18900
#                                               batch_size=params["batch_size"],
#                                               noise_multiplier=params["noise_multiplier"],
#                                               epochs=max(epochs_dict.values()), # siehe unten
#                                               delta=1e-5)

def build_model(nn_mode, num_signals, num_output_class) -> tf.keras.Model:
    if nn_mode == "CNN":
        model = build_cnn(num_signals, num_output_class)
    elif nn_mode == "Transformer":
        model = model = build_transformer(num_signals, num_output_class)
    elif nn_mode == "CNN-LSTM":
        model = build_cnn_lstm(num_signals, num_output_class)
    else:
        raise Exception("not a valid model selection")
    return model

def compile_model(model, learning_rate, eps, num_unique_windows, batch_size, epochs, l2_norm_clip) -> Tuple[tf.keras.Model, float, float]:
    if not eps:
        optimizer = Adam(learning_rate=learning_rate)
        delta, noise_multiplier = None, None
    else:
        # calculate relevant training sample number
        delta = compute_delta(num_unique_windows)
        noise_multiplier = compute_noise(
            n=num_unique_windows, # unique 60 second windows before ffe sampling
            batch_size=batch_size,
            target_epsilon=eps,
            epochs=epochs,
            delta=delta)
        optimizer = VectorizedDPAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=batch_size,
            learning_rate=learning_rate
        )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=['accuracy', Precision(), Recall()],
    )
    return model, delta, noise_multiplier

def train_model(model, X_train, Y_train, epochs, batch_size, class_weight=True, validation_split=None, verbose=2) -> Tuple[tf.keras.Model, dict]:
    if class_weight: 
        if not (np.argmax(Y_train, axis=1).tolist().count(1)): # time gan generates only one class (non-stress)
            class_weight = None
        else:
            class_weight = {0: 1,
                            1: (np.argmax(Y_train, axis=1).tolist().count(0) / np.argmax(Y_train, axis=1).tolist().count(1)),}

    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        class_weight=class_weight, # to address the imbalance of the class labels
    )
    return model, history

def evaluate_model(model, X_test, Y_test, verbose=0) -> dict:
    # get acc, prec, rec, f1
    evaluation_metrics = model.evaluate(X_test, Y_test, verbose=verbose)
    accuracy = evaluation_metrics[1]
    precision = evaluation_metrics[2]
    recall = evaluation_metrics[3]
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,}


def train(
    realData: Tuple,
    real_subj_cnt: int,
    synData: Tuple,
    syn_subj_cnt: int,
    gan_mode: str,
    nn_mode: str,
    eval_mode: str,
    num_signals: int = 6,
    num_output_class: int = 2,
    epochs: int = 10,
    batch_size: int = 50,
    learning_rate: float = 1e-3,
    eps: float = None,
    num_unique_windows: int = None,
    l2_norm_clip: float = 1.0,
    prepare_environment_func: Optional[Callable] = None,
) -> Tuple[dict, float, float]:
    assert(
        num_output_class == 2
    ), "Only binary classification supported right now (num_output_class is >2 right now)"
    print(f"*** Training {nn_mode} model in {eval_mode} mode ***")

    realX, realY = realData
    synX, synY = synData

    if real_subj_cnt > 0: real_ids = ["S"+str(i) for i in constants.SUBJECT_IDS[:real_subj_cnt]]
    if syn_subj_cnt > 0: syn_ids = ["SYN"+str(i) for i in range(1,syn_subj_cnt+1)]

    #if test_sub is not None:
    #    test_sub_index = constants.SUBJECT_IDS.index(test_sub)

    results = {}
    if eval_mode == "LOSO":
        assert(
            real_subj_cnt >= 2
        ), "At least 2 real subjects needed for LOSO"

        print(f"LOSO train on {real_subj_cnt} real and {syn_subj_cnt} synthetic {gan_mode} subjects - with eps={eps}")
        for test_idx in range(0, len(real_ids)):
            
            # if test_sub is not None:
            #     if test_sub_index != test_index:
            #         print("SKIP training for test_index", test_index)
            #         continue

            print(f"LOSO on {real_ids[test_idx]} from WESAD ({test_idx+1}/{len(real_ids)})...")
            start_time = time.monotonic()

            # get real training data without LOSO test subject
            X_train = np.concatenate(np.delete(realX, test_idx, 0))
            Y_train = np.concatenate(np.delete(realY, test_idx, 0))

            # test data is the removed LOSO test subject
            X_test = realX[test_idx]
            Y_test = realY[test_idx]

            # add synthetic subjects
            if syn_subj_cnt > 0:
                X_train = np.concatenate((X_train, synX)) # inside parenthesis because matrice-like structure
                Y_train = np.concatenate((Y_train, synY))

                assert ( # check correct shape
                    X_train.shape == (np.concatenate(np.delete(realX, test_idx, 0)).shape[0] + synX.shape[0], synX.shape[1], synX.shape[2])
                ), f"got shape: {X_train.shape} instead of expected {(np.concatenate(np.delete(realX, test_idx, 0)).shape[0] + synX.shape[0], synX.shape[1], synX.shape[2])}"

            Y_train = tf.keras.utils.to_categorical(Y_train, num_output_class)
            Y_test = tf.keras.utils.to_categorical(Y_test, num_output_class)

            if prepare_environment_func: prepare_environment_func()

            model = build_model(nn_mode, num_signals, num_output_class)
            model, delta, noise_multiplier = compile_model(model, learning_rate, eps, num_unique_windows, batch_size, epochs, l2_norm_clip)
            model, history = train_model(model, X_train, Y_train, epochs, batch_size, gan_mode, validation_split=None, verbose=2)
            results[real_ids[test_idx]] = evaluate_model(model, X_test, Y_test)

            end_time = time.monotonic()
            print(f"--subj duration :{timedelta(seconds=end_time - start_time)}")

        # TODO test transformer, test private training, cluster, main

    elif eval_mode == "TSTR": #TODO test
        assert(
            syn_subj_cnt > 0
        ), "At least 1 synthetic subject needed for TSTR"
        assert(
            real_subj_cnt == 15
        ), "All 15 real subjects needed for TSTR (for now)" #TODO

        print(f"TSTR train on {syn_subj_cnt} synthetic {gan_mode} and test on {real_subj_cnt} real subjects - with eps={eps}")

        # synthetic data is train set
        X_train, Y_train = synX, synY
        Y_train = tf.keras.utils.to_categorical(Y_train, num_output_class)

        if prepare_environment_func: prepare_environment_func()

        model = build_model(nn_mode, num_signals, num_output_class)

        if eps and False: # TODO
            gan_num_unique_windows = 545
            if gan_mode == "CGAN":
                # epochs from GAN times 2 because of sliding windows doubling appearance of each sample
                # add epochs from this training because data appears again in GAN data
                gan_epochs = 1600*2 #+ epochs #TODO
            elif gan_mode == "DGAN":
                gan_epochs == 1
            elif gan_mode == "TIMEGAN":
                raise Exception("TimeGAN not supported for TSTR private training due to unclear epoch count")
            elif gan_mode == "DPCGAN":
                raise Exception("DPCGAN not supported TSTR private training")
            else:
                raise Exception("not a valid gan_mode")
        else: gan_num_unique_windows, gan_epochs = None, None

        #todo validation split
        model, delta, noise_multiplier = compile_model(model, learning_rate, eps, gan_num_unique_windows, batch_size, gan_epochs, l2_norm_clip)
        model, history = train_model(model, X_train, Y_train, epochs, batch_size, validation_split=None, verbose=2)
        
        # real data is test set
        # create loso-like results for TSTR
        for test_idx in range(0, len(real_ids)):
            print(f"TSTR-LOSO on {real_ids[test_idx]} from WESAD ({test_idx+1}/{len(real_ids)})...")
            X_test = realX[test_idx]
            Y_test = realY[test_idx]
            Y_test = tf.keras.utils.to_categorical(Y_test, num_output_class)
            results[real_ids[test_idx]] = evaluate_model(model, X_test, Y_test)

    else: raise Exception(f"Unknown eval_mode: {eval_mode}")

    df = pd.DataFrame.from_dict(results, orient='index', columns=["accuracy", "precision", "recall", "f1"])
    averages = df.mean(axis=0)
    print(f"***Subjects:\n{df}")
    print(f"***Averages:\n{averages}")

    results["average"] = {
        "accuracy": averages[0],
        "precision": averages[1],
        "recall": averages[2],
        "f1": averages[3],
    }

    return results, delta, noise_multiplier
