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
from stress_detector.model import build_cnn, build_transformer

import time
from datetime import timedelta


from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise as tfp_computer_noise
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasAdamOptimizer

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
        optimizer = VectorizedDPKerasAdamOptimizer(
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
    print(f"LOSO train on {real_subj_cnt} real and {syn_subj_cnt} synthetic {gan_mode} subjects")

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

        for test_idx in range(0, len(real_ids)):
            start_time = time.monotonic()
            # if test_sub is not None:
            #     if test_sub_index != test_index:
            #         print("SKIP training for test_index", test_index)
            #         continue

            print(f"LOSO on {real_ids[test_idx]} from WESAD ({test_idx+1}/{len(real_ids)})...")

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

            print(X_train.shape)

            model = build_model(nn_mode, num_signals, num_output_class)
            model, delta, noise_multiplier = compile_model(model, learning_rate, eps, num_unique_windows, batch_size, epochs, l2_norm_clip)
            model, history = train_model(model, X_train, Y_train, epochs, batch_size, validation_split=None, verbose=2)
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

        # synthetic data is train set
        X_train, Y_train = synX, synY
        Y_train = tf.keras.utils.to_categorical(y_train, num_output_class)

        if prepare_environment_func: prepare_environment_func()

        model = build_model(nn_mode, num_signals, num_output_class)

        if eps:
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
        else: gan_num_unique_windows, gan_epochs == None, None

        model, delta, noise_multiplier = compile_model(model, learning_rate, eps, gan_num_unique_windows, batch_size, gan_epochs, l2_norm_clip)
        model, history = train_model(model, X_train, Y_train, epochs, batch_size, validation_split=None, verbose=2)
        
        # real data is test set
        # create loso-like results for TSTR
        for test_idx in range(0, len(real_ids)):
            print(f"TSTR-LOSO on {real_ids[test_idx]} from WESAD ({test_idx+1}/{len(real_ids)})...")
            X_test = realX[test_idx]
            Y_test = realY[test_idx]
            Y_test = tf.keras.utils.to_categorical(y_test, num_output_class)
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


def evaluate( #TODO
    gan_scores_acc,
    gan_scores_f1,
    gan_scores_precision,
    gan_scores_recall,
    all_subjects_X,
    all_subjects_y,
    data_type: DataType,
    num_epochs: int,
    num_subjects: int = 1,
    with_loso: bool = True,
    subject_ids=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
    additional_name: str = None,
):
    model_path = ""

    all_subjects_X_os = all_subjects_X
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    num_output_class = 2

    if not with_loso:
        print("*** Train on Synth, test on Real ***")
    else:
        print("*** Evaluate using 'Leave One Subject Out'-Method ***")

    if not with_loso:
        print(f"DATATYPE: {data_type}")
        if data_type == DataType.DGAN:
            model_path = (
                f"models/stress_detector/tstr/syn/dgan_30000/{num_epochs}/wesad.h5"
            )
        if data_type == DataType.CGAN_LSTM:
            model_path = (
                f"models/stress_detector/tstr/syn/cgan/no_dp/lstm/{num_epochs}/wesad.h5"
            )
        if data_type == DataType.CGAN_FCN:
            model_path = (
                f"models/stress_detector/tstr/syn/cgan/no_dp/fcn/{num_epochs}/wesad.h5"
            )
        if data_type == DataType.CGAN_TRANSFORMER:
            model_path = f"models/stress_detector/tstr/syn/cgan/no_dp/transformer/{num_epochs}/wesad.h5"
        if data_type == DataType.DPCGAN:
            model_path = f"models/stress_detector/tstr/syn/cgan/dp/{num_epochs}/{additional_name}_wesad.h5"
        if data_type == DataType.TIMEGAN:
            model_path = (
                f"models/stress_detector/tstr/syn/timegan/{num_epochs}/wesad.h5"
            )
        model = tf.keras.models.load_model(model_path)
        print(f"LOADED: {model_path}")

    all_confusion_matrices = []
    for i, subject_id in enumerate(subject_ids):
        test_index = constants.SUBJECT_IDS.index(subject_id)
        X_test = np.asarray(all_subjects_X_os[test_index])
        y_test = np.asarray(all_subjects_y[test_index])
        y_test = tf.keras.utils.to_categorical(y_test, num_output_class)

        if with_loso:
            if data_type == DataType.REAL:
                model_path = f"models/stress_detector/real/{num_epochs}/wesad_s{subject_id}.h5"  # Path to save the model file
            if data_type == DataType.DGAN:
                model_path = f"models/stress_detector/loso/syn/dgan_30000/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.CGAN_LSTM:
                model_path = f"models/stress_detector/loso/syn/cgan/no_dp/lstm/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.CGAN_FCN:
                model_path = f"models/stress_detector/loso/syn/cgan/no_dp/fcn/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.CGAN_TRANSFORMER:
                model_path = f"models/stress_detector/loso/syn/cgan/no_dp/transformer/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.DPCGAN:
                model_path = f"models/stress_detector/loso/syn/cgan/dp/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.TIMEGAN:
                model_path = f"models/stress_detector/loso/syn/timegan/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"

            print("MODEL_PATH:", model_path)
            model = tf.keras.models.load_model(model_path)

        ## Create confusion matrix
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        confusion = confusion_matrix(y_true, y_pred)
        all_confusion_matrices.append(confusion)

        # get acc, prec, rec, f1
        evaluation_metrics = model.evaluate(X_test, y_test, verbose=0)

        accuracy = evaluation_metrics[1]
        precision = evaluation_metrics[2]
        recall = evaluation_metrics[3]

        f1 = 2 * precision * recall / (precision + recall)
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    print(f"GAN: {data_type.name}")
    print(f"Evaluation of CNN model trained on {num_epochs} epochs\n")
    print(f"Subject\t\t Accuracy\tPrecision\tRecall\t\tF1-Score")
    print("************************************************************************")
    for i in range(len(all_accuracies)):
        print(
            f"S{subject_ids[i]}\t\t {round(all_accuracies[i], 5):.5f}\t{round(all_precisions[i], 5):.5f}\t\t{round(all_recalls[i], 5):.5f}\t\t{round(all_f1s[i], 5):.5f}"
        )

    print("************************************************************************")
    print(
        f"Average\t\t {round(np.mean(all_accuracies), 5):.5f}\t{round(np.mean(all_precisions), 5):.5f}\t\t{round(np.mean(all_recalls), 5):.5f}\t\t{round(np.mean(all_f1s), 5):.5f}\n\n\n"
    )

    key_suffix = "_aug" if with_loso else "_tstr"
    gan_scores_acc[f"{data_type.name}{key_suffix}{additional_name}"] = all_accuracies
    gan_scores_f1[f"{data_type.name}{key_suffix}{additional_name}"] = all_f1s
    gan_scores_precision[
        f"{data_type.name}{key_suffix}{additional_name}"
    ] = all_precisions
    gan_scores_recall[f"{data_type.name}{key_suffix}{additional_name}"] = all_recalls

    return {
        "acc": gan_scores_acc,
        "f1": gan_scores_f1,
        "precision": gan_scores_precision,
        "recall": gan_scores_recall,
    }
