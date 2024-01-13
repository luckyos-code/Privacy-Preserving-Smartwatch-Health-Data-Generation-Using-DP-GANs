import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from argparse import Namespace

from stress_detector.data.datatype import DataType

RANDOM_SEED = 42

REAL_PATH = "stress_slurm/data/wesad_1hz.pickle"
RESULTS_FOLDER_PATH = "stress_slurm/results"

def set_random_seeds(random_seed: int = 42) -> None:
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    #print(np.random.get_state()[1][:2])

def prepare_environment(random_seed: int = 42) -> None:
    tf.keras.backend.clear_session()
    set_random_seeds(RANDOM_SEED)

GAN_CONFIG = {
    "noGAN": {
                "gan_mode": "noGAN",
                "name": "WESAD",
                "type": None,
                "path": None,
                "eps": None
            },
    "TIMEGAN": {
                "gan_mode": "TIMEGAN", 
                "name": "TimeGAN", 
                "path": "stress_slurm/data/15_subj_timeGAN.npy",
                "eps": None
            },
    "DGAN": {
                "gan_mode": "DGAN", 
                "name": "DGAN", 
                "path": "stress_slurm/data/100_subj_DGAN.csv",
                "eps": None
            },
    "CGAN": {
                "gan_mode": "CGAN", 
                "name": "CGAN", 
                "path": "stress_slurm/data/100_subj_cGAN.csv",
                "eps": None
            },
    "DPCGAN-e-10": {
                "gan_mode": "DPCGAN-e-10", 
                "name": "DP-CGAN \u03B5=10",
                "path": "stress_slurm/data/100_subj_DP-cGAN-e-10.csv",
                "eps": 10
            },
    "DPCGAN-e-1": {
                "gan_mode": "DPCGAN-e-1",
                "name": "DP-CGAN \u03B5=1", 
                "path": "stress_slurm/data/100_subj_DP-cGAN-e-1.csv",
                "eps": 1
            },
    "DPCGAN-e-0.1": {
                "gan_mode": "DPCGAN-e-0.1", 
                "name": "DP-CGAN \u03B5=0.1",
                "path": "stress_slurm/data/100_subj_DP-cGAN-e-0.1.csv",
                "eps": 0.1
    },
}

TRAIN_CONFIG = {
    "CNN": {
                "nn_mode": "CNN", 
                "epochs": 10,
                "batch_size": 50,
                "learning_rate": 1e-3,
            },
    "Transformer": {
                "nn_mode": "Transformer", 
                "epochs": 110,
                "batch_size": 50,
                "learning_rate": 1e-4,
            },
    "CNN-LSTM": {
                "nn_mode": "CNN-LSTM", 
                "epochs": 20,
                "batch_size": 50,
                "learning_rate": 1e-3,
            },            
}

PRIVACY_CONFIG = {
    "l2_norm_clip": 1.0,
}

def create_arg_parse_instance() -> ArgumentParser:
    parser = ArgumentParser(prog="Augmented Private Stress Detection using GANs and DP")

    parser.add_argument(
        "-i",
        "--id",
        type=int,
        help=
        "Specific experiment id to run from config.",
        metavar="I",
    )

    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        help=
        "Number of experiment runs for CI calculation.",
        metavar="N",
    )
    parser.add_argument(
        "-r",
        "--real",
        type=int,
        help=
        "Number of real subjects.",
        metavar="R",
    )
    parser.add_argument(
        "-s",
        "--syn",
        type=int,
        help=
        "Number of synthetic subjects.",
        metavar="S",
    )
    parser.add_argument(
        "-g",
        "--gan",
        type=str,
        choices=["TIMEGAN", "DGAN", "CGAN", "DPCGAN-e-10", "DPCGAN-e-1", "DPCGAN-e-0.1"],
        help=
        "Specify which GAN should be used for synthetic data. Only one can be selected!",
        metavar="G",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["CNN", "Transformer", "CNN-LSTM"],
        help=
        "Model for classification. Only one can be selected!",
        metavar="M",
    )
    parser.add_argument(
        "--sliding",
        help=
        "Set this flag if sliding windows should be used for real data.",
        action="store_true"
    )
    parser.add_argument(
        "--saving",
        help=
        "Set this flag if sliding windows should be used for real data.",
        action="store_true"
    )
    parser.add_argument(
        "-e",
        "--eval",
        type=str,
        choices=["LOSO", "TSTR"],
        help=
        "Evaluation mode to be used. Only one can be selected!",
        metavar="E",
    )
    parser.add_argument(
        "-p",
        "--privacy",
        type=float,
        choices=[0, 10, 1, 0.1],
        help="Epsilon used for privacy. No selection for non-private.",
        metavar="P",
    )
    return parser

