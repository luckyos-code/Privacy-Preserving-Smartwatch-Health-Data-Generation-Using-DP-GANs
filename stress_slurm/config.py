import numpy as np
import tensorflow as tf

from stress_detector.data.datatype import DataType

REAL_PATH = "stress_slurm/data/wesad_1hz.pickle"
RANDOM_SEED = 42

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
                "type": None,
                "path": None,
                "eps": None
            },
    "TIMEGAN": {
                "gan_mode": "TIMEGAN", 
                "type": DataType.TIMEGAN,
                "path": "stress_slurm/data/15_subj_timeGAN.npy",
                "eps": None
            },
    "DGAN": {
                "gan_mode": "DGAN", 
                "type": DataType.DGAN,
                "path": "stress_slurm/data/100_subj_DGAN.csv",
                "eps": None
            },
    "CGAN": {
                "gan_mode": "CGAN", 
                "type": DataType.CGAN,
                "path": "stress_slurm/data/100_subj_cGAN.csv",
                "eps": None
            },
    "DPCGAN-e-10": {
                "gan_mode": "DPCGAN-e-10", 
                "type": DataType.DPCGAN,
                "path": "stress_slurm/data/100_subj_cGAN-e-10.csv",
                "eps": 10
            },
    "DPCGAN-e-1": {
                "gan_mode": "DPCGAN-e-1", 
                "type": DataType.DPCGAN,
                "path": "stress_slurm/data/100_subj_cGAN-e-1.csv",
                "eps": 1
            },
    "DPCGAN-e-0.1": {
                "gan_mode": "DPCGAN-e-0.1", 
                "type": DataType.DPCGAN,
                "path": "stress_slurm/data/100_subj_cGAN-e-0.1.csv",
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
                "epochs": 10, # 110
                "batch_size": 50,
                "learning_rate": 1e-3, # 1e-4
            },
}

PRIVACY_CONFIG = {
    "l2_norm_clip": 1.0,
}

