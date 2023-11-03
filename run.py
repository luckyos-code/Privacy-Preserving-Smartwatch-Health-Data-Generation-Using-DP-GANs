import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import os
import sys
from stress_slurm import config
from stress_detector.data.datatype import DataType
from stress_detector.preprocessing import load_data
from stress_detector.training import train, evaluate
from stress_detector import constants
from synthesizers.preprocessing.wesad import WESADDataset

# check if cluster is in correct env
assert(
    pd.__version__ == "2.0.1"
), f"got pandas {pd.__version__} but expected 2.0.1, probably means wrong cluster env"

# hide annoying messages
tf.get_logger().setLevel('ERROR')

# look for cluster gpu
print(tf.config.list_physical_devices('GPU'))

config.prepare_environment()

# GAN subjs counts [1,5,10,15,30,50,100]
real_subj_cnt, syn_subj_cnt = 15 , 15
gan_mode = "DGAN" if syn_subj_cnt > 0 else "noGAN" # or any other gan name
sliding_windows=False
eval_mode = "TSTR" # LOSO or TSTR
nn_mode = "CNN" # CNN or transformer
eps = None


# TODO scenario config
run_config = {
    # real data
    "real_subj_cnt": real_subj_cnt, # max 15 wesad
    "real_path": config.REAL_PATH,
    "sliding_windows": sliding_windows,
    # gan data
    "syn_subj_cnt": syn_subj_cnt,  # timegan max 30, others max 100
    "gan_config": config.GAN_CONFIG[gan_mode],
    # privacy
    "privacy_config": {**(config.PRIVACY_CONFIG), **{"eps": eps, "num_unique_windows": None}},
    # training
    "train_config": {**(config.TRAIN_CONFIG[nn_mode]), **{"eval_mode": eval_mode, "random_seed": config.RANDOM_SEED}},
}

# realX and realY is nested subject-wise for LOSO training
# unnest using: np.concatenate(realX)
# synX and synY is ready for training
realX, realY, synX, synY = load_data(
    real_subj_cnt=run_config["real_subj_cnt"],
    real_path=run_config["real_path"],
    syn_subj_cnt=run_config["syn_subj_cnt"],
    syn_path=run_config["gan_config"]["path"],
    gan_mode=run_config["gan_config"]["gan_mode"],
    use_sliding_windows=run_config["sliding_windows"]
)

if eps and real_subj_cnt > 0: # TODO double epochs if sliding window
    run_config["privacy_config"]["num_unique_windows"] = np.concatenate(realX).shape[0] if not sliding_windows else np.concatenate(realX).shape[0] // 2

(run_config["results"],
 run_config["privacy_config"]["delta"],
 run_config["privacy_config"]["noise_multiplier"]) = train(
    realData=(realX, realY),
    real_subj_cnt=run_config["real_subj_cnt"],
    synData=(synX, synY),
    syn_subj_cnt=run_config["syn_subj_cnt"],
    gan_mode=run_config["gan_config"]["gan_mode"],
    nn_mode=run_config["train_config"]["nn_mode"],
    eval_mode=run_config["train_config"]["eval_mode"],
    epochs=run_config["train_config"]["epochs"],
    batch_size=run_config["train_config"]["batch_size"],
    learning_rate=run_config["train_config"]["learning_rate"],
    eps=run_config["privacy_config"]["eps"],
    num_unique_windows=run_config["privacy_config"]["num_unique_windows"],
    l2_norm_clip=run_config["privacy_config"]["l2_norm_clip"],
    prepare_environment_func=config.prepare_environment,
) #TODO get results dict over multiple runs and then CI of these results as final result dict

