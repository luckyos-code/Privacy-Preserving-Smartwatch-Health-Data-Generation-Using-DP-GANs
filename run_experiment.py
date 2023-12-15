import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import os, sys, copy
from contextlib import nullcontext
from contextlib import contextmanager
import time
from datetime import timedelta
from typing import Optional, Tuple

from stress_slurm import config
from stress_detector.data.datatype import DataType
from stress_detector.preprocessing import load_data
from stress_detector.training import train
from stress_detector import constants
from synthesizers.preprocessing.wesad import WESADDataset

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def run(
    real_subj_cnt: int = 15,
    syn_subj_cnt: int = 0,
    gan_mode: str = "CGAN",
    sliding_windows: bool = False,
    eval_mode: str = "LOSO", # LOSO or TSTR
    nn_mode: str = "CNN", # CNN or transformer
    eps: float = None,
    loaded_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    data_noise_parameter: Optional[float] = None

) -> dict:
    gan_mode = gan_mode if syn_subj_cnt > 0 else "noGAN" # or any other gan name

    # check if cluster is in correct env
    assert(
        pd.__version__ == "2.0.1" or pd.__version__ == "1.4.1"
    ), f"got pandas {pd.__version__} but expected 2.0.1 or 1.4.1, probably means wrong cluster env"

    # look for cluster gpu
    print(tf.config.list_physical_devices('GPU'))

    config.prepare_environment()

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
        "train_config": {**(config.TRAIN_CONFIG[nn_mode]), **{"eval_mode": eval_mode, "random_seed": config.RANDOM_SEED, "data_noise_parameter":data_noise_parameter}},
    }

    # realX and realY is nested subject-wise for LOSO training
    # unnest using: np.concatenate(realX)
    # synX and synY is ready for training
    if loaded_data: # dont reload data if CI experiment
        (realX, realY, synX, synY) = loaded_data
    else: 
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
        data_noise_parameter=data_noise_parameter
    )

    return run_config

def ci_experiment(
    num_runs: int = 10,
    real_subj_cnt: int = 15,
    syn_subj_cnt: int = 0,
    gan_mode: str = "CGAN",
    sliding_windows: bool = False,
    eval_mode: str = "LOSO",
    nn_mode: str = "CNN",
    eps: float = None,
    silent_runs: bool = True,
    data_noise_parameter: Optional[float] = None
) -> dict:
    # load experiment data
    gan_mode = gan_mode if syn_subj_cnt > 0 else "noGAN" # or any other gan name
    realX, realY, synX, synY = load_data(
        real_subj_cnt=real_subj_cnt,
        real_path=config.REAL_PATH,
        syn_subj_cnt=syn_subj_cnt,
        syn_path=config.GAN_CONFIG[gan_mode]["path"],
        gan_mode=gan_mode,
        use_sliding_windows=sliding_windows
    )

    # run an experiment num_runs times for ci
    run_configs_lst = []
    for i in range(num_runs):
        print(f"*CI run {i+1}/{num_runs}...")
        start_time = time.monotonic()
        with suppress_stdout() if silent_runs else nullcontext():
            single_run_config = run(
                real_subj_cnt,
                syn_subj_cnt,
                gan_mode,
                sliding_windows,
                eval_mode,
                nn_mode,
                eps,
                loaded_data=(realX, realY, synX, synY),
                data_noise_parameter=data_noise_parameter
            )
        run_configs_lst.append(single_run_config)
        end_time = time.monotonic()
        print(f"--CI run duration :{timedelta(seconds=end_time - start_time)}\n")

    # TODO
    # get final results: 95-ci, average, max, min, unusable model ratio
    dfs = [pd.DataFrame.from_dict(run_config["results"], 
                                  orient='index',
                                  columns=["accuracy", "precision", "recall", "f1"]) for run_config in run_configs_lst]
    df = pd.concat(dfs, axis=0)

    grp_df = df.groupby(df.index)

    ci_run_config = copy.deepcopy(run_configs_lst[0])
    ci_run_config.pop("results")
    ci_run_config["results"] = {
        "num_runs": num_runs,
        "mean": {},
        "std": {},
        "poor_ratio": None
    }

    real_ids = run_configs_lst[0]["results"].keys()

    # add mean and std
    mean_df = grp_df.mean()
    std_df = grp_df.std()
    for key in real_ids:
        ci_run_config["results"]["mean"][key] = mean_df.loc[key].to_dict()
        ci_run_config["results"]["std"][key] = std_df.loc[key].to_dict()

    # add poor_ratio for models with f1 <= 0.3, since target class stress is 0.3
    cnt = 0
    for run_config in run_configs_lst:
        if run_config["results"]["average"]["f1"] <= 0.3:
            cnt += 1
    ci_run_config["results"]["poor_ratio"] = cnt / len(run_configs_lst)

    # print ci results
    print(f"***CI results:\n")
    df = pd.DataFrame.from_dict(ci_run_config["results"]["mean"], orient='index', columns=["accuracy", "precision", "recall", "f1"])
    print(df)
    print(f"poor model ratio: {ci_run_config['results']['poor_ratio']}")

    return ci_run_config

