# tensorflow-privacy==0.7.3
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise as tfp_computer_noise
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

import numpy as np

from stress_detector.preprocessing import load_data

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

"""
Add noise to data based on multiplier or epsilon

Args:
    data (np.ndarray): training data to noise, e.g. full WESAD shape should be of shape (15,) with unterlying concatenated data of (530,6,210).
    target_epsilon (float): wanted epsilon replaces noise_multiplier arg, if given the noise is calculated assuming a machine learning task.
    noise_multiplier (float): wanted noise_multipier if no target_epsilon given, directly states the wanted noise.
    noise_type (str of "laplace" or "gaussian"): wanted noise distribution.
    clip_max (bool): clipping to max of each signal for dp, if false clip to 1 as sensitity of similarity func.

Returns:
    tuple[np.ndarray, float, target_epsilon]: Windows with physiological measurements and corresponding labels.
"""
def create_noisy_data(data: np.ndarray,
                      target_epsilon: float=None,
                      noise_multiplier: float=None,
                      noise_type: str="laplace",
                      clip_max: bool=False):

    if len(data.shape) == 1: # if data is saved per subject
        points_per_subj = [subj.shape[0] for subj in data] # save number of data points belonging to each subject
        data = np.concatenate(data) # flatten data to (n,6,210) for DP
    else: points_per_subj = None

    n = data.shape[0] # number of training samples
    delta = compute_delta(n)
    if target_epsilon:
        noise_multiplier = compute_noise( # simulate one round of dp training to get aquivalent noise multiplier
            n=n,
            target_epsilon=target_epsilon,
            delta=delta,
            batch_size=n, # all data is evaluated at once and not in batches = sampling rate 100%
            epochs=1 # all data is only evaluated in one query during attack = 1 step
        )
    else: # calculate assumed epsilon based on given noise
        target_epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=n,
            batch_size=n, # all data is evaluated at once and not in batches = sampling rate 100%
            noise_multiplier=noise_multiplier,
            epochs=1, # all data is only evaluated in one query during attack = 1 step
            delta=delta
        )

    signal_splits = np.split(data, data.shape[1], axis=1) # per signal clipping preparation
    
    for idx, signal_data in enumerate(signal_splits):
        clip = 1 if not clip_max else np.max(signal_data) # 1 as sensitity of similarity func or max of each signal as clip for DP

        if noise_type == "laplace":
            signal_splits[idx] = signal_data + np.random.laplace(loc=clip*noise_multiplier, size=signal_data.shape)
        elif noise_type == "gaussian":
            signal_splits[idx] = signal_data + np.random.normal(loc=clip*noise_multiplier, size=signal_data.shape)

    noisy_data = np.concatenate(signal_splits, axis=1)

    if points_per_subj: # reverse flattening back to subject-based array
        noisy_data = np.asarray([noisy_data[:idx] for idx in points_per_subj], dtype=object)

    return noisy_data, noise_multiplier, target_epsilon


def main():
    realX, realY, synX, synY = load_data(
        real_subj_cnt=15,
        real_path="stress_slurm/data/wesad_1hz.pickle",
        syn_subj_cnt=0,
        syn_path=None,
        gan_mode=None,
        use_sliding_windows=False
    )

    noisyX, noise_multiplier, target_epsilon = create_noisy_data(realX, noise_multiplier=1.0, clip_max=False)

    # print(f"Noise: {noise_multiplier}")
    # print(f"Epsilon: {target_epsilon}")
    print("\nExcerpt of change in real data:")
    print(realX[0][0]-noisyX[0][0])


if __name__ == "__main__":
    main()