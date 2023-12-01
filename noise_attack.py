import numpy as np

"""
Add noise to data based on multiplier or epsilon

Args:
    data (np.ndarray): data to noise, in two dimensional np.array of signals e.g. (6,100) for the six signals and 100 data points each.
    noise_multiplier (float): wanted noise_multipier for the random distribution.
    noise_type (str of "laplace" or "gaussian"): wanted noise distribution.
    clip_max (bool): clipping to max of each signal for dp, if false clip to 1 as sensitity of similarity func.

Returns:
    np.ndarray: noised data.
"""
def create_noisy_data(data: np.ndarray, noise_multiplier: float=None, noise_type: str="laplace", clip_max: bool=False):
    noisy_data = np.empty(data.shape)
    for idx, signal_data in enumerate(data):
        clip = 1 if not clip_max else np.max(signal_data) # 1 as sensitity of similarity func or max of each signal as clip for DP
        if noise_type == "laplace":
            noisy_data[idx] = signal_data + np.random.laplace(scale=clip*noise_multiplier, size=signal_data.shape)
        elif noise_type == "gaussian":
            noisy_data[idx] = signal_data + np.random.normal(scale=clip*noise_multiplier, size=signal_data.shape)
    return noisy_data

def main():
    noise_multiplier = 100.0
    data = np.random.rand(6,100)

    noisy_data = create_noisy_data(data, noise_multiplier)
    diff = noisy_data[0][:100] - data[0][:100]
    print("diff: ", diff)
    print("min: ", diff.min())
    print("max: ", diff.max())
    


if __name__ == "__main__":
    main()