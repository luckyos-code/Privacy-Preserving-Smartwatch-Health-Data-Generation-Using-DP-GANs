"""
This module provides classes for handling and processing the WESAD (Wearable Stress and Affect Detection) dataset.

This includes the 'Subject' class, which encapsulates the data and behavior of a single subject in the WESAD dataset,
and the 'WESADDataset' class, which is a container for multiple 'Subject' instances.

The main functionalities provided by these classes are:
- Reading and preprocessing of the raw data
- Resampling of the data to match the BVP data sampling rate using Fourier method
- Normalization of the data using min-max scaling
- Getting the preprocessed data of a subject as a DataFrame
- Creating windows from the data frame and returning the windows and the labels
- Creating a sliding window from physiological measurement data

Example:
    main_path = "/path/to/wesad/dataset"
    subject_numbers = [2, 3, 4, 5]
    wesad_dataset = WESADDataset(main_path, subject_numbers)
    subject_dataframes = wesad_dataset.get_subject_dataframes(sampling_rate=64, normalize=True)
    all_data = wesad_dataset.get_all_data(sampling_rate=64, normalize=True)
    windows, labels = WESADDataset.create_windows(dataframe=all_data, samples_per_sec=64)
"""


import os
import pickle
from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy

from synthesizers.utils.preprocessing import get_max_value_from_list, most_common


class LabelType(Enum):
    """Enum representing the method of determining the label for a window."""

    MOST_COMMON = "most_common"
    MAX_VALUE = "max_value"


LABEL_SAMPLING_RATE = 700

label_functions = {
    "most_common": most_common,  # Function to find most common label
    "max_value": get_max_value_from_list,  # Function to find max value
}


class Subject:
    """Subject of the WESAD dataset.
    Subject Class inspired by: https://github.com/WJMatthew/WESAD"""

    def __init__(self, main_path, subject_number):
        self.name = f"S{subject_number}"
        self.subject_keys = ["signal", "label", "subject"]
        self.signal_keys = ["chest", "wrist"]
        self.chest_keys = ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"]
        self.wrist_keys = ["ACC", "BVP", "EDA", "TEMP"]

        file_path = os.path.join(main_path, self.name, f"{self.name}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as file:
            self.data = pickle.load(file, encoding="latin1")
        self.labels = self.data["label"]

    def get_wrist_data(self):
        """Returns data measured by the E4 Empatica"""

        data = self.data["signal"]["wrist"]
        return data

    def resample_data(self, signal, num_samples):
        """Resample data to match BVP data sampling rate using Fourier method"""
        return scipy.signal.resample(signal, num_samples)

    # Min-max scaling function
    def normalize_min_max(self, data):
        return (data - data.min()) / (data.max() - data.min()) * 2 - 1

    def get_subject_dataframe(self, sampling_rate: int = 64, normalize: bool = True):
        """Returns a dataframe with the preprocessed data of the subject"""
        wrist_data = self.get_wrist_data()
        bvp_signal = wrist_data["BVP"][:, 0]
        eda_signal = wrist_data["EDA"][:, 0]
        acc_x_signal = wrist_data["ACC"][:, 0]
        acc_y_signal = wrist_data["ACC"][:, 1]
        acc_z_signal = wrist_data["ACC"][:, 2]
        temp_signal = wrist_data["TEMP"][:, 0]

        num_samples = (len(self.labels) // LABEL_SAMPLING_RATE) * sampling_rate
        bvp_resampled = self.resample_data(bvp_signal, num_samples)
        eda_resampled = self.resample_data(eda_signal, num_samples)
        temp_resampled = self.resample_data(temp_signal, num_samples)
        acc_x_resampled = self.resample_data(acc_x_signal, num_samples)
        acc_y_resampled = self.resample_data(acc_y_signal, num_samples)
        acc_z_resampled = self.resample_data(acc_z_signal, num_samples)
        label_df = pd.DataFrame(self.labels, columns=["label"])
        label_df.index = [
            (1 / LABEL_SAMPLING_RATE) * i for i in range(len(label_df))
        ]  # Sampling rate of the label
        label_df.index = pd.to_datetime(label_df.index, unit="s")

        data_arrays = zip(
            bvp_resampled,
            eda_resampled,
            acc_x_resampled,
            acc_y_resampled,
            acc_z_resampled,
            temp_resampled,
        )
        subject_data = pd.DataFrame(
            data=data_arrays, columns=["BVP", "EDA", "ACC_x", "ACC_y", "ACC_z", "TEMP"]
        )
        subject_data.index = [
            (1 / sampling_rate) * i for i in range(len(subject_data))
        ]  # Sampling rate of BVP
        subject_data.index = pd.to_datetime(subject_data.index, unit="s")
        subject_data = subject_data.join(label_df)
        subject_data["label"] = subject_data["label"].fillna(method="ffill")
        subject_data.reset_index(drop=True, inplace=True)
        subject_data.drop(
            subject_data[subject_data["label"].isin([0.0, 4.0, 5.0, 6.0, 7.0])].index,
            inplace=True,
        )
        subject_data["label"] = subject_data["label"].replace(
            [1.0, 2.0, 3.0], [0, 1, 2]
        )
        subject_data.reset_index(drop=True, inplace=True)

        if normalize:
            # Normalize data (no train test leakage since data frame per subject)
            feature_cols = ["BVP", "EDA", "ACC_x", "ACC_y", "ACC_z", "TEMP"]
            subject_data[feature_cols] = (
                subject_data[feature_cols] - subject_data[feature_cols].min()
            ) / (subject_data[feature_cols].max() - subject_data[feature_cols].min())

        return subject_data


class WESADDataset:
    """Wrapper class for multiple Subject instances from the WESAD dataset."""

    def __init__(self, main_path: str, subject_numbers: List[int]):
        self.subjects = {}
        for num in subject_numbers:
            self.subjects[num] = Subject(main_path, num)

    def get_subject_dataframes(
        self,
        sampling_rate: int = 64,
        normalize: bool = True,
    ):
        """Returns a dictionary of preprocessed dataframes for each subject."""
        subject_dataframes = {}
        for num, subject in self.subjects.items():
            subject_dataframes[num] = subject.get_subject_dataframe(
                sampling_rate=sampling_rate, normalize=normalize
            )
        return subject_dataframes

    def get_all_data(self, sampling_rate: int = 64, normalize: bool = True):
        """Returns a concatenated dataframe of preprocessed data from all subjects."""
        subject_dataframes = self.get_subject_dataframes(
            sampling_rate=sampling_rate, normalize=normalize
        )
        concatenated_df = pd.concat(subject_dataframes.values())
        concatenated_df = concatenated_df.reset_index(drop=True)
        return concatenated_df

    def get_all_data_and_windows(
        self,
        sampling_rate: int = 64,
        normalize: bool = True,
        window_function: callable = None,  # Add window function argument
        label_type: LabelType = LabelType.MOST_COMMON,
    ):
        """Returns a concatenated dataframe and windows of preprocessed data from all subjects."""
        windows_list = []
        labels_list = []
        for num, subject in self.subjects.items():
            subject_data = subject.get_subject_dataframe(
                sampling_rate=sampling_rate, normalize=normalize
            )
            if window_function:  # Only create windows if window function is given
                windows, labels = window_function(subject_data, label_type=label_type)
                windows_list.append(windows)
                labels_list.append(labels)
        all_windows = np.concatenate(windows_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        return all_windows, all_labels

    @staticmethod
    def create_windows(
        dataframe: pd.DataFrame, label_type: LabelType, samples_per_sec: int = 1
    ) -> Tuple[np.ndarray, list]:
        """Creates windows from the data frame and returns the windows and the labels.
        If the window is assigned to multiple labels, the maximum label for that period is selected.
        We decide for this handling, because already the presence of a stress impulse is evaluated for us as a stress window
        and what is consistent with the handling of stress by Ehrhart et al. (https://www.mdpi.com/1424-8220/22/16/5969)

        Args:
            dataframe (pd.DataFrame): Subject DataFrame
            samples_per_sec (int): Samples per second
            label_type (LabelType): Enum to decide which function to use for label selection

        Returns:
            tuple[np.ndarray,list]: Windows representing the activity of the subject in one minute and the corresponding labels.
        """
        # Create an empty list for the windows and labels
        windows = []
        labels = []

        # Calculate the window length in samples
        window_len = samples_per_sec * 60
        label_func = label_functions[label_type.value]

        # Loop over the rows in the DataFrame to create the windows
        for i in range(0, dataframe.shape[0] - window_len, window_len):
            # Get the window data and label
            window = dataframe[i : i + window_len]

            label = int(label_func(dataframe["label"][i : i + window_len].to_list()))

            # Convert the window data to a numpy array
            window = window.to_numpy()

            # Add the window and label to the list
            windows.append(window)
            labels.append(label)

        # Convert the windows and labels to numpy arrays
        windows = np.array(windows)
        labels = np.array(labels)

        # Return the windows and labels as a tuple
        return windows, labels

    @staticmethod
    def sliding_windows(
        data: pd.DataFrame, label_type: LabelType
    ) -> Tuple[np.array, np.array]:
        """Create a sliding window from physiological measurement data.

        Args:
            data (pd.DataFrame): Pandas DataFrame object containing physiological measurements.
            label_type (LabelType): Enum to decide which function to use for label selection

        Returns:
            tuple[np.array, np.array]: Windows with physiological measurements and corresponding labels.
        """
        container_array = np.array(data)

        window_size = 60  # Window size in seconds
        overlap_size = 30  # Overlap size in seconds

        # Calculate the number of data points per window and overlap
        window_points = window_size * 1  # Since the capturing frequency is 1 Hz
        overlap_points = overlap_size * 1

        windows = []
        labels = []
        label_func = label_functions[label_type.value]

        for i in range(0, container_array.shape[0] - window_points + 1, overlap_points):
            window = container_array[i : i + window_points]
            windows.append(window)

            label = int(label_func(data["label"][i : i + window_points].to_list()))
            labels.append(label)

        return np.array(windows), np.array(labels)
