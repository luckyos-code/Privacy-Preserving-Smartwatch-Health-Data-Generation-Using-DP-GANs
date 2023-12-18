from __future__ import annotations

import io
import zipfile
from enum import Enum

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd()))
#raise Exception(os.path.join(os.getcwd()))
from synthesizers.dgan.dgan import DGAN
from synthesizers.dgan.config import DGANConfig

import keras
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


class StressType(Enum):
    BOTH = "Both"
    NON_STRESS = "Non-Stress"
    STRESS = "Stress"
    ALL = "All" # including amusement

MODEL_DICT = {
    "cGAN": "cgan/resilient_sweep-1",
    "DP-cGAN-e-0.1": "dp-cgan-e-0_1/light-sweep-1",
    "DP-cGAN-e-1": "dp-cgan-e-1/revived-sweep-2",
    "DP-cGAN-e-10": "dp-cgan-e-10/usual-sweep-3",
    
    "new-10": "new_cgan/10",
    "new-1": "new_cgan/1",
}  # Update with your model names or paths

ITOSIG = {
    0: "BVP",
    1: "EDA",
    2: "ACC_x",
    3: "ACC_y",
    4: "ACC_z",
    5: "TEMP",
    6: "Label",
}


def generate_samples(
    model: keras.models.Model | DGAN, num_samples: int, latent_dim: int, label_value: int
) -> np.ndarray:
    """
    Generate synthetic data samples.

    Args:
        model: Trained model used to generate the synthetic samples.
        num_samples: The number of synthetic samples to generate.
        latent_dim: The dimension of the latent space in the model.
        label_value: The label value of the synthetic samples to generate.

    Returns:
        A numpy array of the generated synthetic samples.
    """

    # Check that num_samples and latent_dim are positive integers
    assert (
        isinstance(num_samples, int) and num_samples > 0
    ), "num_samples should be a positive integer"
    assert (
        isinstance(latent_dim, int) and latent_dim > 0
    ), "latent_dim should be a positive integer"

    if type(model) is DGAN:
        _, synth_samples = model.generate_numpy(num_samples)
        synth_samples[:, :, 6] = np.where(synth_samples[:, :, 6] >= 0.5, 1, 0)
    else:
        labels = tf.fill([num_samples, 1], label_value)
        append_labels = np.full([num_samples, 60, 1], label_value)

        random_vector = tf.random.normal(shape=(num_samples, latent_dim))
        synth_samples = model([random_vector, labels])
        synth_samples = np.append(np.array(synth_samples), append_labels, axis=2)

    return synth_samples


def generate(
    model: keras.models.Model | DGAN,
    num_syn_samples: int,
    latent_dim: int,
    stress_type: StressType,
) -> np.ndarray:
    if not isinstance(stress_type, StressType):
        raise ValueError(
            f"stress_type must be an instance of StressType Enum, got {stress_type}"
        )

    if stress_type in [StressType.NON_STRESS, StressType.STRESS]:
        label_value = 0 if stress_type == StressType.NON_STRESS else 1
        synth_samples = generate_samples(
            model, num_syn_samples, latent_dim, label_value
        )
    elif stress_type in [StressType.BOTH]:
        if type(model) is DGAN:
            synth_samples = generate_samples(model, num_syn_samples, latent_dim, 0)
        else:
            # wesad subject has 70% non-stress and 30% stress data on average
            num_samples_non =  int(np.floor(0.7 * num_syn_samples))
            num_samples_stress = int(np.floor(0.3 * num_syn_samples))
            non_stress_samples = generate_samples(model, num_samples_non, latent_dim, 0)
            stress_samples = generate_samples(model, num_samples_stress, latent_dim, 1)
            synth_samples = np.concatenate((non_stress_samples, stress_samples))
    elif stress_type in [StressType.ALL]:
        num_samples_third = num_syn_samples // 3
        base_samples = generate_samples(model, num_samples_third, latent_dim, 0)
        amuse_samples = generate_samples(model, num_samples_third, latent_dim, 2)
        stress_samples = generate_samples(model, num_samples_third, latent_dim, 1)
        synth_samples = np.concatenate((base_samples, amuse_samples, stress_samples))

    return synth_samples


def plot_generated_data(data: np.ndarray, title: str) -> None:
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[0]):
        plt.plot(data[i, :, 0], label=f"Sample {i+1}")
    plt.title(title)
    plt.xlabel("Time in seconds (s)")
    plt.ylabel("Signal Value")
    plt.ylim([0, 1])
    plt.grid(True)
    st.pyplot()


def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def load_model(model_name: str) -> keras.models.Model | DGAN:
    if model_name == "dgan":
        return DGAN.load(f"models/{model_name}/model_load_generator")
    else:
        return keras.models.load_model(f"models/{model_name}/generator")


def generate_synthetic_data(
    model: keras.models.Model | DGAN, num_syn_samples: int, latent_dim: int, stress_type: str
) -> pd.DataFrame:
    synth_data = generate(model, num_syn_samples, latent_dim, stress_type)
    df = pd.DataFrame(synth_data.reshape(-1, synth_data.shape[-1]))

    # Add column names
    df.columns = [ITOSIG[i] for i in range(synth_data.shape[-1])]

    # Add index
    df.index = pd.RangeIndex(1, len(df) + 1)

    if stress_type in [StressType.ALL]:
        df['Label'] = df['Label'].replace(2, 0.5)

    return df


def display_dataframe(df: pd.DataFrame, show_profile: bool = True) -> None:
    st.dataframe(df)
    # For each signal in the synthetic data
    if 'sid' in df.columns: st.line_chart(df.drop(columns=["sid"]), height=200)
    else: st.line_chart(df, height=200)

    if show_profile:
        pr = ProfileReport(df, explorative=True)
        st_profile_report(pr)


def download_dataframe(df: pd.DataFrame, model_name: str) -> None:
    csv = convert_df(df)
    st.download_button(
        "Download synthetic data as CSV",
        csv,
        file_name=f"synthetic_{model_name}.csv",
        mime="text/csv",
        key="down-load-csv",
    )

def download_dataframe_zipped(subjs: np.ndarray, model_name: str) -> None:
    # create one big csv
    df = pd.concat(subjs)
    subjs_csv = [(f"{len(subjs)}_subj_synthetic_{model_name}.csv", convert_df(df))]

    # create many single csvs
    #subjs_csv = [(f"g{df['sid'].iloc[0]}.csv", convert_df(df)) for df in subjs]

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in subjs_csv:
            zip_file.writestr(file_name, data.decode("utf-8"))  # Add each CSV to the zip

    zip_buffer.seek(0)
    zip_data = zip_buffer.getvalue()

    st.download_button(
        "Download synthetic data as zipped CSVs",
        zip_data,
        file_name=f"{len(subjs)}_subj_synthetic_{model_name}.zip",
        mime="application/zip",
        key="down-load-zip",
    )


def run():
    st.subheader("Generate synthetic data from a trained model")
    col1, col2 = st.columns([4, 2])
    with col1:
        model_selection = st.selectbox(
            "Select the model", list(MODEL_DICT.keys())
        )
        model_name = MODEL_DICT[model_selection]

        latent_dim = st.number_input(
            "Length of windows in seconds", min_value=0, value=60
        )
        num_syn_samples = st.number_input(
            "Number of synthetic windows to generate", min_value=0, value=36
        )
        stress_type_str = st.selectbox(
            "Select type of data to generate", [e.value for e in StressType]
        )
        num_gen_rounds = st.number_input(
            "Select number of generation rounds (i.e. subjects)", min_value=1, value=1
        )

        # Map string back to StressType Enum
        stress_type = StressType(stress_type_str)

    if st.button("Generate samples"):
        if num_gen_rounds == 1:
            try:
                model = load_model(model_name)
                st.success(
                    "The model was properly loaded and is now ready to generate synthetic samples!"
                )
                with st.spinner("Generating samples... This might take time."):
                    df = generate_synthetic_data(
                        model, num_syn_samples, latent_dim, stress_type
                    )

                display_dataframe(df)
                st.success("Synthetic data has been generated successfully!")
                download_dataframe(df, model_selection)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else: # generae multiple csvs at once
            try:
                model = load_model(model_name)
                st.success(
                    "The model was properly loaded and is now ready to generate synthetic samples!"
                )
                with st.spinner("Generating samples... This might take time."):
                    subjs = []
                    for i in range(1, num_gen_rounds+1):
                        sid = 1000 + i # create ids starting at 1000
                        if model_name != "dgan":
                            df = generate_synthetic_data(
                                model, num_syn_samples, latent_dim, stress_type
                            )
                        else: # DGAN needs optimized stress ratio by hand
                            stress_ratio = 0.0 
                            while not (0.29 <= stress_ratio <= 0.32): # wesad min: 0.29, wesad max: 0.32
                                df = generate_synthetic_data(
                                    model, num_syn_samples, latent_dim, stress_type
                                )
                                stress_cnt = df['Label'].value_counts()[1.0]
                                stress_ratio = stress_cnt / len(df['Label'])
                        # add subject id as column
                        df.insert(0, column="sid", value=np.full(shape=len(df.index), fill_value=sid))
                        subjs.append(df)
                
                [display_dataframe(df, show_profile=False) for df in subjs[:3]]
                st.success("Synthetic data has been generated successfully!")
                download_dataframe_zipped(subjs, model_selection)
            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run()
