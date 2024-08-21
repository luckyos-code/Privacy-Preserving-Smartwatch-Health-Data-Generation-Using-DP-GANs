# Privacy-Preserving Smartwatch Health Data Generation For Stress Detection Using GANs

This repository contains the code and documentation for a research project on generating synthetic smartwatch health data while preserving the privacy of the original data owners. The project uses a combination of differential privacy and generative adversarial networks (DP-GANs) to create synthetic data that closely resembles the original data in terms of statistical properties and data distributions.

![AI generated smartwatch image](images/smartwatch.png)

## Abstract

Smartwatch health sensor data are increasingly utilized in smart health applications and patient monitoring, including stress detection. However, such medical data often comprise sensitive personal information and are resource-intensive to acquire for research purposes. In response to this challenge, we introduce the privacy-aware synthetization of multi-sensor smartwatch health readings related to moments of stress, employing Generative Adversarial Networks (GANs) and Differential Privacy (DP) safeguards. Our method not only protects patient information but also enhances data availability for research. To ensure its usefulness, we test synthetic data from multiple GANs and employ different data enhancement strategies on an actual stress detection task. Our GAN-based augmentation methods demonstrate significant improvements in model performance, with private DP training scenarios observing an 11.90–15.48% increase in F1-score, while non-private training scenarios still see a 0.45% boost. These results underline the potential of differentially private synthetic data in optimizing utility–privacy trade-offs, especially with the limited availability of real training samples. Through rigorous quality assessments, we confirm the integrity and plausibility of our synthetic data, which, however, are significantly impacted when increasing privacy requirements.

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Generator Frontend](#generator-frontend)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Background

Smartwatch health data has become an increasingly popular source of information for healthcare research and personalized medicine. However, the use of such data raises concerns about privacy, as the data often contains sensitive information about individuals' health and fitness. In this project, we aim to address these privacy concerns by generating synthetic health data that can be used in research and analysis while protecting the privacy of the original data owners.

Our approach uses a combination of differential privacy and generative adversarial networks (GANs).

## Requirements

The following dataset is required to run the code in this repository:

- [WESAD Dataset (2,1GB)](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download)
- Python 3.8


Download the WESAD dataset [here](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/) and save the WESAD directory inside the [`data directory`](data).

## Installation

To install the required dependencies for GAN training, run the following command:

```bash
pip install -r requirements.txt
```

To install the required dependencies for stress detection training, run the following command:

```bash
pip install -r /stress_slurm/requirements-docker.txt
```

## Usage

The repository consists of multiple notebooks representing the workflow of this work. Every notebook is one step of this workflow starting with the data preprocessing going over to the model training, synthesizing of the new generated dataset, to evaluating it with a newly trained respective stress detection model.

**[01-Data](01-Data.ipynb)**

The data is loaded from the original WESAD dataset preprocessed and saved within a new file under a new named file [wesad_preprocessed_1hz.csv](data/wesad/wesad_preprocessed_1hz.csv). You can skip downloading the 2,1GB WESAD dataset and preprocessing and work with the already preprocessed WESAD dataset. This consists of two numpy arrays [wesad_windows.npy](data/wesad/wesad_windows.npy) and [wesad_labels.npy](data/wesad/wesad_labels.npy).

**[02-cGAN](02-cGAN-Model.ipynb)**

This notebook focuses on training the cGAN model. It loads the preprocessed data from the previous 01-Data notebook and runs the training for the cGAN model.

**[02-TimeGAN](02-TimeGAN-Model.ipynb)**

This notebook focuses on training the TimeGAN model. It loads the preprocessed data from the previous 01-Data notebook and runs the training for the TimeGAN model.


**[02-DGAN](02-DGAN-Model.ipynb)**

This notebook focuses on training the DGAN model. It loads the preprocessed data from the previous 01-Data notebook and runs the training for the DGAN model.


**[03-Generator](03-Generator.ipynb)**

The generator notebook is responsible for synthesizing a new dataset based on the trained GAN model. The generated data is saved separately in the [syn data folder](data/syn).

**[04-Evaluation](04-Evaluation.ipynb)**

In the evaluation notebook, we assess the quality of the synthetically generated dataset using visual and statistical metrics. The usefulness evaluation takes place in the [05-Stress_Detection](05-Stress_Detection.ipynb) notebook.

**[05-Stress_Detection](05-Stress_Detection.ipynb)**

This notebook focuses on training a CNN model to perform stress detection on the synthetic dataset, simulating a real-world use case.

**ATTENTION:** The actual stress detectuib experiments were run on a server using slurm. Therefore the most recent implementations are using the code starting in the [main.py](main.py) file and following the lead of functions from there. The [slurm.job](slurm.job) file shows how to run the program.

## Generator Frontend

We have also developed a frontend for the generator using Streamlit, which provides a user-friendly interface to interact with the trained GAN model. You can specify different parameters, generate synthetic data, and visualize the results.

To run the Streamlit app, navigate to the streamlit_app directory in your terminal, and run the following command:

```bash
streamlit run streamlit_app/About.py
```

This will start the Streamlit server and open the app in your default web browser.

## Deliverables

The research artifacts resulting from this work are available in a condensed format in this repository.

The results regarding synthetic datasets and stress detection are located in the [results folder](results/).

The trained models can be found in the [model directory](models/). These can be used in the Generator frontend to generate new synthetic data.

## Acknowledgement

I would like to extend my sincere thanks to Maximilian Ehrhart and Bernd Resch for sharing their code related to their paper titled ["A Conditional GAN for Generating Time Series Data for Stress Detection in Wearable Physiological Sensor Data"](https://www.mdpi.com/1424-8220/22/16/5969). Their work on implementing the cGAN architecture and their insights on training it have been important to the success of our project.

## License
[MIT License](LICENSE)
