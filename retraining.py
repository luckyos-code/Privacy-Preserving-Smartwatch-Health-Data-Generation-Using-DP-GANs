import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

import wandb
from wandb.keras import WandbCallback


from synthesizers.cgan.model import (
    ConditionalGAN, 
    GANMonitor
)
from synthesizers.preprocessing.wesad import (
    WESADDataset, 
    LabelType
)
from synthesizers.utils.training import data_split



def main():

    parser = ArgumentParser(prog="Augmented Private Stress Detection using GANs and DP")
    parser.add_argument(
        "-n",
        "--noise_mult",
        type=int,
        help=
        "noise_mult one of [None, 10, 1, 0.1].",
        metavar="N",
    )
    args = parser.parse_args()
    
    noise_mult = args.noise_mult

    SAMPLING_RATE = 1
    USE_SLIDING_WINDOWS = True

    # Training Hyperparameters
    DP_TRAINING = True
    NUM_FEATURES = 6
    SEQ_LENGTH = 60
    LATENT_DIM = SEQ_LENGTH
    BATCH_SIZE = 8
    HIDDEN_UNITS = 64
    EPOCHS = 420
    ACTIVATION = "relu"
    RANDOM_SEED = 42
    LEARNING_RATE = 0.0002
    LOSS_FN = "binary_cross_entropy"
    D_ARCHITECTURE = "lstm"
    LOSO_TRAINING_WITHOUT_SUBJECT = False # "14"

    # DP Training Hyperparameter
    L2_NORM_CLIP = 1.0
    NUM_MICROBATCHES = BATCH_SIZE
    DP_LEARNING_RATE = 1e-3
    DELTA = 1e-4


    # Define run config
    config = {
        "activation_function": ACTIVATION,
        "hidden_units": HIDDEN_UNITS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "num_features": NUM_FEATURES,
        "seq_length": SEQ_LENGTH,
        "dp_training": DP_TRAINING,
        "learning_rate": LEARNING_RATE,
        "loss_function": LOSS_FN,
        "d_architecture": D_ARCHITECTURE,
        "use_sliding_windows": USE_SLIDING_WINDOWS
    }

    if LOSO_TRAINING_WITHOUT_SUBJECT:
        config["WESAD_WITHOUT_SUBJ"] = LOSO_TRAINING_WITHOUT_SUBJECT

    if DP_TRAINING:
        config["l2_norm_clip"] = L2_NORM_CLIP
        config["num_microbatches"] = NUM_MICROBATCHES
        config["dp_learning_rate"] = DP_LEARNING_RATE

    windows = np.load('data/wesad/wesad_windows.npy')
    labels = np.load('data/wesad/wesad_labels.npy')

    if USE_SLIDING_WINDOWS:
        mos = windows[labels == 1]
        non_mos = windows[labels == 0]
    else:
        mos = windows[labels == 1]
        non_mos = windows[labels == 0]

    windows = np.delete(windows, 6, axis=2)
    mos = np.delete(mos, 6, axis=2)
    non_mos = np.delete(non_mos, 6, axis=2)

    num_split = 0.8
    trainmos, testmos = data_split(mos, num_split)
    trainnomos, testnomos = data_split(non_mos, num_split)

    print(trainmos.shape)
    print(testmos.shape)
    print(trainnomos.shape)
    print(testnomos.shape)

    # get needed noise for target epsilon
    min_noise = 1e-5
    target_epsilons = [0.1, 1, 10]
    noise_multipliers = {target_epsilon : compute_noise(
        windows.shape[0] // 2,
        BATCH_SIZE,
        target_epsilon,
        EPOCHS * 2,
        DELTA,
        min_noise
    ) for target_epsilon in target_epsilons}
    print(noise_multipliers)

    compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=windows.shape[0] // 2,
                                                  batch_size=BATCH_SIZE,
                                                                  noise_multiplier=noise_multipliers[target_epsilons[0]],
                                                  epochs=EPOCHS*2,
                                                  delta=DELTA)

    # Load dataset into tf dataset
    dataset = tf.data.Dataset.from_tensor_slices((windows, labels))

    # Shuffle, cache, and batch the dataset
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    tf.random.set_seed(RANDOM_SEED)
    randomTrainMos = tf.random.normal(shape=(trainmos.shape[0], LATENT_DIM))

    tf.random.set_seed(RANDOM_SEED)
    randomTrainNoMos = tf.random.normal(shape=(trainnomos.shape[0], LATENT_DIM))

    tf.random.set_seed(RANDOM_SEED)
    randomTestMos = tf.random.normal(shape=(testmos.shape[0], LATENT_DIM))

    tf.random.set_seed(RANDOM_SEED)
    randomTestNoMos = tf.random.normal(shape=(testnomos.shape[0], LATENT_DIM))

    cond_gan = ConditionalGAN(
        num_features=NUM_FEATURES,
        seq_length=SEQ_LENGTH,
        latent_dim=LATENT_DIM,
        discriminator=ConditionalGAN.conditional_discriminator(
            hidden_units=SEQ_LENGTH, 
            seq_length=SEQ_LENGTH, 
            num_features=NUM_FEATURES,
            filters=[32, 64, 32],
            activation_function= ACTIVATION,
            architecture=D_ARCHITECTURE, 
            #head_size=wandb.config.head_size#wandb.config.d_architecture
            #filters=[wandb.config.filter1, wandb.config.filter2, wandb.config.filter3],
            #kernel_sizes=[wandb.config.kernel_size1, wandb.config.kernel_size2, wandb.config.kernel_size3]
            ),
        generator=ConditionalGAN.conditional_generator(
            hidden_units=SEQ_LENGTH, 
            seq_length=SEQ_LENGTH, 
            latent_dim=LATENT_DIM,
            num_features=NUM_FEATURES,
            activation_function=ACTIVATION
        )
    )
    if DP_TRAINING:
        config["noise_multiplier"] = noise_multipliers[noise_mult]

        d_optimizer = dp_optimizer_vectorized.VectorizedDPAdamOptimizer( #vectorized adam am schnellsten
            l2_norm_clip=L2_NORM_CLIP,
            noise_multiplier=noise_multipliers[noise_mult],
            num_microbatches=NUM_MICROBATCHES,
            learning_rate=DP_LEARNING_RATE
        )
    else:
        d_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.5) # get_optimizer(0.0002, wandb.config.optimizer)#

    g_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.5) # get_optimizer(0.0002, wandb.config.optimizer)#

    cond_gan.compile(
        d_optimizer= d_optimizer, # Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer= g_optimizer, # Adam(learning_rate=0.0002, beta_1=0.5), #optimizer
        loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )

    print(f"{cond_gan.d_optimizer} is used")

    history = cond_gan.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[
            # GANMonitor(
            #     trainmos,
            #     trainnomos,
            #     testmos,
            #     testnomos,
            #     randomTrainMos,
            #     randomTrainNoMos,
            #     randomTestMos,
            #     randomTestNoMos,
            #     num_seq=50,
            #     save_path=None,
            #     batch_size=BATCH_SIZE,
            #     seq_length=SEQ_LENGTH,
            #     num_features=NUM_FEATURES,
            #     dp=DP_TRAINING,
            # )
        ],
    )

    if DP_TRAINING:
        base_path = f"models/new_cgan/{noise_mult}/"
        cond_gan.generator.save(f"{base_path}generator")
        cond_gan.discriminator.save(f"{base_path}discriminator")
    elif LOSO_TRAINING_WITHOUT_SUBJECT:
        base_path = f"models/no_dp/loso/sub{LOSO_TRAINING_WITHOUT_SUBJECT}/{wandb.run.name}/"
        cond_gan.generator.save(f"{base_path}cgan_generator")
        cond_gan.discriminator.save(f"{base_path}cgan_discriminator")
    else:
        base_path = f"models/no_dp/{wandb.run.name}/"
        cond_gan.generator.save(f"{base_path}cgan_generator")
        cond_gan.discriminator.save(f"{base_path}cgan_discriminator")


if __name__ == "__main__":
    main()