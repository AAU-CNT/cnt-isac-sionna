import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_CNN_encoder(input_shape=(15, 200, 2)):

    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        64,
        (6, 1),
        strides=(2, 1),
        padding="same",
        activation="relu"
    )(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 1), padding="same")(x)
    x = layers.Conv2D(
        32,
        (10, 2),
        strides=(1, 1),
        padding="same",
        activation="relu"
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), padding="same")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(16, activation="relu")(x)

    return keras.Model([inputs], [x], name="standard_cnn_encoder")


def _cnn_subset_branch(x, name_prefix, out_dim=None):
    x = layers.Conv2D(
        64, (6, 1), strides=(2, 1), padding="same",
        activation="relu", name=f"{name_prefix}_conv1",
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), padding="same", name=f"{name_prefix}_pool1")(x)
    x = layers.Conv2D(
        32, (10, 2), strides=(1, 1), padding="same",
        activation="relu", name=f"{name_prefix}_conv2",
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), padding="same", name=f"{name_prefix}_pool2")(x)
    x = layers.Dropout(0.25, name=f"{name_prefix}_drop1")(x)
    x = layers.Flatten(name=f"{name_prefix}_flatten")(x)
    x = layers.Dense(256, activation="relu", name=f"{name_prefix}_dense1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Dropout(0.15, name=f"{name_prefix}_drop2")(x)
    x = layers.Dense(32, activation="relu", name=f"{name_prefix}_dense2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Dropout(0.15, name=f"{name_prefix}_drop3")(x)
    x = layers.Dense(16, activation="relu", name=f"{name_prefix}_dense3")(x)
    if out_dim is not None:
        x = layers.Dense(out_dim, activation=None, name=f"{name_prefix}_out")(x)
    return x


def build_parallel_CNN_encoder(input_shape=(15, 200, 2), out_dim=None, average=True):
    num_anchors = input_shape[0]
    if num_anchors < 3:
        raise ValueError(
            f"build_parallel_CNN_encoder needs at least 3 anchors, got {num_anchors}"
        )

    inputs = keras.Input(shape=input_shape)

    R = 3
    branch_outputs = []
    for i in range(num_anchors):
        # Consecutive anchors [i, i+1, i+2] with wrap-around. gather (not crop)
        # so the wrapping subsets, e.g. [N-1, 0, 1], are handled too.
        idx = [(i + j) % num_anchors for j in range(R)]
        subset = layers.Lambda(
            lambda t, idx=idx: tf.gather(t, idx, axis=1),
            output_shape=(R,) + tuple(input_shape[1:]),
            name=f"anchor_subset_{i}",
        )(inputs)
        branch_outputs.append(
            _cnn_subset_branch(subset, name_prefix=f"branch_{i}", out_dim=out_dim)
        )

    feat_dim = out_dim if out_dim is not None else 16
    stacked = [
        layers.Reshape((1, feat_dim), name=f"branch_{i}_expand")(b)
        for i, b in enumerate(branch_outputs)
    ]
    x = stacked[0] if len(stacked) == 1 else layers.Concatenate(
        axis=1, name="branch_stack"
    )(stacked)

    return keras.Model([inputs], [x], name="parallel_cnn_encoder")
