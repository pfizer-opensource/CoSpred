import logging
import numpy
import json
import os
import warnings
import tensorflow as tf
from argparse import ArgumentParser

import params.constants_location as constants_location
import params.constants as constants
import prosit_model.layers as layers
import prosit_model.utils as utils
from prosit_model.attention import CustomAttention


def is_weight_name(w):
    return w.startswith("weight_") and w.endswith(".hdf5")


def get_loss(x):
    return float(x.split("_")[-1][:-5])


def get_best_weights_path(model_dir):
    weights = list(filter(is_weight_name, os.listdir(model_dir)))
    if len(weights) == 0:
        logging.info(f"[STATUS] No weights was found in {model_dir}.")
        return None
    else:
        d = {get_loss(w): w for w in weights}
        weights_path = "{}/{}".format(model_dir, d[min(d)])
        logging.info(f"[STATUS] Best weights was loaded as {weights_path}")
        return weights_path


def load(model_dir, model_path, config_path, trained=False):

    # model_path = os.path.join(model_dir, model_name)
    # config_path = os.path.join(model_dir, config_name)
    weights_path = get_best_weights_path(model_dir)
    with open(config_path, "r") as f:
        config = json.load(f)
        # config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    with open(model_path, "r") as f:
        model = tf.keras.models.model_from_json(
            f.read(), custom_objects={"CustomAttention": layers.CustomAttention}
        )
        # model = keras.models.model_from_yaml(
        #     f.read(), custom_objects={"CustomAttention": layers.CustomAttention}
        # )
    if trained and weights_path is not None:
        model.load_weights(weights_path)
    return model, config


def save(model, config, model_path, config_path, model_dir):
    # config_name = config_name_json
    # model_name = model_name_json
    # model_dir = dir_out
    # model_path = os.path.join(model_dir, model_name)
    # config_path = os.path.join(model_dir, config_name)
    utils.check_mandatory_keys(config, ["name", "optimizer", "loss", "x", "y"])
    with open(config_path, "w") as f:
        json.dump(config, f, indent=3)
    with open(model_path, "w") as f:
        json.dump(json.loads(model.to_json()), f, indent=3)


def model_build_biGRU():
    # from tf.keras import Model, Input
    # from tf.keras.layers import LeakyReLU, Flatten, Dense, Dropout
    # from tf.keras.layers import Concatenate, Embedding, GRU, Bidirectional
    # from tf.keras.layers import RepeatVector, TimeDistributed, Multiply, Permute
    from keras import Model, Input
    from keras.layers import LeakyReLU, Flatten, Dense, Dropout
    from keras.layers import Concatenate, Embedding, GRU, Bidirectional
    from keras.layers import RepeatVector, TimeDistributed, Multiply, Permute
    
    # fix random seed for reproducibility
    seed = 100
    numpy.random.seed(seed)

    peplen = 30
    # max_features = 22
    max_features = len(constants.ALPHABET)+10

    # this embedding layer will encode the input sequence into a sequence of dense 32-dimensional vectors.
    peptides_in = Input(shape=(peplen,), dtype='int32',
                        name='sequence_integer')
    embedding = Embedding(max_features, 32, name='embedding')(peptides_in)
    encoder1 = Bidirectional(
        GRU(512, return_sequences=True, name='encoder1_gru'), name='encoder1')(embedding)
    dropout_1 = Dropout(0.3, name='dropout_1')(encoder1)
    encoder2 = GRU(512, return_sequences=True, name='encoder2')(dropout_1)
    dropout_2 = Dropout(0.3, name='dropout_2')(encoder2)
    encoder_att = CustomAttention(name='encoder_att')(dropout_2)

    collision_energy_in = Input(
        shape=(1,), dtype='float32', name='collision_energy_aligned_normed')
    precursor_charge_in = Input(
        shape=(6,), dtype='float32', name='precursor_charge_onehot')
    meta_in = Concatenate(
        axis=-1, name='meta_in')([collision_energy_in, precursor_charge_in])
    meta_dense = Dense(512, name='meta_dense')(meta_in)
    meta_dense_do = Dropout(0.3, name='meta_dense_do')(meta_dense)

    # combine seq, charge, ce embedding
    add_meta = Multiply(name='add_meta')([encoder_att, meta_dense_do])
    repeat = RepeatVector(100, name='repeat')(add_meta)
    decoder = GRU(512, return_sequences=True, name='decoder')(repeat)
    dropout_3 = Dropout(0.3, name='dropout_3')(decoder)

    permute_1 = Permute((2, 1), name='permute_1')(dropout_3)
    dense_1 = Dense(100, activation='softmax', name='dense_1')(permute_1)
    permute_2 = Permute((2, 1), name='permute_2')(dense_1)

    multiply_1 = Multiply(name='multiply_1')([dropout_3, permute_2])
    timedense = TimeDistributed(
        Dense(30, name='dense_2'), name='timedense')(multiply_1)
    activation = LeakyReLU(alpha=0.3, name='activation')(
        timedense)  # names are added here
    out = Flatten(name='intensities_raw')(activation)

    model = Model(inputs=[peptides_in, precursor_charge_in,
                  collision_energy_in], outputs=[out], name='model_1')
    logging.info(model.summary())

    return model


def main():
    # Suppress warning message of tensorflow compatibility
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")

    # Configure logging
    log_file_prep = os.path.join(constants_location.PREDICT_DIR, "cospred_prep.log")
    logging.basicConfig(
        filename=log_file_prep,
        filemode="w",  # Overwrite the log file each time the script runs
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO  # Set the logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    )
    # Optionally, log to both file and console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    
    parser = ArgumentParser()
    parser.parse_args()

    model_dir = constants_location.MODEL_DIR
    model_name = "model.json"
    config_name = "config.json"
    model_path = os.path.join(model_dir, model_name)
    config_path = os.path.join(model_dir, config_name)

    # load the model template
    model, config = load(model_dir, model_path, config_path, trained=False)
    logging.info("Model was loaded successfully.")

    # construct the model
    model = model_build_biGRU()

    # save the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    save(model, config, model_path, config_path, model_dir)
    logging.info(f"[STATUS] New model was saved as {model_path} and configured as {config_path}.")


if __name__ == "__main__":
    main()
