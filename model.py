import os
import json
import re

import params.constants as constants
from prosit_model import layers, utils
from cospred_model.model.transformerEncoder import TransformerConfig, TransformerEncoder

MODEL_NAME = "model.json"
CONFIG_NAME = "config.json"


def is_weight_name(w, flag_prosit, flag_fullspectrum):
    if flag_prosit is True:
        if flag_fullspectrum is True:
            return w.startswith("prosit_full_") and w.endswith(".hdf5")
        else:
            return w.startswith("prosit_byion_") and w.endswith(".hdf5")
    else:
        if flag_fullspectrum is True:
            return w.startswith("transformer_full_") and w.endswith(".pt")
        else:
            return w.startswith("transformer_byion_") and w.endswith(".pt")


def get_loss(x, flag_prosit):
    if flag_prosit is True:
        return float(re.sub('[a-zA-Z]+', '', x.split("_")[-1][:-len('.hdf5')]))
    else:
        return float(re.sub('[a-zA-Z]+', '', x.split("_")[-1][:-len('.pt')]))


def get_best_weights_path(model_dir, flag_prosit, flag_fullspectrum):
    weights = list(filter(lambda x: is_weight_name(x, flag_prosit, flag_fullspectrum),
                          os.listdir(model_dir)))
    if len(weights) == 0:
        print("No existing weight files founded.")
        return None
    else:
        d = {get_loss(w, flag_prosit): w for w in weights}
        weights_path = os.path.join(model_dir, d[min(d)])
        # weights_path = "{}/{}".format(model_dir, d[min(d)])
        return weights_path


def load(model_dir, flag_fullspectrum, flag_prosit, trained=False):

    config_path = os.path.join(model_dir, CONFIG_NAME)
    with open(config_path, "r") as f:
        config = json.load(f)

    weights_path = get_best_weights_path(
        model_dir, flag_prosit, flag_fullspectrum)

    if flag_prosit is True:
        import tensorflow as tf
        model_path = os.path.join(model_dir, MODEL_NAME)
        # load model
        with open(model_path, "r") as f:
            model = tf.keras.models.model_from_json(
                f.read(), custom_objects={"CustomAttention": layers.CustomAttention}
            )
        # load weight
        if trained and (weights_path is not None):
            print('Loading weight from: {}'.format(weights_path))
            model.load_weights(weights_path)
    else:
        import torch
        # load model
        if flag_fullspectrum is True:
            # OPTIONA 1: full spectrum model
            mconf = TransformerConfig(vocab_size=constants.MAX_ALPHABETSIZE, block_size=37,
                                      embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
                                      n_layer=8, n_head=16, n_embd=256,
                                      n_output=constants.SPECTRA_DIMENSION,
                                      max_charge=10, max_ce=100)
        else:
            # OPTION 2: b,y ion model
            mconf = TransformerConfig(vocab_size=constants.MAX_ALPHABETSIZE, block_size=37,
                                      embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
                                      n_layer=8, n_head=16, n_embd=256,
                                      n_output=174,
                                      max_charge=10, max_ce=100)
        model = TransformerEncoder(mconf)

        # take over whatever gpus are on the system
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load weight
        if trained and (weights_path is not None):
            print('Loading weight from: {}'.format(weights_path))
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.eval()
    return model, config, weights_path


def save(model, config, model_dir):
    model_path = MODEL_NAME.format(model_dir)
    config_path = CONFIG_NAME.format(model_dir)
    utils.check_mandatory_keys(config, ["name", "optimizer", "loss", "x", "y"])
    with open(config_path, "w") as f:
        json.dump(config, f, indent=3)
    with open(model_path, "w") as f:
        json.dump(json.loads(model.to_json()), f, indent=3)
