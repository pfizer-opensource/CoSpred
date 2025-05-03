import os
import yaml
import json

from prosit_model import layers, utils
import tensorflow as tf


# MODEL_NAME = "model.yml"
# CONFIG_NAME = "config.yml"
MODEL_NAME = "model.json"
CONFIG_NAME = "config.json"

def is_weight_name(w):
    return w.startswith("weight_") and w.endswith(".hdf5")


def get_loss(x):
    return float(x.split("_")[-1][:-5])


def get_best_weights_path(model_dir):
    weights = list(filter(is_weight_name, os.listdir(model_dir)))
    if len(weights) == 0:
        return None
    else:
        d = {get_loss(w): w for w in weights}
        weights_path = "{}/{}".format(model_dir, d[min(d)])
        return weights_path


def load(model_dir, trained=False):

    model_path = os.path.join(model_dir, MODEL_NAME)
    config_path = os.path.join(model_dir, CONFIG_NAME)
    weights_path = get_best_weights_path(model_dir)
    with open(config_path, "r") as f:
        config = json.load(f)
        # config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    with open(model_path, "r") as f:
        model = tf.keras.models.model_from_json(
            f.read(), custom_objects={"CustomAttention": layers.CustomAttention}
        )
    if trained and (weights_path is not None):
        model.load_weights(weights_path)
    return model, config


def save(model, config, model_dir):
    model_path = MODEL_NAME.format(model_dir)
    config_path = CONFIG_NAME.format(model_dir)
    utils.check_mandatory_keys(config, ["name", "optimizer", "loss", "x", "y"])       
    with open(config_path, "w") as f:
        json.dump(config, f, indent=3) 
    with open(model_path, "w") as f:
        json.dump(json.loads(model.to_json()), f, indent=3)

