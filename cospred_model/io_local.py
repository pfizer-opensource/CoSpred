import tensorflow as tf
import numpy as np
import h5py
from datasets import Dataset

from prosit_model import utils
import params.constants as constants

def get_array(tensor, keys):
    utils.check_mandatory_keys(tensor, keys)
    return [tensor[key] for key in keys]


def to_hdf5(dictionary, path):
    import h5py

    with h5py.File(path, "w") as f:
        for key, data in dictionary.items():
            # f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
            f.create_dataset(key, data=data, dtype=data.dtype)
            

def from_hdf5_to_transformer(file_path, model_config, tensorformat='torch'):
    # from tensorflow.keras.utils.io_utils import HDF5Matrix    
    f = h5py.File(file_path, 'r')
    
    # Get a list of the keys for the datasets
    dataset_list_set = set(list(f.keys()))
    target_list_set = set(model_config["x"] + model_config["y"])
    dataset_list = list(target_list_set.intersection(dataset_list_set))

    # Assemble into a dictionary
    dataset = dict()
    for feature in dataset_list:
        dataset[feature] = np.array(f[feature])
    f.close()
    print("Construct dictrionary DONE")
    
    # construct hugginface dataset from dictionary
    dataset = Dataset.from_dict(dataset)
    print("Construct Dataset DONE")

    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    # split training and validation set
    ds_split = dataset.train_test_split(test_size=1-constants.VAL_SPLIT, shuffle=True)

    if (tensorformat == 'torch'):
        # ALTERNATIVE 1: pytorch tensor representation of dataset
        ds_train = ds_split['train'].with_format(
                    type='torch', 
                    columns=model_config["x"]+model_config["y"],
                    # label_cols=model_config["y"],
                    # batch_size=constants.TRAIN_BATCH_SIZE,
                    # shuffle=True
                    )
        ds_train.features
        # ds_train = ds_train.rename_column(model_config["y"][0], "label")
        ds_train.format
        # ds_train.format['type']
        ds_val = ds_split['test'].with_format(
                    type='torch', 
                    columns=model_config["x"]+model_config["y"],
                    # label_cols=model_config["y"],
                    # batch_size=constants.PRED_BATCH_SIZE,
                    # shuffle=True
                    )
        ds_val.format
    elif (tensorformat == 'tf'):
        # ALTERNATIVE 2: tf tensor representation of dataset
        # ds = dataset.with_format("tf")       
        ds_train = ds_split['train'].to_tf_dataset(
                    columns=model_config["x"],
                    label_cols=model_config["y"],
                    batch_size=constants.TRAIN_BATCH_SIZE,
                    shuffle=True
                    )
        ds_val = ds_split['test'].to_tf_dataset(
                    columns=model_config["x"],
                    label_cols=model_config["y"],
                    batch_size=constants.PRED_BATCH_SIZE,
                    shuffle=True
                    )
    
    return ds_train, ds_val



# def subset_hdf5(file_path):
#     dataset_names_x = ["sequence_integer", "precursor_charge_onehot", 'collision_energy_aligned_normed']
#     dataset_names_y = ["intensities_raw"]
#     f = h5py.File(file_path, 'r')
#     # dataset_list = list(f.keys())
#     data = dict()
#     for dataset in dataset_names_x:
#         data[dataset] = f[dataset]
#     to_hdf5(data, '../data/traintest_hcd_100_3col.hdf5')   
#     data = dict()
#     for dataset in dataset_names_y:
#         data[dataset] = f[dataset]
#     to_hdf5(data, '../data/traintest_hcd_100_y.hdf5')   
#     f.close()