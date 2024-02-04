import numpy as np
import h5py
import pandas as pd
from datasets import Dataset, load_from_disk, concatenate_datasets
import os
import re

from prosit_model import utils
import params.constants as constants


def get_array(tensor, keys):
    utils.check_mandatory_keys(tensor, keys)
    return [tensor[key] for key in keys]


def to_hdf5(dictionary, path):
    dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(path, "w") as f:
        for key, data in dictionary.items():
            if (data.dtype == 'object'):
                f.create_dataset(key, data=data, dtype=dt, compression="gzip")
            else:
                f.create_dataset(
                    key, data=data, dtype=data.dtype, compression="gzip")


def from_hdf5(file_path, model_config, tensorformat='torch'):
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
    ds_split = dataset.train_test_split(
        test_size=1-constants.VAL_SPLIT, shuffle=True)

    if (tensorformat == 'torch'):
        # ALTERNATIVE 1: pytorch tensor representation of dataset
        ds_train = ds_split['train'].with_format(
            type='torch',
            columns=model_config["x"]+model_config["y"],
            # label_cols=model_config["y"],
            # batch_size=constants.TRAIN_BATCH_SIZE,
            # shuffle=True
        )
        # ds_train.features
        # ds_train.format
        ds_val = ds_split['test'].with_format(
            type='torch',
            columns=model_config["x"]+model_config["y"],
        )
        # ds_val.format
    elif (tensorformat == 'tf'):
        # ALTERNATIVE 2: tf tensor representation of dataset
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
    f.close()

    return ds_train, ds_val


def train_val_split(dataset, model_config, tensorformat='torch'):
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    # split training and validation set
    ds_split = dataset.train_test_split(
        test_size=1-constants.VAL_SPLIT, shuffle=True)

    if (tensorformat == 'torch'):
        # ALTERNATIVE 1: pytorch tensor representation of dataset
        ds_train = ds_split['train'].with_format(
            type='torch',
            columns=model_config["x"]+model_config["y"],
        )
        ds_train.features
        # ds_train = ds_train.rename_column(model_config["y"][0], "label")
        ds_train.format
        # ds_train.format['type']
        ds_val = ds_split['test'].with_format(
            type='torch',
            columns=model_config["x"]+model_config["y"],
        )
        ds_val.format
    elif (tensorformat == 'tf'):
        # ALTERNATIVE 2: tf tensor representation of dataset
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


def pdfile_to_arrow(datasetdictfile, data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for i, df_chunk in enumerate(pd.read_hdf(datasetdictfile, iterator=False,
                                             chunksize=constants.CHUNKSIZE)):
        chunkfile = data_path + "/chunk_{}".format(i)
        dset = Dataset.from_pandas(df_chunk)
        dset.save_to_disk(chunkfile)


def to_arrow(dataset, chunk_path):
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)

    chunksize = constants.CHUNKSIZE
    chunk = {}
    keys_list = [i for i in dataset.keys()]
    feature_len = dataset[keys_list[0]].shape[0]
    i = 0
    last_idx = 0
    # iterate by chunk
    while last_idx < feature_len - 1 - chunksize:
        chunkfile = chunk_path + "/chunk_{}".format(i)
        # chunking each feature
        for feature in dataset.keys():
            chunk[feature] = dataset[feature][last_idx: (last_idx+chunksize)]
        print(chunk[feature].shape)
        # store chunk into dataset and store
        dset = Dataset.from_dict(chunk)
        dset.save_to_disk(chunkfile)
        last_idx += chunksize
        i += 1
    if (last_idx < feature_len - 1):
        chunkfile = chunk_path + "/chunk_{}".format(i)
        for feature in dataset.keys():
            chunk[feature] = dataset[feature][last_idx:]
        print(chunk[feature].shape)
        dset = Dataset.from_dict(chunk)
        dset.save_to_disk(chunkfile)


def genDataset(file_path, chunk_path, flag_chunk):
    if flag_chunk is True:
        # BEST METHOD: Read arrow chunk files into dataset
        if not os.path.exists(chunk_path):
            os.makedirs(chunk_path)
            
            # read from hdf5 file
            f = h5py.File(file_path, 'r')
            # Assemble into a dictionary
            dataset = dict()
            for feature in set(list(f.keys())):
                dataset[feature] = np.array(f[feature])
            f.close()
            print("Construct dictrionary DONE")
            # chunking dataset
            to_arrow(dataset, chunk_path)
            print("Construct chunk files DONE")

        dsets = []
        for filename in os.listdir(chunk_path):
            print(filename)
            if (re.search('chunk_', filename) is not None):
                chunkfile = os.path.join(chunk_path, filename)
                print(chunkfile)
                dset = load_from_disk(chunkfile)
                dsets.append(dset)
        dataset = concatenate_datasets(dsets)
    else:
        f = h5py.File(file_path, 'r')

        # Assemble into a dictionary
        dataset = dict()
        for feature in set(list(f.keys())):
            dataset[feature] = np.array(f[feature])
        f.close()
        print("Construct dictrionary DONE")

        # construct hugginface dataset from dictionary
        dataset = Dataset.from_dict(dataset)
    print("Construct Dataset DONE")
    return dataset


def from_arrow(file_path, model_config, n_samples=None):
    # BEST METHOD: Read arrow chunk files into dataset
    dsets = []
    for filename in os.listdir(file_path):
        print(filename)
        if (re.search('chunk_', filename) is not None):
            chunkfile = os.path.join(file_path, filename)
            print(chunkfile)
            dset = load_from_disk(chunkfile)
            dsets.append(dset)
    dataset = concatenate_datasets(dsets)

    # # ALTERNATIVE 1: construct hugginface dataset from SINGLE arrow file
    # dataset = load_from_disk(file_path)
    # # ALTERNATIVE 2: from dictionary in memory
    # dataset = Dataset.from_dict(f)
    print("Construct Dataset DONE")

    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    # split training and validation set
    ds_split = dataset.train_test_split(
        test_size=1-constants.VAL_SPLIT, shuffle=True)

    tf_ds_train = ds_split['train'].to_tf_dataset(
        columns=model_config["x"],
        label_cols=model_config["y"],
        batch_size=constants.TRAIN_BATCH_SIZE,
        shuffle=True
    )
    tf_ds_val = ds_split['test'].to_tf_dataset(
        columns=model_config["x"],
        label_cols=model_config["y"],
        batch_size=constants.PRED_BATCH_SIZE,
        shuffle=True
    )

    return tf_ds_train, tf_ds_val
