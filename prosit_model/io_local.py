import tensorflow as tf
import numpy as np
import h5py
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
import os
import re

from prosit_model import utils
import params.constants as constants


def get_array(tensor, keys):
    utils.check_mandatory_keys(tensor, keys)
    return [tensor[key] for key in keys]


# def to_hdf5(dictionary, path):
#     import h5py
#     with h5py.File(path, "w") as f:
#         for key, data in dictionary.items():
#             # f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
#             f.create_dataset(key, data=data, dtype=data.dtype)
            

def to_hdf5(dictionary, path):
    dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(path, "w") as f:
        for key, data in dictionary.items():
            # f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
            if (data.dtype == 'object'):
                f.create_dataset(key, data=data, dtype=dt, compression="gzip")
            else:
                f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")


# # DEBUG
# def read_hdf5(path, n_samples=None):
#     # Get a list of the keys for the datasets
#     with h5py.File(path, 'r') as f:
#         print(f.keys())
#         dataset_list = list(f.keys())
#         for dset_name in dataset_list:
#             print(dset_name)
#             print(f[dset_name][:6])
#         f.close()
#     return dataset_list
# #
            

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


def from_hdf5(file_path, model_config, n_samples=None):
    # from tensorflow.keras.utils.io_utils import HDF5Matrix
    
    # # DEBUG
    # file_path = '/Users/xuel12/Documents/Projects/seq2spec/CoSpred/data/massiveKBv2synthetic/test.hdf5'
    # model_dir = '/Users/xuel12/Documents/Projects/seq2spec/CoSpred/prosit_model/model_spectra/'
    # model, model_config = model_lib.load(model_dir, args.trained)
    # #
    
    f = h5py.File(file_path, 'r')
    
    # Get a list of the keys for the datasets
    dataset_list_set = set(list(f.keys()))
    target_list_set = set(model_config["x"] + model_config["y"])
    dataset_list = list(target_list_set.intersection(dataset_list_set))

    # Obsolete: Hugginface Dataset can directly read from hdf5 dictionary
    # Assemble into a dictionary
    dataset = dict()
    for feature in dataset_list:
        print(feature)
        # data[feature] = np.array(f[feature])
        dataset[feature] = np.array(f[feature])
    dataset['intensities_raw'] = dataset['intensities_raw'].astype(np.float32)
    print("Construct dictrionary DONE")
    
    # construct hugginface dataset from dictionary
    # dataset = Dataset.from_dict(f)
    dataset = Dataset.from_dict(dataset)
    print("Construct Dataset DONE")
    
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    # split training and validation set
    ds_split = dataset.train_test_split(test_size=1-constants.VAL_SPLIT, shuffle=True)

    # ds = dataset.with_format("tf")       # tf tensor representation of dataset
    tf_ds_train = ds_split['train'].to_tf_dataset(
                # columns=model_config["x"],
                # label_cols=model_config["y"],
                columns=model_config["x"],
                label_cols=model_config["y"],
                batch_size=constants.TRAIN_BATCH_SIZE,
                shuffle=True
                )
    tf_ds_val = ds_split['test'].to_tf_dataset(
                # columns=model_config["x"],
                # label_cols=model_config["y"],
                columns=model_config["x"],
                label_cols=model_config["y"],
                batch_size=constants.PRED_BATCH_SIZE,
                shuffle=True
                )
    f.close()

    return tf_ds_train, tf_ds_val


def pdfile_to_arrow(datasetdictfile, data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    dsets = []
    for i, df_chunk in enumerate(pd.read_hdf(datasetdictfile, iterator=False, 
                                             chunksize=constants.CHUNKSIZE)):
        chunkfile = data_path + "/chunk_{}".format(i)
        dset = Dataset.from_pandas(df_chunk)
        dset.save_to_disk(chunkfile)
        
    # for feature in dictionary.keys():
    #     print(dictionary[feature].shape)
        # dictionary[feature] = np.array(dictionary[feature])
    # dictionary['intensities_raw'] = dictionary['intensities_raw'].astype(np.float32)
    # dictionary['masses_pred'] = dictionary['masses_pred'].astype(np.float32)
    # dictionary.pop('masses_pred')
    # print(dictionary.keys())

    # dataset = Dataset.from_dict(dictionary)
    # dataset.save_to_disk(path)
    
    # # DEBUG
    # df = load_from_disk(constants_local.DATA_DIR + 'prosit_dataset.arrow')
    # sys.getsizeof(df)      # check the size of dataset in memory
    # #
    

def to_arrow(dataset, data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    chunksize = constants.CHUNKSIZE
    chunk = {}
    keys_list = [i for i in dataset.keys()]
    feature_len = dataset[keys_list[0]].shape[0]
    i = 0
    last_idx = 0
    # iterate by chunk
    while last_idx < feature_len - 1 - chunksize:
        chunkfile = data_path + "/chunk_{}".format(i)
        # chunking each feature
        for feature in dataset.keys():
            # dictionary['intensities_raw'] = dictionary['intensities_raw'].astype(np.float32)
            # dictionary['masses_pred'] = dictionary['masses_pred'].astype(np.float32)
            chunk[feature] = dataset[feature][last_idx : (last_idx+chunksize) ]
        print(chunk[feature].shape)
        # store chunk into dataset and store
        dset = Dataset.from_dict(chunk)
        dset.save_to_disk(chunkfile)
        last_idx += chunksize
        i += 1
    if (last_idx < feature_len - 1):
        chunkfile = data_path + "/chunk_{}".format(i)
        for feature in dataset.keys():
            chunk[feature] = dataset[feature][last_idx:]
        print(chunk[feature].shape)
        dset = Dataset.from_dict(chunk)
        dset.save_to_disk(chunkfile)
                
        
def from_arrow(file_path, model_config, n_samples=None):
    # BEST METHOD: Read arrow chunk files into dataset
    dsets = []    
    for filename in os.listdir(file_path):
        print(filename)
        if (re.search('chunk_',filename) is not None):
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
    ds_split = dataset.train_test_split(test_size=1-constants.VAL_SPLIT, shuffle=True)

    # ds = dataset.with_format("tf")       # tf tensor representation of dataset
    tf_ds_train = ds_split['train'].to_tf_dataset(
                # columns=model_config["x"],
                # label_cols=model_config["y"],
                columns=model_config["x"],
                label_cols=model_config["y"],
                batch_size=constants.TRAIN_BATCH_SIZE,
                shuffle=True
                )
    tf_ds_val = ds_split['test'].to_tf_dataset(
                # columns=model_config["x"],
                # label_cols=model_config["y"],
                columns=model_config["x"],
                label_cols=model_config["y"],
                batch_size=constants.PRED_BATCH_SIZE,
                shuffle=True
                )

    return tf_ds_train, tf_ds_val


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