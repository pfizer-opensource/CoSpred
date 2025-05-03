import sys
import numpy as np
import h5py
import pandas as pd
from datasets import Dataset, load_from_disk, concatenate_datasets
import os
import shutil
import re

from prosit_model import utils, sanitize, tensorize
import params.constants as constants
from params.constants import (
    ALPHABET,
    AMINO_ACID,
    CHARGES,
    MAX_SEQUENCE,
)


def get_numbers(vals, dtype=np.float16):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_precursor_charge_onehot(charges):
    array = np.zeros([len(charges), max(CHARGES)])
    for i, precursor_charge in enumerate(charges):
        if precursor_charge > max(CHARGES):
            pass
        else:
            array[i, int(precursor_charge) - 1] = 1
    return array


def get_sequence_integer(sequences, dtype='i1'):
    array = np.zeros([len(sequences), MAX_SEQUENCE])
    for i, sequence in enumerate(sequences):
        try:
            if len(sequence) > MAX_SEQUENCE:
                pass
            else:
                for j, s in enumerate(utils.peptide_parser(sequence)):
                    # # POC: uppercase all amino acid, so no PTM
                    # array[i, j] = ALPHABET[s.upper()]
                    # #
                    array[i, j] = ALPHABET[s]
        except:
            next
    return array


def get_array(tensor, keys):
    utils.check_mandatory_keys(tensor, keys)
    return [tensor[key] for key in keys]


def fixEncoding(df, column_name):
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' is missing from the DataFrame.")
    if isinstance(df[column_name].iloc[0], bytes):
        # print(f"Decoding '{column_name}' from bytes to string...")
        df[column_name] = df[column_name].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df


def read_hdf5_to_dict(file_path):
    """
    Read an HDF5 file and convert its datasets into a Python dictionary.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary containing the datasets from the HDF5 file.
    """
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data = f[key][:]
            data_dict[key] = np.array(data)  # Convert to NumPy array
            # print(f"Key: {key}, Data Type: {type(data_dict[key])}, Shape: {data_dict[key].shape}")
    return data_dict


def read_hdf5(path, n_samples=None):
    # Get a list of the keys for the datasets
    with h5py.File(path, 'r') as f:
        # print(f.keys())
        dataset_list = list(f.keys())
        for dset_name in dataset_list:
            print(dset_name)
            # print(f[dset_name][:3])
        f.close()
    return dataset_list


def to_hdf5(dictionary, path):
    dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(path, "w") as f:
        for key, data in dictionary.items():
            if data.dtype.kind == 'O':  # Object type
                data = np.array(data, dtype='S')  # Convert to fixed-length strings
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

    # construct hugginface dataset from dictionary
    dataset = Dataset.from_dict(dataset)

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


def dict_to_arrow_chunks(data_dict, chunk_path):
    """
    Convert a dictionary to Arrow chunks and save them to disk.

    Args:
        data_dict (dict): The dictionary to convert.
        chunk_path (str): The directory to save the Arrow chunks.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(chunk_path, exist_ok=True)
    chunksize = constants.CHUNKSIZE

    # Convert the dictionary to a Hugging Face Dataset
    dataset = Dataset.from_dict(data_dict)

    # Split the dataset into chunks
    for i in range(0, len(dataset), chunksize):
        chunk = dataset.select(range(i, min(i + chunksize, len(dataset))))
        chunk_file = os.path.join(chunk_path, f"chunk_{i // chunksize}")
        chunk.save_to_disk(chunk_file)
        # print(f"Saved chunk {i // chunksize} to {chunk_file}")


def genDataset(file_path, chunk_path, flag_chunk):
    if flag_chunk is True:
        # BEST METHOD: Read arrow chunk files into dataset
        if not os.path.exists(chunk_path):
            os.makedirs(chunk_path)
        else:
            try:
                shutil.rmtree(chunk_path)
                # print(f"Folder '{chunk_path}' and its contents deleted successfully.")
            except OSError as e:
                print(f"Error deleting folder '{chunk_path}': {e}")
       
        # read from hdf5 file
        f = h5py.File(file_path, 'r')
        # Assemble into a dictionary
        dataset = dict()
        # for feature in set(list(f.keys())):
        #     dataset[feature] = np.array(f[feature])
        for feature in set(f.keys()):
            dataset[feature] = f[feature]
        # chunking dataset
        # to_arrow(dataset, chunk_path)
        dict_to_arrow_chunks(dataset, chunk_path)
        f.close()

        dsets = []
        for filename in os.listdir(chunk_path):
            if (re.search('chunk_', filename) is not None):
                chunkfile = os.path.join(chunk_path, filename)
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

        # construct hugginface dataset from dictionary
        dataset = Dataset.from_dict(dataset)
    return dataset


def arrow_chunk_to_hdf5(chunkfile, hdf5_file):
    """
    Convert an Arrow chunk file to HDF5 format.

    Args:
        chunkfile (str): Path to the Arrow chunk file.
        hdf5_file (str): Path to save the HDF5 file.

    Returns:
        None
    """
    # Step 1: Load the Arrow chunk file
    dset = load_from_disk(chunkfile)

    # Step 2: Convert the Arrow dataset to a dictionary
    chunk_dict = {column: dset[column] for column in dset.column_names}

    # Step 3: Save the dictionary to HDF5
    with h5py.File(hdf5_file, "w") as h5f:
        for key, value in chunk_dict.items():
            # Convert lists to NumPy arrays if necessary, fix the 1-dimension as the length of the list
            if isinstance(value, list):
                value = np.array(value)
                # Fix the first dimension as the length of the list
                value = value.reshape(len(value), -1) if value.ndim == 1 else value
            # Save the dataset to HDF5
            h5f.create_dataset(key, data=value)


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
    print("Construct Dataset ... DONE")

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


def sanitizePeptide(peptide_df, predict_csv):
    assert "modified_sequence" in peptide_df.columns
    assert "collision_energy" in peptide_df.columns
    assert "precursor_charge" in peptide_df.columns

    peptide_df.dropna(subset=['modified_sequence', 'collision_energy', 'precursor_charge'], inplace=True)
    peptide_df.columns = peptide_df.columns.str.replace('[\r]', '')

    # get overlap of AMINO_ACID and ALPHABET
    overlap_keys = set(AMINO_ACID.keys()).intersection(ALPHABET.keys())
    # print("overlap amino acids: ", overlap_keys)

    # remove the rows when modified_sequence has letter not in AMINO_ACID and ALPHABET
    peptide_df = peptide_df[peptide_df['modified_sequence'].str.contains(
        '[^' + ''.join(overlap_keys) + ']', na=False) == False]
    
    # Ensure 'modified_sequence' is decoded if it's in bytes
    if isinstance(peptide_df['modified_sequence'].iloc[0], bytes):
        peptide_df['modified_sequence'] = peptide_df['modified_sequence'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    # write a csv file
    peptide_df.to_csv(predict_csv, index=False)

    return peptide_df


def constructDataset_frompep(csvfile, predict_csv):
    df = pd.read_csv(csvfile, sep=',', index_col=False)

    df = sanitizePeptide(df, predict_csv)

    dataset = {
        "collision_energy": df['collision_energy'].astype(np.uint8),
        "collision_energy_aligned_normed": get_numbers(df['collision_energy']/100.0, dtype=np.float16),
        "precursor_charge":df['precursor_charge'].astype(np.uint8),
        "precursor_charge_onehot": get_precursor_charge_onehot(df['precursor_charge']).astype(np.uint8),
        "modified_sequence": df['modified_sequence'].astype(str),
        "sequence_integer": get_sequence_integer(df['modified_sequence']).astype(np.uint8),
    }

    return dataset

def remove_keys_with_2darray(chunk_dict):
    keys_to_remove = []
    for key, value in chunk_dict.items():
        if isinstance(value, np.ndarray) and value.ndim > 1 and value.shape[1] > 1:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del chunk_dict[key]
    
    # flatten to 1d array in chunk_dict
    for key, value in chunk_dict.items():
        if isinstance(value, np.ndarray) and value.ndim > 1:
            # print(f"Flattening multi-dimensional array for key: {key}")
            chunk_dict[key] = value.flatten()
    return chunk_dict


def print_combined_data_sizes(combined_data):
    """
    Print the size of each key in the combined_data dictionary or Dataset.

    Args:
        combined_data (dict or Dataset): Combined data to inspect.
    """
    print("Combined Prediction Data Sizes:")
    if isinstance(combined_data, dict):
        print(f"Dictionary contains {len(combined_data)} keys.")
        for key, value in combined_data.items():
            if isinstance(value, np.ndarray):
                print(f"Key: {key}, Shape: {value.shape}, Size: {value.nbytes / (1024 * 1024):.2f} MB")
            elif isinstance(value, list):
                print(f"Key: {key}, Length: {len(value)}, Approx. Size: {sum(sys.getsizeof(v) for v in value) / (1024 * 1024):.2f} MB")
            else:
                print(f"Key: {key}, Type: {type(value)}, Size: {sys.getsizeof(value) / (1024 * 1024):.2f} MB")
    elif isinstance(combined_data, Dataset):
        print(f"Dataset contains {len(combined_data)} rows.")
        for column in combined_data.column_names:
            print(f"Column: {column}, Length: {len(combined_data[column])}")
    else:
        print("Unsupported data type.")
