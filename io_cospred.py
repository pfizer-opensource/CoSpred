import logging
import sys
import time
import random
import glob
import numpy as np
import h5py
import pandas as pd
from datasets import Dataset, load_from_disk, concatenate_datasets
import os
import ntpath
import shutil
import re
import copy
from pyteomics import mzml, mgf

from prosit_model import utils, sanitize, tensorize
import params.constants as constants
from params.constants import (
    ALPHABET,
    AMINO_ACID,
    CHARGES,
    METHODS,
    MAX_SEQUENCE,
)


# def peptide_parser(p):
#     p = p.replace("_", "")
#     if p[0] == "(":
#         raise ValueError("sequence starts with '('")
#     n = len(p)
#     i = 0
#     while i < n:
#         if i < n - 3 and p[i + 1] == "(":
#             j = p[i + 2:].index(")")
#             offset = i + j + 3
#             yield p[i:offset]
#             i = offset
#         else:
#             yield p[i]
#             i += 1


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
    array = array.astype(dtype)
    return array


def get_float(vals, dtype=np.float16):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_boolean(vals, dtype=bool):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_number(vals, dtype='i1'):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_string(vals, dtype=str):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


# def get_numbers(vals, dtype=np.float16):
#     a = np.array(vals).astype(dtype)
#     return a.reshape([len(vals), 1])


def get_precursor_charge_onehot(charges, dtype='i1'):
    array = np.zeros([len(charges), max(CHARGES)])
    for i, precursor_charge in enumerate(charges):
        if precursor_charge > max(CHARGES):
            pass
        else:
            array[i, int(precursor_charge) - 1] = 1
    array = array.astype(dtype)
    return array


def get_method_onehot(methods, dtype = bool):
    array = np.zeros([len(methods), len(METHODS)])
    for i, method in enumerate(methods):
        for j, methodstype in enumerate(METHODS):
            if method == methodstype:
                array[i, j] = int(1)
    return array.astype(dtype)


def get_sequence_onehot(sequences, dtype = bool):
    array = np.zeros([len(sequences), MAX_SEQUENCE, len(ALPHABET)+1])
    for i, sequence in enumerate(sequences):
        j = 0
        for aa in utils.peptide_parser(p=sequence):
            if aa in ALPHABET.keys():
                array[i, j, ALPHABET[aa]] = int(1)
            j += 1
        while j < MAX_SEQUENCE:
            array[i, j, 0] = int(1)
            j += 1
    return array.astype(dtype)


# def get_precursor_charge_onehot(charges):
#     array = np.zeros([len(charges), max(CHARGES)])
#     for i, precursor_charge in enumerate(charges):
#         if precursor_charge > max(CHARGES):
#             pass
#         else:
#             array[i, int(precursor_charge) - 1] = 1
#     return array


# def get_sequence_integer(sequences, dtype='i1'):
#     array = np.zeros([len(sequences), MAX_SEQUENCE])
#     for i, sequence in enumerate(sequences):
#         try:
#             if len(sequence) > MAX_SEQUENCE:
#                 pass
#             else:
#                 for j, s in enumerate(utils.peptide_parser(sequence)):
#                     # # POC: uppercase all amino acid, so no PTM
#                     # array[i, j] = ALPHABET[s.upper()]
#                     # #
#                     array[i, j] = ALPHABET[s]
#         except:
#             next
#     return array


def get_array(tensor, keys):
    utils.check_mandatory_keys(tensor, keys)
    return [tensor[key] for key in keys]


def get_2darray(vals, dtype=np.float16):
    a = np.array(vals.values.tolist())
    a = a.astype(dtype)
    return a


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
        "collision_energy_aligned_normed": get_float(df['collision_energy']/100.0, dtype=np.float16),
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



def filter_dbsearch_by_mgf_files(dbsearch, mgf_dir):
    """
    Filter dbsearch DataFrame to keep rows where the 'file' column matches the basename of files in mgf_dir.

    Args:
        dbsearch (pd.DataFrame): DataFrame containing PSM data.
        mgf_dir (str): Path to the directory containing .mgf files.

    Returns:
        pd.DataFrame: Filtered dbsearch DataFrame.
    """
    # Get the basenames of all .mgf files in mgf_dir
    mgf_files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(mgf_dir) if f.endswith('.mgf')]

    # Remove the suffix (file extension) from dbsearch["file"]
    dbsearch["file_basename"] = dbsearch["file"].apply(lambda x: os.path.splitext(x)[0])

    # Filter dbsearch to keep rows where 'file' matches the basename of files in mgf_dir
    filtered_dbsearch = dbsearch[dbsearch["file_basename"].isin(mgf_files)]

    return filtered_dbsearch


def getPSM(psmfile, mgf_dir, mappingfile=None):
    """
    Read PSM file and return a DataFrame with relevant columns.
    Args:
        psmfile (str): Path to the PSM file.
        mappingfile (str, optional): Path to the mapping file. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame containing relevant columns from the PSM file.
    """
    if not os.path.exists(psmfile):
        raise FileNotFoundError(f"PSM file '{psmfile}' does not exist.")
    
    target_cols = {
        "Annotated Sequence": "seq",
        "Modifications": "modifications",
        "m/z [Da]": "mz",
        "Charge": "charge",
        "RT [min]": "retentiontime",
        "Checked": "reverse",
        "PEP": "score",
        "First Scan": "scan",
        "Spectrum File": "file",
    }

    dbsearch = pd.read_csv(psmfile, sep="\t", keep_default_na=False, na_values=["NaN"], index_col=False)

    # Make sure 'Percolator PEP' or 'PEP' existed
    if "PEP" not in dbsearch.columns:
        if "Percolator PEP" in dbsearch.columns:
            dbsearch.rename(columns={"Percolator PEP": "PEP"}, inplace=True)
        else:
            raise KeyError("Neither 'PEP' nor 'Percolator PEP' column found in dbsearch.")

    # Handle mapping file if provided
    if os.path.exists(mappingfile):
        mapping_df = pd.read_csv(mappingfile, sep="\t", keep_default_na=False, na_values=["NaN"], index_col=False)
        if "Spectrum File" not in dbsearch.columns and "File ID" in dbsearch.columns:
            dbsearch = dbsearch.merge(
                mapping_df[["Study File ID", "File Name"]],
                how="left",
                left_on="File ID",
                right_on="Study File ID",
            )
            dbsearch["Spectrum File"] = dbsearch["File Name"].apply(
                lambda x: ntpath.basename(x) if "\\" in x else os.path.basename(x)
            )
        elif "File ID" not in dbsearch.columns:
            raise KeyError("'File ID' is missing in dbsearch, cannot join with mapping_df to get 'Spectrum File'.")
    else:
        if "Spectrum File" not in dbsearch.columns:
            raise KeyError("'Spectrum File' is missing in dbsearch and no mapping file provided to retrieve it.")

    # Filter and rename columns
    existing_cols = [col for col in target_cols.keys() if col in dbsearch.columns]
    filtered_target_cols = {col: target_cols[col] for col in existing_cols}
    dbsearch = dbsearch[dbsearch["Confidence"] == "High"][existing_cols]
    dbsearch = dbsearch.rename(columns=filtered_target_cols)

    dbsearch["seq"] = dbsearch["seq"].str.replace(r"\[\S+\]\.", "", regex=True)
    dbsearch["seq"] = dbsearch["seq"].str.replace(r"\.\[\S+\]", "", regex=True)
    dbsearch['seq'] = dbsearch['seq'].str.upper()

    dbsearch = dbsearch[dbsearch["seq"].str.len() <= 30]  # Remove sequences longer than 30

    # Parse modifications dynamically
    seq_list = dbsearch["seq"].tolist()
    mod_list = dbsearch["modifications"].tolist()
    modseq_list = []
    proforma_list = []
    modnum_list = []

    for k in range(len(dbsearch)):
        letter = [x for x in seq_list[k]]
        modseq_list.append(letter)
        modnum_list.append(0)
    proforma_list = copy.deepcopy(modseq_list)
    # make sure later concatenate aligns
    dbsearch = dbsearch.reset_index(drop=True)

    # Dynamically parse modifications
    for mod_name, mod_format in constants.VARMOD_PROFORMA.items():
        if "(" in mod_name:
            targetmod = f"[{mod_name.split('(')[0]}][0-9]+\\({mod_name.split('(')[1].rstrip(')')}\\)"
        elif "+" in mod_name:
            targetmod = f"{mod_name.split('+')[0]}\\+{mod_name.split('+')[1]}"

        for k in range(len(dbsearch)):
            if mod_list[k] and mod_list[k] != "":
                matchMod = re.findall(targetmod, mod_list[k])  # Locate mod site
                matchDigit = [int(re.search(r"([0-9]+)", x).group(0)) for x in matchMod]
                for i in reversed(matchDigit):
                    if "(" in mod_name:
                        modseq_list[k][i - 1] = modseq_list[k][i - 1] + f"({mod_name.split('(')[1].rstrip(')')})"
                    elif "+" in mod_name:
                        modseq_list[k][i - 1] = modseq_list[k][i - 1] + f"(+{mod_name.split('+')[1]})"
                    proforma_list[k][i - 1] = proforma_list[k][i - 1] + f"[{mod_format.split('[')[1].rstrip(']')}]"
                    modnum_list[k] += 1

    # Ensure all lists have the same length as seq_list
    assert len(seq_list) == len(modseq_list) == len(proforma_list), "List lengths are inconsistent!"

    # Add the processed data back to dbsearch
    dbsearch["modifiedseq"] = pd.Series(["".join(x) for x in modseq_list])
    dbsearch["proforma"] = pd.Series(["".join(x) for x in proforma_list])
    dbsearch["mod_num"] = pd.Series(modnum_list).astype(str)

    dbsearch = filter_dbsearch_by_mgf_files(dbsearch, mgf_dir)

    # Reset index and recreate title for mzML matching
    dbsearch = dbsearch.reset_index(drop=True)
    dbsearch["title"] = "mzspec:repoID:" + dbsearch["file"] + ":scan:" + dbsearch["scan"].astype(str)
    
    return dbsearch



def get_mgf_files(mgf_dir):
    """
    Retrieve all .mgf files from the specified directory.

    Args:
        mgf_dir (str): Path to the directory containing .mgf files.

    Returns:
        list: List of paths to .mgf files.
    """
    if not os.path.exists(mgf_dir):
        raise FileNotFoundError(f"Directory '{mgf_dir}' does not exist.")
    
    mgf_files = glob.glob(os.path.join(mgf_dir, "*.mgf"))
    if not mgf_files:
        raise FileNotFoundError(f"No .mgf files found in directory '{mgf_dir}'.")
    
    return mgf_files


def splitMGF(mgf_dir, combined_mgffile, trainsetfile, testsetfile, test_ratio=0.2):
    """
    Combine all .mgf files in mgf_dir, write to a combined .mgf file, and split into train and test sets.

    Args:
        mgf_dir (str): Path to the directory containing .mgf files.
        combined_mgffile (str): Path to save the combined .mgf file.
        trainsetfile (str): Path to save the train set MGF file.
        testsetfile (str): Path to save the test set MGF file.
        test_ratio (float): Ratio of test records (e.g., 0.2 for 20% test set).
    """
    # Step 1: Combine all .mgf files in mgf_dir and write to disk
    logging.info(f"Checking directory: {mgf_dir}")
    mgf_files = get_mgf_files(mgf_dir)
    if not mgf_files:
        logging.error(f"No .mgf files found in directory: {mgf_dir}")
        return
    
    logging.info(f"Combining all .mgf files in directory: {mgf_dir}")
    logging.info(f"Found {len(mgf_files)} .mgf files in directory: {mgf_dir}")
    logging.info(f"Concatenating files into: {combined_mgffile}")

    # OPTION 1: Simple concatenation
    # Open the output file in write mode
    with open(combined_mgffile, "w") as outfile:
        for mgf_file in mgf_files:
            logging.info(f"Appending file: {mgf_file}")
            with open(mgf_file, "r") as infile:
                # Write the content of each .mgf file to the output file
                outfile.write(infile.read())
                outfile.write("\n")  # Ensure a newline between files

    # # OPTION 2: Use pyteomics.mgf to read and write MGF files
    # mgf.write([], output=combined_mgffile, file_mode='w')
    # for mgf_file in mgf_files:
    #     spectra_tmp = []
    #     i = 0
    #     logging.info(f"Reading MGF file: {mgf_file}")
    #     for spectrum in mgf.read(mgf_file):
    #         spectra_tmp.append(spectrum)
    #         if len(spectra_tmp) % 20000 == 0:
    #             # Append a chunk of spectra to the test MGF file
    #             mgf.write(spectra_tmp, output=combined_mgffile, file_mode='a')
    #             spectra_tmp = []
    #             # logging.info(f"Spectrum index {i} added to Combined MGF.")
    #         i += 1
    #     # Write any remaining spectra
    #     if len(spectra_tmp) > 0:
    #         mgf.write(spectra_tmp, output=combined_mgffile, file_mode='a')
    #     logging.info(f"MGF file {mgf_file} read and appended to combined MGF.")

    logging.info(f"Combined MGF file written to: {combined_mgffile}")

    # Step 2: Read combined MGF file
    seed = 42
    np.random.seed(seed)
    combined_spectra = list(mgf.read(combined_mgffile))
    total_spectra = len(combined_spectra)

    # Calculate the number of test records based on the split ratio
    n_test = int(total_spectra * test_ratio)
    logging.info(f"Total spectra: {total_spectra}, Test set size: {n_test}")

    # Randomly sample indices for the test set
    test_index = sorted(random.sample(range(total_spectra), n_test))

    # Initialize lists for train and test spectra
    spectra_train = []
    spectra_test = []

    # Write test and train spectra to their respective files
    mgf.write(spectra_test, output=testsetfile, file_mode='w')
    mgf.write(spectra_train, output=trainsetfile, file_mode='w')

    test_index_list = []
    i = 0
    for spectrum in combined_spectra:
        if i in test_index:
            spectra_test.append(spectrum)
            test_index_list.append(test_index.pop(0))
            if len(spectra_test) % 1000 == 0:
                # Append a chunk of spectra to the test MGF file
                mgf.write(spectra_test, output=testsetfile, file_mode='a')
                spectra_test = []
                logging.info(f"Spectrum index {i} added to test set.")
        else:
            spectra_train.append(spectrum)
            if len(spectra_train) % 1000 == 0:
                # Append a chunk of spectra to the train MGF file
                mgf.write(spectra_train, output=trainsetfile, file_mode='a')
                spectra_train = []
                logging.info(f"Spectrum index {i} added to train set.")
        i += 1

    # Write any remaining spectra
    if len(spectra_test) > 0:
        mgf.write(spectra_test, output=testsetfile, file_mode='a')
    if len(spectra_train) > 0:
        mgf.write(spectra_train, output=trainsetfile, file_mode='a')

    logging.info(f"[USER] Splitting MGF Progress ... DONE: Total {i} records.")
    return test_index_list


def reformatMGF(mgffile, mzml_dir, dbsearch_df, reformatmgffile, temp_dir):
    """
    Reformat an MGF file by adding TITLE, SEQ, CE, and other metadata from mzML files.

    Args:
        mgffile (str): Path to the input MGF file.
        mzml_dir (str): Path to the directory containing mzML files.
        dbsearch_df (pd.DataFrame): DataFrame containing PSM data.
        reformatmgffile (str): Path to save the reformatted MGF file.
        temp_dir (str): Path to a temporary directory for intermediate files.
    """
    # Rewrite TITLE for the MGF
    logging.info('Creating temp MGF file with new TITLE...')
    # # DEBUG:
    # reformatmgffile_temp = os.path.join(temp_dir, 'example_temp.mgf')
    # #
    reformatmgffile_temp = os.path.join(temp_dir, time.strftime("%Y%m%d%H%M%S") + '.mgf')

    spectra_origin = mgf.read(mgffile)
    spectra_temp = []

    # Try parsing the first record to determine the correct parser
    first_spectrum = next(spectra_origin)
    parser = None
    try:
        # Try MSconvert-generated format
        title_split = first_spectrum['params']['title'].split(' ')
        repoid = re.sub('\W$', '', title_split[1].split('"')[1])
        scan_number = re.sub('\W+', '', title_split[0].split('.')[1])
        parser = "MSconvert"
        logging.info("Using MSconvert parser.")
    except Exception as e:
        logging.warning(f"Failed MSconvert format for title: {first_spectrum['params']['title']}. Trying PD format.")
        try:
            # Try PD-generated format
            title_split = first_spectrum['params']['title'].split(';')
            repoid = re.sub("\W$", '', title_split[0].split('\\')[-1])
            scan_number = re.sub('\W+', '', title_split[-1].split('scans')[-1])
            parser = "PD"
            logging.info("Using PD parser.")
        except Exception as e:
            logging.error(f"Failed both MSconvert and PD formats for title: {first_spectrum['params']['title']}. Error: {e}")
            raise ValueError("Unable to determine the correct parser for the MGF file.")

    # Process the first spectrum
    if parser == "MSconvert":
        repoid = re.sub('\W$', '', title_split[1].split('"')[1])
        scan_number = re.sub('\W+', '', title_split[0].split('.')[1])
    elif parser == "PD":
        repoid = re.sub("\W$", '', title_split[0].split('\\')[-1])
        scan_number = re.sub('\W+', '', title_split[-1].split('scans')[-1])
    first_spectrum['params']['title'] = ':'.join(['mzspec', 'repoID', repoid, 'scan', scan_number])
    first_spectrum['params']['scans'] = scan_number
    
    spectra_temp.append(first_spectrum)

    # Process the remaining spectra using the identified parser
    for spectrum in spectra_origin:
        try:
            if parser == "MSconvert":
                title_split = spectrum['params']['title'].split(' ')
                repoid = re.sub('\W$', '', title_split[1].split('"')[1])
                scan_number = re.sub('\W+', '', title_split[0].split('.')[1])
            elif parser == "PD":
                title_split = spectrum['params']['title'].split(';')
                repoid = re.sub("\W$", '', title_split[0].split('\\')[-1])
                scan_number = re.sub('\W+', '', title_split[-1].split('scans')[-1])
            spectrum['params']['title'] = ':'.join(['mzspec', 'repoID', repoid, 'scan', scan_number])
            spectrum['params']['scans'] = scan_number
            spectra_temp.append(spectrum)
        except Exception as e:
            logging.error(f"Error processing spectrum with title {spectrum['params']['title']}: {e}")
            continue

    # Write the reformatted spectra to a temporary MGF file
    mgf.write(spectra_temp, output=reformatmgffile_temp)
    spectra_origin.close()
    logging.info('Temp MGF file with new TITLE was created!')

    # Add SEQ and CE to the reformatted MGF
    mzml_files = [os.path.join(mzml_dir, f) for f in os.listdir(mzml_dir) if f.endswith('.mzML')]
    if not mzml_files:
        raise FileNotFoundError(f"No mzML files found in directory: {mzml_dir}")

    # check PSMs.txt and mzML inputs
    logging.info(f"Total number of identified PSMs are: {len(dbsearch_df)}")
    logging.info(f"Loading mzML files from directory: {mzml_dir}")
    mzml_readers = {os.path.basename(mzml_file): mzml.MzML(mzml_file) for mzml_file in mzml_files}
    logging.info(f"Loaded {len(mzml_readers)} mzML files.")

    spectra = mgf.read(reformatmgffile_temp)
    mgf.write([], output=reformatmgffile, file_mode='w')
    for index, row in dbsearch_df.iterrows():
        if index % 100 == 0:
            logging.info(f'Reformatting MGF Progress: {index / dbsearch_df.shape[0] * 100:.2f}%')

        try:
            # Retrieve spectrum of PSM from MGF
            spectrum = spectra.get_spectrum(row['title'])
            spectrum['params']['seq'] = row['modifiedseq']
            spectrum['params']['proforma'] = row['proforma']
            spectrum['params']['mod_num'] = str(row['mod_num'])
            
            # Extract repoid from the spectrum title
            title_parts = spectrum['params']['title'].split(':')
            if len(title_parts) < 3:
                logging.warning(f"Invalid title format: {spectrum['params']['title']}")
                continue
            repoid = title_parts[2]  # Extract the repoid (e.g., "phospho_sample1.raw")
            repoid = os.path.splitext(repoid)[0]  # Remove the suffix (e.g., ".raw" or ".mzML")

            # Find the corresponding mzML file
            mzml_file_key = f"{repoid}.mzML"
            if mzml_file_key not in mzml_readers:
                logging.warning(f"RepoID '{repoid}' not found in mzML files.")
                continue

            # Retrieve the spectrum from the mzML file
            mzml_reader = mzml_readers[mzml_file_key]
            controller_str = 'controllerType=0 controllerNumber=1 '
            mzml_spectrum = mzml_reader.get_by_id(controller_str + f"scan={spectrum['params']['scans']}")

            # Enrich the spectrum with mzML metadata
            precursor = mzml_spectrum['precursorList']['precursor'][0]
            spectrum['params']['pepmass'] = spectrum['params']['pepmass'][0]
            spectrum['params']['rtinseconds'] = str(mzml_spectrum['scanList']['scan'][0]['scan start time'] * 60)  # Convert minutes to seconds
            spectrum['params']['ce'] = str(precursor['activation']['collision energy'])
            spectrum['params']['charge'] = re.sub(
                '\D+', '', str(precursor['selectedIonList']['selectedIon'][0]['charge state']))
            filter_string = mzml_spectrum['scanList']['scan'][0]['filter string']
            if re.search("hcd", filter_string, re.IGNORECASE):
                method = "HCD"
            elif re.search("cid", filter_string, re.IGNORECASE):
                method = "CID"
            elif re.search("etd", filter_string, re.IGNORECASE):
                method = "ETD"
            else:
                method = "Unknown"
            spectrum['params']['method'] = method

            # Write the enriched spectra to the final MGF file
            mgf.write(spectrum, output=reformatmgffile, file_mode='a')
        except Exception as e:
            # logging.error(f"Error processing spectrum with title {row['title']}: {e}")
            continue

    # Close mzML readers
    for mzml_reader in mzml_readers.values():
        mzml_reader.close()
    spectra.close()

    # # Remove the temporary MGF file
    # if os.path.exists(reformatmgffile_temp):
    #     os.remove(reformatmgffile_temp)
    # else:
    #     logging.error("The temp reformatted MGF file does not exist")

    logging.info(f"Reformatted MGF file written to: {reformatmgffile}")
    return 1


def modifyMGFtitle(usimgffile, reformatmgffile):
    # Rewrite TITLE for the MGF
    if os.path.exists(usimgffile):
        logging.info('Loading USI compatible MGF file ...')

        spectra_origin = mgf.read(usimgffile)
        spectra_new = []
        for spectrum in spectra_origin:
            peptide = spectrum['params']['seq']
            ce = spectrum['params']['ce']
            mod_num = str(spectrum['params']['mod_num'])
            charge = re.sub('\D+', '', str(spectrum['params']['charge'][0]))
            # To facilitate Spectrum predicition evaluation, convert title format from USI to seq/charge_ce_0
            spectrum['params']['title'] = peptide + \
                '/' + charge + '_' + ce + '_' + mod_num
            spectra_new.append(spectrum)
        mgf.write(spectra_new, output=reformatmgffile)
        spectra_origin.close()
    else:
        logging.error("[ERROR] The input USI compatible MGF file does not exist")

    logging.info('[STATUS] Reformatted MGF file with new TITLE was created!')

