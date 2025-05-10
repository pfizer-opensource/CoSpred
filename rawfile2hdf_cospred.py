import logging
import numpy as np
import pandas as pd
import os
import warnings
from argparse import ArgumentParser
from pyteomics import mzml, mgf

from prosit_model import io_local
import io_cospred
import params.constants_location as constants_location
from params.constants import (
    VAL_SPLIT, SPECTRA_DIMENSION, BIN_MAXMZ, BIN_SIZE,
)


def filterPSM(dbsearch_df, csvfile):
    """
    Filter the dbsearch_df DataFrame to include only rows where the combination of
    `file`, `scan`, `modifiedseq`, and `charge` matches the corresponding
    columns in the `csvfile`. Retain all columns in dbsearch_df and avoid column name collisions.

    Args:
        dbsearch_df (pd.DataFrame): DataFrame containing PSM data.
        csvfile (str): Path to the CSV file containing the required columns.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Load the CSV file
    if not os.path.exists(csvfile):
        raise FileNotFoundError(f"File not found: {csvfile}")

    csv_df = pd.read_csv(csvfile)
    
    # Ensure the required columns exist in both DataFrames
    csv_to_dbsearch_mapping = {
        'raw_file': 'file',
        'scan_number': 'scan',
        'modified_sequence': 'modifiedseq',
        'precursor_charge': 'charge'
    }
    
    for csv_col, db_col in csv_to_dbsearch_mapping.items():
        assert db_col in dbsearch_df.columns, f"dbsearch_df must contain '{db_col}' column."
        assert csv_col in csv_df.columns, f"csvfile must contain '{csv_col}' column."
    
    # Subset and rename columns in csv_df to match dbsearch_df
    csv_df = csv_df[list(csv_to_dbsearch_mapping.keys())].rename(columns=csv_to_dbsearch_mapping)
    
    # Perform the filtering based on all specified columns
    filtered_df = dbsearch_df.merge(csv_df, on=list(csv_to_dbsearch_mapping.values()), how='inner')
    
    logging.info(f"[USER] Filtered PSM DataFrame: {len(filtered_df)} rows remaining out of {len(dbsearch_df)}.")
    return filtered_df


def constructCospredVec(mz_arr, intensity_arr):
    intensity_arr = intensity_arr / np.max(intensity_arr)
    vector_intensity = np.zeros(SPECTRA_DIMENSION, dtype=np.float16)
    vector_mass = np.arange(0, BIN_MAXMZ, BIN_SIZE, dtype=np.float16)
    index_arr = mz_arr / BIN_SIZE
    index_arr = np.around(index_arr).astype('int16')

    for i, index in enumerate(index_arr):
        if (index > SPECTRA_DIMENSION - 1):         # add intensity to last bin for high m/z
            vector_intensity[-1] += intensity_arr[i]
        else:
            vector_intensity[index] += intensity_arr[i]
    return vector_intensity, vector_mass


# Contruct ML friendly spectra matrix for transformer full prediction
def generateHDF5_transformer(usimgffile, reformatmgffile, dbsearch_df, csvfile, contrastcsvfile):
    assert "file" in dbsearch_df.columns
    assert "scan" in dbsearch_df.columns
    assert "charge" in dbsearch_df.columns
    assert "seq" in dbsearch_df.columns
    assert "modifiedseq" in dbsearch_df.columns
    assert "proforma" in dbsearch_df.columns
    assert "score" in dbsearch_df.columns
    assert "reverse" in dbsearch_df.columns

    # retrieve spectrum of PSM from MGF
    spectra = mgf.read(usimgffile)
    mzs_df = []

    for index, row in dbsearch_df.iterrows():
        if (index % 100 == 0):
            logging.info('Generating CSV Progress: {}%'.format(
                index/dbsearch_df.shape[0]*100))

        try:
            # title format:'mzspec:repoID:phospho_sample1.raw:scan:9'
            spectrum = spectra.get_spectrum(row['title'])
            retention_time = spectrum['params']['rtinseconds']
            collision_energy = float(spectrum['params']['ce'])
            charge_state = int(spectrum['params']['charge'][0])
            method = spectrum['params']['method']
            mod_num = spectrum['params']['mod_num']
            raw_file = row['file']
            scan_number = row['scan']
            sequence = row['seq']
            score = row['score']
            modified_sequence = row['modifiedseq']
            proforma = row['proforma']
            reverse = row['reverse']

            # Transformer specific vector
            intensity_vec, mz_vec = constructCospredVec(
                spectrum['m/z array'], spectrum['intensity array'])
            masses = mz_vec
            intensities = intensity_vec

            mzs_df.append(pd.Series([raw_file, scan_number, sequence, score,
                                     modified_sequence, proforma,
                                     mod_num, reverse,
                                     collision_energy, charge_state,
                                     masses, intensities,
                                     retention_time, method
                                     ]))
        except:
            next

    # construct CSV
    mzs_df = pd.concat(mzs_df, axis=1).transpose()
    mzs_df.columns = ['raw_file', 'scan_number', 'sequence', 'score',
                      'modified_sequence', 'proforma',
                      'mod_num', 'reverse',
                      'collision_energy', 'precursor_charge',
                      'masses', 'intensities',
                      'retention_time', 'method']
    mzs_df = mzs_df.reset_index(drop=True)
    mzs_df = mzs_df.dropna()
    mzs_df.columns = mzs_df.columns.str.replace('[\r]', '')
    # To prevent data leaking, only keep the peptides that are not in the contrast dataset
    if (contrastcsvfile is not None):
        constrast_dataset = pd.read_csv(contrastcsvfile, sep=',')
        mzs_df = mzs_df[~mzs_df['proforma'].isin(constrast_dataset['proforma'])]
    mzs_df.to_csv(csvfile, index=False)      # CSV discards values in large vec
    logging.info('[STATUS] Generating peptide list CSV ... DONE!')

    # construct Dataset based on CoSpred Transformer definition
    dataset = {
        "collision_energy": mzs_df['collision_energy'].astype(np.uint8),
        "collision_energy_aligned": io_cospred.get_float(mzs_df['collision_energy']),
        "collision_energy_aligned_normed": io_cospred.get_float(mzs_df['collision_energy']/100.0),
        "intensities_raw": io_cospred.get_2darray(mzs_df['intensities']).astype(np.float16),
        "masses_pred": io_cospred.get_2darray(mzs_df['masses']).astype(np.float16),
        "masses_raw": io_cospred.get_2darray(mzs_df['masses']).astype(np.float16),
        "method": io_cospred.get_method_onehot(mzs_df['method']).astype(np.uint8),
        "precursor_charge": mzs_df['precursor_charge'].astype(np.uint8),
        "precursor_charge_onehot": io_cospred.get_precursor_charge_onehot(mzs_df['precursor_charge']).astype(np.uint8),
        "raw_file": mzs_df['raw_file'].astype('S32'),
        "reverse": io_cospred.get_boolean(mzs_df['reverse']),
        "scan_number": io_cospred.get_number(mzs_df['scan_number']).astype(np.uint8),
        "score": io_cospred.get_float(mzs_df['score']),
        "modified_sequence": mzs_df['modified_sequence'].astype('S32'),
        "sequence_integer": io_cospred.get_sequence_integer(mzs_df['modified_sequence']).astype(np.uint8),
        "sequence_onehot": io_cospred.get_sequence_onehot(mzs_df['modified_sequence']).astype(np.uint8),
    }

    io_cospred.modifyMGFtitle(usimgffile, reformatmgffile)
    return dataset


# def load_dataframe(input_data):
#     """
#     Load data into a pandas DataFrame based on the input type.

#     Args:
#         input_data (str or dict): Input data, either a file path to a CSV or a dictionary.

#     Returns:
#         pd.DataFrame: Loaded DataFrame.
#     """
#     if isinstance(input_data, dict):
#         # Input is a dictionary
#         logging.info("Input is a dictionary. Converting to DataFrame...")
#         df = pd.DataFrame.from_dict(input_data)
#     elif isinstance(input_data, str) and os.path.isfile(input_data):
#         # Input is a file path
#         logging.info("Input is a file path. Reading CSV...")
#         df = pd.read_csv(input_data, sep=',', index_col=False)
#     else:
#         raise ValueError("Input must be either a dictionary or a valid file path to a CSV.")
    
#     return df


def main():
    # Suppress warning message of tensorflow compatibility
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")

    # Configure logging
    log_file_prep = os.path.join(constants_location.LOGS_DIR, "cospred_prep.log")
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
    parser.add_argument('-w', '--workflow', default='test',
                        help='workflow to use')
    args = parser.parse_args()

    workflow = args.workflow

    temp_dir = constants_location.TEMP_DIR
    trainsetfile = constants_location.TRAINFILE
    testsetfile = constants_location.TESTFILE
    traincsvfile = constants_location.TRAINCSV_PATH
    testcsvfile = constants_location.TESTCSV_PATH
    
    mgf_dir = constants_location.MGF_DIR
    mgffile = constants_location.MGF_PATH
    mzml_dir = constants_location.MZML_DIR
    psmfile = constants_location.PSM_PATH
    mappingfile = constants_location.MAPPINGFILE_PATH

    if (workflow == 'split'):
        pass
    elif (workflow == 'train'):
        usimgffile = constants_location.REFORMAT_TRAIN_USITITLE_PATH
        reformatmgffile = constants_location.REFORMAT_TRAIN_PATH
        hdf5file = constants_location.TRAINDATA_PATH
        datasetfile = trainsetfile
        csvfile = traincsvfile
    elif (workflow == 'test'):
        usimgffile = constants_location.REFORMAT_TEST_USITITLE_PATH
        reformatmgffile = constants_location.REFORMAT_TEST_PATH
        hdf5file = constants_location.TESTDATA_PATH
        datasetfile = testsetfile
        csvfile = testcsvfile
    elif (workflow == 'predict'):
        csvfile = constants_location.PREDICT_ORIGINAL
        predict_csv = constants_location.PREDICTCSV_PATH
        hdf5file = constants_location.PREDDATA_PATH
    else:
        logging.error("Unknown workflow choice.")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # get psm result
    # if (workflow != 'predict' and workflow != 'combine'):
    #     dbsearch_df = io_cospred.getPSM(psmfile, mappingfile)

    if (workflow == 'split'):
        # hold out N records as testset
        logging.info('[INFO] Workflow: Splitting the dataset ...')        
        io_cospred.splitMGF(mgf_dir, mgffile, trainsetfile, testsetfile, test_ratio=1-VAL_SPLIT)
        logging.info('[STATUS] Splitting train vs test set ... DONE!')
    # reformat the Spectra
    elif (workflow == 'train' or workflow == 'test'):
        logging.info(f'[INFO] Workflow: Annotating {workflow} set ...')        
        dbsearch_df = io_cospred.getPSM(psmfile, mgf_dir, mappingfile)
        # regenerate header for downstream annotation
        if not os.path.exists(usimgffile):
            io_cospred.reformatMGF(datasetfile, mzml_dir, dbsearch_df, usimgffile, temp_dir)
        # match peptide from PSM with spectra MGF to generate CSV with full spectra bins
        if not os.path.exists(reformatmgffile):
            if (workflow == 'train'):
                dataset = generateHDF5_transformer(usimgffile, reformatmgffile, dbsearch_df,
                                                csvfile, None)
            elif (workflow == 'test'):
                dataset = generateHDF5_transformer(usimgffile, reformatmgffile, dbsearch_df,
                                                csvfile, traincsvfile)
            io_local.to_hdf5(dataset, hdf5file)
            logging.info(f'[STATUS] Generating {workflow} set ... DONE!')
        else:
            logging.info(f'[USER] {workflow} set is already existed')
    elif (workflow == 'predict'):
        logging.info('[INFO] Workflow: Spectrum prediction ...')        
        if not os.path.exists(csvfile):
            logging.error('No peptide list available!')
        else:
            # transform to hdf5
            dataset = io_cospred.constructDataset_frompep(csvfile, predict_csv)
            io_cospred.to_hdf5(dataset, hdf5file)
            # check generated hdf
            io_cospred.read_hdf5(hdf5file)
            logging.info(f'[STATUS] Generating {workflow} set ... DONE!')
    else:
        logging.error("Unknown workflow choice.")


if __name__ == "__main__":
    main()
