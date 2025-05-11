import logging
import numpy as np
import pandas as pd
import os
import re
import functools
import warnings
from argparse import ArgumentParser
from pyteomics import mzml, mgf
import spectrum_utils.spectrum as sus

import io_cospred
import params.constants_location as constants_location
from params.constants import (
    VAL_SPLIT,
    MAX_SEQUENCE,
    MAX_ION,
    NLOSSES,
    ION_TYPES,
    MAX_FRAG_CHARGE,
)


def parse_ion(string):
    # string=ion
    ion_type = ION_TYPES.index(string[0])
    ion_fr = 1        # default ion charge is 1
    suffix = ''
    if ("-") in string:
        ion_n, suffix = string[1:].split("-")
    elif ('^') in string:
        ion_n = string[1:].split("^")[0]
        suffix = ""
    else:
        ion_n = re.sub('^\D+', '', string)

    if ("+") in string:
        ion_frag = string.count("+")
        if ion_frag <= 3:
            ion_fr = ion_frag
        else:
            pass
    elif ('^') in string:
        ion_frag = int(string.split("^")[1])
        if ion_frag <= 3:
            ion_fr = ion_frag
        else:
            pass
    else:
        ion_fr = 1

    return int(ion_n) - 1, ion_type, NLOSSES.index(suffix), int(ion_fr)-1


def reshape_dims(array):
    n, dims = array.shape
    assert dims == 174
    nlosses = 1
    return array.reshape(
        [array.shape[0], MAX_SEQUENCE - 1,
            len(ION_TYPES), nlosses, MAX_FRAG_CHARGE]
    )


def reshape_flat(array):
    s = array.shape
    flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
    return array.reshape(flat_dim)


def mask_outofrange(array, lengths, mask=-1.0):
    for i in range(array.shape[0]):
        # array[0,7:].shape
        array[i, (lengths[i] - 1):, :, :, :] = mask
    return array


# restrict nloss and charge to be considered
def cap(array, nlosses=1, z=3):
    return array[:, :, :, :nlosses, :z]


def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        array[i, :, :, :, (charges[i]):] = mask
    return array


# # Extract spectra from mzml
# def readMZML(mzmlfile, dbsearch_df):
#     f = mzml.MzML(mzmlfile)

#     mzs_df = []

#     for index, row in dbsearch_df.iterrows():
#         try:
#             controller_str = 'controllerType=0 controllerNumber=1 '
#             spectrum_id = controller_str + "scan=" + str(row.scan)
#             if spectrum_id not in f:
#                 print(f"Spectrum ID {spectrum_id} not found in the dataset.")
#                 continue
#             p = f.get_by_id(spectrum_id)
#             dfg = p.get('precursorList')
#             fg = dfg['precursor']
#             collision_energy = fg[0].get('activation').get('collision energy')
#             charge_state = fg[0].get('selectedIonList').get(
#                 'selectedIon')[0].get('charge state')
#             filter_string = p.get('scanList').get('scan')[0].get('filter string')
#             retention_time = p.get('scanList').get('scan')[
#                 0].get('scan start time')
#             method = "Unknown"
#             if re.search("hcd", filter_string):
#                 method = "HCD"
#             elif re.search("cid", filter_string):
#                 method = "CID"
#             elif re.search("etd", filter_string):
#                 method = "ETD"
#             mzs_df.append(
#                 pd.Series([collision_energy, charge_state, retention_time, method]))
#         except Exception as e:
#             print(f"Error processing row {index}: {e}")
#             continue

#     mzs_df = pd.concat(mzs_df, axis=1).transpose()
#     mzs_df.columns = ['collision_energy',
#                       'charge_state', 'retention_time', 'method']

#     return mzs_df


# Annotate b and y ions to MGF file
def annotateMGF(reformatmgffile, dbsearch_df, temp_dir):

    mgfile = mgf.read(reformatmgffile)

    fragment_tol_mass = 0.35
    fragment_tol_mode = 'Da'
    min_mz = 100
    max_mz = 1400
    min_intensity = 0.05

    mzs_df = []

    for index, row in dbsearch_df.iterrows():
        if (index % 100 == 0):
            logging.info('MS2 Annotation Progress: {}%'.format(
                index/dbsearch_df.shape[0]*100))

        try:
            # retrieve spectrum of PSM from MGF
            proforma = row['proforma']
            seq = row['modifiedseq']
            spectrum_dict = mgfile.get_spectrum(row['title'])
            # modifications = {}
            identifier = spectrum_dict['params']['title']
            # peptide = spectrum_dict['params']['seq']
            # ce = spectrum_dict['params']['ce']
            # method = spectrum_dict['params']['method']
            # scan = spectrum_dict['params']['scans']
            precursor_mz = spectrum_dict['params']['pepmass'][0]
            precursor_charge = spectrum_dict['params']['charge'][0]
            retention_time = float(spectrum_dict['params']['rtinseconds'])
            mz = spectrum_dict['m/z array']
            intensity = spectrum_dict['intensity array']

            # Create the MS/MS spectrum.
            # sus.reset_modifications()
            spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge, mz, intensity,
                                        retention_time=retention_time,
                                        # peptide=peptide,
                                        # modifications=modifications
                                        )

            # Filter and clean up the MS/MS spectrum.
            spectrum = spectrum.set_mz_range(min_mz=min_mz, max_mz=max_mz). \
                remove_precursor_peak(fragment_tol_mass, fragment_tol_mode). \
                filter_intensity(min_intensity=min_intensity, max_num_peaks=50)
            # Annotate the MS2 spectrum.
            spectrum = spectrum.annotate_proforma(proforma,
                                                  fragment_tol_mass=10,
                                                  fragment_tol_mode="ppm",
                                                  ion_types="by"
                                                  )

            intensity_annotations = ";".join(
                [str(element) for element in spectrum.intensity])
            mz_annotations = ";".join([str(element)
                                      for element in spectrum.mz])
            ion_annotations = ";".join(
                [re.sub('/\S+', '', str(element)) for element in spectrum.annotation.tolist()])
            mzs_df.append(
                pd.Series([seq, intensity_annotations, mz_annotations, ion_annotations]))
        except:
            next

    # construct dataframe for annotated MS2
    mzs_df = pd.concat(mzs_df, axis=1).transpose()
    mzs_df.columns = ['seq', 'intensity_annotations',
                      'mz_annotations', 'ion_annotations']
    mzs_df.to_csv(temp_dir+'annotatedMGF.csv', index=False)

    return mzs_df


# Contruct ML friendly spectra matrix
def generateCSV(usimgffile, reformatmgffile, dbsearch_df, annotation_results, csvfile, temp_dir, contrastcsvfile):
    assert "file" in dbsearch_df.columns
    assert "scan" in dbsearch_df.columns
    assert "charge" in dbsearch_df.columns
    assert "seq" in dbsearch_df.columns
    assert "modifiedseq" in dbsearch_df.columns
    assert "proforma" in dbsearch_df.columns
    assert "score" in dbsearch_df.columns
    assert "reverse" in dbsearch_df.columns

    # get annotation MS2
    annotation_results.columns = [
        'seq', 'intensities', 'masses', 'matches_raw']

    # retrieve spectrum of PSM from MGF
    spectra = mgf.read(usimgffile)
    spectra[0]
    mzs_df = []
    for index, row in dbsearch_df.iterrows():
        if (index % 100 == 0):
            logging.info('Generating CSV Progress: {}%'.format(
                index/dbsearch_df.shape[0]*100))

        try:
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
            mod_num = row['mod_num']
            reverse = row['reverse']
            mzs_df.append(pd.Series([raw_file, scan_number, sequence, score,
                                     modified_sequence, proforma,
                                     mod_num, reverse,
                                     collision_energy, charge_state,
                                     retention_time, method, mod_num]))
        except:
            next

    mzs_df = pd.concat(mzs_df, axis=1).transpose()
    mzs_df.columns = ['raw_file', 'scan_number', 'sequence', 'score',
                      'modified_sequence', 'proforma',
                      'mod_num', 'reverse',
                      'collision_energy', 'precursor_charge', 'retention_time',
                      'method', 'mod_num']
    mzs_df['collision_energy_aligned_normed'] = mzs_df['collision_energy']/100.0

    # construct CSV
    annotation_results_new = annotation_results.reset_index(drop=True)
    mzs_df_new = mzs_df.reset_index(drop=True)

    dataset = pd.concat([mzs_df_new, annotation_results_new], axis=1)
    dataset = dataset.dropna()

    # To prevent data leaking, only keep the peptides that are not in the contrast dataset
    if (contrastcsvfile is not None):
        constrast_dataset = pd.read_csv(contrastcsvfile, sep=',', index_col=False)
        dataset = dataset[~dataset['proforma'].isin(constrast_dataset['proforma'])]
    dataset.to_csv(csvfile, index=False)

    logging.info('[STATUS] Generating peptide list CSV ... DONE!')

    io_cospred.modifyMGFtitle(usimgffile, reformatmgffile)
    return dataset


def get_PrositArray(df, vectype):
    array_series = []
    for index, row in df.iterrows():
        array = np.zeros(
            [MAX_ION, len(ION_TYPES), len(NLOSSES), MAX_FRAG_CHARGE])
        lstions = str(row.matches_raw).split(";")
        # lstions = str(row.masses).split(";")
        lstmasses = str(row[vectype]).split(";")
        for i in ION_TYPES:
            patternn = r"^" + i + "[0-9]+"
            ions_regex = re.compile(patternn)
            for index, ion in enumerate(lstions):
                if ions_regex.match(ion):
                    ion_n, ion_type, nloss, ion_charge = parse_ion(ion)
                    array[ion_n, ion_type, nloss, ion_charge] = float(
                        lstmasses[index])
        if (vectype == 'intensities'):
            max_value = np.max(array)
            array = array/max_value if max_value > 0 else array
        array_series.append(array)
    out = np.squeeze(np.stack(array_series))
    if len(out.shape) == 4:
        out = out.reshape([1] + list(out.shape))
    return out


def constructPrositVec(df, vectype):
    nlosses = 1
    z = 3
    lengths = [len(x) for x in df["modified_sequence"]]
    array = get_PrositArray(df, vectype)
    # # DEBUG
    # array.shape
    # array[0,0,0,0]     # 1st record, y1
    # array[0,1,1,0]     # 1st record, b2
    # #
    array = cap(array, nlosses, z)   # do not consider nloss, limit charge 1-3
    array = mask_outofrange(array, lengths)    # mask impossible fragment as -1
    # # DEBUG
    # array[0,0,0,0]     # 1st record, y1
    # array[0,9,1,0]     # 1st record, b9
    # #
    array = reshape_flat(array)     # flatten to 174 dimension b,y ion vector
    # # DEBUG
    # array.shape
    # array[0]     # 1st record, y1
    # #
    return array


def constructDataset_byion(csvfile):
    df = pd.read_csv(csvfile, sep=',', index_col=False)
    
    assert "modified_sequence" in df.columns
    assert "collision_energy" in df.columns
    assert "precursor_charge" in df.columns
    assert "intensities" in df.columns
    assert "masses" in df.columns

    df.dropna(subset=['intensities', 'masses'], inplace=True)
    df.columns = df.columns.str.replace('[\r]', '')

    # construct Dataset based on Prosit definition
    dataset = {
        "collision_energy": df['collision_energy'].astype(np.float16),
        "collision_energy_aligned": io_cospred.get_float(df['collision_energy']),
        "collision_energy_aligned_normed": io_cospred.get_float(df['collision_energy']/100.0),
        "intensities_raw": constructPrositVec(df, 'intensities'),
        "masses_pred": constructPrositVec(df, vectype='masses'),
        "masses_raw": constructPrositVec(df, vectype='masses'),
        "method": io_cospred.get_method_onehot(df['method']).astype(np.uint8),
        "precursor_charge": df['precursor_charge'].astype(np.uint8),
        "precursor_charge_onehot": io_cospred.get_precursor_charge_onehot(df['precursor_charge']).astype(np.uint8),
        "raw_file": df['raw_file'].astype('S32'),
        "reverse": io_cospred.get_boolean(df['reverse']),
        "scan_number": io_cospred.get_number(df['scan_number']),
        "score": io_cospred.get_float(df['score']),
        "modified_sequence": df['modified_sequence'].astype('S32'),
        "sequence_integer": io_cospred.get_sequence_integer(df['modified_sequence']).astype(np.uint8),
        "sequence_onehot": io_cospred.get_sequence_onehot(df['modified_sequence']).astype(np.uint8),
    }

    # # Assuming `dictionary` is your dictionary
    # logging.info("Dictionary after constructDataset_byion:")
    # for key, value in dataset.items():
    #     logging.info(f"Key: {key}, Data Type: {type(dataset[key])}, Shape: {dataset[key].shape}, Length: {len(value)}, Approx. Size: {sum(sys.getsizeof(v) for v in value) / (1024 * 1024):.2f} MB")

    return dataset


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
    parser.add_argument(
        '-m', '--mgffile', default=constants_location.MGF_PATH, help='raw file MGF')
    parser.add_argument(
        '-z', '--mzmlfile', default=constants_location.MZML_PATH, help='raw file mzML')
    parser.add_argument('-p', '--psmfile',
                        default=constants_location.PSM_PATH, help='PSM file')
    parser.add_argument('-w', '--workflow', default='split',
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
    if (workflow == 'train'):
        usimgffile = constants_location.REFORMAT_TRAIN_USITITLE_PATH
        reformatmgffile = constants_location.REFORMAT_TRAIN_PATH
        datasetfile = trainsetfile
        csvfile = traincsvfile
        hdf5file = constants_location.TRAINDATA_PATH
    elif (workflow == 'test'):
        usimgffile = constants_location.REFORMAT_TEST_USITITLE_PATH
        reformatmgffile = constants_location.REFORMAT_TEST_PATH
        datasetfile = testsetfile
        csvfile = testcsvfile
        hdf5file = constants_location.TESTDATA_PATH
    elif (workflow == 'predict'):
        # datasetfile = testsetfile
        csvfile = constants_location.PREDICT_ORIGINAL
        predict_csv = constants_location.PREDICTCSV_PATH
        hdf5file = constants_location.PREDDATA_PATH
    else:
        logging.error("Unknown workflow choice.")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # # get psm result
    # if (workflow != 'predict'):
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
        if not os.path.isfile(usimgffile):
            io_cospred.reformatMGF(datasetfile, mzml_dir, dbsearch_df, usimgffile, temp_dir)
            annotation_results = annotateMGF(usimgffile, dbsearch_df, temp_dir)
        else:
            annotation_results = pd.read_csv(temp_dir+'annotatedMGF.csv', index_col=False)
        # match peptide from PSM with spectra MGF
        if not os.path.isfile(reformatmgffile):
            if (workflow == 'train'):
                dataset = generateCSV(usimgffile, reformatmgffile, dbsearch_df, annotation_results,
                                      csvfile, temp_dir, None)
            elif (workflow == 'test'):
                dataset = generateCSV(usimgffile, reformatmgffile, dbsearch_df, annotation_results,
                                      csvfile, temp_dir, traincsvfile)
            # transform to hdf5
            dataset = constructDataset_byion(csvfile)
            io_cospred.to_hdf5(dataset, hdf5file)
            logging.info(f'[STATUS] Generating {workflow} set ... DONE!')
        else:
            logging.info(f'[USER] {workflow} set is already existed')
    elif (workflow == 'predict'):
        logging.info('[INFO] Workflow: Spectrum prediction ...')        
        if not os.path.isfile(csvfile):
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
