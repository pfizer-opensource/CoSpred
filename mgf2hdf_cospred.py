import logging
import re
from pyteomics import mgf
import numpy as np
import pandas as pd
import os
import time
from argparse import ArgumentParser
import random
import shutil

import params.constants_location as constants_location
from params.constants import (
    SPECTRA_DIMENSION, BIN_MAXMZ, BIN_SIZE,
    CHARGES,
    MAX_SEQUENCE,
    ALPHABET,
    FIXMOD_PROFORMA,
    VARMOD_PROFORMA,
    METHODS,
)
from preprocess import utils
from prosit_model import io_local

import warnings
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

def peptide_parser(p):
    p = p.replace("_", "")
    if p[0] == "(":
        raise ValueError("sequence starts with '('")
    n = len(p)
    i = 0
    while i < n:
        if i < n - 3 and p[i + 1] == "(":
            j = p[i + 2:].index(")")
            offset = i + j + 3
            yield p[i:offset]
            i = offset
        else:
            yield p[i]
            i += 1


def get_sequence_integer(sequences, dtype='i1'):
    array = np.zeros([len(sequences), MAX_SEQUENCE])
    for i, sequence in enumerate(sequences):
        if len(sequence) > MAX_SEQUENCE:
            pass
        else:
            for j, s in enumerate(utils.peptide_parser(sequence)):
                array[i, j] = ALPHABET[s]
    array = array.astype(dtype)
    return array


def get_float(vals, dtype=np.float16):
    a = np.array(vals).astype(dtype)
    a = a.reshape([len(vals), 1])
    return a


def get_boolean(vals, dtype=bool):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_number(vals, dtype='i1'):
    a = np.array(vals).astype(dtype)
    a = a.reshape([len(vals), 1])
    return a


def get_2darray(vals, dtype=np.float16):
    a = np.array(vals.values.tolist())
    a = a.astype(dtype)
    return a


def get_precursor_charge_onehot(charges, dtype='i1'):
    array = np.zeros([len(charges), max(CHARGES)])
    for i, precursor_charge in enumerate(charges):
        if precursor_charge > max(CHARGES):
            pass
        else:
            array[i, int(precursor_charge) - 1] = 1
    array = array.astype(dtype)
    return array


def get_method_onehot(methods, dtype=np.uint8):
    array = np.zeros([len(methods), len(METHODS)])
    for i, method in enumerate(methods):
        for j, methodstype in enumerate(METHODS):
            if method == methodstype:
                array[i, j] = int(1)
    array = array.astype(dtype)
    return array


def get_sequence_onehot(sequences, dtype=np.uint8):
    array = np.zeros([len(sequences), MAX_SEQUENCE, len(ALPHABET)+1])
    for i, sequence in enumerate(sequences):
        j = 0
        for aa in peptide_parser(p=sequence):
            if aa in ALPHABET.keys():
                array[i, j, ALPHABET[aa]] = int(1)
            j += 1
        while j < MAX_SEQUENCE:
            array[i, j, 0] = int(1)
            j += 1
    array = array.astype(dtype)
    return array


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


def parseSeq(seq, fixmod_proforma, varmod_proforma):
    import re
    nonmod_seq = seq.upper()
    nonmod_seq = re.sub('[^A-Z]', '', nonmod_seq)
    mod_num = 0
    for target_aa in fixmod_proforma.keys():
        mod_seq = re.sub(target_aa, fixmod_proforma[target_aa], seq)
    for target_aa in varmod_proforma.keys():
        mod_num += len(re.findall(target_aa, mod_seq))
        proforma = re.sub(target_aa, varmod_proforma[target_aa], mod_seq)
    return proforma, mod_num, seq, mod_seq, nonmod_seq


""" This function is used for convert non-compliant massiveKB mgf to standard mgf format"""


def reformatMGF_wSeq(mgffile, reformatmgffile):
    import re
    if (mgffile == reformatmgffile):
        originalmgffile = re.sub('.mgf', '_backup.mgf', mgffile)
        shutil.copy2(mgffile, originalmgffile)
        mgffile = originalmgffile
    spectra = []
    spectrum = {}
    mzs, intensities = [], []
    params = {}
    # initiate the file writing
    mgf.write(spectra, output=reformatmgffile, file_mode='w')
    with open(mgffile) as fp:
        count = 0
        for line in fp:
            line = line.strip().strip('\n')
            if line != '':
                if 'BEGIN IONS' in line:
                    # start processing
                    peptide, pepmass, charge, scans, ce, rawfilename = None, None, None, None, None, None
                    mzs, intensities = [], []
                    spectrum = {}
                    params = {}
                elif 'SEQ=' in line:
                    peptide = line.replace('SEQ=', '')
                    proforma, mod_num, seq, mod_seq, nonmod_seq = parseSeq(
                        peptide, FIXMOD_PROFORMA, VARMOD_PROFORMA)
                elif 'CHARGE=' in line:
                    charge = line.replace('CHARGE=', '')
                    charge = re.sub(r'\D+', '', charge)  # Remove all non-numeric characters
                    charge = int(charge)
                elif 'SCANS=' in line:
                    scans = line.replace('SCANS=', '')
                    scans = int(scans)
                elif 'COLLISION_ENERGY=' in line:
                    ce = line.replace('COLLISION_ENERGY=', '')
                    ce = float(ce)
                elif 'CE=' in line:
                    ce = line.replace('CE=', '')
                    ce = float(ce)
                elif 'PEPMASS=' in line:
                    pepmass = line.replace('PEPMASS=', '')
                    pepmass = float(pepmass)
                elif 'FILENAME=' in line:
                    rawfilename = line.replace('FILENAME=', '')
                    rawfilename = re.sub("\W$", '', rawfilename.split('/')[-1])
                elif 'FILE=' in line:
                    rawfilename = line.replace('FILE=', '')
                    rawfilename = re.sub("\W$", '', rawfilename.split('/')[-1])
                elif 'END IONS' in line:
                    # remove sequence length > 30 or with unexpected amino acids
                    if re.search('[^A-Z]+', mod_seq) is None and len(nonmod_seq) <= 30 and charge < 5 and charge > 1:

                        params['TITLE'] = ':'.join(['mzspec', 'repoID', rawfilename,
                                                    'scan', str(scans)])
                        params['PEPMASS'] = pepmass
                        params['RTINSECONDS'] = 0.0
                        params['CHARGE'] = charge
                        params['MASS'] = 'Monoisotopic'
                        params['FILE'] = rawfilename
                        params['SCANS'] = scans
                        params['CE'] = ce
                        params['SEQ'] = seq
                        params['NONMOD_SEQ'] = nonmod_seq
                        params['MOD_SEQ'] = mod_seq
                        params['PROFORMA'] = proforma
                        params['MOD_NUM'] = mod_num
                        params['METHOD'] = 'HCD'

                        spectrum['params'] = params
                        spectrum['m/z array'] = mzs
                        spectrum['intensity array'] = intensities
                        spectra.append(spectrum)
                        count += 1
                        if (count % 1000 == 0):
                            logging.info('Reformatting MGF Progress: {} records'.format(count))
                            # append a chunk of spectra to new MGF
                            mgf.write(
                                spectra, output=reformatmgffile, file_mode='a')
                            spectra = []

                elif re.search('=', line) is None:
                    mz, intensity = line.split()
                    mzs.append(float(mz))
                    intensities.append(float(intensity))
                    assert len(mzs) == len(intensities)
        if (len(spectra) > 0):
            mgf.write(spectra, output=reformatmgffile, file_mode='a')
        logging.info('[USER] Reformatting MGF Progress DONE: total {} records'.format(count))
    return spectra


def splitMGF(mgffile, trainsetfile, testsetfile, n_test=5000):
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    spectra = mgf.read(mgffile)
    spectra_train = []
    spectra_test = []

    # initiate the file writing
    mgf.write(spectra_test, output=testsetfile, file_mode='w')
    mgf.write(spectra_train, output=trainsetfile, file_mode='w')

    test_index = sorted(random.sample(range(0, len(spectra)), n_test))
    test_index_list = []
    i = 0
    for spectrum in spectra:
        if i in test_index:
            spectra_test.append(spectrum)
            test_index_list.append(test_index.pop(0))
            if (len(spectra_test) % 100 == 0):
                # append a chunk of spectra to new MGF
                mgf.write(spectra_test, output=testsetfile, file_mode='a')
                spectra_test = []
                logging.info('spectrum index {} in testset'.format(i))
        else:
            spectra_train.append(spectrum)
            if (len(spectra_train) % 1000 == 0):
                # append a chunk of spectra to new MGF
                mgf.write(spectra_train, output=trainsetfile, file_mode='a')
                spectra_train = []
        i += 1
    if (len(spectra_test) > 0):
        mgf.write(spectra_test, output=testsetfile, file_mode='a')
    if (len(spectra_train) > 0):
        mgf.write(spectra_train, output=trainsetfile, file_mode='a')
    logging.info('[USER] Splitting MGF Progress DONE: total {} records'.format(i))

    spectra.close()
    return test_index_list


def modifyMGFtitle(usimgffile, reformatmgffile, temp_dir):
    # Rewrite TITLE for the MGF
    if os.path.exists(usimgffile):
        logging.info('Creating temp MGF file with new TITLE...')

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
        # reformatmgffile_new = temp_dir+time.strftime("%Y%m%d%H%M%S")+'.mgf'
        mgf.write(spectra_new, output=reformatmgffile)
        spectra_origin.close()
    else:
        logging.error("The reformatted MGF file does not exist")

    logging.info('[STATUS] MGF file with new TITLE was created!')


# Contruct ML friendly spectra matrix for transformer full prediction
def generateHDF5_transformer_wSeq(usimgffile, reformatmgffile, csvfile,
                                  hdf5file, temp_dir, contrastcsvfile):
    # retrieve spectrum of PSM from MGF
    start_time = time.time()        # start time for parsing
    spectra = mgf.read(usimgffile)
    spectra[0]
    mzs_df = []

    index = 0
    for spectrum in spectra:
        index += 1
        if (index % 100 == 0):
            logging.info('Generating CSV Progress: {} records'.format(index))
        try:
            retention_time = spectrum['params']['rtinseconds']
            collision_energy = float(spectrum['params']['ce'])
            charge_state = int(spectrum['params']['charge'][0])
            method = spectrum['params']['method']
            mod_num = spectrum['params']['mod_num']
            raw_file = spectrum['params']['file']
            scan_number = spectrum['params']['scans']
            sequence = spectrum['params']['seq']
            score = 0
            modified_sequence = spectrum['params']['mod_seq']
            proforma = spectrum['params']['proforma']
            reverse = 'FALSE'

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

    mzs_df = pd.concat(mzs_df, axis=1)
    mzs_df = mzs_df.transpose()
    mzs_df.columns = ['raw_file', 'scan_number', 'sequence', 'score',
                      'modified_sequence', 'proforma',
                      'mod_num', 'reverse',
                      'collision_energy', 'precursor_charge',
                      'masses', 'intensities',
                      'retention_time', 'method']

    # construct CSV
    mzs_df = mzs_df.reset_index(drop=True)
    mzs_df = pd.concat([mzs_df], axis=1)
    mzs_df['precursor_charge'] = mzs_df['precursor_charge'].astype(np.uint8)
    mzs_df['collision_energy_aligned_normed'] = mzs_df['collision_energy']/100.0
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
        "collision_energy_aligned_normed": get_number(mzs_df['collision_energy_aligned_normed']),
        "intensities_raw": get_2darray(mzs_df['intensities']),
        "masses_pred": get_2darray(mzs_df['masses']),
        "precursor_charge_onehot": get_precursor_charge_onehot(mzs_df['precursor_charge']),
        "sequence_integer": get_sequence_integer(mzs_df['modified_sequence'])
    }

    return dataset


def main():

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
    mgffile = constants_location.MGF_PATH
    if (workflow == 'split' or workflow == 'split_usi'):
        logging.info('[INFO] Workflow: Splitting the dataset ...')        
        if (workflow == 'split'):
            mgffile = constants_location.MGF_PATH
            trainsetfile = constants_location.TRAINFILE
            testsetfile = constants_location.TESTFILE
        else:
            mgffile = constants_location.REFORMAT_USITITLE_PATH
            trainsetfile = constants_location.REFORMAT_TRAIN_USITITLE_PATH
            testsetfile = constants_location.REFORMAT_TEST_USITITLE_PATH
        # hold out N records as testset
        splitMGF(mgffile, trainsetfile, testsetfile, n_test=5000)
        logging.info(f'[STATUS] Workflow {workflow} ... DONE!')
    elif (workflow == 'train' or workflow == 'test'):
        if workflow == 'train':
            usimgffile = constants_location.REFORMAT_TRAIN_USITITLE_PATH
            reformatmgffile = constants_location.REFORMAT_TRAIN_PATH
            hdf5file = constants_location.TRAINDATA_PATH
            csvfile = traincsvfile
            datasetfile = constants_location.TRAINDATASET_PATH
        else:
            usimgffile = constants_location.REFORMAT_TEST_USITITLE_PATH
            reformatmgffile = constants_location.REFORMAT_TEST_PATH
            hdf5file = constants_location.TESTDATA_PATH
            csvfile = testcsvfile
            datasetfile = constants_location.TESTDATASET_PATH
        # generate CSV for train and test list
        if not os.path.exists(reformatmgffile):
            logging.info(f'Workflow: Annotating {workflow} set ...')        
            if (workflow == 'train'):
                dataset = generateHDF5_transformer_wSeq(usimgffile, reformatmgffile,
                                                csvfile, hdf5file, temp_dir, None)
            elif (workflow == 'test'):
                dataset = generateHDF5_transformer_wSeq(usimgffile, reformatmgffile,
                                                csvfile, hdf5file, temp_dir, traincsvfile)
            logging.info(f'Saving {workflow} set to HDF5 ...')
            io_local.to_hdf5(dataset, hdf5file)
            logging.info(f'Saving {workflow} set to Arrow chunks ...')
            io_local.to_arrow(dataset, datasetfile)
            logging.info(f'[STATUS] {workflow} set preparation ... DONE!')
        else:
            logging.error(f'{workflow} set is already existed')            
    elif (workflow == 'reformat'):
        logging.info(f'WORKFLOW: {workflow} ... ')
        usimgffile = constants_location.REFORMAT_USITITLE_PATH
        reformatMGF_wSeq(mgffile, usimgffile)
        logging.info(f'[STATUS] Workflow {workflow} ... DONE!')
    else:
        logging.error("Unknown workflow choice.")


if __name__ == "__main__":
    main()
