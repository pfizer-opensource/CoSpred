import re
from pyteomics import mzml
from pyteomics import mgf
import numpy as np
import pandas as pd
import os
import time
from argparse import ArgumentParser
import random
import copy

import params.constants_location as constants_location
from params.constants import (
    SPECTRA_DIMENSION, BIN_MAXMZ, BIN_SIZE,
    CHARGES,
    MAX_SEQUENCE,
    ALPHABET,
    AMINO_ACID,
    METHODS,
)

from preprocess import utils
from prosit_model import io_local
import io_cospred

COL_SEP = "\t"


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
    start_time = time.time()
    array = np.zeros([len(sequences), MAX_SEQUENCE])
    for i, sequence in enumerate(sequences):
        if len(sequence) > MAX_SEQUENCE:
            pass
        else:
            for j, s in enumerate(utils.peptide_parser(sequence)):
                array[i, j] = ALPHABET[s]
    array = array.astype(dtype)
    print('sequence interger: ' + str(time.time()-start_time))
    return array


def get_float(vals, dtype=np.float32):
    start_time = time.time()
    a = np.array(vals).astype(dtype)
    a = a.reshape([len(vals), 1])
    print('get float: ' + str(time.time()-start_time))
    return a


def get_boolean(vals, dtype=bool):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_number(vals, dtype='i1'):
    start_time = time.time()
    a = np.array(vals).astype(dtype)
    a = a.reshape([len(vals), 1])
    print('get number: ' + str(time.time()-start_time))
    return a


def get_2darray(vals, dtype=np.float32):
    start_time = time.time()
    # a = np.vstack(vals)
    a = np.array(vals.values.tolist())
    # a = a.reshape([len(vals), len(vals.iloc[0])])
    a = a.astype(dtype)
    print('2d array: ' + str(time.time()-start_time))
    return a


def get_precursor_charge_onehot(charges, dtype='i1'):
    start_time = time.time()
    array = np.zeros([len(charges), max(CHARGES)])
    for i, precursor_charge in enumerate(charges):
        if precursor_charge > max(CHARGES):
            pass
        else:
            array[i, int(precursor_charge) - 1] = 1
    array = array.astype(dtype)
    print('onehot assignment charge: ' + str(time.time()-start_time))
    return array


def get_method_onehot(methods):
    array = np.zeros([len(methods), len(METHODS)])
    for i, method in enumerate(methods):
        for j, methodstype in enumerate(METHODS):
            if method == methodstype:
                array[i, j] = int(1)
    return array


def get_sequence_onehot(sequences):
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
    return array


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
                print('spectrum index {} in testset'.format(i))
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
    print('Splitting MGF Progress DONE: total {} records'.format(i))

    spectra.close()
    return test_index_list


def getPSM(psmfile):
    target_cols = {"Annotated Sequence": "seq", "Modifications": "modifications",
                   "m/z [Da]": "mz", 'Charge': 'charge', 'RT [min]': 'retentiontime',
                   'Percolator PEP': 'score', 'Checked': 'reverse',
                   'First Scan': 'scan', 'Spectrum File': 'file'}
    target_cols.keys()
    dbsearch = pd.read_csv(
        psmfile, sep='\t', keep_default_na=False, na_values=['NaN'])
    dbsearch = dbsearch[dbsearch['Confidence'] == 'High'][target_cols.keys()]
    dbsearch = dbsearch.rename(columns=target_cols)

    # remove N, C terimal of flanking amino acids
    dbsearch['seq'] = dbsearch['seq'].str.replace(
        "\\[\\S+\\]\\.", '', regex=True)
    dbsearch['seq'] = dbsearch['seq'].str.replace(
        "\\.\\[\\S+\\]", '', regex=True)
    dbsearch['seq'] = dbsearch['seq'].str.upper()
    # remove sequence length > 30
    dbsearch = dbsearch[dbsearch['seq'].str.len() <= 30]

    # parse Modified Peptide
    seq_list = dbsearch['seq'].tolist()
    mod_list = dbsearch['modifications'].tolist()
    modseq_list = []
    proforma_list = []
    modnum_list = []
    for k in range(len(dbsearch)):
        letter = [x for x in seq_list[k]]
        modseq_list.append(letter)
        modnum_list.append(0)
    proforma_list = copy.deepcopy(modseq_list)
    # make sure later concatenate aligns
    dbsearch.reset_index(drop=True, inplace=True)

    # parse phospho
    targetmod = ''.join(["[", 'STY', "]", "[0-9]+\\(", 'Phospho', "\\)"])
    for k in range(len(dbsearch)):
        if (mod_list[k] != ''):
            matchMod = re.findall(targetmod, mod_list[k])  # locate mod site
            matchChr = [re.search("([STY])", x).group(0) for x in matchMod]
            matchDigit = [int(re.search("([0-9]+)", x).group(0))
                          for x in matchMod]
            for i in reversed(matchDigit):
                modseq_list[k][i-1] = modseq_list[k][i-1] + '(ph)'
                proforma_list[k][i-1] = proforma_list[k][i-1] + '[Phospho]'
                modnum_list[k] += 1

    # parse oxidation
    targetmod = ''.join(["[", 'M', "]", "[0-9]+\\(", 'Oxidation', "\\)"])
    for k in range(len(dbsearch)):
        if (mod_list[k] is not None):
            if (mod_list[k] != ''):
                matchMod = re.findall(
                    targetmod, mod_list[k])  # locate mod site
                # matchChr = [re.search("([M])", x).group(0) for x in matchMod]
                matchDigit = [int(re.search("([0-9]+)", x).group(0))
                              for x in matchMod]
                # test_str = tmp_row['seq']
                for i in reversed(matchDigit):
                    modseq_list[k][i-1] = modseq_list[k][i-1] + '(ox)'
                    proforma_list[k][i-1] = proforma_list[k][i -
                                                             1] + '[Oxidation]'
                    modnum_list[k] += 1

    dbsearch['modifiedseq'] = pd.Series([''.join(x) for x in modseq_list])
    dbsearch['proforma'] = pd.Series([''.join(x) for x in proforma_list])
    dbsearch['mod_num'] = pd.Series(modnum_list).astype(str)

    # reset index and recreate title for mzml matching
    dbsearch['title'] = 'mzspec:repoID:'+dbsearch['file'] + \
        ':scan:'+dbsearch['scan'].astype(str)
    return dbsearch


# Add TITLE, SEQ, CE to raw file MGF
def reformatMGF(mgffile, mzmlfile, dbsearch_df, reformatmgffile, temp_dir):
    f = mzml.MzML(mzmlfile)

    # Rewrite TITLE for the MGF
    print('Creating temp MGF file with new TITLE...')

    spectra_origin = mgf.read(mgffile)
    spectra_temp = []
    for spectrum in spectra_origin:
        # for MSconvert generated MGF
        title_split = spectrum['params']['title'].split(' ')
        repoid = re.sub('\W$', '', title_split[1].split('"')[1])
        scan_number = re.sub('\W+', '', title_split[0].split('.')[1])
        # # for PD generated MGF
        # title_split = spectrum['params']['title'].split(';')
        # repoid = re.sub("\W$",'',title_split[0].split('\\')[-1])
        # scan_number = re.sub('\W+','',title_split[-1].split('scans')[-1])
        # # spectrum['params']['title'] = ':'.join(['mzspec', 'repoID',
        # #                   re.sub("\W$",'',title_split[0].split('\\')[-1]),
        # #                   'scan',
        # #                   re.sub('\W+','',title_split[-1].split('scans')[-1])])        
        spectrum['params']['title'] = ':'.join(['mzspec', 'repoID',
                                                repoid, 'scan', scan_number])
        spectrum['params']['scans'] = scan_number
        spectra_temp.append(spectrum)
    reformatmgffile_temp = temp_dir+time.strftime("%Y%m%d%H%M%S")+'.mgf'
    mgf.write(spectra_temp, output=reformatmgffile_temp)
    spectra_origin.close()

    print('Temp MGF file with new TITLE was created!')

    # Add SEQ and CE to the reformatted MGF
    spectra = mgf.read(reformatmgffile_temp)
    for spectrum in spectra:
        pass
    spectra_new = []
    for index, row in dbsearch_df.iterrows():
        if (index % 100 == 0):
            print('Reformatting MGF Progress: {}%'.format(
                index/dbsearch_df.shape[0]*100))
        try:
            # retrieve spectrum of PSM from MGF and MZML
            spectrum = spectra.get_spectrum(row['title'])
            spectrum['params']['seq'] = row['modifiedseq']
            spectrum['params']['proforma'] = row['proforma']
            spectrum['params']['mod_num'] = str(row['mod_num'])

            controller_str = 'controllerType=0 controllerNumber=1 '
            p = f.get_by_id(controller_str + "scan=" +
                            str(spectrum['params']['scans']))
            fg = p.get('precursorList').get('precursor')
            spectrum['params']['pepmass'] = spectrum['params']['pepmass'][0]
            spectrum['params']['rtinseconds'] = str(
                spectrum['params']['rtinseconds'])
            spectrum['params']['ce'] = str(
                fg[0].get('activation').get('collision energy'))
            spectrum['params']['charge'] = re.sub(
                '\D+', '', str(fg[0].get('selectedIonList').get('selectedIon')[0].get('charge state')))
            filter_string = p.get('scanList').get('scan')[
                0].get('filter string')
            if re.search("hcd", filter_string):
                method = "HCD"
            if re.search("cid", filter_string):
                method = "CID"
            if re.search("etd", filter_string):
                method = "ETD"
            spectrum['params']['method'] = method

            spectra_new.append(spectrum)
        except:
            next

    mgf.write(spectra_new, output=reformatmgffile)
    f.close()
    spectra.close()

    if os.path.exists(reformatmgffile_temp):
        os.remove(reformatmgffile_temp)
    else:
        print("The temp reformatted MGF file does not exist")

    return spectra_new


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


def modifyMGFtitle(usimgffile, reformatmgffile):
    # Rewrite TITLE for the MGF
    if os.path.exists(usimgffile):
        print('Creating temp MGF file with new TITLE...')

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
        print("The reformatted MGF file does not exist")

    print('MGF file with new TITLE was created!')


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
    start_time = time.time()        # start time for parsing
    spectra = mgf.read(usimgffile)
    spectra[0]
    mzs_df = []

    for index, row in dbsearch_df.iterrows():
        if (index % 100 == 0):
            print('Generating CSV Progress: {}%'.format(
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

    print('generate list: ' + str(time.time()-start_time))
    mzs_df = pd.concat(mzs_df, axis=1).transpose()
    print('transpost: ' + str(time.time()-start_time))
    mzs_df.columns = ['raw_files', 'scan_number', 'sequence', 'score',
                      'modified_sequence', 'proforma',
                      'mod_num', 'reverse',
                      'collision_energy', 'precursor_charge',
                      'masses', 'intensities',
                      'retention_time', 'method']

    # construct CSV
    mzs_df = mzs_df.reset_index(drop=True)
    print('reset index: ' + str(time.time()-start_time))

    mzs_df = pd.concat([mzs_df], axis=1)
    print('pd.concat: ' + str(time.time()-start_time))

    mzs_df['precursor_charge'] = mzs_df['precursor_charge'].astype(np.uint8)
    mzs_df['collision_energy_aligned_normed'] = mzs_df['collision_energy']/100.0
    mzs_df.info()

    mzs_df = mzs_df.dropna()
    print('dropna: ' + str(time.time()-start_time))

    mzs_df.columns = mzs_df.columns.str.replace('[\r]', '')
    print('replace newline: ' + str(time.time()-start_time))

    # To prevent data leaking, only keep the peptides that are not in the contrast dataset
    if (contrastcsvfile is not None):
        constrast_dataset = pd.read_csv(contrastcsvfile, sep=',')
        mzs_df = mzs_df[~mzs_df['proforma'].isin(constrast_dataset['proforma'])]

    mzs_df.to_csv(csvfile, index=False)      # CSV discards values in large vec
    print('Write CSV: ' + str(time.time()-start_time))
    print('Generating CSV Done!')

    # construct Dataset based on CoSpred Transformer definition
    dataset = {
        "collision_energy_aligned_normed": get_number(mzs_df['collision_energy_aligned_normed']),
        "intensities_raw": get_2darray(mzs_df['intensities']),
        "masses_pred": get_2darray(mzs_df['masses']),
        "precursor_charge_onehot": get_precursor_charge_onehot(mzs_df['precursor_charge']),
        "sequence_integer": get_sequence_integer(mzs_df['modified_sequence'])
    }

    print('Assembling dataset dictionary: ' + str(time.time()-start_time))

    modifyMGFtitle(usimgffile, reformatmgffile)
    return dataset


def constructDataset_fullspectrum(csvfile, predict_csv):
    df = pd.read_csv(csvfile, sep=',', index_col=False)

    df = io_cospred.sanitizePeptide(df, predict_csv)

    assert "modified_sequence" in df.columns
    assert "collision_energy" in df.columns
    assert "precursor_charge" in df.columns
    assert "intensities" in df.columns
    assert "masses" in df.columns

    df.dropna(subset=['intensities', 'masses'], inplace=True)
    df.columns = df.columns.str.replace('[\r]', '')

    intensity_vec, mz_vec = constructCospredVec(
                df['masses'], df['intensities'])

    # construct Dataset based on Prosit definition
    dataset = {
        "collision_energy": get_float(df['collision_energy']),
        "collision_energy_aligned": get_float(df['collision_energy']),
        "collision_energy_aligned_normed": get_float(df['collision_energy']/100.0),
        "intensities_raw": intensity_vec,
        "masses_pred": mz_vec,
        "masses_raw": mz_vec,
        "method": get_method_onehot(df['method']).astype(int),
        "precursor_charge_onehot": get_precursor_charge_onehot(df['precursor_charge']).astype(int),
        "rawfile": df['raw_file'].astype('S32'),
        "reverse": get_boolean(df['reverse']),
        "scan_number": get_number(df['scan_number']),
        "score": get_float(df['score']),
        "sequence_integer": get_sequence_integer(df['modified_sequence']).astype(int),
        "sequence_onehot": get_sequence_onehot(df['modified_sequence']).astype(int),
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
    mzmlfile = constants_location.MZML_PATH
    psmfile = constants_location.PSM_PATH
    
    if (workflow == 'train'):
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
        print("Unknown workflow choice.")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # get psm result
    if (workflow != 'predict'):
        dbsearch = getPSM(psmfile)
        dbsearch_df = dbsearch

    if (workflow == 'split'):
        # hold out N records as testset
        splitMGF(mgffile, trainsetfile, testsetfile, n_test=5000)
        print('Splitting train vs test set Done!')
    # reformat the Spectra
    elif (workflow == 'train' or workflow == 'test'):
        if not os.path.isfile(usimgffile):
            reformatMGF(datasetfile, mzmlfile, dbsearch_df, usimgffile, temp_dir)
        # match peptide from PSM with spectra MGF to generate CSV with full spectra bins
        if not os.path.isfile(reformatmgffile):
            if (workflow == 'train'):
                dataset = generateHDF5_transformer(usimgffile, reformatmgffile, dbsearch_df,
                                                csvfile, None)
            elif (workflow == 'test'):
                dataset = generateHDF5_transformer(usimgffile, reformatmgffile, dbsearch_df,
                                                csvfile, traincsvfile)
        io_local.to_hdf5(dataset, hdf5file)
        print('Generating HDF5 Done!')
    elif (workflow == 'predict'):
        if not os.path.isfile(csvfile):
            print('No peptide list available!')
        else:
            # transform to hdf5
            dataset = io_cospred.constructDataset_frompep(csvfile, predict_csv)
            io_cospred.to_hdf5(dataset, hdf5file)
            # check generated hdf
            io_cospred.read_hdf5(hdf5file)
            print('Generating HDF5 ... DONE!')
    else:
        print("Unknown workflow choice.")


if __name__ == "__main__":
    main()
