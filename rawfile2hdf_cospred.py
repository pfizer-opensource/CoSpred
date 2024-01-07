import re
import sys, getopt
from pyteomics import mzml
from pyteomics import mgf
import numpy as np
import spectrum_utils.spectrum as sus
import pandas as pd
import os
import h5py
import time
from argparse import ArgumentParser
import random
import copy
import math

# try: 
#     os.chdir('/Users/xuel12/Documents/Projects/seq2spec/CoSpred/')
#     print("Current directory is {}".format(os.getcwd()))
# except: 
#     print("Something wrong with specified directory. Exception- ", sys.exc_info())
  
import params.constants_local as constants_local
import params.constants_gcp as constants_gcp
from params.constants import (
    SPECTRA_DIMENSION, BIN_MAXMZ, BIN_SIZE,
    CHARGES,
    DEFAULT_MAX_CHARGE,
    MAX_SEQUENCE,
    ALPHABET,
    ALPHABET_S,
    FIXMOD_PROFORMA,
    VARMOD_PROFORMA,
    # MAX_ION,
    NLOSSES,
    ION_TYPES,
    # ION_OFFSET,
    METHODS,
)
from preprocess import utils, annotate, match
from prosit_model import io_local

COL_SEP = "\t"


def peptide_parser(p):
    p = p.replace("_", "")
    if p[0] == "(":
        raise ValueError("sequence starts with '('")
    n = len(p)
    i = 0
    while i < n:
        if i < n - 3 and p[i + 1] == "(":
            j = p[i + 2 :].index(")")
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
                # # POC: uppercase all amino acid, so no PTM
                # array[i, j] = ALPHABET[s.upper()]
                # #
                array[i, j] = ALPHABET[s]
    array = array.astype(dtype)    
    print('sequence interger: ' + str(time.time()-start_time))
    return array


# def get_sequence_integer(sequences):
#     # sequences = df.sequence
#     array = np.zeros([len(sequences), MAX_SEQUENCE])
#     for i, sequence in enumerate(sequences):
#         if len(sequence) > MAX_SEQUENCE:
#             pass
#         else:
#             for j, s in enumerate(utils.peptide_parser(sequence)):
#                 # # POC: uppercase all amino acid, so no PTM
#                 # array[i, j] = ALPHABET[s.upper()]
#                 # #
#                 array[i, j] = ALPHABET[s]
#     return array


def get_float(vals, dtype=np.float32):
    start_time = time.time()
    a = np.array(vals).astype(dtype)
    a = a.reshape([len(vals), 1])
    print('get float: ' + str(time.time()-start_time))
    return a


# def get_float(vals, dtype=float):
#     a = np.array(vals).astype(dtype)
#     return a.reshape([len(vals), 1])


def get_boolean(vals, dtype=bool):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_number(vals, dtype='i1'):
    start_time = time.time()
    a = np.array(vals).astype(dtype)
    a = a.reshape([len(vals), 1])
    print('get number: ' + str(time.time()-start_time))
    return a


# def get_number(vals, dtype=int):
#     a = np.array(vals).astype(dtype)
#     return a.reshape([len(vals), 1])


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

# def get_precursor_charge_onehot(charges):
#     array = np.zeros([len(charges), max(CHARGES)])
#     for i, precursor_charge in enumerate(charges):
#         if precursor_charge > max(CHARGES):
#             pass
#         else:
#             array[i, int(precursor_charge) - 1] = 1
#     return array

def get_method_onehot(methods):
    array = np.zeros([len(methods), len(METHODS)])
    for i, method in enumerate(methods):
        for j, methodstype in enumerate(METHODS):
            if method == methodstype:
                array[i,j]=int(1)
    return array

def get_sequence_onehot(sequences):
    array = np.zeros([len(sequences), MAX_SEQUENCE, len(ALPHABET)+1])
    for i, sequence in enumerate(sequences):
        j = 0
        for aa in peptide_parser(p=sequence):
            if aa in ALPHABET.keys():
                array[i,j,ALPHABET[aa]]=int(1) 
            j += 1
        while j < MAX_SEQUENCE:
            array[i,j,0]=int(1)
            j += 1
    return array


def splitMGF(mgffile, trainsetfile, testsetfile, n_test = 1000):
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    spectra=mgf.read(mgffile)
    spectra_train = []
    spectra_test = []
        
    # initiate the file writing
    mgf.write(spectra_test, output = testsetfile, file_mode='w')
    mgf.write(spectra_train, output = trainsetfile, file_mode='w')

    test_index = sorted(random.sample(range(0, len(spectra)), n_test))
    test_index_list = []
    i = 0
    for spectrum in spectra:
        if i in test_index:
            spectra_test.append(spectrum)
            test_index_list.append(test_index.pop(0))
            if (len(spectra_test) % 100 == 0):
                # append a chunk of spectra to new MGF
                mgf.write(spectra_test, output = testsetfile, file_mode='a')
                spectra_test = []
                print('spectrum index {} in testset'.format(i))
        else:
            spectra_train.append(spectrum)
            if (len(spectra_train) % 1000 == 0):
                # append a chunk of spectra to new MGF
                mgf.write(spectra_train, output = trainsetfile, file_mode='a')
                spectra_train = []
        i += 1
    if (len(spectra_test) > 0):
        mgf.write(spectra_test, output = testsetfile, file_mode='a')
    if (len(spectra_train) > 0):
        mgf.write(spectra_train, output = trainsetfile, file_mode='a')
    print('Splitting MGF Progress DONE: total {} records'.format(i))
    
    spectra.close()
    # mgf.write(spectra_test, output = testsetfile)
    # mgf.write(spectra_train, output = trainsetfile)
    return test_index_list


# def splitMGF(mgffile, trainsetfile, testsetfile, n_test = 1000):
#     # fix random seed for reproducibility
#     seed = 42
#     np.random.seed(seed)
    
#     spectra=mgf.read(mgffile)
#     spectra_train = []
#     spectra_test = []
#     test_index = sorted(random.sample(range(0, len(spectra)), n_test))
#     test_index_list = []
#     i = 0
#     for spectrum in spectra:
#         if i in test_index:
#             spectra_test.append(spectrum)
#             test_index_list.append(test_index.pop(0))
#             print('spectrum index {} in testset'.format(i))
#         else:
#             spectra_train.append(spectrum)
#         i += 1
#     mgf.write(spectra_test, output = testsetfile)
#     mgf.write(spectra_train, output = trainsetfile)
#     return test_index_list
    

def getPSM(psmfile):
    target_cols = {"Annotated Sequence":"seq", "Modifications":"modifications",
                   "m/z [Da]": "mz",'Charge':'charge','RT [min]':'retentiontime',
                   'Percolator PEP':'score','Checked':'reverse',
                   'First Scan':'scan','Spectrum File':'file'}
    target_cols.keys()
    dbsearch = pd.read_csv(psmfile, sep = '\t', keep_default_na=False,na_values=['NaN'])
    dbsearch = dbsearch[dbsearch['Confidence']=='High'][target_cols.keys()]
    dbsearch = dbsearch.rename(columns=target_cols)
    # dbsearch['mod_num'] =  (dbsearch['modifications'].str.count(';')+1).fillna(0).astype(int)
    
    # modfile = 'data/phospho/phospho_sample1_ModificationSites.txt'
    # mod_df = pd.read_csv(modfile, sep = '\t')
    
    # remove N, C terimal of flanking amino acids
    dbsearch['seq'] = dbsearch['seq'].str.replace("\\[\\S+\\]\\.",'', regex=True)
    dbsearch['seq'] = dbsearch['seq'].str.replace("\\.\\[\\S+\\]",'', regex=True)
    dbsearch['seq'] = dbsearch['seq'].str.upper()

    # dbsearch['modifiedseq'] = dbsearch['seq'].str.upper()
    # dbsearch['seq'] = dbsearch['seq'].str.upper()
    dbsearch = dbsearch[dbsearch['seq'].str.len()<=30]      # remove sequence length > 30

    # parse Modified Peptide    
    # dbsearch['modifiedseq'] = dbsearch['seq']
    # dbsearch['modifiedseq'].tolist()
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
    dbsearch.reset_index(drop=True, inplace=True)       # make sure later concatenate aligns

    # parse phospho
    targetmod = ''.join(["[",'STY',"]","[0-9]+\\(",'Phospho',"\\)"])
    for k in range(len(dbsearch)):
        # if (mod_list[k] is not None):
        #     pass
        # else:
        #     print(k)
        if (mod_list[k] != ''):
            matchMod = re.findall(targetmod, mod_list[k])     #locate mod site
            matchChr = [re.search("([STY])",x).group(0) for x in matchMod]
            matchDigit = [int(re.search("([0-9]+)",x).group(0)) for x in matchMod]     
            # test_str = tmp_row['seq']
            for i in reversed(matchDigit):
                modseq_list[k][i-1] = modseq_list[k][i-1] + '(ph)'
                proforma_list[k][i-1] = proforma_list[k][i-1] + '[Phospho]'
                modnum_list[k] += 1
    
    # parse oxidation 
    targetmod = ''.join(["[",'M',"]","[0-9]+\\(",'Oxidation',"\\)"])
    for k in range(len(dbsearch)):
        if (mod_list[k] is not None):
            if (mod_list[k] != ''):
                matchMod = re.findall(targetmod, mod_list[k])     #locate mod site
                matchChr = [re.search("([M])",x).group(0) for x in matchMod]
                matchDigit = [int(re.search("([0-9]+)",x).group(0)) for x in matchMod]
                # test_str = tmp_row['seq']
                for i in reversed(matchDigit):
                    modseq_list[k][i-1] = modseq_list[k][i-1] + '(ox)'
                    proforma_list[k][i-1] = proforma_list[k][i-1] + '[Oxidation]'
                    modnum_list[k] += 1

    dbsearch['modifiedseq'] = pd.Series([''.join(x) for x in modseq_list])
    dbsearch['proforma'] = pd.Series([''.join(x) for x in proforma_list])
    dbsearch['mod_num'] = pd.Series(modnum_list).astype(str)

    # reset index and recreate title for mzml matching
    dbsearch['title'] = 'mzspec:repoID:'+dbsearch['file']+':scan:'+dbsearch['scan'].astype(str)      
    return dbsearch

def constructCospredVec(mz_arr, intensity_arr):
    intensity_arr = intensity_arr / np.max(intensity_arr)
    vector_intensity = np.zeros(SPECTRA_DIMENSION, dtype=np.float32)
    # vector_intensity.dtype
    # vector_mass = np.zeros(SPECTRA_DIMENSION, dtype='float')
    vector_mass = np.arange(0, BIN_MAXMZ, BIN_SIZE, dtype=np.float32)
    # vector_mass.dtype
    # vector_count = np.zeros(SPECTRA_DIMENSION, dtype='int16')
    
    index_arr = mz_arr / BIN_SIZE
    index_arr = np.around(index_arr).astype('int16')

    for i, index in enumerate(index_arr):
        if (index > SPECTRA_DIMENSION - 1):         # add intensity to last bin for high m/z
            vector_intensity[-1] += intensity_arr[i]
            # vector_mass[-1] += mz_arr[i]
            # vector_count[-1] += 1
        else:
            vector_intensity[index] += intensity_arr[i]
            # vector_mass[index] += mz_arr[i]
            # vector_count[index] += 1
    # vector_count[vector_count == 0] = int(1)
    # vector_mass = vector_mass/vector_count
    return vector_intensity, vector_mass

# def constructCospredVec(mz_arr, intensity_arr):
#     intensity_arr = intensity_arr / np.max(intensity_arr)
#     vector_intensity = np.zeros(SPECTRA_DIMENSION, dtype='float')
#     # vector_mass = np.zeros(SPECTRA_DIMENSION, dtype='float')
#     vector_mass = np.arange(0, BIN_MAXMZ, BIN_SIZE)
#     # vector_count = np.zeros(SPECTRA_DIMENSION, dtype='int16')
    
#     index_arr = mz_arr / BIN_SIZE
#     index_arr = np.around(index_arr).astype('int16')

#     for i, index in enumerate(index_arr):
#         if (index > SPECTRA_DIMENSION - 1):         # add intensity to last bin for high m/z
#             vector_intensity[-1] += intensity_arr[i]
#             # vector_mass[-1] += mz_arr[i]
#             # vector_count[-1] += 1
#         else:
#             vector_intensity[index] += intensity_arr[i]
#             # vector_mass[index] += mz_arr[i]
#             # vector_count[index] += 1
#     # vector_count[vector_count == 0] = int(1)
#     # vector_mass = vector_mass/vector_count
#     return vector_intensity, vector_mass


def modifyMGFtitle(usimgffile, reformatmgffile):
    # Rewrite TITLE for the MGF
    if os.path.exists(usimgffile):
        print('Creating temp MGF file with new TITLE...')
        
        spectra_origin=mgf.read(usimgffile)
        spectra_new = []
        for spectrum in spectra_origin:
            peptide = spectrum['params']['seq']
            ce = spectrum['params']['ce']
            mod_num = str(spectrum['params']['mod_num'])
            charge = re.sub('\D+','', str(spectrum['params']['charge'][0]))
            # To facilitate Spectrum predicition evaluation, convert title format from USI to seq/charge_ce_0
            spectrum['params']['title'] = peptide+ '/' + charge + '_' + ce + '_' + mod_num
            spectra_new.append(spectrum)
        # reformatmgffile_new = temp_dir+time.strftime("%Y%m%d%H%M%S")+'.mgf'
        mgf.write(spectra_new, output = reformatmgffile)
        spectra_origin.close()

        # os.remove(reformatmgffile)
        # os.rename(reformatmgffile_new, reformatmgffile)
    else:
        print("The reformatted MGF file does not exist")
        
    print('MGF file with new TITLE was created!')
    
    
## Contruct ML friendly spectra matrix for transformer full prediction
def generateHDF5_transformer(usimgffile, reformatmgffile, dbsearch_df, csvfile):
    # # debug
    # df = csvfile.copy()
    # dbsearch['Modifiedsequence'] = dbsearch.Sequence
    # #
    
    assert "file" in dbsearch_df.columns
    assert "scan" in dbsearch_df.columns
    assert "charge" in dbsearch_df.columns
    assert "seq" in dbsearch_df.columns
    assert "modifiedseq" in dbsearch_df.columns
    assert "proforma" in dbsearch_df.columns
    assert "score" in dbsearch_df.columns
    assert "reverse" in dbsearch_df.columns
    
    # # get annotation MS2
    # annotation_results.columns = ['seq','intensities','masses','matches_raw']
    # ## DEBUG
    # annotation_results.to_pickle('annotation_result.pkl')
    # annotation_results = pd.read_pickle("annotation_result.pkl") 
    # ##
    
    # retrieve spectrum of PSM from MGF
    start_time = time.time()        # start time for parsing
    spectra=mgf.read(usimgffile)
    spectra[0]
    mzs_df = []
    
    for index, row in dbsearch_df.iterrows():  
        if (index % 100 == 0):
            print('Generating CSV Progress: {}%'.format(index/dbsearch_df.shape[0]*100))
                
        try:
            spectrum = spectra.get_spectrum(row['title'])    # title format:'mzspec:repoID:phospho_sample1.raw:scan:9'            
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
            intensity_vec, mz_vec = constructCospredVec(spectrum['m/z array'],spectrum['intensity array'])
            masses = mz_vec
            intensities = intensity_vec
            
            # "masses_pred": constructPrositVec(df, vectype='masses'),
            # mz_arr.append(spectrum['m/z array'])
            # intensity_arr.append(spectrum['intensity array'])
            #massbin = calc_mass_int_from_mzmlfile(mz_arr,intensity_arr)
            # massbin.append(spectrum2vector(spectrum['m/z array'],spectrum['intensity array']))
            # print(massbin[500:600])
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
    mzs_df = pd.concat(mzs_df, axis = 1).transpose()
    print('transpost: ' + str(time.time()-start_time))
    mzs_df.columns =['raw_files', 'scan_number', 'sequence', 'score' , 
                             'modified_sequence', 'proforma', 
                             'mod_num', 'reverse',
                             'collision_energy','precursor_charge',  
                              'masses', 'intensities', 
                             'retention_time', 'method']

   # construct CSV
    mzs_df = mzs_df.reset_index(drop=True)
    print('reset index: ' + str(time.time()-start_time))

    # # No need for transformer complete prediction
    # annotation_results_new = annotation_results.reset_index(drop=True)
    # dataset = pd.concat([mzs_df_new, annotation_results_new], axis = 1)
    # #
    
    mzs_df = pd.concat([mzs_df], axis = 1)
    print('pd.concat: ' + str(time.time()-start_time))
    
    mzs_df['precursor_charge'] = mzs_df['precursor_charge'].astype(np.uint8)        
    mzs_df['collision_energy_aligned_normed'] = mzs_df['collision_energy']/100.0
    # mzs_df['collision_energy_aligned_normed'] = mzs_df['collision_energy_aligned_normed'].astype(np.uint8)
    mzs_df.info()
    
    mzs_df = mzs_df.dropna()
    print('dropna: ' + str(time.time()-start_time))

    mzs_df.columns = mzs_df.columns.str.replace('[\r]', '')
    print('replace newline: ' + str(time.time()-start_time))
    
    # # Obsolete: Can't recover to PD consistently anyway
    # start_time = time.time()        # start time for the training
    # mzs_df.to_json(jsonfile, index=False)      # to Json format for future use
    # # mzs_df = pd.read_json(hdf5file) 
    # print('Write JSON: ' + str(time.time()-start_time))
    # #
    
    # # to HDF5 format for future use
    # hdf5file = '/Users/xuel12/Documents/Projects/seq2spec/CoSpred/data/massiveKBv2synthetic/test_pd.hdf5'
    # # mzs_df.columns
    # # mzs_df.to_hdf(hdf5file, key = 'mzs_df', index=False)      
    # # mzs_df = pd.read_hdf(hdf5file, 'mzs_df') 
    # print('Write HDF5: ' + str(time.time()-start_time))
    # #
    
    mzs_df.to_csv(csvfile, index=False)      # CSV discards values in large vec
    print('Write CSV: ' + str(time.time()-start_time))
    print('Generating CSV Done!')
    
    # construct Dataset based on CoSpred Transformer definition
    dataset = {
        # "collision_energy": get_float(mzs_df['collision_energy']),
        # "collision_energy_aligned": get_float(mzs_df['collision_energy']),
        "collision_energy_aligned_normed":get_number(mzs_df['collision_energy_aligned_normed']),
        "intensities_raw": get_2darray(mzs_df['intensities']),
        "masses_pred": get_2darray(mzs_df['masses']),
        # "masses_raw": np.vstack(mzs_df['masses']),
        # "method": get_method_onehot(mzs_df['method']).astype(int),
        "precursor_charge_onehot":get_precursor_charge_onehot(mzs_df['precursor_charge']),
        # "precursor_charge":get_number(mzs_df['precursor_charge']),
        # "rawfile": mzs_df['raw_file'].astype('S32'),
        # "reverse": get_boolean(mzs_df['reverse']),
        # "scan_number": get_number(mzs_df['scan_number']),
        # "score": get_float(mzs_df['score']),
        "sequence_integer": get_sequence_integer(mzs_df['modified_sequence'])
        # "sequence_onehot": get_sequence_onehot(mzs_df['modified_sequence']).astype(int),
    }
        
    # # OPTIONAL: when train_x doesn't need hierachy
    # dataset_df = []
    # for feature in dataset.keys():
    #     if feature != 'masses_pred':
    #         dataset_df.append(pd.DataFrame(dataset[feature]))
    # dataset_df = pd.concat(dataset_df, axis = 1)
    # dataset_df.columns = range(dataset_df.shape[1])
    # dataset_df.info()
    # dataset_df.to_hdf(hdf5file, key = 'train', format = 'table')
    print('Assembling dataset dictionary: ' + str(time.time()-start_time))

    # # construct CSV
    # mzs_df_new = mzs_df.reset_index(drop=True)
    
    # # # No need for transformer complete prediction
    # # annotation_results_new = annotation_results.reset_index(drop=True)
    # # dataset = pd.concat([mzs_df_new, annotation_results_new], axis = 1)
    # # #
    
    # df = pd.concat([mzs_df_new], axis = 1)
    
    # df = df.dropna()
    # # df.iloc[0]['masses'][0].shape
    
    # df.columns = df.columns.str.replace('[\r]', '')

    # df.to_csv(csvfile, index=False)      # CSV discards values in large vec
    # print('Generating CSV Done!')
    
    # # construct Dataset based on CoSpred Transformer definition
    # dataset = {
    #     "collision_energy": get_float(df['collision_energy']),
    #     "collision_energy_aligned": get_float(df['collision_energy']),
    #     "collision_energy_aligned_normed":get_float(df['collision_energy']/100.0),
    #     "intensities_raw": np.vstack(df['intensities']),
    #     "masses_pred": np.vstack(df['masses']),
    #     "masses_raw": np.vstack(df['masses']),
    #     "method": get_method_onehot(df['method']).astype(int),
    #     "precursor_charge_onehot":get_precursor_charge_onehot(df['precursor_charge']).astype(int),
    #     "rawfile": df['raw_files'].astype('S32'),
    #     "reverse": get_boolean(df['reverse']),
    #     "scan_number": get_number(df['scan_number']),
    #     "score": get_float(df['score']),
    #     "sequence_integer": get_sequence_integer(df['modified_sequence']).astype(int),
    #     "sequence_onehot": get_sequence_onehot(df['modified_sequence']).astype(int),
    # }

    modifyMGFtitle(usimgffile, reformatmgffile)
    return dataset
        

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

            
# def to_hdf5(dictionary, path):
#     dt = h5py.string_dtype(encoding='utf-8')

#     with h5py.File(path, "w") as f:
#         for key, data in dictionary.items():
#             # f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
#             if (data.dtype == 'object'):
#                 f.create_dataset(key, data=data, dtype=dt, compression="gzip")
#             else:
#                 f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
     

def main():
    
    parser = ArgumentParser()
    # parser.add_argument('-m', '--mgffile', default=constants_local.MGF_PATH, help='raw file MGF')
    # parser.add_argument('-r', '--reformatmgffile', default=constants_local.REFORMAT_TRAIN_PATH, help='reformat MGF')
    # parser.add_argument('-l', '--mzmlfile', default=constants_local.MZML_PATH, help='raw file mzML')
    parser.add_argument('-l', '--local', default=False, action='store_true',
                    help='execute in local computer')
    # parser.add_argument('-p', '--psmfile', default=constants_local.PSM_PATH, help='PSM file')
    # parser.add_argument('-c', '--csvfile', default=constants_local.CSV_PATH, help='csv file for ML')
    parser.add_argument('-w', '--workflow', default='test', help='workflow to use')
    args = parser.parse_args()    

    workflow = args.workflow

    if args.local is True:
        # basedir = constants_local.BASE_PATH
        # datadir = constants_local.DATA_DIR
        temp_dir = constants_local.TEMP_DIR   
        trainsetfile = constants_local.TRAINFILE
        testsetfile = constants_local.TESTFILE
        traincsvfile = constants_local.TRAINCSV_PATH
        testcsvfile = constants_local.TESTCSV_PATH
        mgffile = constants_local.MGF_PATH
        psmfile = constants_local.PSM_PATH
        if (workflow == 'split'):
            # hold out N records as testset
            splitMGF(mgffile, trainsetfile, testsetfile, n_test = 1000)
        elif (workflow == 'train'):
            usimgffile = constants_local.REFORMAT_TRAIN_USITITLE_PATH
            reformatmgffile = constants_local.REFORMAT_TRAIN_PATH
            hdf5file = constants_local.TRAINDATA_PATH
            datasetfile = trainsetfile
            csvfile = traincsvfile
        elif (workflow == 'test'):
            usimgffile = constants_local.REFORMAT_TEST_USITITLE_PATH
            reformatmgffile = constants_local.REFORMAT_TEST_PATH
            hdf5file = constants_local.TESTDATA_PATH
            datasetfile = testsetfile
            csvfile = testcsvfile
        else:
            print("Unknown workflow choice.")        
    else:
        # basedir = constants_gcp.BASE_PATH
        # datadir = constants_gcp.DATA_DIR
        temp_dir = constants_gcp.TEMP_DIR   
        trainsetfile = constants_gcp.TRAINFILE
        testsetfile = constants_gcp.TESTFILE
        traincsvfile = constants_gcp.TRAINCSV_PATH
        testcsvfile = constants_gcp.TESTCSV_PATH
        mgffile = constants_gcp.MGF_PATH
        psmfile = constants_gcp.PSM_PATH
        if (workflow == 'split'):
            # hold out N records as testset
            splitMGF(mgffile, trainsetfile, testsetfile, n_test = 1000)
        elif (workflow == 'train'):
            usimgffile = constants_gcp.REFORMAT_TRAIN_USITITLE_PATH
            reformatmgffile = constants_gcp.REFORMAT_TRAIN_PATH
            hdf5file = constants_gcp.TRAINDATA_PATH
            datasetfile = trainsetfile
            csvfile = traincsvfile
        elif (workflow == 'test'):
            usimgffile = constants_gcp.REFORMAT_TEST_USITITLE_PATH
            reformatmgffile = constants_gcp.REFORMAT_TEST_PATH
            hdf5file = constants_gcp.TESTDATA_PATH
            datasetfile = testsetfile
            csvfile = testcsvfile
        else:
            print("Unknown workflow choice.")        
            
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # get psm result
    dbsearch = getPSM(psmfile)
    dbsearch_df = dbsearch
    # dbsearch_df = dbsearch.iloc[:1000]
    
    # ## NO NEED OF ANNOTATION FOR TRANSFORMER
    # # reformat the Spectra
    # if not os.path.isfile(usimgffile):
    #     reformatMGF(datasetfile, mzmlfile, dbsearch_df, usimgffile, temp_dir)
    #     annotation_results = annotateMGF(usimgffile, dbsearch_df, temp_dir)
    # else:
    #     annotation_results = pd.read_csv(temp_dir+'annotatedMGF.csv') 
    # ##
    
    # match peptide from PSM with spectra MGF to generate CSV with full spectra bins
    # print(usimgffile)
    # print(reformatmgffile)
    # print(dbsearch_df)
    # print(csvfile)
    # print(temp_dir)
    dataset = generateHDF5_transformer(usimgffile, reformatmgffile, dbsearch_df, 
                                       csvfile)
    io_local.to_hdf5(dataset, hdf5file)        
    # to_hdf5(dataset, hdf5file)        
    print('Generating HDF5 Done!')

    
    # ## Sanity check Dataset stored in hdf5
    # # examine inputs
    # example = h5py.File(constants_local.EXAMPLE_DIR + 'traintest_hcd_100.hdf5', 'r')
    # for i in example.keys():
    #     print(example[i][:3])
    # example['masses_raw'][:10]
    # example['sequence_integer'][0]
    # example['sequence_onehot'][0].shape
    # example.close()


    # dataset_new = h5py.File(hdf5file, 'r')
    # dataset_new.keys()
    # for i in dataset_new.keys():
    #     print(dataset_new[i][:3])
        
    # example.keys()
    # example['sequence_onehot']    
    
    # dataset_new['sequence_onehot']
    # i = 'sequence_onehot'
    # for i in dataset_new.keys():
    #     # print(dataset_new[i][0])
    #     print(dataset_new[i][0].shape)
    #     print(example[i][0])
    #     print(example[i][0].shape)
    # len(dataset_new['sequence_integer'])
    # print(dataset_new['sequence_integer'][39])
    # dataset_new.close()
    
    
    
        
    # dfy = readmzml(mzmlfile, csvfile)
    # Rawfile = dfy[0][:].tolist()
    # Scannumber = dfy[1][:].tolist()
    # Sequence = dfy[2][:].tolist()
    # Score = dfy[3][:].tolist()
    # Reverse = dfy[4][:].tolist()
    # collision_energy = dfy[5][:].tolist()
    # charge_state = dfy[6][:].tolist()
    # retention_time = dfy[7][:].tolist()
    # method = dfy[8][:].tolist()
    # massbin = dfy[9][:].tolist()
    # dataset = pd.DataFrame(
    #     {"Rawfile": Rawfile, "Scan_number": Scannumber, "sequence_integer": Sequence, "Score": Score, "Reverse": Reverse,
    #      "collision_energy_aligned_normed": collision_energy, "precursor_charge_onehot": charge_state, "retention_time": retention_time,
    #      "method": method, "intensities_raw": massbin})

    # dataset.to_csv(outputfile, index=False)


if __name__ == "__main__":
    main()
