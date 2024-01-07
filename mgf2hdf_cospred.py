import re
# import sys, getopt
# from pyteomics import mzml
from pyteomics import mgf
import numpy as np
# import spectrum_utils.spectrum as sus
import pandas as pd
import os
# import h5py
import time
from argparse import ArgumentParser
import random
# import copy
# import math
import shutil
# import json
# from datasets import Dataset, load_dataset, load_from_disk

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

# COL_SEP = "\t"

# def asnp(x): return np.asarray(x)
# def asnp32(x): return np.asarray(x, dtype='float32')
# def peptide_parser(p):
#     p = p.replace("_", "")
#     if p[0] == "(":
#         raise ValueError("sequence starts with '('")
#     n = len(p)
#     i = 0
#     while i < n:
#         if i < n - 3 and p[i + 1] == "(":
#             j = p[i + 2 :].index(")")
#             offset = i + j + 3
#             yield p[i:offset]
#             i = offset
#         else:
#             yield p[i]
#             i += 1


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


# def readmgf(fn):
#     file = open(fn, "r")
#     data = mgf.read(file, convert_arrays=1, read_charges=False,
#                     dtype='float32', use_index=False)

#     codes = parse_spectra(data)
#     file.close()
#     return codes

# def calc_mass_int_from_mzmlfile(massarr,intensityarr):
#     massbin=dict()
#     lstmasses=massarr.tolist()
#     lstintensities=intensityarr.tolist()
#     massbin_vals=[]
#     for x in range(SPECTRA_DIMENSION):
#         massbin[x] = 0.0
#     for index, ion in enumerate(lstmasses):
#         if math.floor(float(ion)) in massbin.keys():
#             massbin[math.floor(float(ion))] = float(lstintensities[index])
#     values_list = list(massbin.values())
#     for i in values_list:
#         if float(i) < 0.03 * max(values_list):
#             values = 0
#         else:
#             values = float(i)/max(values_list)
#         massbin_vals.append(values)
#     return massbin_vals


# def spectrum2vector(mz_arr, intensity_arr, mass, bin_size, charge):
#     intensity_arr = intensity_arr / np.max(intensity_arr)
#     vector = np.zeros(SPECTRA_DIMENSION, dtype='float16')
#     # mz_list = np.asarray(mz_list)
#     index_arr = mz_arr / BIN_SIZE
#     index_arr = np.around(index_arr).astype('int16')

#     for i, index in enumerate(index_arr):
#         if (index > SPECTRA_DIMENSION - 1):         # add intensity to last bin for high m/z
#             vector[-1] += intensity_arr[i]
#         else:
#             vector[index] += intensity_arr[i]
    
#     # # normalize (WHY TO NORMALIZE and remove precursor ???)
#     # vector = np.sqrt(vector)
#     # # remove precursors, including isotropic peaks
#     # for delta in (0, 1, 2):
#     #     precursor_mz = mass + delta / charge
#     #     if precursor_mz > 0 and precursor_mz < 2000:
#     #         vector[round(precursor_mz / bin_size)] = 0
#     return vector


# def constructDataset(csvfile):
#     #df = pd.read_csv('C:/Users/tiwars46/PycharmProjects/prosit_PfizerRD/finaldatafolder/csv_for_prosit_training/val_100000.csv',sep=',')
#     # df = pd.read_csv('C:/Users/tiwars46/PycharmProjects/data/SEARCH_Ymod_Phospho/Yph_mod_prosit.csv',sep=',')
#     # df = pd.read_csv('/Users/xuel12/Documents/Projects/seq2spec/CosPred_dataset/Preprocessing_dataset_from_rawfiles/01625b_GA1-TUM_first_pool_1_01_01-DDA-1h-R2_mod.csv',sep=',')
#     df = pd.read_csv(csvfile,sep=',')

#     df['massbin'].iloc[0]
    
#     assert "modified_sequence" in df.columns
#     assert "collision_energy" in df.columns
#     assert "precursor_charge" in df.columns
#     assert "intensities" in df.columns
#     assert "masses" in df.columns
    
#     # ## DEBUG: Ignore modification on the peptide
#     # df.modified_sequence = df.sequence
#     # ##
    
#     df.dropna(subset=['intensities','matches_raw'],inplace=True)
#     # df.dropna(subset=['matches_raw'],inplace=True)
#     # df.dropna(subset=['matches_raw'], inplace=True)
#     df.columns = df.columns.str.replace('[\r]', '')
#     #df['reverse'] = df['reverse'].str.replace('[\r]', '')

#     # construct Dataset based on Prosit definition
#     dataset = {
#         "collision_energy": get_float(df['collision_energy']),
#         "collision_energy_aligned": get_float(df['collision_energy']),
#         #"collision_energy_aligned_normed": np.asarray([float(i[0]) for i in df.collision_energy_aligned_normed.map(lambda x: ast.literal_eval(x))]),
#         "collision_energy_aligned_normed":get_float(df['collision_energy']/100.0),
#         "intensities_raw": constructPrositVec(df, 'intensities'),
#         "masses_pred": constructPrositVec(df, vectype='masses'),
#         "masses_raw": constructPrositVec(df, vectype='masses'),
#         "method": get_method_onehot(df['method']).astype(int),
#         "precursor_charge_onehot":get_precursor_charge_onehot(df['precursor_charge']).astype(int),
#         # "rawfile": get_string(df['raw_files']),
#         "rawfile": df['raw_files'],
#         "reverse": get_boolean(df['reverse']),
#         "scan_number": get_number(df['scan_number']),
#         "score": get_float(df['score']),
#         "sequence_integer": get_sequence_integer(df['modified_sequence']).astype(int),
#         "sequence_onehot": get_sequence_onehot(df['modified_sequence']).astype(int),
#     }

#     # dataset = pd.DataFrame({
#     #     "raw_files": dbsearch_df.file,
#     #     "scan_number": get_number(dbsearch_df.scan).tolist(),
#     #     "sequence": dbsearch_df.seq,
#     #     "score": get_float(dbsearch_df.score).tolist(),
#     #     "modified_sequence": dbsearch_df.modifiedseq,
#     #     "reverse":get_boolean(dbsearch_df.reverse).tolist(),
#     #     # Needs spectrum util
#     #     "intensities": annotation_results['intensity_annotations'],
#     #     "masses": annotation_results['mz_annotations'],
#     #     "matches_raw": annotation_results['ion_annotations'],
#     #     # Needs pyteomics mzml
#     #     "precursor_charge": get_precursor_charge_onehot(mzml_results['charge_state']).astype(int).tolist(),
#     #     "retention_time": mzml_results['retention_time'],
#     #     "method": get_method_onehot(mzml_results['method']).astype(int).tolist(),
#     #     "collision_energy": get_float(mzml_results['collision_energy']).tolist(),
#     #     "collision_energy_aligned_normed": get_float(mzml_results['collision_energy']/ 100.0).tolist(),
#     #     # "collision_energy": mzml_results['collision_energy'],
#     #     # "collision_energy_aligned_normed": mzml_results['collision_energy']/100.0,
#     #     # "charge_state": mzml_results[1],
#     #     # "retention_time": get_numbers(mzml_results[2]),
#     #     # "method": mzml_results[3],
#     # })
#     return dataset



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


def parseSeq(seq, fixmod_proforma, varmod_proforma):
    import re
    # fixmod_proforma = FIXMOD_PROFORMA
    # varmod_proforma = VARMOD_PROFORMA
    # seq = 'GLYVAAQGAC+57.021R'
    # fixmod_proforma = {'C\\+57.021':'C'}
    # varmod_proforma = {'C\\+57.021':'C'}
    nonmod_seq = seq.upper()
    nonmod_seq = re.sub('[^A-Z]','',nonmod_seq)
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
        originalmgffile = re.sub('.mgf','_backup.mgf', mgffile)
        shutil.copy2(mgffile, originalmgffile) 
        mgffile = originalmgffile        
    spectra = []
    spectrum = {}
    mzs, intensities = [], []
    params = {}
    # initiate the file writing
    mgf.write(spectra, output = reformatmgffile, file_mode='w')
    with open(mgffile) as fp:
        count = 0
        for line in fp:
            # # DEBUG
            # if (count > 10000):
            #     break
            # #
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
                    # for target_aa in fixmod_proforma.keys():
                    #     peptide = re.sub(target_aa, fixmod_proforma[target_aa], peptide) 
                    proforma, mod_num, seq, mod_seq, nonmod_seq = parseSeq(peptide, FIXMOD_PROFORMA, VARMOD_PROFORMA)
                elif 'CHARGE=' in line:
                    charge = line.replace('CHARGE=', '')
                    charge = int(charge)
                elif 'SCANS=' in line:
                    scans = line.replace('SCANS=', '')
                    scans = int(scans)                    
                elif 'COLLISION_ENERGY=' in line:
                    ce = line.replace('COLLISION_ENERGY=', '')
                    ce = float(ce)    
                elif 'PEPMASS=' in line:
                    pepmass = line.replace('PEPMASS=', '')
                    pepmass = float(pepmass)
                elif 'FILENAME=' in line:
                    rawfilename = line.replace('FILENAME=', '')
                    rawfilename = re.sub("\W$",'',rawfilename.split('/')[-1])
                # elif 'PROVENANCE_DATASET_PXD' in line:
                #     mass_flag = True
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
                            print('Reformatting MGF Progress: {} records'.format(count))
                            # append a chunk of spectra to new MGF
                            mgf.write(spectra, output = reformatmgffile, file_mode='a')
                            spectra = []

                elif re.search('=', line) is None:
                    mz, intensity = line.split()
                    mzs.append(float(mz))
                    intensities.append(float(intensity))
                    assert len(mzs) == len(intensities)
        if (len(spectra) > 0):
            mgf.write(spectra, output = reformatmgffile, file_mode='a')
        print('Reformatting MGF Progress DONE: total {} records'.format(count))
    return spectra
                    

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


def modifyMGFtitle(usimgffile, reformatmgffile, temp_dir):
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
def generateHDF5_transformer_wSeq(usimgffile, reformatmgffile, csvfile, 
                                  hdf5file, temp_dir):
    # # DEBUG
    # df = csvfile.copy()
    # dbsearch['Modifiedsequence'] = dbsearch.Sequence
    # #
    
    # # Obsolete: Can't recover to PD consistently anyway
    # dtype_dict = {'collision_energy_aligned_normed':np.uint8,
    #               'precursor_charge':np.uint8,  
    #               'masses':np.float16,
    #               'intensities':np.float16
    #               }
    # if os.path.exists(jsonfile):
    #     mzs_df = pd.read_json(jsonfile, dtype = dtype_dict)
    #     mzs_df.info()
    #     print('Loading JSON to pandas: ' + str(time.time()-start_time))

    # retrieve spectrum of PSM from MGF
    start_time = time.time()        # start time for parsing
    spectra=mgf.read(usimgffile)
    spectra[0]
    mzs_df = []
    
    index = 0
    for spectrum in spectra:
        index += 1
        if (index % 100 == 0):
            print('Generating CSV Progress: {} records'.format(index))
        try:
            retention_time = spectrum['params']['rtinseconds']
            collision_energy = float(spectrum['params']['ce'])
            # collision_energy_aligned_normed = float(spectrum['params']['ce']/100)
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
            intensity_vec, mz_vec = constructCospredVec(spectrum['m/z array'],spectrum['intensity array'])
            masses = mz_vec
            intensities = intensity_vec
            
            mzs_df.append(pd.Series([raw_file, scan_number, sequence, score, 
                                      modified_sequence, proforma, 
                                      mod_num, reverse, 
                                      collision_energy, charge_state,
                                      masses, intensities, 
                                      retention_time, method
                                      ]))
            # mzs_df.append(pd.Series([modified_sequence, collision_energy_aligned_normed, charge_state,
            #                           masses, intensities
            #                           ]))
        except:
            next
    
    print('generate list: ' + str(time.time()-start_time))
    mzs_df = pd.concat(mzs_df, axis = 1)
    print('Concat list: ' + str(time.time()-start_time))
    
    mzs_df=mzs_df.transpose()
    print('transpost: ' + str(time.time()-start_time))
    mzs_df.columns =['raw_file', 'scan_number', 'sequence', 'score' , 
                              'modified_sequence', 'proforma', 
                              'mod_num', 'reverse',
                              'collision_energy','precursor_charge',  
                              'masses', 'intensities', 
                              'retention_time', 'method']
    # mzs_df.columns =['modified_sequence', 'collision_energy_aligned_normed','precursor_charge',  
    #                           'masses', 'intensities']
    
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

    # modifyMGFtitle(usimgffile, reformatmgffile, temp_dir)
    return dataset
            

def main():
    
    parser = ArgumentParser()
    parser.add_argument('-l', '--local', default=False, action='store_true',
                    help='execute in local computer')
    parser.add_argument('-w', '--workflow', default='test', help='workflow to use')
    args = parser.parse_args()    

    workflow = args.workflow

    if args.local is True:
        temp_dir = constants_local.TEMP_DIR   
        traincsvfile = constants_local.TRAINCSV_PATH
        testcsvfile = constants_local.TESTCSV_PATH
        mgffile = constants_local.MGF_PATH
        psmfile = constants_local.PSM_PATH
        if (workflow == 'split' or workflow == 'split_usi'):
            if (workflow == 'split'):
                mgffile = constants_local.MGF_PATH
                trainsetfile = constants_local.TRAINFILE
                testsetfile = constants_local.TESTFILE
            else:
                mgffile = constants_local.REFORMAT_USITITLE_PATH
                trainsetfile = constants_local.REFORMAT_TRAIN_USITITLE_PATH
                testsetfile = constants_local.REFORMAT_TEST_USITITLE_PATH
            # hold out N records as testset
            splitMGF(mgffile, trainsetfile, testsetfile, n_test = 1000)
        elif (workflow == 'train' or workflow == 'test'):
            if workflow == 'train':
                usimgffile = constants_local.REFORMAT_TRAIN_USITITLE_PATH
                reformatmgffile = constants_local.REFORMAT_TRAIN_PATH
                hdf5file = constants_local.TRAINDATA_PATH
                csvfile = traincsvfile
                datasetfile = constants_local.TRAINDATASET_PATH
            else:
                usimgffile = constants_local.REFORMAT_TEST_USITITLE_PATH
                reformatmgffile = constants_local.REFORMAT_TEST_PATH
                hdf5file = constants_local.TESTDATA_PATH
                csvfile = testcsvfile
                datasetfile = constants_local.TESTDATASET_PATH
            dataset = generateHDF5_transformer_wSeq(usimgffile, reformatmgffile, 
                                                    csvfile, hdf5file, temp_dir)
            io_local.to_hdf5(dataset, hdf5file)        # generate HDF5 format
            io_local.to_arrow(dataset, datasetfile)      # generate arrow format with chunking  
            print('Saving Dataset Done!')
        elif (workflow == 'reformat'):
            usimgffile = constants_local.REFORMAT_USITITLE_PATH
            reformatMGF_wSeq(mgffile, usimgffile)
        else:
            print("Unknown workflow choice.")        
    else:
        temp_dir = constants_gcp.TEMP_DIR   
        trainsetfile = constants_gcp.TRAINFILE
        testsetfile = constants_gcp.TESTFILE
        traincsvfile = constants_gcp.TRAINCSV_PATH
        testcsvfile = constants_gcp.TESTCSV_PATH
        mgffile = constants_gcp.MGF_PATH
        psmfile = constants_gcp.PSM_PATH
        if (workflow == 'split' or workflow == 'split_usi'):
            if (workflow == 'split'):
                mgffile = constants_gcp.MGF_PATH
                trainsetfile = constants_gcp.TRAINFILE
                testsetfile = constants_gcp.TESTFILE
            else:
                mgffile = constants_gcp.REFORMAT_USITITLE_PATH
                trainsetfile = constants_gcp.REFORMAT_TRAIN_USITITLE_PATH
                testsetfile = constants_gcp.REFORMAT_TEST_USITITLE_PATH
            # hold out N records as testset
            splitMGF(mgffile, trainsetfile, testsetfile, n_test = 1000)
        elif (workflow == 'train' or workflow == 'test'):
            if workflow == 'train':
                usimgffile = constants_gcp.REFORMAT_TRAIN_USITITLE_PATH
                reformatmgffile = constants_gcp.REFORMAT_TRAIN_PATH
                hdf5file = constants_gcp.TRAINDATA_PATH
                csvfile = traincsvfile
                datasetfile = constants_gcp.TRAINDATASET_PATH
            else:
                usimgffile = constants_gcp.REFORMAT_TEST_USITITLE_PATH
                reformatmgffile = constants_gcp.REFORMAT_TEST_PATH
                hdf5file = constants_gcp.TESTDATA_PATH
                csvfile = testcsvfile
                datasetfile = constants_gcp.TESTDATASET_PATH
            dataset = generateHDF5_transformer_wSeq(usimgffile, reformatmgffile, 
                                                    csvfile, hdf5file, temp_dir)
            io_local.to_hdf5(dataset, hdf5file)        
            io_local.to_arrow(dataset, datasetfile)        
            print('Saving Dataset Done!')
        elif (workflow == 'reformat'):
            usimgffile = constants_gcp.REFORMAT_USITITLE_PATH
            reformatMGF_wSeq(mgffile, usimgffile)
        else:
            print("Unknown workflow choice.")   
            
            
# # From Shivani: get info from MGF
# def main():
#     print('Reading mgf...', mgffile)
#     spectra = readmgf(mgffile)
#     y = [spectrum2vector(sp['mz'], sp['it'], sp['mass'], BIN_SIZE, sp['charge']) for sp in spectra]
#     # get_sequence_integer(df['modified_sequence']).astype(int)    
#     x = [get_sequence_integer(sp['pep']) for sp in spectra]
#     ce = [sp['nce'] for sp in spectra]
#     charge = [sp['charge'] for sp in spectra]
#     print(ce)

#     # Get info from CSV
#     sequence_integer = torch.tensor(np.stack(x), dtype=torch.long)
#     precursor_charge_onehot = torch.tensor(np.stack(charge), dtype=torch.long)
#     collision_energy_aligned_normed = torch.tensor(np.stack(ce))
#     intensities_raw = torch.tensor(np.stack(y), dtype=torch.float32)
#     m = {'sequence_integer': sequence_integer, 'precursor_charge_onehot': precursor_charge_onehot, 'intensities_raw': intensities_raw,'collision_energy_aligned_normed':collision_energy_aligned_normed}
#     torch.save(m, "C:/Users/tiwars46/PycharmProjects/data/valu_tensor.csv")
    
    
if __name__ == "__main__":
    main()
