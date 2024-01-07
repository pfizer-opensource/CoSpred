import sys, getopt
from pyteomics import mzml
from pyteomics import mgf
import numpy as np
import spectrum_utils.spectrum as sus
import pandas as pd
import os
import re
import time
from argparse import ArgumentParser
import random
import copy
import functools
import ast
import h5py

try: 
    os.chdir('/Users/xuel12/Documents/Projects/seq2spec/CoSpred/')
    print("Current directory is {}".format(os.getcwd()))
except: 
    print("Something wrong with specified directory. Exception- ", sys.exc_info())
  
from preprocess import utils, annotate, match
import params.constants_local as constants_local
import params.constants_gcp as constants_gcp
from params.constants import (
    ALPHABET_S,
    CHARGES,
    MAX_SEQUENCE,
    ALPHABET,
    MAX_ION,
    NLOSSES,
    ION_TYPES,
    ION_OFFSET,
    MAX_FRAG_CHARGE,
    METHODS,
)

COL_SEP = "\t"

def get_float(vals, dtype=float):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_string(vals, dtype=str):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_boolean(vals, dtype=bool):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_number(vals, dtype=int):
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


def check_mandatory_keys(dictionary, keys):
    for key in keys:
        if key not in dictionary.keys():
            raise KeyError("key {} is missing".format(key))
    return True


# def reshape_dims(array, nlosses=1, z=3):
#     return array.reshape([array.shape[0], MAX_ION, len(ION_TYPES), nlosses, z])

# from sequence_interger to sequence
def get_sequence(sequence):
    d = ALPHABET_S
    return "".join([d[i] if i in d else "" for i in sequence])


def sequence_integer_to_str(array):
    sequences = [get_sequence(array[i]) for i in range(array.shape[0])]
    return sequences


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
            
def get_sequence_integer(sequences):
    # sequences = df.sequence
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
    return array


# def parse_ion(string):
#     ion_type = ION_TYPES.index(string[0])
#     if ("-") in string:
#         ion_n, suffix = string[1:].split("-")
#     else:
#         ion_n = string[1:].split("+")[0]
#         suffix = ""
#     if ("+") in string:
#         ion_frag=string.count("+")
#     return ion_type, int(ion_n) - 1, NLOSSES.index(suffix), int(ion_frag)-1


def parse_ion(string):
    # string=ion
    ion_type = ION_TYPES.index(string[0])
    ion_fr=1        # default ion charge is 1
    suffix = ''
    if ("-") in string:
        ion_n, suffix = string[1:].split("-")
    elif ('^') in string:
        ion_n = string[1:].split("^")[0]
        suffix = ""
    else:
        ion_n = re.sub('^\D+','',string)
        
    if ("+") in string:
        ion_frag=string.count("+")
        if ion_frag <= 3:
            ion_fr=ion_frag
        else:
            pass
    elif ('^') in string:
        ion_frag=int(string.split("^")[1])
        if ion_frag <= 3:
            ion_fr=ion_frag
        else:
            pass
    else:
        ion_fr = 1
        
    return int(ion_n) - 1, ion_type, NLOSSES.index(suffix), int(ion_fr)-1
    #return ion_type, int(ion_n) - 1, int(ion_frag)-1
    

# def parse_ions(string):
#     ion_type = ION_TYPES.index(string[0])
#     if ("-") in string:
#         ion_n, suffix = string[1:].split("-")
#     else:
#         ion_n = string[1:]
#         suffix = ""
#     return ion_type, int(ion_n) - 1, NLOSSES.index(suffix)


def reshape_dims(array):
    n, dims = array.shape
    assert dims == 174
    nlosses = 1
    return array.reshape(
        [array.shape[0], MAX_SEQUENCE - 1, len(ION_TYPES), nlosses, MAX_FRAG_CHARGE]
    )

def reshape_flat(array):
    s = array.shape
    flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
    return array.reshape(flat_dim)

def mask_outofrange(array, lengths, mask=-1.0):
    # dim
    # array[0,0,0,0]
    # lengths[0]
    for i in range(array.shape[0]):
        # array[0,7:].shape
        array[i, (lengths[i] - 1):, :, :, :] = mask
    return array

# restrict nloss and charge to be considered
def cap(array, nlosses=1, z=3):
    return array[:, :, :, :nlosses, :z]

# not applicable since PROSIT model doesn't make out of charge ion, meaning \
    # charge 2 precursor could still have charge 3 fragments
def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        # if int(charges[i]) > 3:      # obsolete since charge was capped at 3
        # array[0,0,0]
        array[i, :, :, :, (charges[i]):] = mask
    return array


def splitMGF(mgffile, trainsetfile, testsetfile, n_test = 1000):
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    spectra=mgf.read(mgffile)
    spectra_train = []
    spectra_test = []
    test_index = sorted(random.sample(range(0, len(spectra)), n_test))
    test_index_list = []
    i = 0
    for spectrum in spectra:
        if i in test_index:
            spectra_test.append(spectrum)
            test_index_list.append(test_index.pop(0))
            print('spectrum index {} in testset'.format(i))
        else:
            spectra_train.append(spectrum)
        i += 1
    mgf.write(spectra_test, output = testsetfile)
    mgf.write(spectra_train, output = trainsetfile)
    return test_index_list
    

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


## Extract spectra from mzml
def readMZML(mzmlfile, dbsearch_df):
    f = mzml.MzML(mzmlfile)

    mzs_df = []
    
    for index, row in dbsearch_df.iterrows():  
        controller_str = 'controllerType=0 controllerNumber=1 '
        p = f.get_by_id(controller_str + "scan=" + str(row.scan))
        dfg = p.get('precursorList')
        fg = dfg['precursor']
        collision_energy = fg[0].get('activation').get('collision energy')
        # charge_state = get_precursor_charge_onehot(fg[0].get('selectedIonList').get('selectedIon')[0].get('charge state'))
        charge_state = fg[0].get('selectedIonList').get('selectedIon')[0].get('charge state')
        filter_string = p.get('scanList').get('scan')[0].get('filter string')
        retention_time = p.get('scanList').get('scan')[0].get('scan start time')
        if re.search("hcd", filter_string):
            method = "HCD"
        if re.search("cid", filter_string):
            method = "CID"
        if re.search("etd", filter_string):
            method = "ETD"
        #return pd.Series([collision_energy, charge_state.tolist(), retention_time, method])
        # return pd.Series([collision_energy, charge_state, retention_time, method])
        mzs_df.append(pd.Series([collision_energy, charge_state, retention_time, method]))

    mzs_df = pd.concat(mzs_df, axis = 1).transpose()
    mzs_df.columns =['collision_energy','charge_state', 'retention_time', 'method']
    
    # def calc_row(row):
    #     try:
    #         controller_str = 'controllerType=0 controllerNumber=1 '
    #         p = f.get_by_id(controller_str + "scan=" + str(row.scan))
    #         dfg = p.get('precursorList')
    #         fg = dfg['precursor']
    #         collision_energy = fg[0].get('activation').get('collision energy')
    #         charge_state = get_precursor_charge_onehot(fg[0].get('selectedIonList').get('selectedIon')[0].get('charge state'))
    #         charge_state = fg[0].get('selectedIonList').get('selectedIon')[0].get('charge state')
    #         filter_string = p.get('scanList').get('scan')[0].get('filter string')
    #         retention_time = p.get('scanList').get('scan')[0].get('scan start time')
    #         if re.search("hcd", filter_string):
    #             method = "HCD"
    #         if re.search("cid", filter_string):
    #             method = "CID"
    #         if re.search("etd", filter_string):
    #             method = "ETD"
    #         #return pd.Series([collision_energy, charge_state.tolist(), retention_time, method])
    #         return pd.Series([collision_energy, charge_state, retention_time, method])
    #     except KeyError:
    #         pass
    # mzs_series = dbsearch_df.apply(calc_row, 1)
    
    return mzs_df


## Add TITLE, SEQ, CE to raw file MGF
def reformatMGF(mgffile, mzmlfile, dbsearch_df, reformatmgffile, temp_dir):
    f = mzml.MzML(mzmlfile)
    # spectra=mgf.read(reformatmgffile)
    
    # Rewrite TITLE for the MGF
    print('Creating temp MGF file with new TITLE...')
    
    spectra_origin=mgf.read(mgffile)
    spectra_temp = []
    for spectrum in spectra_origin:
        # for MSconvert generated MGF
        title_split = spectrum['params']['title'].split(' ')
        repoid = re.sub('\W$','',title_split[1].split('"')[1])
        scan_number = re.sub('\W+','',title_split[0].split('.')[1])
        spectrum['params']['title'] = ':'.join(['mzspec', 'repoID',
                          repoid, 'scan', scan_number])
        spectrum['params']['scans'] = scan_number                       
        # # for PD generated MGF
        # title_split = spectrum['params']['title'].split(';')
        # spectrum['params']['title'] = ':'.join(['mzspec', 'repoID',
        #                   re.sub("\W$",'',title_split[0].split('\\')[-1]),
        #                   'scan',
        #                   re.sub('\W+','',title_split[-1].split('scans')[-1])])
        spectra_temp.append(spectrum)
    reformatmgffile_temp = temp_dir+time.strftime("%Y%m%d%H%M%S")+'.mgf'
    mgf.write(spectra_temp, output = reformatmgffile_temp)
    spectra_origin.close()

    print('Temp MGF file with new TITLE was created!')

    # Add SEQ and CE to the reformatted MGF
    spectra=mgf.read(reformatmgffile_temp)
    for spectrum in spectra:
        pass
    spectra_new = []
    for index, row in dbsearch_df.iterrows():  
        if (index % 100 == 0):
            print('Reformatting MGF Progress: {}%'.format(index/dbsearch_df.shape[0]*100))
        try:
            # retrieve spectrum of PSM from MGF and MZML
            spectrum = spectra.get_spectrum(row['title'])
            spectrum['params']['seq'] = row['modifiedseq']
            spectrum['params']['proforma'] = row['proforma']
            spectrum['params']['mod_num'] = str(row['mod_num'])

            controller_str = 'controllerType=0 controllerNumber=1 '
            p = f.get_by_id(controller_str + "scan=" + str(spectrum['params']['scans']))
            fg = p.get('precursorList').get('precursor')
            spectrum['params']['pepmass'] = spectrum['params']['pepmass'][0]
            spectrum['params']['rtinseconds'] = str(spectrum['params']['rtinseconds'])
            spectrum['params']['ce'] = str(fg[0].get('activation').get('collision energy'))
            spectrum['params']['charge'] = re.sub('\D+','',str(fg[0].get('selectedIonList').get('selectedIon')[0].get('charge state')))
            filter_string = p.get('scanList').get('scan')[0].get('filter string')
            # retention_time = p.get('scanList').get('scan')[0].get('scan start time')
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

    mgf.write(spectra_new, output = reformatmgffile)
    f.close()
    spectra.close()
    
    if os.path.exists(reformatmgffile_temp):
        os.remove(reformatmgffile_temp)
    else:
        print("The temp reformatted MGF file does not exist")
  
    return spectra_new


## Annotate b and y ions to MGF file
def annotateMGF(reformatmgffile, dbsearch_df, temp_dir):

    mgfile=mgf.read(reformatmgffile)
    # # debug
    # rawfilels = rawfile + '.' + scan_number + '.' + scan_number + '.' + charge + ' File:"' + rawfile + '.raw' + '"' + ',' + ' NativeID:"controllerType=0 controllerNumber=1 ' + 'scan=' + scan_number + '"'
    # dbsearch_df  = dbsearch.iloc[0:10].copy()
    # mgfile[0]
    # mgfile[0]['params']
    # row = dbsearch.iloc[0]
    # #
    
    fragment_tol_mass = 0.35
    fragment_tol_mode = 'Da'
    min_mz = 100
    max_mz = 1400
    min_intensity = 0.05
    
    # intensity_annotations = []
    # mz_annotations = []
    # ion_annotations = []
    mzs_df = []
    
    for index, row in dbsearch_df.iterrows():  
        if (index % 100 == 0):
            print('MS2 Annotation Progress: {}%'.format(index/dbsearch_df.shape[0]*100))

        try:
            # retrieve spectrum of PSM from MGF
            proforma = row['proforma']
            seq = row['modifiedseq']
            spectrum_dict = mgfile.get_spectrum(row['title'])
            modifications = {}
            identifier = spectrum_dict['params']['title']
            peptide = spectrum_dict['params']['seq']
            ce = spectrum_dict['params']['ce']
            method = spectrum_dict['params']['method']
            scan = spectrum_dict['params']['scans']
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
                
            # intensity_ann = spectrum.intensity.tolist()
            # intensity_annotations = [str(element) for element in intensity_ann]
            intensity_annotations = ";".join([str(element) for element in spectrum.intensity])
            # len(intensity_annotations.split(';'))
            # mz_ann = spectrum.mz.tolist()
            # mz_annotations = [str(element) for element in mz_ann]
            mz_annotations = ";".join([str(element) for element in spectrum.mz])
            # len(mz_annotations.split(';'))
            # ion_ann = spectrum.annotation.tolist()
            # ion_annotations = [str(element) for element in ion_ann]
            ion_annotations = ";".join([re.sub('/\S+','', str(element)) for element in spectrum.annotation.tolist()])
            # len(ion_annotations.split(';'))
            # mzs_df.append([intensity_annotations, mz_annotations, ion_annotations])
            mzs_df.append(pd.Series([seq, intensity_annotations, mz_annotations, ion_annotations]))
            # mzs_df.append('|'.join([intensity_annotations, mz_annotations, ion_annotations]))
        except:
            next
            
    # construct dataframe for annotated MS2
    # mzs_df = pd.DataFrame(columns=['intensity_annotations', 'mz_annotations','ion_annotations'], 
    #                   data=[row.split('|') for row in mzs_df])
    mzs_df = pd.concat(mzs_df, axis = 1).transpose()
    mzs_df.columns =['seq','intensity_annotations', 'mz_annotations', 'ion_annotations']
    mzs_df.to_csv(temp_dir+'annotatedMGF.csv', index=False)

    return mzs_df


## Contruct ML friendly spectra matrix
def generateCSV(usimgffile, reformatmgffile, dbsearch_df, annotation_results, csvfile, temp_dir):
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
    
    # get annotation MS2
    annotation_results.columns = ['seq','intensities','masses','matches_raw']
    # ## DEBUG
    # annotation_results.to_pickle('annotation_result.pkl')
    # annotation_results = pd.read_pickle("annotation_result.pkl") 
    # ##
    
    # # NO NEED anymore: get spectra from mzML
    # mzml_results = readMZML(mzmlfile, dbsearch_df)
    
    # retrieve spectrum of PSM from MGF
    spectra=mgf.read(usimgffile)
    spectra[0]
    mzs_df = []
    for index, row in dbsearch_df.iterrows():  
        if (index % 100 == 0):
            print('Generating CSV Progress: {}%'.format(index/dbsearch_df.shape[0]*100))
                
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
            mzs_df.append(pd.Series([raw_file, scan_number, sequence, score , 
                                     modified_sequence, proforma, 
                                     mod_num, reverse, 
                                     collision_energy, charge_state, 
                                     retention_time, method, mod_num]))
        except:
            next
            
    mzs_df = pd.concat(mzs_df, axis = 1).transpose()
    mzs_df.columns =['raw_file', 'scan_number', 'sequence', 'score' , 
                             'modified_sequence', 'proforma', 
                             'mod_num', 'reverse',
                             'collision_energy','precursor_charge', 'retention_time', 
                             'method', 'mod_num']
    mzs_df['collision_energy_aligned_normed'] = mzs_df['collision_energy']/100.0
    
    # construct CSV
    annotation_results_new = annotation_results.reset_index(drop=True)
    mzs_df_new = mzs_df.reset_index(drop=True)
    
    dataset = pd.concat([mzs_df_new, annotation_results_new], axis = 1)
    
    # dataset = pd.DataFrame({
    #     "raw_file": dbsearch_df.file,
    #     "scan_number": dbsearch_df.scan,
    #     "sequence": dbsearch_df.seq,
    #     "score": dbsearch_df.score,
    #     "modified_sequence": dbsearch_df.modifiedseq,
    #     "mod_num": dbsearch_df.mod_num,
    #     "reverse":dbsearch_df.reverse,
    #     # Needs spectrum util
    #     "intensities": annotation_results['intensity_annotations'],
    #     "masses": annotation_results['mz_annotations'],
    #     "matches_raw": annotation_results['ion_annotations'],
    #     # Needs pyteomics mzml
    #     "precursor_charge": mzs_df['charge_state'],
    #     "retention_time": mzs_df['retention_time'],
    #     "method": mzs_df['method'],
    #     "collision_energy": mzs_df['collision_energy'],
    #     "collision_energy_aligned_normed": mzs_df['collision_energy']/ 100.0,
    #     # "collision_energy": mzml_results['collision_energy'],
    #     # "collision_energy_aligned_normed": mzml_results['collision_energy']/100.0,
    #     # "charge_state": mzml_results[1],
    #     # "retention_time": get_numbers(mzml_results[2]),
    #     # "method": mzml_results[3],
    # })
    
    dataset = dataset.dropna()
    dataset.to_csv(csvfile, index=False)

    print('Generating CSV Done!')

    modifyMGFtitle(usimgffile, reformatmgffile, temp_dir)
    return dataset


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


def get_PrositArray(df, vectype):
    # row = df.iloc[0]
    array_series = []
    for index, row in df.iterrows():
        array = np.zeros([MAX_ION, len(ION_TYPES), len(NLOSSES), MAX_FRAG_CHARGE])
        lstions = str(row.matches_raw).split(";")
        lstmasses = str(row[vectype]).split(";")
        for i in ION_TYPES:
            patternn = r"^" + i + "[0-9]+"
            ions_regex = re.compile(patternn)
            for index, ion in enumerate(lstions):
                if ions_regex.match(ion):
                    ion_n, ion_type, nloss, ion_charge = parse_ion(ion)
                    array[ion_n, ion_type, nloss, ion_charge] = float(lstmasses[index])
        if (vectype == 'intensities'):
            max_value = np.max(array)
            array = array/max_value if max_value > 0 else array
        array_series.append(array)
    out = np.squeeze(np.stack(array_series))
    # out.shape
    if len(out.shape) == 4:
        out = out.reshape([1] + list(out.shape))
    return out

    
def constructPrositVec(df, vectype):
    nlosses = 1
    z = 3
    lengths = [len(x) for x in df["modified_sequence"]]
    # lengths = (data["sequence_integer"] > 0).sum(1)
    
    # dimension: [record, ion number(1-29), ion type(y=0,b=1), nloss(1), charge(1-3)]    
    array = get_PrositArray(df, vectype)
    # # DEBUG
    # array.shape
    # array[0,0,0,0]     # 1st record, y1
    # array[0,1,1,0]     # 1st record, b2
    # df.iloc[0][vectype]
    # #
    array = cap(array, nlosses, z)   # do not consider nloss, limit charge 1-3    
    array = mask_outofrange(array, lengths)    # mask impossible fragment as -1
    # array = mask_outofcharge(masses_raw, df.precursor_charge)
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
    
def constructDataset(csvfile):
    #df = pd.read_csv('C:/Users/tiwars46/PycharmProjects/prosit_PfizerRD/finaldatafolder/csv_for_prosit_training/val_100000.csv',sep=',')
    # df = pd.read_csv('C:/Users/tiwars46/PycharmProjects/data/SEARCH_Ymod_Phospho/Yph_mod_prosit.csv',sep=',')
    # df = pd.read_csv('/Users/xuel12/Documents/Projects/seq2spec/CoSpred/data/heladigest/peptidelist_test.csv',sep=',')
    df = pd.read_csv(csvfile,sep=',')

    assert "modified_sequence" in df.columns
    assert "collision_energy" in df.columns
    assert "precursor_charge" in df.columns
    assert "intensities" in df.columns
    assert "masses" in df.columns
    
    # ## DEBUG: Ignore modification on the peptide
    # df.modified_sequence = df.sequence
    # ##
    
    df.dropna(subset=['intensities','matches_raw'],inplace=True)
    # df.dropna(subset=['matches_raw'],inplace=True)
    # df.dropna(subset=['matches_raw'], inplace=True)
    df.columns = df.columns.str.replace('[\r]', '')
    #df['reverse'] = df['reverse'].str.replace('[\r]', '')

    # construct Dataset based on Prosit definition
    dataset = {
        "collision_energy": get_float(df['collision_energy']),
        "collision_energy_aligned": get_float(df['collision_energy']),
        #"collision_energy_aligned_normed": np.asarray([float(i[0]) for i in df.collision_energy_aligned_normed.map(lambda x: ast.literal_eval(x))]),
        "collision_energy_aligned_normed":get_float(df['collision_energy']/100.0),
        "intensities_raw": constructPrositVec(df, 'intensities'),
        "masses_pred": constructPrositVec(df, vectype='masses'),
        "masses_raw": constructPrositVec(df, vectype='masses'),        
        "method": get_method_onehot(df['method']).astype(int),
        "precursor_charge_onehot":get_precursor_charge_onehot(df['precursor_charge']).astype(int),
        # "rawfile": get_string(df['raw_files']),
        "rawfile": df['raw_file'].astype('S32'),
        "reverse": get_boolean(df['reverse']),
        "scan_number": get_number(df['scan_number']),
        "score": get_float(df['score']),
        "sequence_integer": get_sequence_integer(df['modified_sequence']).astype(int),
        "sequence_onehot": get_sequence_onehot(df['modified_sequence']).astype(int),
    }

    # dataset = pd.DataFrame({
    #     "raw_files": dbsearch_df.file,
    #     "scan_number": get_number(dbsearch_df.scan).tolist(),
    #     "sequence": dbsearch_df.seq,
    #     "score": get_float(dbsearch_df.score).tolist(),
    #     "modified_sequence": dbsearch_df.modifiedseq,
    #     "reverse":get_boolean(dbsearch_df.reverse).tolist(),
    #     # Needs spectrum util
    #     "intensities": annotation_results['intensity_annotations'],
    #     "masses": annotation_results['mz_annotations'],
    #     "matches_raw": annotation_results['ion_annotations'],
    #     # Needs pyteomics mzml
    #     "precursor_charge": get_precursor_charge_onehot(mzml_results['charge_state']).astype(int).tolist(),
    #     "retention_time": mzml_results['retention_time'],
    #     "method": get_method_onehot(mzml_results['method']).astype(int).tolist(),
    #     "collision_energy": get_float(mzml_results['collision_energy']).tolist(),
    #     "collision_energy_aligned_normed": get_float(mzml_results['collision_energy']/ 100.0).tolist(),
    #     # "collision_energy": mzml_results['collision_energy'],
    #     # "collision_energy_aligned_normed": mzml_results['collision_energy']/100.0,
    #     # "charge_state": mzml_results[1],
    #     # "retention_time": get_numbers(mzml_results[2]),
    #     # "method": mzml_results[3],
    # })
    return dataset
   

def read_hdf5(path, n_samples=None):
    # Get a list of the keys for the datasets
    with h5py.File(path, 'r') as f:
        print(f.keys())
        dataset_list = list(f.keys())
        for dset_name in dataset_list:
            print(dset_name)
            print(f[dset_name][:6])
        f.close()
    return dataset_list

            
def to_hdf5(dictionary, path):
    dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(path, "w") as f:
        for key, data in dictionary.items():
            # f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
            if (data.dtype == 'object'):
                f.create_dataset(key, data=data, dtype=dt, compression="gzip")
            else:
                f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")
       
        
def main():    
    
    parser = ArgumentParser()
    parser.add_argument('-m', '--mgffile', default=constants_local.MGF_PATH, help='raw file MGF')
    # parser.add_argument('-r', '--reformatmgffile', default=constants_local.REFORMAT_TRAIN_PATH, help='reformat MGF')
    parser.add_argument('-z', '--mzmlfile', default=constants_local.MZML_PATH, help='raw file mzML')
    parser.add_argument('-p', '--psmfile', default=constants_local.PSM_PATH, help='PSM file')
    # parser.add_argument('-c', '--csvfile', default=constants_local.CSV_PATH, help='csv file for ML')
    parser.add_argument('-l', '--local', default=False, action='store_true',
                    help='execute in local computer')
    parser.add_argument('-w', '--workflow', default='split', help='workflow to use')
    args = parser.parse_args()    
    
    workflow = args.workflow
    
    if args.local is True:
        temp_dir = constants_local.TEMP_DIR   
        trainsetfile = constants_local.TRAINFILE
        testsetfile = constants_local.TESTFILE
        traincsvfile = constants_local.TRAINCSV_PATH
        testcsvfile = constants_local.TESTCSV_PATH
        
        mgffile = constants_local.MGF_PATH
        mzmlfile = constants_local.MZML_PATH
        psmfile = constants_local.PSM_PATH
        
        if (workflow == 'train'):
            usimgffile = constants_local.REFORMAT_TRAIN_USITITLE_PATH
            reformatmgffile = constants_local.REFORMAT_TRAIN_PATH
            datasetfile = trainsetfile
            csvfile = traincsvfile
            hdf5file = constants_local.TRAINDATA_PATH
        elif (workflow == 'test'):
            usimgffile = constants_local.REFORMAT_TEST_USITITLE_PATH
            reformatmgffile = constants_local.REFORMAT_TEST_PATH
            datasetfile = testsetfile
            csvfile = testcsvfile
            hdf5file = constants_local.TESTDATA_PATH
        else:
            print("Unknown workflow choice.")
    else:
        temp_dir = constants_gcp.TEMP_DIR   
        trainsetfile = constants_gcp.TRAINFILE
        testsetfile = constants_gcp.TESTFILE
        traincsvfile = constants_gcp.TRAINCSV_PATH
        testcsvfile = constants_gcp.TESTCSV_PATH
        
        mgffile = constants_gcp.MGF_PATH
        mzmlfile = constants_gcp.MZML_PATH
        psmfile = constants_gcp.PSM_PATH
 
        if (workflow == 'train'):
            usimgffile = constants_gcp.REFORMAT_TRAIN_USITITLE_PATH
            reformatmgffile = constants_gcp.REFORMAT_TRAIN_PATH
            datasetfile = trainsetfile
            csvfile = traincsvfile
            hdf5file = constants_gcp.TRAINDATA_PATH
        elif (workflow == 'test'):
            usimgffile = constants_gcp.REFORMAT_TEST_USITITLE_PATH
            reformatmgffile = constants_gcp.REFORMAT_TEST_PATH
            datasetfile = testsetfile
            csvfile = testcsvfile
            hdf5file = constants_gcp.TESTDATA_PATH
        else:
            print("Unknown workflow choice.")
           
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # get psm result
    dbsearch = getPSM(psmfile)
    dbsearch_df = dbsearch
    # dbsearch_df = dbsearch.iloc[:10]       # trial run
        
    if (workflow == 'split'):
        # hold out N records as testset
        splitMGF(mgffile, trainsetfile, testsetfile, n_test = 1000)
        print('Splitting train vs test set Done!')
    # reformat the Spectra
    elif (workflow == 'train' or workflow == 'test'):
        if not os.path.isfile(usimgffile):
            reformatMGF(datasetfile, mzmlfile, dbsearch_df, usimgffile, temp_dir)
            annotation_results = annotateMGF(usimgffile, dbsearch_df, temp_dir)
        else:
            annotation_results = pd.read_csv(temp_dir+'annotatedMGF.csv') 
        # match peptide from PSM with spectra MGF
        if not os.path.isfile(reformatmgffile):
            dataset = generateCSV(usimgffile, reformatmgffile, dbsearch_df, annotation_results, 
                                csvfile, temp_dir)
        # transform to hdf5
        dataset = constructDataset(csvfile)
        to_hdf5(dataset, hdf5file)
        print('Generating HDF5 Done!')
    else:
        print("Unknown workflow choice.")

    
if __name__ == "__main__":
    main()
