#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:51:37 2020

@author: xuel12
"""

import os
import re

# try: 
#     os.chdir('/Users/xuel12/Documents/Projects/seq2spec/prosit/local_training_tf2')
#     print("Current directory is {}".format(os.getcwd()))
# except: 
#     print("Something wrong with specified directory. Exception- ", sys.exc_info())
          
import params.constants as constants
import params.constants_local as constants_local


def from_msp_prosit(ofile):
    # Get a list of the keys for the datasets
    tmp_dict = {}
    final_dict = {}
    f = open(ofile, 'r')
    # use readline() to read the first line 
    line = f.readline()
    # use the read line to read further.
    # If the file is not empty keep reading one line at a time, till the file is empty
    while line:
        if re.search("^Name",line) is not None:
            tmp_dict['Name'] = line.rstrip('\n').split(' ')[-1]
            line = f.readline()
        elif re.search("^MW",line) is not None:
            tmp_dict['MW'] = round(float(line.rstrip('\n').split(' ')[-1]),4)
            tmp_dict['Precursor'] = tmp_dict['MW']
            line = f.readline()
        elif re.search("^Comment",line) is not None:
            tmp_dict['Comment'] = re.sub('Comment: ','',line.rstrip('\n'))
            tmp_dict['Comment'] = re.sub('; ',';',tmp_dict['Comment'])
            tmpline = tmp_dict['Comment'].split(' ')
            parent, ce, mod, modseq = [x.split('=')[1] for x in tmpline]
            ce = round(float(ce),1)
            mod_num = mod.split('/')[0]
            line = f.readline()
        elif re.search("^Num peaks",line) is not None:
            tmp_dict['Num_peaks'] = line.rstrip('\n').split(' ')[-1]
            line = f.readline()
            mz = []
            intensity = []
            anno = []
            while (line and line.strip() and re.search("^Name",line) is None):
                tmpline = line.rstrip('\n').split('\t')
                mz.append(tmpline[0])
                intensity.append(tmpline[1])
                anno.append(tmpline[2])
                line = f.readline()
            # one record per peptide_ce_mod_num
            peptide_ce_modstring = '_' .join([tmp_dict['Name'], str(ce), mod_num])
            # final_dict['peptide_ce_modstring'] = '_' .join([tmp_dict['Name'],tmp_dict['Comment'][1],tmp_dict['Comment'][3]])
            final_dict[peptide_ce_modstring] = {}
            final_dict[peptide_ce_modstring]['precursor'] = tmp_dict['Precursor']
            final_dict[peptide_ce_modstring]['mz'] = mz
            final_dict[peptide_ce_modstring]['intensity'] = intensity
            final_dict[peptide_ce_modstring]['anno'] = anno
    f.close()
    return final_dict
    

def from_msp_propel(ofile):
    
    # Get a list of the keys for the datasets
    with open(ofile, 'r') as f:
        tmp_dict = dict()
        final_dict = {}
        # use readline() to read the first line 
        line = f.readline()
        # use the read line to read further.
        # If the file is not empty keep reading one line at a time, till the file is empty
        counter = 0
        max_count = 10000
        ce = 0
        while (line is not None):
            if not line.strip():
                line = f.readline()
            elif counter >= max_count:
                break
            else:
                if re.search("^Name",line) is not None:
                    tmpline = line.rstrip('\n').split(' ')[-1]
                    tmp_dict['Name'] = tmpline.split('_')[0]
                    # print(tmp_dict['Name'])
                    mod, ce = [tmpline.split('_')[1], tmpline.split('_')[2]]
                    ce = round(float(ce.strip('%')),1)
                    line = f.readline()
                elif re.search("^MW",line) is not None:
                    tmp_dict['MW'] = round(float(line.rstrip('\n').split(' ')[-1]),4)
                    line = f.readline()
                elif re.search("^Comment",line) is not None:
                    tmpline = re.sub('Comment: ','',line.rstrip('\n'))
                    precursor = round(float(tmpline.split(' ')[5].split('=')[1]),4)
                    # tmpline = [x.split('=')[1] for x in tmpline]
                    tmp_dict['Comment'] = tmpline
                    tmp_dict['Precursor'] = precursor
                    line = f.readline()
                elif re.search("^Num peaks",line) is not None:
                    tmp_dict['Num_peaks'] = line.rstrip('\n').split(' ')[-1]
                    line = f.readline()
                    mz = []
                    intensity = []
                    anno = []
                    while (line is not None):
                        if not line.strip():
                            # one record per peptide_ce_modstring
                            peptide_ce_modstring = '_'.join([tmp_dict['Name'], str(ce), mod])
                            final_dict[peptide_ce_modstring] = {}
                            batch_dict[peptide_ce_modstring]['precursor'] = tmp_dict['Precursor']
                            final_dict[peptide_ce_modstring]['mz'] = mz
                            final_dict[peptide_ce_modstring]['intensity'] = intensity
                            final_dict[peptide_ce_modstring]['anno'] = anno
                            # tmp_dict = dict()
                            counter += 1
                            break
                        else:
                            tmpline = line.rstrip('\n').split('\t')
                            mz.append(tmpline[0])
                            intensity.append(tmpline[1])
                            anno.append(tmpline[2])
                            line = f.readline()
    return final_dict



def dict2mgf(spectrum_dict, ofile, mode='w'):   
    # take spedtrum list, write to mgf
    # ofile = example_dir+'peptidelist_pred.mgf'
    with open(ofile, mode) as f:
        rtinseconds = 1
        for name in spectrum_dict:
            f.write('BEGIN IONS\n')
            tmpname = name.split('_')
            seq = tmpname[0].split('/')[0]
            charge = tmpname[0].split('/')[1]
            precursor = spectrum_dict[name]['precursor']
            f.write('SEQ={}\n'.format(seq))
            f.write('PEPMASS={}\n'.format(precursor))
            f.write('CHARGE={}+\n'.format(charge))
            f.write('TITLE={}\n'.format(name))
            f.write('RTINSECONDS={}\n'.format(rtinseconds))
            for i in range(len(spectrum_dict[name]['mz'])):
                tmpion = spectrum_dict[name]['mz'][i] + ' ' + spectrum_dict[name]['intensity'][i] + '\n'
                f.write(tmpion)
            f.write('END IONS\n')
            rtinseconds += 1
    f.close()
    


def copy_from_msp(n_record, file, ofile=None):
    counter = 0
    sign = 0
    with open (file, 'r') as f1:
        line = f1.readline()
        if (ofile is not None):
            f2 = open (ofile, 'w')
        while(line is not None and counter < n_record):
            if (line == ''):
                print('DONE')
                sign = 1
                break
            if not line.strip():
                if (ofile is not None):
                    f2.write(line)
                counter += 1
                line = f1.readline()
            if (counter < n_record):
                if (ofile is not None):
                    f2.write(line)
                line = f1.readline()
            else:
                if (ofile is not None):
                    f2.close()
                break
        if (sign == 1):
            print("All {} records processed")
        else:
            print("{} records processed".format(counter))
    return (counter)


def msp2mgf(n_record, file, ofile, peplistfile):
    
    # Get a list of the keys for the datasets
    # f1 = open(example_dir+'human_synthetic_hcd_selected_sub.msp', 'r')
    # f1.close()
    with open(file, 'r') as f1:
        tmp_dict = dict()
        batch_dict = {}
        # use readline() to read the first line 
        line = f1.readline()
        # use the read line to read further.
        # If the file is not empty keep reading one line at a time, till the file is empty
        counter = 0
        sign = 0
        ce = 0
        batch_size = 40000
        peptidelist = []
        while (line is not None and counter < n_record):
            if line == '':
                sign = 1
                break
            if not line.strip():
                line = f1.readline()
            else:
                if re.search("^Name",line) is not None:
                    tmpline = line.rstrip('\n').split(' ')[-1]
                    tmp_dict['Name'] = tmpline.split('_')[0]
                    # print(tmp_dict['Name'])
                    mod, ce = [tmpline.split('_')[1], tmpline.split('_')[2]]
                    ce = round(float(ce.strip('%')),1)
                    line = f1.readline()
                elif re.search("^MW",line) is not None:
                    tmp_dict['MW'] = round(float(line.rstrip('\n').split(' ')[-1]),4)
                    line = f1.readline()
                elif re.search("^Comment",line) is not None:
                    tmpline = re.sub('Comment: ','',line.rstrip('\n'))
                    precursor = round(float(tmpline.split(' ')[5].split('=')[1]),4)
                    # tmpline = [x.split('=')[1] for x in tmpline]
                    tmp_dict['Comment'] = tmpline
                    tmp_dict['Precursor'] = precursor
                    line = f1.readline()
                elif re.search("^Num peaks",line) is not None:
                    tmp_dict['Num_peaks'] = line.rstrip('\n').split(' ')[-1]
                    line = f1.readline()
                    mz = []
                    intensity = []
                    anno = []
                    while (line is not None and counter < n_record):
                        if line == '':
                            sign = 1
                            break
                        if not line.strip():
                            # one record per peptide_ce_modstring
                            peptide_ce_modstring = '_'.join([tmp_dict['Name'], str(ce), mod])
                            peptidelist.append(peptide_ce_modstring)
                            batch_dict[peptide_ce_modstring] = {}
                            batch_dict[peptide_ce_modstring]['precursor'] = tmp_dict['Precursor']
                            batch_dict[peptide_ce_modstring]['mz'] = mz
                            batch_dict[peptide_ce_modstring]['intensity'] = intensity
                            batch_dict[peptide_ce_modstring]['anno'] = anno
                            # tmp_dict = dict()
                            counter += 1
                            if (counter % batch_size == 0):
                                if (counter == batch_size):
                                    dict2mgf(batch_dict, ofile, 'w')
                                else:
                                    dict2mgf(batch_dict, ofile, 'a')  
                                batch_dict = {}
                            break
                        else:
                            tmpline = line.rstrip('\n').split('\t')
                            mz.append(tmpline[0])
                            intensity.append(tmpline[1])
                            anno.append(tmpline[2])
                        line = f1.readline()
                        
        if (counter % batch_size != 0):
            if (counter < batch_size):
                dict2mgf(batch_dict, ofile, 'w')
            else:
                dict2mgf(batch_dict, ofile, 'a')  
                
        if (sign == 1):
            print('Processed all {} records.'.format(counter))
        else:
            print('Processed {} records.'.format(counter))

    with open (peplistfile, 'w') as f:
        for peptide in peptidelist:
            f.write(peptide+'\n')
        
    return peptidelist


def sampling_peptidelist(n_record, ifile, ofile):
    import random
    random.seed(100)
    
    peptidelist = []
    # f1 = open (example_dir+'peptidelist_library.txt', 'r')
    with open (ifile, 'r') as f1:
        for line in f1:
            if line !='':
                peptidelist.append(line.rstrip('\n'))
    peptidelist_nomod = [x for x in peptidelist if re.search('\\(',x) is None]
    peptidelist_len = [x for x in peptidelist_nomod if len(x.split('/')[0])<=30]
    peptidelist_sample = random.sample(peptidelist_len, n_record)
    with open (ofile, 'w') as f:
        f.write('modified_sequence,collision_energy,precursor_charge\n')
        for peptide in peptidelist_sample:
            modseq, ce, mod = peptide.split('_')
            seq, charge = modseq.split('/')
            f.write(seq+','+ce+','+charge+'\n')
    return (peptidelist_sample)


def main():
    data_dir = constants_local.DATA_DIR
    example_dir = constants_local.EXAMPLE_DIR
    if not os.path.exists(example_dir):
        os.makedirs(example_dir)
    
    # get smaller set and count records number
    copy_from_msp(1000, data_dir+'human_synthetic_hcd_selected.msp', example_dir+'human_synthetic_hcd_selected_sub.msp')
    
    # get record number
    n_propel = copy_from_msp(1000000, data_dir+'human_synthetic_hcd_selected.msp')
    # n_propel = 1000000
    
    # store msp files to dictionary from prosit prediction
    spectrum_prosit = from_msp_prosit(example_dir+'peptidelist_pred.msp')
    dict2mgf(spectrum_prosit, example_dir+'peptidelist_pred.mgf')
    
    # # OPTION 1: when msp size is small (limiting to 10000 records)
    # spectrum_propel = from_msp_propel(data_dir+'human_synthetic_hcd_selected.msp')
    # dict2mgf(spectrum_propel, example_dir+'human_synthetic_hcd_selected.mgf')

    # OPTION 2: transform full msp directly to mgf
    ## NOT WORKING
    peptidelist_propel = msp2mgf(n_propel+1000, data_dir+'human_synthetic_hcd_selected.msp',\
                                 data_dir+'human_synthetic_hcd_selected.mgf', \
                                 data_dir+'peptidelist_library.txt')

    # Create peptidelist example
    peptidelist_propel_sample = sampling_peptidelist(100, data_dir+'peptidelist_library.txt', \
                                                     example_dir+'peptidelist.csv')       
            
            
        
if __name__ == "__main__":
    main()
            
            
    

