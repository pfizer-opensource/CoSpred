# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import h5py
import sys
import os

try: 
    os.chdir('/Users/xuel12/Documents/Projects/seq2spec/prosit/local_training_tf2')
    print("Current directory is {}".format(os.getcwd()))
except: 
    print("Something wrong with specified directory. Exception- ", sys.exc_info())
    
    
import constants
import constants_local

def slice_hdf5(path, ofile, n_samples=None):
    import random  

    # path='/Users/xuel12/Documents/Projects/seq2spec/prosit/local_training/data/holdout_hcd.hdf5'
    # n_samples=100
    # Get a list of the keys for the datasets
    random.seed(100)
    with h5py.File(path, 'r') as f:
        # with h5py.File('/Users/xuel12/Documents/MSdatascience/CS7180AI/project/data/traintest_hcd_1m.hdf5', 'w') as f2:
        with h5py.File(ofile, 'w') as f2:
            print(f.keys())
            dataset_list = list(f.keys())
            
#            fulllen = f[dataset_list[0]].shape[0]
#            idx = sorted(random.sample(range(0,fulllen), int(fulllen*prop)))
            idx = range(n_samples)
            
            
            for dset_name in dataset_list:
                # print(dset_name)
                # print(f[dset_name].dtype)
                # print(len(f[dset_name].shape))
                # print(f[dset_name].shape)
    
                if (len(f[dset_name].shape) == 2):
                    # print(f[dset_name][:1, :])
                    dset = f2.create_dataset(dset_name, shape=(len(idx), f[dset_name].shape[1]), dtype=f[dset_name].dtype, compression="gzip", compression_opts=9)
                    dset.write_direct(f[dset_name][:n_samples, :])
                elif (len(f[dset_name].shape) == 3):
                    # print(f[dset_name][:1, :, :])
                    dset = f2.create_dataset(dset_name, shape=(len(idx), f[dset_name].shape[1], f[dset_name].shape[2]), dtype=f[dset_name].dtype, compression="gzip", compression_opts=9)
                    dset.write_direct(f[dset_name][:n_samples, :, :])
                else:
                    # print(f[dset_name][:1, ])
                    dset = f2.create_dataset(dset_name, shape=(len(idx), ), dtype=f[dset_name].dtype, compression="gzip", compression_opts=9)
                    dset.write_direct(f[dset_name][:n_samples, ])
            f2.close()
        f.close()
            
#        sequence_integer_dset = f['sequence_integer']
#        print(sequence_integer_dset.shape)
#        print(sequence_integer_dset.dtype)
#
#        ce_dset = f['collision_energy_aligned_normed']
#        print(ce_dset.shape)
#        charge_dset = f['precursor_charge_onehot']
#        print(charge_dset.shape)
    return dataset_list


# Get a list of the keys for the datasets
def read_hdf5(path, n_samples=None):
    with h5py.File(path, 'r') as f:
        print(f.keys())
        dataset_list = list(f.keys())
        for dset_name in dataset_list:
            print(dset_name)
            print(f[dset_name][:6])
#            print(f[dset_name].dtype)
#            print(f[dset_name].shape)
#            if (len(f[dset_name].shape) == 2):
#                print(f[dset_name][:5, :])
#            elif (len(f[dset_name].shape) == 3):
#                print(f[dset_name][:5, :, :])
#            else:
#                print(f[dset_name][:5, ])
        f.close()

    return dataset_list


# subset smaller input hdf5
data_path = constants_local.TRAINDATA_PATH
subdata_path = constants_local.DATA_DIR + 'train_100.hdf5'

slice_hdf5(path = data_path, 
           ofile = subdata_path,
           n_samples = 100)
# slice_hdf5('/Users/xuel12/Documents/MSdatascience/CS7180AI/project/data/traintest_hcd.hdf5', n_samples = 1000000)

# examine inputs
f = read_hdf5(constants_local.EXAMPLE_DIR + 'traintest_hcd_100.hdf5')

# examine weights
read_hdf5(constants_local.EXAMPLE_DIR + 'weight_32_0.10211.hdf5')
f = h5py.File(constants_local.EXAMPLE_DIR + 'weight_32_0.10211.hdf5', 'r')

list(f.keys())
dset = f['model_weights']
list(dset.keys())
dset_model = dset['encoder2']
print(dset_model.keys())
dset_model['encoder2'].keys()

