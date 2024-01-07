import numpy as np
import h5py
import sys
import os
from argparse import ArgumentParser
import random  

# try: 
#     os.chdir('/Users/xuel12/Documents/Projects/seq2spec/CoSpred/')
#     print("Current directory is {}".format(os.getcwd()))
# except: 
#     print("Something wrong with specified directory. Exception- ", sys.exc_info())
    
import params.constants as constants
import params.constants_local as constants_local
import params.constants_gcp as constants_gcp


def slice_hdf5(path, ofile, n_samples=None):
    if n_samples is None:
        n_samples = 1
    else:
        n_samples = int(n_samples)
    # path='/Users/xuel12/Documents/Projects/seq2spec/prosit/local_training/data/holdout_hcd.hdf5'
    # n_samples=100
    # Get a list of the keys for the datasets
    random.seed(100)
    with h5py.File(path, 'r') as f:
        # with h5py.File('/Users/xuel12/Documents/MSdatascience/CS7180AI/project/data/traintest_hcd_1m.hdf5', 'w') as f2:
        with h5py.File(ofile, 'w') as f2:
            print(f.keys())
            dataset_list = list(f.keys())
            
            fulllen = f[dataset_list[0]].shape[0]
            print('Total number of records in original hdf5 is {}'.format(fulllen))
            if (n_samples < fulllen):
                idx = sorted(random.sample(range(0,fulllen), int(n_samples)))
                # idx = range(n_samples)
            else:
                n_samples = range(fulllen)
            
            
            for dset_name in dataset_list:
                print(dset_name)
                # print(f[dset_name].dtype)
                # print(len(f[dset_name].shape))
                # print(f[dset_name].shape)
    
                if (len(f[dset_name].shape) == 2):
                    # print(f[dset_name][:1, :])
                    dset = f2.create_dataset(dset_name, shape=(len(idx), f[dset_name].shape[1]), dtype=f[dset_name].dtype, compression="gzip", compression_opts=9)
                    # dset.write_direct(f[dset_name][:n_samples, :])
                    dset.write_direct(f[dset_name][idx, :])
                elif (len(f[dset_name].shape) == 3):
                    # print(f[dset_name][:1, :, :])
                    dset = f2.create_dataset(dset_name, shape=(len(idx), f[dset_name].shape[1], f[dset_name].shape[2]), dtype=f[dset_name].dtype, compression="gzip", compression_opts=9)
                    # dset.write_direct(f[dset_name][:n_samples, :, :])
                    dset.write_direct(f[dset_name][idx, :, :])
                else:
                    # print(f[dset_name][:1, ])
                    dset = f2.create_dataset(dset_name, shape=(len(idx), ), dtype=f[dset_name].dtype, compression="gzip", compression_opts=9)
                    # dset.write_direct(f[dset_name][:n_samples, ])
                    dset.write_direct(f[dset_name][idx, ])
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


def main():    
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', default='', 
                        help='HDF5 file to trim')
    parser.add_argument('-l', '--local', default=False, action='store_true',
                    help='execute in local computer')
    parser.add_argument('-n', '--n_samples', default=100, 
                        help='number of records to subset')
    args = parser.parse_args()    

    n_samples = args.n_samples
    # input file choices
    if args.data_path == '':
        if args.local is True:
            data_path = constants_local.TRAINDATA_PATH
        else:
            data_path = constants_gcp.TRAINDATA_PATH
    else:
        data_path = args.data_path
    # output file choices    
    if args.local is True:
        # subset smaller input hdf5
        subdata_path = constants_local.DATA_DIR + 'train_{}.hdf5'.format(n_samples)
    else:
        # subset smaller input hdf5
        subdata_path = constants_gcp.DATA_DIR + 'train_{}.hdf5'.format(n_samples)
    print(data_path)
    
    # subset smaller input hdf5
    slice_hdf5(path = data_path, 
               ofile = subdata_path,
               n_samples = n_samples)    
    print('Generating HDF5 Done!')
    
    # # examine inputs
    # f = read_hdf5(constants_local.EXAMPLE_DIR + 'traintest_hcd_100.hdf5')
    
    # # examine weights
    # read_hdf5(constants_local.EXAMPLE_DIR + 'weight_32_0.10211.hdf5')
    # f = h5py.File(constants_local.EXAMPLE_DIR + 'weight_32_0.10211.hdf5', 'r')
    
    # list(f.keys())
    # dset = f['model_weights']
    # list(dset.keys())
    # dset_model = dset['encoder2']
    # print(dset_model.keys())
    # dset_model['encoder2'].keys()

    
if __name__ == "__main__":
    main()


