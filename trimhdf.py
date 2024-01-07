import torch
import os
import re
from argparse import ArgumentParser
import h5py
import shutil

# try: 
#     os.chdir('/Users/xuel12/Documents/Projects/seq2spec/CoSpred/')
#     print("Current directory is {}".format(os.getcwd()))
# except: 
#     print("Something wrong with specified directory. Exception- ", sys.exc_info())
  
import params.constants_local as constants_local
import params.constants_gcp as constants_gcp
import model as model_lib


def get_path_and_filename(input_file):
    folder_path, file_name = os.path.split(input_file)
    basename, suffix = file_name.split('.')
    return folder_path, file_name, basename, suffix
    

def copy_and_rename(src_file):
    folder_path, file_name, basename, suffix = get_path_and_filename(src_file)
    dst_file = os.path.join(folder_path, basename+'_bk.'+suffix)
    shutil.copy(src_file, dst_file)
    return dst_file
    
    
def subset_hdf5(data_path, model_config):
    input_file = copy_and_rename(data_path)

    keys = model_config['x']+model_config['y'] # replace with your keys
    # output_file = 'output.hdf5'
    with h5py.File(input_file, 'r') as f_in:
        with h5py.File(data_path, 'w') as f_out:
            for key in keys:
                if key in f_in:
                    data = f_in[key][...]
                    print('Copying dataset {}...'.format(key))

                    # f_out.create_dataset(key, data=data)
                    if (data.dtype == 'object'):
                        f_out.create_dataset(key, data=data, dtype=dt, compression="gzip")
                    else:
                        f_out.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")

        
def main():    
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', default='', 
                        help='HDF5 file to trim')
    parser.add_argument('-l', '--local', default=False, action='store_true',
                    help='execute in local computer')
    args = parser.parse_args()    
        
    if args.local is True:
        model_dir = constants_local.MODEL_DIR
    else:
        model_dir = constants_gcp.MODEL_DIR
    # load model    
    model, model_config, weights_path = model_lib.load(model_dir, flag_fullspectrum=False, 
                                                       flag_prosit=True, trained=False)
    
    # input file choices
    if args.data_path == '':
        if args.local is True:
            data_path = constants_local.TRAINDATA_PATH
        else:
            data_path = constants_gcp.TRAINDATA_PATH
    else:
        data_path = args.data_path
    print(data_path)
    
    # trim HDF5, only keep required column
    subset_hdf5(data_path, model_config)
    print('Generating HDF5 Done!')
    
if __name__ == "__main__":
    main()
