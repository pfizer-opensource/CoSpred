import os
import torch
import sys
import numpy as np
import pandas as pd
import re
from pyteomics import mgf
import spectrum_utils.spectrum as sus

import tensorflow as tf
import keras     
import h5py
from argparse import ArgumentParser
import csv 
          
import params.constants as constants
import params.constants_location as constants_location

import io_cospred
import model as model_lib
import rawfile2hdf_prosit

from prosit_model import sanitize, tensorize
from prosit_model.converters import maxquant, msp, diannoutput, generic

from cospred_model.metrics import ComputeMetrics_CPU

global d_spectra
#global d_irt
d_spectra = {}
#d_irt = {}
def prediction_prosit(data, d_spectra, flag_fullspectrum, flag_evaluate=False):
    # check for mandatory keys
    x = io_cospred.get_array(data, d_spectra["config"]["x"])
    # y = io_cospred.get_array(data, d_spectra["config"]["y"])
    keras.backend.set_session(d_spectra["session"])
    with d_spectra["graph"].as_default():
        prediction = d_spectra["model"].predict(
            x, verbose=True, batch_size=constants.PRED_BATCH_SIZE
        )

    if d_spectra["config"]["prediction_type"] == "intensity":
        data["intensities_pred"] = prediction
        data = sanitize.prediction(data, flag_fullspectrum, flag_evaluate)
   # elif d_model["config"]["prediction_type"] == "iRT":
   #     scal = float(d_model["config"]["iRT_rescaling_var"])
   #     mean = float(d_model["config"]["iRT_rescaling_mean"])
   #     data["iRT"] = prediction * np.sqrt(scal) + mean
    else:
        raise ValueError("model_config misses parameter")
    return data


def prediction_transformer(data, d_spectra, flag_fullspectrum=True, flag_evaluate=False):
    # check for mandatory keys
    x = io_cospred.get_array(data, d_spectra["config"]["x"])
    y = io_cospred.get_array(data, d_spectra["config"]["y"])
    x_tr = [torch.tensor(data[x]) for x in d_spectra["config"]["x"]]
    x_tr = torch.cat(x_tr, dim=1)
    print(x_tr.shape)

    # take over whatever gpus are on the system
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        d_spectra["model"] = torch.nn.DataParallel(d_spectra["model"]).to(device)
    x_tr = x_tr.to(device)

    # create batch
    prediction_list = []
    for i in range(0, x_tr.shape[0], constants.PRED_BATCH_SIZE):
        x_batch = x_tr[i : i + constants.PRED_BATCH_SIZE]
        prediction = d_spectra["model"].forward(x_batch)[0]
        prediction_list.append(prediction)
    prediction = torch.cat(prediction_list, dim=0)
    if d_spectra["config"]["prediction_type"] == "intensity":
        data["intensities_pred"] = prediction.cpu().detach().numpy()
        data = sanitize.prediction(data, flag_fullspectrum, flag_evaluate)
   # elif d_model["config"]["prediction_type"] == "iRT":
   #     scal = float(d_model["config"]["iRT_rescaling_var"])
   #     mean = float(d_model["config"]["iRT_rescaling_mean"])
   #     data["iRT"] = prediction * np.sqrt(scal) + mean
    else:
        raise ValueError("model_config misses parameter")
    return data



## Annotate b and y ions to MGF file
def annotateMGF_wSeq(usimgffile, testcsvfile, temp_dir):

    mgfile=mgf.read(usimgffile)
    fragment_tol_mass = 0.35
    fragment_tol_mode = 'Da'
    min_mz = 100
    max_mz = 1400
    min_intensity = 0.05
    
    csv_df = pd.read_csv(testcsvfile)
    csv_df['title'] = 'mzspec:repoID:'+csv_df['raw_file']+':scan:'+csv_df['scan_number'].astype(str)      
    csv_df['modifiedseq'] = csv_df['modified_sequence']
    
    mzs_df = []
    
    for index, row in csv_df.iterrows():  
        if (index % 100 == 0):
            print('MS2 Annotation Progress: {}%'.format(index/csv_df.shape[0]*100))

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
            spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge, mz, intensity,
                                    retention_time=retention_time, 
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
                
            intensity_annotations = ";".join([str(element) for element in spectrum.intensity])
            mz_annotations = ";".join([str(element) for element in spectrum.mz])
            ion_annotations = ";".join([re.sub('/\S+','', str(element)) for element in spectrum.annotation.tolist()])
            mzs_df.append(pd.Series([seq, intensity_annotations, mz_annotations, ion_annotations]))
        except:
            next
            
    # construct dataframe for annotated MS2
    mzs_df = pd.concat(mzs_df, axis = 1).transpose()
    mzs_df.columns =['seq','intensity_annotations', 'mz_annotations', 'ion_annotations']
    mzs_df.to_csv(temp_dir+'annotatedMGF.csv', index=False)

    return mzs_df



## Contruct ML friendly spectra matrix
def generateCSV_wSeq(usimgffile, reformatmgffile, predict_input, annotation_results, csvfile, temp_dir):
    csv_df = pd.read_csv(predict_input)
    csv_df['title'] = 'mzspec:repoID:'+csv_df['raw_file']+':scan:'+csv_df['scan_number'].astype(str)      
    csv_df['file'] = csv_df['raw_file']
    csv_df['scan'] = csv_df['scan_number']
    csv_df['charge'] = csv_df['precursor_charge']
    csv_df['seq'] = csv_df['sequence']
    csv_df['modifiedseq'] = csv_df['modified_sequence']
    
    assert "file" in csv_df.columns
    assert "scan" in csv_df.columns
    assert "charge" in csv_df.columns
    assert "seq" in csv_df.columns
    assert "modifiedseq" in csv_df.columns
    assert "proforma" in csv_df.columns
    assert "score" in csv_df.columns
    assert "reverse" in csv_df.columns
    
    # get annotation MS2
    annotation_results.columns = ['seq','intensities','masses','matches_raw']
    
    # retrieve spectrum of PSM from MGF
    spectra=mgf.read(usimgffile)
    spectra[0]
    mzs_df = []
    for index, row in csv_df.iterrows():  
        if (index % 100 == 0):
            print('Generating CSV Progress: {}%'.format(index/csv_df.shape[0]*100))
                
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
        mgf.write(spectra_new, output = reformatmgffile)
        spectra_origin.close()
    else:
        print("The reformatted MGF file does not exist")
        
    print('MGF file with new TITLE was created!')



def predict(predict_input, predict_dir, predict_format, testdata, 
            flag_prosit, flag_fullspectrum, flag_evaluate):
    from statistics import mean 

    if os.path.exists(predict_input):
        df = pd.read_csv(predict_input)
        if flag_evaluate is True:
            data = tensorize.hdf5(df, hdf5file=testdata)
        else:
            data = tensorize.csv(df, flag_fullspectrum)
    else:
        pass       

    if flag_prosit is True:
        pred = prediction_prosit(data, d_spectra, flag_fullspectrum, flag_evaluate)
    else: 
        pred = prediction_transformer(data, d_spectra, flag_fullspectrum, flag_evaluate)
    
    if flag_evaluate is True:
        y_true = torch.tensor(data['intensities_raw'])
        y_pred = torch.tensor(pred['intensities_pred'])
        seq, charge, ce = df['modified_sequence'], df['precursor_charge'], df['collision_energy']
        
        # calculate prediction metrics
        metrics = ComputeMetrics_CPU(true=y_true, pred=y_pred, seq=seq, charge=charge, ce=ce)
        metrics_byrecord = pd.DataFrame(metrics.return_metrics_byrecord())

        # calculate mean of metrics
        metrics_mean = metrics.return_metrics_mean()
        metrics_df = pd.DataFrame.from_dict(metrics_mean, orient='index')
        
        # OPTIONAL: calculate spectral angle
        spectralangle_df = pd.DataFrame([{'spectral_angle': mean(pred['spectral_angle'])}]).T
        metrics_df = pd.concat([metrics_df, spectralangle_df], ignore_index=False)
        
        model_name = d_spectra['weights_path'].split('/')[-1]
        metrics_df.columns = [model_name]
        metrics_df[model_name] = metrics_df[model_name].astype(float)

        # store metrics in csv file
        metrics_folder = predict_dir + model_name + '/'
        os.makedirs(metrics_folder, exist_ok=True) 
        metrics_byrecord.to_csv(metrics_folder + 'metrics_byrecord.csv')
        metrics_df.to_csv(metrics_folder + 'metrics.csv')

        # plot Precision-Recall curve, ROC curve
        metrics.plot_PRcurve_micro(metrics_folder)
        metrics.plot_PRcurve_sample(metrics_folder)
        metrics.plot_PRcurve_macro(metrics_folder)
        metrics.plot_ROCcurve_macro(metrics_folder)
        metrics.plot_ROCcurve_micro(metrics_folder)

    if (predict_format == 'maxquant'):
        df_pred = maxquant.convert_prediction(pred)
        maxquant.write(df_pred, predict_dir+'peptidelist_pred.txt')
    elif (predict_format == 'msp'):
        df_pred = msp.Converter(pred, predict_dir+'peptidelist_pred.msp', 
                                flag_fullspectrum).convert()        
    else:
        print("Unknown Formatted Requested.")
    print("Spectrum predicition DONE!")
    return df_pred
    
    
def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--trained', default=True, action='store_false',
                        help='turn off loading best existing model')
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help='full spectrum presentation')
    parser.add_argument('-p', '--prosit', default=False, action='store_true',
                        help='predict with prosit model')
    parser.add_argument('-e', '--evaluate', default=False, action='store_true',
                        help='evaulate model with metrics')
    args = parser.parse_args()    
    
    model_dir = constants_location.MODEL_DIR
    predict_format = constants_location.PREDICT_FORMAT
    predict_input = constants_location.PREDICT_INPUT
    predict_dir = constants_location.PREDICT_DIR
    testdata = constants_location.TESTDATA_PATH
    
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    
    if args.prosit is True:   
        d_spectra["graph"] = tf.Graph()
        with d_spectra["graph"].as_default():
            d_spectra["session"] = tf.compat.v1.Session()
            with d_spectra["session"].as_default():
                d_spectra["model"], d_spectra["config"], d_spectra['weights_path'] = model_lib.load(
                    model_dir, 
                    args.full,
                    args.prosit,
                    args.trained
                )
        # d_irt["graph"] = tf.Graph()
        # with d_irt["graph"].as_default():
        #    d_irt["session"] = tf.Session()
        #    with d_irt["session"].as_default():
        #        d_irt["model"], d_irt["config"] = model.load(constants.MODEL_IRT,
        #                    trained=True)
        #       d_irt["model"].compile(optimizer="adam", loss="mse")
    else:
        d_spectra["model"], d_spectra["config"], d_spectra['weights_path'] = model_lib.load(
            model_dir, 
            args.full,
            args.prosit,
            args.trained
        )
        
    # create prediction list
    if not os.path.isfile(predict_input):
        temp_dir = constants_location.TEMP_DIR                   
        testpeptides = constants_location.TESTPEPTIDES_PATH
        usimgffile = constants_location.REFORMAT_TEST_USITITLE_PATH
        reformatmgffile = constants_location.REFORMAT_TEST_PATH
        
        # if b,y ion prediction, annotation is required
        annotation_results = annotateMGF_wSeq(usimgffile, testpeptides, temp_dir)
        # match peptide from PSM with spectra MGF
        dataset = generateCSV_wSeq(usimgffile, reformatmgffile, testpeptides, annotation_results, 
                            predict_input, temp_dir)
        # transform to hdf5
        dataset = rawfile2hdf_prosit.constructDataset(predict_input)
        rawfile2hdf_prosit.to_hdf5(dataset, testdata)
        print('Generating HDF5 Testset Done!')
        
    print('MODEL LOADING finished. Start PREDICTION...')
    
    if predict_format == 'maxquant':
        # Maxquant output
        df_pred = predict(predict_input, predict_dir, 'maxquant', testdata,
                          args.prosit, args.full, args.evaluate)
    elif predict_format == 'msp':  
        # MSP output
        df_pred = predict(predict_input, predict_dir, 'msp', testdata, 
                          args.prosit, args.full, args.evaluate)
    else:
        print('PREDICT_FORMAT could only be maxquant or msp')
        

if __name__ == "__main__":
    main()
