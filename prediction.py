import os
import sys
import logging
import warnings
import shutil
import time
import torch
import pandas as pd
import re
from pyteomics import mgf
import spectrum_utils.spectrum as sus
from datasets import Dataset
import h5py
import numpy as np
from datasets import Dataset
from statistics import mean

import tensorflow as tf
import keras
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader

import params.constants as constants
import params.constants_location as constants_location

import io_cospred
import model as model_lib
import rawfile2hdf_prosit, rawfile2hdf_cospred

from prosit_model import sanitize, tensorize
from prosit_model.converters import maxquant, msp

from cospred_model.metrics import ComputeMetrics_CPU


# initialize global variables
global d_spectra
# global d_irt
d_spectra = {}
# d_irt = {}

def combine_hdf5_files(predict_input, predict_result_file, combined_file):
    """
    Combine result HDF5 files with prediction input variable into a single HDF5 file.

    Args:
        predict_input (dictionary/Hugginface Dataset/HDF5 file): Input as dictionary, Hugging Face Dataset, or HDF5 file with data and metadata.
        predict_result_file (str): Path to the batch prediction HDF5 file.
        combined_file (str): Path to save the combined HDF5 file.
   """       
    # Define the keys to keep from result_h5 and combined_h5
    combined_keys_to_keep = ["collision_energy_aligned_normed", "intensities_raw", 
                           "masses_pred", "precursor_charge_onehot", "sequence_integer"]
    result_keys_to_keep = ["intensities_pred"]

    with h5py.File(predict_result_file, "r") as result_h5:
        # Open the combined HDF5 file for writing
        with h5py.File(combined_file, "w") as combined_h5:
            # Copy only the specified datasets from result_h5 to combined_h5
            for key in result_keys_to_keep:
                if key in result_h5:
                    combined_h5.create_dataset(key, data=result_h5[key][:])
                else:
                    logging.warning(f"Warning: Key '{key}' not found in result_h5.")
            # Append or merge datasets from predict_input
            if isinstance(predict_input, Dataset):  # Handle Hugging Face Dataset objects
                for column in predict_input.column_names:
                    if column in combined_keys_to_keep:
                        data = np.array(predict_input[column])
                        if column in combined_h5:
                            combined_h5[column][...] = np.concatenate((combined_h5[column][:], data), axis=0)
                        else:
                            combined_h5.create_dataset(column, data=data)
            elif isinstance(predict_input, dict):  # Handle dictionary input
                for key in predict_input.keys():
                    if key in combined_keys_to_keep:
                        if key in combined_h5:
                            combined_h5[key][...] = np.concatenate((combined_h5[key][:], predict_input[key][:]), axis=0)
                        else:
                            combined_h5.create_dataset(key, data=predict_input[key])                              
            elif isinstance(predict_input, str) and (predict_input.endswith(".h5") or predict_input.endswith(".hdf5")):  # Handle HDF5 file input
                with h5py.File(predict_input, "r") as input_h5:
                    for key in input_h5.keys():
                        if key in combined_keys_to_keep:
                            if key in combined_h5:
                                # If the dataset already exists, concatenate the data
                                combined_h5[key][...] = np.concatenate((combined_h5[key][:], input_h5[key][:]), axis=0)
                            else:
                                # If the dataset does not exist, create it
                                combined_h5.create_dataset(key, data=input_h5[key][:])
            else:
                raise ValueError("Unsupported predict_input type. Must be a dictionary, Hugging Face Dataset, or HDF5 file.")
        # logging.info(f"Batch Prediction saved to {combined_file}")


def concatenate_hdf5(output_dir, predict_input, predict_result_file, combined_file, flag_fullspectrum=False):
    # Step 1: Read all HDF5 batch files
    batch_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".h5")]
    logging.info(f"Found {len(batch_files)} batch files to concatenate.")

    # Initialize lists to store concatenated data
    all_predictions = []
    # all_batch_indices = []

    # Read and concatenate all batch files
    for batch_file in batch_files:
        with h5py.File(batch_file, "r") as h5f:
            all_predictions.append(h5f["intensities_pred"][:])
            
    # Combine all predictions and batch indices
    combined_predictions = np.vstack(all_predictions)
    logging.info(f"Predictions Dimension: {combined_predictions.shape}")

    # Step 2: Save the batch data to a single HDF5 file
    with h5py.File(predict_result_file, "w") as h5f:
        h5f.create_dataset("intensities_pred", data=combined_predictions)
    # logging.info(f"Prediction Values saved to {predict_result_file}")

    # Step 3: Read the single result HDF5 file for combination
    combine_hdf5_files(predict_input, predict_result_file, combined_file)
    
    combined_dict = {}
    with h5py.File(combined_file, "r") as h5f:
        for key in h5f.keys():
            combined_dict[key] = h5f[key][:]

    return combined_dict


def concatenate_hdf5_chunk(hdf5_dir, predict_result_file, flag_resume):
    """
    Concatenate all HDF5 files in a directory by their keys and save to a single HDF5 file.

    Args:
        hdf5_dir (str): Directory containing HDF5 files to concatenate.
        predict_result_file (str): Path to save the concatenated chunk HDF5 file.

    Returns:
        combined_dict (dict): Dictionary containing concatenated data from all HDF5 files.
    """

    # Step 1: Collect all HDF5 files in the directory
    hdf5_files = [os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith(".h5")]
    logging.info(f"Found {len(hdf5_files)} HDF5 files to concatenate.")

    # Step 2: Open the output file in append mode
    with h5py.File(predict_result_file, "w") as combined_h5:
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, "r") as h5f:
                for key in h5f.keys():
                    data = h5f[key][:]
                    if key in combined_h5:
                        # Append data to the existing dataset
                        combined_h5[key].resize((combined_h5[key].shape[0] + data.shape[0]), axis=0)
                        combined_h5[key][-data.shape[0]:] = data
                    else:
                        # Create a new dataset and enable resizing
                        combined_h5.create_dataset(
                            key, data=data, maxshape=(None,) + data.shape[1:], chunks=True
                        )
    logging.info(f"Prediction Values saved to {predict_result_file}")

    combined_dict = {}
    # if the size is too big has to be done by resuming, not practical to load entire set in memory
    if not flag_resume:
        with h5py.File(predict_result_file, "r") as h5f:
            for key in h5f.keys():
                combined_dict[key] = h5f[key][:]

    return combined_dict


# Annotate b and y ions to MGF file
def annotateMGF_wSeq(usimgffile, testcsvfile, temp_dir):

    mgfile = mgf.read(usimgffile)
    fragment_tol_mass = 0.35
    fragment_tol_mode = 'Da'
    min_mz = 100
    max_mz = 1400
    min_intensity = 0.05

    csv_df = pd.read_csv(testcsvfile, index_col=False)
    csv_df['title'] = 'mzspec:repoID:'+csv_df['raw_file'] + \
        ':scan:'+csv_df['scan_number'].astype(str)
    csv_df['modifiedseq'] = csv_df['modified_sequence']

    mzs_df = []

    for index, row in csv_df.iterrows():
        if (index % 100 == 0):
            logging.info('MS2 Annotation Progress: {}%'.format(
                index/csv_df.shape[0]*100))

        try:
            # retrieve spectrum of PSM from MGF
            proforma = row['proforma']
            seq = row['modifiedseq']
            spectrum_dict = mgfile.get_spectrum(row['title'])
            # modifications = {}
            identifier = spectrum_dict['params']['title']
            # peptide = spectrum_dict['params']['seq']
            # ce = spectrum_dict['params']['ce']
            # method = spectrum_dict['params']['method']
            # scan = spectrum_dict['params']['scans']
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

            intensity_annotations = ";".join(
                [str(element) for element in spectrum.intensity])
            mz_annotations = ";".join([str(element)
                                      for element in spectrum.mz])
            ion_annotations = ";".join(
                [re.sub('/\S+', '', str(element)) for element in spectrum.annotation.tolist()])
            mzs_df.append(
                pd.Series([seq, intensity_annotations, mz_annotations, ion_annotations]))
        except:
            next

    # construct dataframe for annotated MS2
    mzs_df = pd.concat(mzs_df, axis=1).transpose()
    mzs_df.columns = ['seq', 'intensity_annotations',
                      'mz_annotations', 'ion_annotations']
    mzs_df.to_csv(temp_dir+'annotatedMGF.csv', index=False)

    return mzs_df


# Contruct ML friendly spectra matrix
def generateCSV_wSeq(usimgffile, reformatmgffile, predict_csv, annotation_results, csvfile, temp_dir):
    csv_df = pd.read_csv(predict_csv, index_col=False)
    csv_df['title'] = 'mzspec:repoID:'+csv_df['raw_file'] + \
        ':scan:'+csv_df['scan_number'].astype(str)
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
    annotation_results.columns = [
        'seq', 'intensities', 'masses', 'matches_raw']

    # retrieve spectrum of PSM from MGF
    spectra = mgf.read(usimgffile)
    spectra[0]
    mzs_df = []
    for index, row in csv_df.iterrows():
        if (index % 100 == 0):
            logging.info('Generating CSV Progress: {}%'.format(
                index/csv_df.shape[0]*100))

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
            mzs_df.append(pd.Series([raw_file, scan_number, sequence, score,
                                     modified_sequence, proforma,
                                     mod_num, reverse,
                                     collision_energy, charge_state,
                                     retention_time, method, mod_num]))
        except:
            next

    mzs_df = pd.concat(mzs_df, axis=1).transpose()
    mzs_df.columns = ['raw_file', 'scan_number', 'sequence', 'score',
                      'modified_sequence', 'proforma',
                      'mod_num', 'reverse',
                      'collision_energy', 'precursor_charge', 'retention_time',
                      'method', 'mod_num']
    mzs_df['collision_energy_aligned_normed'] = mzs_df['collision_energy']/100.0

    # construct CSV
    annotation_results_new = annotation_results.reset_index(drop=True)
    mzs_df_new = mzs_df.reset_index(drop=True)

    dataset = pd.concat([mzs_df_new, annotation_results_new], axis=1)
    dataset = dataset.dropna()
    dataset.to_csv(csvfile, index=False)

    logging.info('Generating CSV ... DONE!')

    modifyMGFtitle(usimgffile, reformatmgffile, temp_dir)
    return dataset


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
        mgf.write(spectra_new, output=reformatmgffile)
        spectra_origin.close()
    else:
        logging.error("The reformatted MGF file does not exist")

    logging.info('MGF file with new TITLE was created!')


def prediction_prosit(predict_batch_dir, predict_result_file, combined_file, 
                      predict_input, hf_dataset, d_spectra, flag_fullspectrum, 
                      flag_evaluate=False, flag_chunk=False):
    """
    Perform batch predictions using either HDF5 or Hugging Face Dataset based on flag_chunk.

    Args:
        predict_batch_dir (str): Directory to save batch predictions.
        predict_result_file (str): Path to save the combined prediction result.
        combined_file (str): Path to save the final combined HDF5 file.
        predict_input (str or dict): Input data, either as an HDF5 file path or a dictionary.
        hf_dataset (Dataset): Hugging Face Dataset containing input data but no metadata.
        d_spectra (dict): Model and configuration dictionary.
        flag_fullspectrum (bool): Whether to use full spectrum.
        flag_evaluate (bool): Whether to evaluate predictions.
        flag_chunk (bool): Whether to use Hugging Face Dataset for chunked processing.
    """
    # Ensure the batch directory exists
    if os.path.exists(predict_batch_dir):
        shutil.rmtree(predict_batch_dir)
    os.makedirs(predict_batch_dir, exist_ok=True)
    batch_size = constants.PRED_BATCH_SIZE

    if flag_chunk:
        # Use Hugging Face Dataset for chunked processing
        num_batches = int(np.ceil(len(hf_dataset) / batch_size))
    else:
        # Use HDF5-based processing or dictionary input
        if isinstance(predict_input, str) and (predict_input.endswith(".h5") or predict_input.endswith(".hdf5")):
            with h5py.File(predict_input, "r") as h5f:
                x = [np.array(h5f[key]) for key in d_spectra["config"]["x"]]
        elif isinstance(predict_input, dict):
            x = [np.array(predict_input[key]) for key in d_spectra["config"]["x"]]
        else:
            raise ValueError("Unsupported predict_input type. Must be an HDF5 file path or a dictionary.")
        num_batches = int(np.ceil(len(x[0]) / batch_size))
    
    # Set the session for Keras
    keras.backend.set_session(d_spectra["session"])
    # Initialize all variables in the session
    with d_spectra["graph"].as_default():
        d_spectra["session"].run(tf.compat.v1.global_variables_initializer())

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        if flag_chunk:
            end_idx = min((batch_idx + 1) * batch_size, len(hf_dataset))
            # Select the batch
            batch = hf_dataset.select(range(start_idx, end_idx))
            x_batch = [np.array(batch[column]) for column in d_spectra["config"]["x"]]
        else:
            end_idx = min((batch_idx + 1) * batch_size, len(x[0]))
            x_batch = [element[start_idx:end_idx] for element in x]
        logging.info(f"Processing batch {batch_idx + 1}/{num_batches}, batch size: {len(x_batch[0])}")

        # Perform prediction for the batch, ensure prediction is in float16
        with d_spectra["graph"].as_default():
            prediction = d_spectra["model"].predict(x_batch, verbose=True, batch_size=batch_size).astype(np.float16)

        # Save the batch predictions to an HDF5 file
        batch_file = os.path.join(predict_batch_dir, f"batch_{batch_idx + 1}.h5")
        with h5py.File(batch_file, "w") as h5f:
            h5f.create_dataset("intensities_pred", data=prediction)

    logging.info(f"PREDICTION computing ... DONE! All result batches saved to {predict_batch_dir}")

    combined_dict = concatenate_hdf5(predict_batch_dir, predict_input, predict_result_file, 
                                     combined_file, flag_fullspectrum)

    # Sanitize the combined data
    if d_spectra["config"]["prediction_type"] == "intensity":
        combined_dict = sanitize.prediction(combined_dict, flag_fullspectrum, flag_evaluate)
    else:
        raise ValueError("model_config misses parameter")

    return combined_dict


def prediction_transformer(predict_batch_dir, predict_result_file, combined_file, 
                           predict_input, hf_dataset, d_spectra, flag_fullspectrum=True, 
                           flag_evaluate=False, flag_chunk=False):
    """
    Perform batch predictions using either HDF5 or Hugging Face Dataset based on flag_chunk.

    Args:
        predict_batch_dir (str): Directory to save batch predictions.
        predict_result_file (str): Path to save the combined prediction result.
        combined_file (str): Path to save the final combined HDF5 file.
        predict_input (str or dict): Input data, either as an HDF5 file path or a dictionary.
        hf_dataset (Dataset): Hugging Face Dataset containing input data.
        d_spectra (dict): Model and configuration dictionary.
        flag_fullspectrum (bool): Whether to use full spectrum.
        flag_evaluate (bool): Whether to evaluate predictions.
        flag_chunk (bool): Whether to use Hugging Face Dataset for chunked processing.
    """
    # Ensure the batch directory exists
    if os.path.exists(predict_batch_dir):
        shutil.rmtree(predict_batch_dir)
    os.makedirs(predict_batch_dir, exist_ok=True)
    batch_size = constants.PRED_BATCH_SIZE

    # Set the device (CPU or GPU)
    d_spectra["device"] = 'cpu'
    if torch.cuda.is_available():
        d_spectra["device"] = torch.cuda.current_device()
        d_spectra["model"] = torch.nn.DataParallel(d_spectra["model"]).to(d_spectra["device"])

    # Prepare data for batch processing
    if flag_chunk:
        num_batches = int(np.ceil(len(hf_dataset) / batch_size))
        data_generator = (
            hf_dataset.select(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(hf_dataset))))
            for batch_idx in range(num_batches)
        )
    else:
        # Use HDF5-based processing or dictionary input
        if isinstance(predict_input, str) and (predict_input.endswith(".h5") or predict_input.endswith(".hdf5")):
            with h5py.File(predict_input, "r") as h5f:
                x = [torch.tensor(np.array(h5f[column])) for column in d_spectra["config"]["x"]]
        elif isinstance(predict_input, dict):
            x = [torch.tensor(np.array(predict_input[column])) for column in d_spectra["config"]["x"]]
        else:
            raise ValueError("Unsupported predict_input type. Must be an HDF5 file path or a dictionary.")
        x = torch.cat(x, dim=1)
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        num_batches = len(dataloader)
        data_generator = ((x_batch,) for x_batch, in dataloader)

    # Perform predictions in batches
    for batch_idx, batch in enumerate(data_generator):
        if flag_chunk:
            x_batch = [torch.tensor(np.array(batch[column])) for column in d_spectra["config"]["x"]]
            x_batch = torch.cat(x_batch, dim=1)
        else:
            x_batch = batch[0]

        logging.info(f"Processing batch {batch_idx + 1}/{num_batches}, batch size: {x_batch.shape[0]}")

        # Perform prediction for the batch and ensure x_batch is cast to float16
        x_batch = x_batch.to(d_spectra["device"])

        # Ensure the model is also in float16 precision
        prediction = d_spectra["model"].forward(x_batch)[0].half()

        # Save the batch predictions to an HDF5 file
        batch_file = os.path.join(predict_batch_dir, f"batch_{batch_idx + 1}.h5")
        with h5py.File(batch_file, "w") as h5f:
            h5f.create_dataset("intensities_pred", data=prediction.cpu().detach().numpy())

        # Clear memory
        del x_batch, prediction
        torch.cuda.empty_cache()  # Optional: Clear GPU memory if using CUDA
    # logging.info(f"All prediction batches saved to {predict_batch_dir}")

    # Concatenate batch files
    combined_dict = concatenate_hdf5(predict_batch_dir, predict_input, predict_result_file, 
                                     combined_file, flag_fullspectrum)

    # Process predictions based on the model configuration
    if d_spectra["config"]["prediction_type"] == "intensity":
        combined_dict = sanitize.prediction(combined_dict, flag_fullspectrum, flag_evaluate)
    else:
        raise ValueError("model_config misses parameter")

    return combined_dict


def convert_and_save_predictions(pred, predict_filename, predict_format, flag_fullspectrum):
    if (predict_format == 'maxquant'):
        df_pred = maxquant.convert_prediction(pred)
        maxquant.write(df_pred, predict_filename)
    elif (predict_format == 'msp'):
        df_pred = msp.Converter(pred, predict_filename,
                                flag_fullspectrum).convert()
    else:
        logging.error("Unknown Formatted Requested.")
    return df_pred


def evaluate_predictions(predict_dict, ref_dict, predict_df, predict_dir):
    """
    Evaluate predictions and calculate metrics.

    Args:
        predict_dict (dict): Dictionary containing predicted values (e.g., 'intensities_pred').
        ref_dict (dict): Dictionary containing ground truth values and metadata.
        predict_df (pd.DataFrame): DataFrame containing ground truth values and metadata.
        predict_dir (str): Directory to save evaluation metrics and plots.

    Returns:
        None
    """
    from statistics import mean

    # Extract ground truth and predicted values
    y_pred = torch.tensor(predict_dict['intensities_pred'])
    y_true = torch.tensor(ref_dict['intensities_raw'])
    seq, charge, ce = predict_df['modified_sequence'], predict_df['precursor_charge'], predict_df['collision_energy']

    # Calculate prediction metrics
    metrics = ComputeMetrics_CPU(true=y_true, pred=y_pred, seq=seq, charge=charge, ce=ce)
    metrics_byrecord = pd.DataFrame(metrics.return_metrics_byrecord())

    # Calculate mean of metrics
    metrics_mean = metrics.return_metrics_mean()
    metrics_df = pd.DataFrame.from_dict(metrics_mean, orient='index')

    # OPTIONAL: Calculate spectral angle
    if 'spectral_angle' in predict_dict:
        spectralangle_df = pd.DataFrame([{'spectral_angle': mean(predict_dict['spectral_angle'])}]).T
        metrics_df = pd.concat([metrics_df, spectralangle_df], ignore_index=False)

    # Add model name to metrics
    model_name = d_spectra['weights_path'].split('/')[-1]
    metrics_df.columns = [model_name]
    metrics_df[model_name] = metrics_df[model_name].astype(float)

    # Save metrics to CSV files
    metrics_folder = os.path.join(predict_dir, model_name)
    os.makedirs(metrics_folder, exist_ok=True)
    metrics_byrecord.to_csv(os.path.join(metrics_folder, 'metrics_byrecord.csv'))
    metrics_df.to_csv(os.path.join(metrics_folder, 'metrics.csv'))

    # Plot Precision-Recall curve and ROC curve
    metrics.plot_PRcurve_micro(metrics_folder)
    metrics.plot_PRcurve_sample(metrics_folder)
    metrics.plot_PRcurve_macro(metrics_folder)
    metrics.plot_ROCcurve_macro(metrics_folder)
    metrics.plot_ROCcurve_micro(metrics_folder)

    logging.info(f"[USER] Evaluation metrics saved to {metrics_folder}")


def initializeDir(list_dir, flag_resume):
    # iterate through list of directory to initiate
    for tmp_dir in list_dir:
        if not flag_resume and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    
    # # Directory to save predicted library files
    # if os.path.exists(predict_hdf5_dir):
    #     shutil.rmtree(predict_hdf5_dir)
    # os.makedirs(predict_hdf5_dir, exist_ok=True)

    # # Directory to save chunk HDF5 files
    # if os.path.exists(predict_chunk_dir):
    #     shutil.rmtree(predict_chunk_dir)
    # os.makedirs(predict_chunk_dir, exist_ok=True)
    
    # # Directory to save batch HDF5 files
    # if os.path.exists(predict_batch_dir):
    #     shutil.rmtree(predict_batch_dir)
    # os.makedirs(predict_batch_dir, exist_ok=True)
    
    # # Directory to save predicted library files
    # if os.path.exists(predict_lib_dir):
    #     shutil.rmtree(predict_lib_dir)
    # os.makedirs(predict_lib_dir, exist_ok=True)
    

def arrowchunk_to_chunkdict(arrow_chunk_dir, chunkname, predict_hdf5_dir,
                            flag_evaluate, flag_fullspectrum):
    chunkfile = os.path.join(arrow_chunk_dir, chunkname)
    logging.info(f"Processing chunk: {chunkfile}")
    chunk_hdf5 = os.path.join(predict_hdf5_dir, f"{chunkname}.h5")
    io_cospred.arrow_chunk_to_hdf5(chunkfile, chunk_hdf5)
    chunk_dict = io_cospred.read_hdf5_to_dict(chunk_hdf5)

    # Remove keys from chunk_dict if more than 1D               
    chunk_dict = io_cospred.remove_keys_with_2darray(chunk_dict)
    
    # Convert chunk_dict to pandas DataFrame
    chunk_df = pd.DataFrame.from_dict(chunk_dict)
    chunk_df = io_cospred.fixEncoding(chunk_df, 'modified_sequence')
    
    # ESSENTIAL: to add mass_pred and intensity_raw columns
    if flag_evaluate is True:
        chunk_dict = tensorize.hdf5(chunk_df, chunk_hdf5)
    else:
        chunk_dict = tensorize.csv(chunk_df, flag_fullspectrum)

    # ESSENTIAL: to have sequence_integer column, convert hdf5 to hugginface Dataset (Three array for predication only)
    chunk_ds = io_cospred.genDataset(chunk_hdf5, None, flag_chunk=False)

    return chunk_dict, chunk_ds, chunk_df


def predict(predict_csv, predict_dir, predict_format, predict_hdf5, predict_ds,
            flag_prosit, flag_fullspectrum, flag_evaluate, flag_chunk, flag_resume):
    
    predict_hdf5_dir = constants_location.PREDICT_HDF5_DIR   
    predict_chunk_dir = constants_location.PREDICT_CHUNK_DIR
    predict_batch_dir = constants_location.PREDICT_BATCH_DIR
    predict_lib_dir = constants_location.PREDICT_LIB_DIR  # predicted library directory

    predict_chunk_result_file = constants_location.PREDICT_CHUNK_RESULT_FILE  # chunk result file path
    predict_result_file = constants_location.PREDICT_RESULT_FILE  # chunk combined result file path
    predict_batch_result_file = constants_location.PREDICT_BATCH_RESULT_FILE  # batch result file path
    combined_batch_file = constants_location.PREDICT_BATCH_COMBINED_FILE  # batch combined result file path
    speclib_filename = constants_location.PREDICT_LIB_FILENAME
    arrow_chunk_dir = constants_location.PREDDATASET_PATH
    
    if predict_format == 'msp':
        speclib_file = os.path.join(predict_dir, speclib_filename + ".msp")
    else:
        speclib_file = os.path.join(predict_dir, speclib_filename + ".txt")

    if os.path.exists(predict_csv):       
        dir_list = [predict_hdf5_dir, predict_chunk_dir, predict_batch_dir, predict_lib_dir]
        initializeDir(dir_list, flag_resume)
        predict_df = pd.read_csv(predict_csv, index_col=None)
        if flag_chunk:
            if not flag_resume:
                ### Iterate through Arrow chunks if flag_chunk is True ###           
                logging.info("Processing dataset in Chunks ...")
                for chunkname in os.listdir(arrow_chunk_dir):
                    if chunkname.startswith("chunk_") and (not chunkname.endswith(".h5")):
                        chunk_dict, chunk_ds, chunk_df = arrowchunk_to_chunkdict(
                            arrow_chunk_dir, chunkname, predict_hdf5_dir, flag_evaluate, flag_fullspectrum)

                        # Perform predictions on the chunk
                        if flag_prosit:
                            pred = prediction_prosit(
                                predict_batch_dir, predict_batch_result_file, combined_batch_file,
                                chunk_dict, chunk_ds, d_spectra, flag_fullspectrum, flag_evaluate,
                                flag_chunk=True)
                        else:
                            pred = prediction_transformer(
                                predict_batch_dir, predict_batch_result_file, combined_batch_file,
                                chunk_dict, chunk_ds, d_spectra, flag_fullspectrum, flag_evaluate,
                                flag_chunk=True)
                            
                        # Save the chunk output to a unique file, all keys include mass, intensity and sequence_integer
                        chunk_output_file = os.path.join(predict_chunk_dir, f"prediction_output_{chunkname}.h5")
                        with h5py.File(chunk_output_file, "w") as h5f:
                            for key, value in pred.items():
                                h5f.create_dataset(key, data=value)
                        logging.info(f"{chunkname} Prediction output saved to {chunk_output_file}")

                        # Save the chunk predictions in the requested format  
                        if predict_format == 'msp':
                            speclib_chunk_filename = os.path.join(predict_lib_dir, f"{speclib_filename}_{chunkname}.msp")
                        else:
                            speclib_chunk_filename = os.path.join(predict_lib_dir, f"{speclib_filename}_{chunkname}.txt")
                        convert_and_save_predictions(pred, speclib_chunk_filename, predict_format, flag_fullspectrum)

                        # Evaluate chunk predictions if flag_evaluate is True
                        if flag_evaluate is True:
                            eval_dir = os.path.join(predict_lib_dir, f"evaluation_{speclib_filename}_{chunkname}")
                            evaluate_predictions(pred, chunk_dict, chunk_df, eval_dir)
                ### End of processing all Arrow chunks ###
            else:
                logging.info("Using Preprocessed Chunks ...")

            # Combine all chunk files into a single HDF5 file
            combined_dict = concatenate_hdf5_chunk(predict_chunk_dir, predict_result_file, flag_resume)

            # # remove all chunk files after combining, except for the combined file
            # for filename in os.listdir(predict_chunk_dir):
            #     file_path = os.path.join(predict_chunk_dir, filename)
            #     if file_path != predict_chunk_result_file:  # Keep the combined chunk result file
            #         os.remove(file_path)
            # logging.info(f"[USER] All chunk files removed except for the combined prediction file: {predict_result_file}")
            
            # Combine all spectra library chunk files into a single file
            with open(speclib_file, 'w') as outfile:
                logging.info(f"Combining predicted spectra library files from: {predict_lib_dir}")
                for speclib_chunk in os.listdir(predict_lib_dir):
                    if speclib_chunk.startswith(speclib_filename+"_chunk") and (speclib_chunk.endswith(".msp") or speclib_chunk.endswith(".txt")):
                        speclib_chunk_path = os.path.join(predict_lib_dir, speclib_chunk)
                        logging.info(f"Combining file path: {speclib_chunk_path}")
                        with open(speclib_chunk_path, 'r') as infile:
                            outfile.write(infile.read())
                            # os.remove(speclib_chunk_path)     # remove the chunk file after combining
            logging.info(f"[USER] All predicted spectra library files were combined into: {speclib_file}")
            
            # Evaluate predictions if flag_evaluate is True
            if flag_evaluate is True and len(combined_dict.keys() > 0):
                eval_dir = os.path.join(predict_lib_dir, f"evaluation_{speclib_filename}")
                evaluate_predictions(combined_dict, combined_dict, predict_df, eval_dir)

        else:
            # Process the entire dataset at once
            if flag_evaluate is True:
                logging.info('Prediction list with reference for evaluation')
                predict_dict = tensorize.hdf5(predict_df, predict_hdf5)
            else:
                logging.info('Prediction list without evaluation')
                predict_dict = tensorize.csv(predict_df, flag_fullspectrum)
            
            # Perform predictions on the entire dataset
            if flag_prosit:
                pred = prediction_prosit(
                    predict_batch_dir, predict_chunk_result_file, predict_result_file,
                    predict_dict, predict_ds, d_spectra, flag_fullspectrum, flag_evaluate,
                    flag_chunk=False)
            else:
                pred = prediction_transformer(
                    predict_batch_dir, predict_chunk_result_file, predict_result_file,
                    predict_dict, predict_ds, d_spectra, flag_fullspectrum, flag_evaluate,
                    flag_chunk=False)
            convert_and_save_predictions(pred, speclib_file, predict_format, flag_fullspectrum)

            # Evaluate whole predictions if flag_evaluate is True
            if flag_evaluate is True:
                evaluate_predictions(pred, predict_dict, predict_df, predict_lib_dir)
    else:
        logging.info(f"Prediction CSV file {predict_csv} does not exist. Please provide a valid CSV file.")
        return 0
    
    return 1


def main():
    # Suppress warning message of tensorflow compatibility
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    # warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore")

    # Configure logging
    log_file_predict = os.path.join(constants_location.LOGS_DIR, "cospred_predict.log")
    logging.basicConfig(
        filename=log_file_predict,
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

    # create prediction directory if it does not exist
    predict_dir = constants_location.PREDICT_DIR
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir, exist_ok=True)
    logging.info("Pediction result directory created: {}".format(predict_dir))

    sys.setrecursionlimit(20000)  # Increase the recursion limit
    start_time = time.time()

    # Code to be timed
    parser = ArgumentParser()
    parser.add_argument('-t', '--trained', default=True, action='store_false',
                        help='turn off loading best existing model')
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help='full spectrum presentation')
    parser.add_argument('-b', '--bigru', default=False, action='store_true',
                        help='predict with BiGRU model')
    parser.add_argument('-c', '--chunk', default=False, action='store_true',
                        help='prediction list in chunk')
    parser.add_argument('-r', '--resume', default=False, action='store_true',
                        help='resume the prediction post chunking')
    parser.add_argument('-e', '--evaluate', default=False, action='store_true',
                        help='evaulate model with metrics')
    args = parser.parse_args()

    model_dir = constants_location.MODEL_DIR
    predict_format = constants_location.PREDICT_FORMAT
    predict_csv = constants_location.PREDICTCSV_PATH
    predict_hdf5 = constants_location.PREDDATA_PATH
    chunk_dir = constants_location.PREDDATASET_PATH     # chunk file path

    ### Initialize to store model and session ###
    if args.bigru is True:
        d_spectra["graph"] = tf.Graph()
        with d_spectra["graph"].as_default():
            d_spectra["session"] = tf.compat.v1.Session()
            logging.info("Tensorflow session created successfully.")
            with d_spectra["session"].as_default():
                d_spectra["model"], d_spectra["config"], d_spectra['weights_path'] = model_lib.load(
                    model_dir,
                    args.full,
                    args.bigru,
                    args.trained
                )
        # d_irt["graph"] = tf.Graph()
        # with d_irt["graph"].as_default():
        #    d_irt["session"] = tf.Session()
        #    with d_irt["session"].as_default():
        #        d_irt["model"], d_irt["config"] = model.load(constants.MODEL_IRT,
        #                    trained=True)
        #       d_irt["model"].compile(optimizer="adam", loss="mse")
        logging.info("[RESULT] BiGRU model was loaded successfully.")
    else:
        d_spectra["model"], d_spectra["config"], d_spectra['weights_path'] = model_lib.load(
            model_dir,
            args.full,
            args.bigru,
            args.trained
        )
        logging.info("Transformer model was loaded successfully.")
    logging.info('[USER] Loaded weight from: {}'.format(d_spectra['weights_path']))
    logging.info('[STATUS] MODEL LOADING finished.')

    ### Input preparation ###
    if args.evaluate is True: 
        logging.info("[STATUS] EVALUATION MODE: Generating prediction list with reference ...")
        csvfile = constants_location.TESTCSV_PATH
        test_hdf5 = constants_location.TESTDATA_PATH
        usimgffile = constants_location.REFORMAT_TEST_USITITLE_PATH
        psmfile = constants_location.PSM_PATH
        if (os.path.exists(csvfile) and os.path.exists(test_hdf5)):
            predict_csv = csvfile
            predict_hdf5 = test_hdf5
            logging.info(f"Reference CSV {csvfile} and HDF5 {test_hdf5} were provided. Move on to prediction ...")
        elif (os.path.exists(csvfile) and os.path.exists(usimgffile) and os.path.exists(psmfile)):
            logging.info(f"Reference CSV {csvfile} was provided.")
            logging.info(f"Generating HDF5 {predict_hdf5} ...")
            temp_dir = constants_location.TEMP_DIR
            reformatmgffile = constants_location.REFORMAT_TEST_PATH
            if args.full is True:
                # match peptide from PSM with spectra MGF to generate CSV with full spectra bins
                dbsearch_df = rawfile2hdf_cospred.getPSM(psmfile)
                # Filter the PSM DataFrame
                dbsearch_df = rawfile2hdf_cospred.filterPSM(dbsearch_df, csvfile)
                dataset = rawfile2hdf_cospred.generateHDF5_transformer(
                        usimgffile, reformatmgffile, dbsearch_df,
                        predict_csv, None)
                # transform full spectrum test peptides list to hdf5
                # # Note: not practical since csv is not ideal for storing 3000 dimension full spectrum 
                # dataset = rawfile2hdf_cospred.constructDataset_fullspectrum(dataset)
                io_cospred.to_hdf5(dataset, predict_hdf5)
            else:
                # if b,y ion prediction, annotation is required
                annotation_results = annotateMGF_wSeq(
                    usimgffile, csvfile, temp_dir)
                # match peptide from PSM with spectra MGF
                generateCSV_wSeq(usimgffile, reformatmgffile, csvfile, annotation_results,
                                predict_csv, temp_dir)
                # transform byion test peptides list to hdf5
                dataset = rawfile2hdf_prosit.constructDataset_byion(predict_csv)
                io_cospred.to_hdf5(dataset, predict_hdf5)
        else:
            logging.error(f"Not sufficient inputs were found. Please provide valid files.")
            return 0
    else:
        logging.info("[STATUS] PREDICTION MODE: Generating prediction list without reference.")
        csvfile = constants_location.PREDICT_ORIGINAL
        if (os.path.exists(csvfile)):
            logging.info(f"Reference CSV {csvfile} was provided. Move on to prediction.")
            if not args.resume:
                # filter prediction list to remove non-amino acid and transform to dictionary
                predict_dict = io_cospred.constructDataset_frompep(csvfile, predict_csv)
                # save dataset to hdf5 for prediction usage
                io_cospred.to_hdf5(predict_dict, predict_hdf5)
        else:
            logging.error(f"Reference CSV {csvfile} was not found. Please provide a valid file.")
            return 0
        
    # convert hdf5 to hugginface Dataset (Three array for predication only)
    if not args.resume:
        predict_ds = io_cospred.genDataset(predict_hdf5, chunk_dir, args.chunk)
    else:
        predict_ds = None

    ### prediction process ###
    logging.info('[STATUS] INPUT PREPARATION finished. Start PREDICTION...')
   
    if predict_format == 'maxquant' or predict_format == 'msp':
        # Maxquant output
        predict(predict_csv, predict_dir, predict_format, predict_hdf5, predict_ds,
                args.bigru, args.full, args.evaluate, args.chunk, args.resume)
    else:
        logging.error('Predicted Spectra library format could only be maxquant or msp')

    logging.info("[STATUS] Whole CoSpred Workflow ... COMPLETE!")
    # disply elapsed time
    logging.info('[STATUS] Elapsed time: {} seconds'.format(time.time()-start_time))
    
if __name__ == "__main__":
    main()
