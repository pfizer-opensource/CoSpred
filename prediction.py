import os
import sys
import shutil
import time
import torch
import pandas as pd
import re
from pyteomics import mgf
import spectrum_utils.spectrum as sus
import h5py
import numpy as np
from datasets import Dataset

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

import warnings
# Suppress warning message of tensorflow compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module="keras.engine.training_v1")

global d_spectra
# global d_irt
d_spectra = {}
# d_irt = {}


def print_combined_data_sizes(combined_data):
    """
    Print the size of each key in the combined_data dictionary.

    Args:
        combined_data (dict): Dictionary containing datasets from the combined HDF5 file.
    """
    print("Combined Prediction Data Sizes:")
    for key, value in combined_data.items():
        if isinstance(value, np.ndarray):
            print(f"Key: {key}, Shape: {value.shape}, Size: {value.nbytes / (1024 * 1024):.2f} MB")
        elif isinstance(value, list):
            print(f"Key: {key}, Length: {len(value)}, Approx. Size: {sum(sys.getsizeof(v) for v in value) / (1024 * 1024):.2f} MB")
        else:
            print(f"Key: {key}, Type: {type(value)}, Size: {sys.getsizeof(value) / (1024 * 1024):.2f} MB")


def combine_hdf5_files(predict_input, predict_result_file, combined_file):
    """
    Combine result HDF5 files with prediction input variable into a single HDF5 file.

    Args:
        predict_input (dictionary/Hugginface Dataset): Input as dictionary/Hugginface Dataset with data and metadata (predict_input).
        predict_result_file (str): Path to the batch prediction HDF5 file (predict_result_file).
        combined_file (str): Path to save the combined HDF5 file.
    """       
    with h5py.File(predict_result_file, "r") as result_h5:
        # Open the combined HDF5 file for writing
        with h5py.File(combined_file, "w") as combined_h5:
            # Copy datasets from predict_ds_file
            for key in result_h5.keys():
                combined_h5.create_dataset(key, data=result_h5[key][:])
            
            # Append or merge datasets from predict_data_file                
            if isinstance(predict_input, Dataset):  # Handle Dataset objects
                for column in predict_input.column_names:
                    data = np.array(predict_input[column])
                    if column in combined_h5:
                        combined_h5[column][...] = np.concatenate((combined_h5[column][:], data), axis=0)
                    else:
                        combined_h5.create_dataset(column, data=data)
            else:
                for key in predict_input.keys():
                    if key in combined_h5:
                        # If the dataset already exists, concatenate the data
                        combined_h5[key][...] = np.concatenate((combined_h5[key][:], predict_input[key][:]), axis=0)
                    else:
                        # If the dataset does not exist, create it
                        combined_h5.create_dataset(key, data=predict_input[key][:])
        print(f"Combined Prediction saved to {combined_file}")


def concatenate_hdf5(output_dir, predict_dict, predict_result_file, combined_file):
    # Step 1: Read all HDF5 batch files
    batch_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".h5")]
    print(f"Found {len(batch_files)} batch files to concatenate.")

    # Initialize lists to store concatenated data
    all_predictions = []
    # all_batch_indices = []

    # Read and concatenate all batch files
    for batch_file in batch_files:
        with h5py.File(batch_file, "r") as h5f:
            all_predictions.append(h5f["intensities_pred"][:])
            # all_batch_indices.append(h5f["batch_idx"][:])

    # Combine all predictions and batch indices
    combined_predictions = np.vstack(all_predictions)
    # combined_batch_indices = np.concatenate(all_batch_indices)

    # Step 2: Save the batch data to a single HDF5 file
    with h5py.File(predict_result_file, "w") as h5f:
        h5f.create_dataset("intensities_pred", data=combined_predictions)
        # h5f.create_dataset("batch_idx", data=combined_batch_indices)
    print(f"Prediction Values saved to {predict_result_file}")

    # Step 3: Read the single result HDF5 file for combination
    combine_hdf5_files(predict_dict, predict_result_file, combined_file)
    combined_data = {}
    with h5py.File(combined_file, "r") as h5f:
        for key in h5f.keys():
            combined_data[key] = h5f[key][:]
    # print(f"Read Prediction Result HDF5 file: {combined_file}")

    return combined_data


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
            print('MS2 Annotation Progress: {}%'.format(
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
            print('Generating CSV Progress: {}%'.format(
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

    print('Generating CSV ... DONE!')

    modifyMGFtitle(usimgffile, reformatmgffile, temp_dir)
    return dataset


def modifyMGFtitle(usimgffile, reformatmgffile, temp_dir):
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


def prediction_prosit(predict_batch_dir, predict_result_file, combined_file, 
                    predict_dict, hf_dataset, d_spectra, flag_fullspectrum, 
                    flag_evaluate=False, flag_chunk=False):
    """
    Perform batch predictions using either HDF5 or Hugging Face Dataset based on flag_chunk.

    Args:
        predict_batch_dir (str): Directory to save batch predictions.
        predict_result_file (str): Path to save the combined prediction result.
        combined_file (str): Path to save the final combined HDF5 file.
        predict_dict (dictionary): Dictionary containing input data and metadata.
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
        # Use HDF5-based processing
        x = io_cospred.get_array(predict_dict, d_spectra["config"]["x"])
        # print(f"Input data shape: {x[0].shape[0]}")
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
        print(f"Processing batch {batch_idx + 1}/{num_batches}, batch size: {len(x_batch[0])}")

        # Perform prediction for the batch
        with d_spectra["graph"].as_default():
            prediction = d_spectra["model"].predict(x_batch, verbose=True, batch_size=batch_size)

        # Save the batch predictions to an HDF5 file
        batch_file = os.path.join(predict_batch_dir, f"batch_{batch_idx + 1}.h5")
        with h5py.File(batch_file, "w") as h5f:
            h5f.create_dataset("intensities_pred", data=prediction)
        # print(f"Batch {batch_idx + 1}/{num_batches} saved to {batch_file}")

    print(f"PREDICTION computing ... DONE! All result batches saved to {predict_batch_dir}")

    # Combine batch files into a single HDF5 file
    combined_data = concatenate_hdf5(predict_batch_dir, predict_dict, predict_result_file, combined_file)
    print_combined_data_sizes(combined_data)

    # Sanitize the combined data
    if d_spectra["config"]["prediction_type"] == "intensity":
        combined_data = sanitize.prediction(combined_data, flag_fullspectrum, flag_evaluate)
    else:
        raise ValueError("model_config misses parameter")

    return combined_data


def prediction_transformer(predict_batch_dir, predict_result_file, combined_file, 
                           predict_dict, hf_dataset, d_spectra, flag_fullspectrum=True, 
                           flag_evaluate=False, flag_chunk=False):
    """
    Perform batch predictions using either HDF5 or Hugging Face Dataset based on flag_chunk.

    Args:
        predict_batch_dir (str): Directory to save batch predictions.
        predict_result_file (str): Path to save the combined prediction result.
        combined_file (str): Path to save the final combined HDF5 file.
        predict_dict (dictionary): Dictionary containing input data and metadata.
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
        print(f"Input Dataset size: {len(hf_dataset)}")
        num_batches = int(np.ceil(len(hf_dataset) / batch_size))
        data_generator = (
            hf_dataset.select(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(hf_dataset))))
            for batch_idx in range(num_batches)
        )
    else:
        x = [torch.tensor(predict_dict[column]) for column in d_spectra["config"]["x"]]
        x = torch.cat(x, dim=1)
        print(f"Input tensor shape: {x.shape}")
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        num_batches = len(dataloader)
        data_generator = ((x_batch,) for x_batch, in dataloader)

    # Perform predictions in batches
    for batch_idx, batch in enumerate(data_generator):
        if flag_chunk:
            x_batch = [torch.tensor(batch[column]) for column in d_spectra["config"]["x"]]
            x_batch = torch.cat(x_batch, dim=1)
        else:
            x_batch = batch[0]

        print(f"Processing batch {batch_idx + 1}/{num_batches}, batch size: {x_batch.shape[0]}")

        # Perform prediction for the batch
        x_batch = x_batch.to(d_spectra["device"])
        prediction = d_spectra["model"].forward(x_batch)[0]

        # Save the batch predictions to an HDF5 file
        batch_file = os.path.join(predict_batch_dir, f"batch_{batch_idx + 1}.h5")
        with h5py.File(batch_file, "w") as h5f:
            h5f.create_dataset("intensities_pred", data=prediction.cpu().detach().numpy())
        # print(f"Batch {batch_idx + 1}/{num_batches} saved to {batch_file}")

        # Clear memory
        del x_batch, prediction
        torch.cuda.empty_cache()  # Optional: Clear GPU memory if using CUDA

    print(f"All prediction batches saved to {predict_batch_dir}")

    # Concatenate batch files
    combined_data = concatenate_hdf5(predict_batch_dir, predict_dict, predict_result_file, combined_file)
    print_combined_data_sizes(combined_data)

    # Process predictions based on the model configuration
    if d_spectra["config"]["prediction_type"] == "intensity":
        combined_data = sanitize.prediction(combined_data, flag_fullspectrum, flag_evaluate)
    else:
        raise ValueError("model_config misses parameter")

    return combined_data


def predict(predict_csv, predict_dir, predict_format, predict_hdf5, predict_ds,
            flag_prosit, flag_fullspectrum, flag_evaluate, flag_chunk):
    from statistics import mean

    # Directory to save batch HDF5 files
    predict_batch_dir = constants_location.PREDICT_BATCH_DIR     # chunk file path
    if os.path.exists(predict_batch_dir):
        shutil.rmtree(predict_batch_dir)
    os.makedirs(predict_batch_dir, exist_ok=True)
    predict_result_file = constants_location.PREDICT_RESULT_FILE     # chunk file path
    combined_file = constants_location.PREDICT_COMBINED_FILE     # chunk file path
    
    if os.path.exists(predict_csv):
        df = pd.read_csv(predict_csv, index_col=None)
        if flag_evaluate is True:
            print('Prediction list with evaluation')
            predict_dict = tensorize.hdf5(df, hdf5file=predict_hdf5)
        else:
            print('Prediction list without evaluation')
            predict_dict = tensorize.csv(df, flag_fullspectrum)
    else:
        pass

    if flag_prosit is True:
        pred = prediction_prosit(
                predict_batch_dir, predict_result_file, combined_file,
                predict_dict, predict_ds, d_spectra, flag_fullspectrum, flag_evaluate,
                flag_chunk)
    else:
        pred = prediction_transformer(
            predict_batch_dir, predict_result_file, combined_file,
            predict_dict, predict_ds, d_spectra, flag_fullspectrum, flag_evaluate,
            flag_chunk)

    if flag_evaluate is True:
        y_true = torch.tensor(predict_dict['intensities_raw'])
        y_pred = torch.tensor(pred['intensities_pred'])
        seq, charge, ce = df['modified_sequence'], df['precursor_charge'], df['collision_energy']

        # calculate prediction metrics
        metrics = ComputeMetrics_CPU(
            true=y_true, pred=y_pred, seq=seq, charge=charge, ce=ce)
        metrics_byrecord = pd.DataFrame(metrics.return_metrics_byrecord())

        # calculate mean of metrics
        metrics_mean = metrics.return_metrics_mean()
        metrics_df = pd.DataFrame.from_dict(metrics_mean, orient='index')

        # OPTIONAL: calculate spectral angle
        spectralangle_df = pd.DataFrame(
            [{'spectral_angle': mean(pred['spectral_angle'])}]).T
        metrics_df = pd.concat(
            [metrics_df, spectralangle_df], ignore_index=False)

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
    print("Whole CoSpred Workflow ... COMPLETE!")
    return df_pred


def main():
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
    parser.add_argument('-e', '--evaluate', default=False, action='store_true',
                        help='evaulate model with metrics')
    args = parser.parse_args()

    model_dir = constants_location.MODEL_DIR
    predict_format = constants_location.PREDICT_FORMAT
    predict_dir = constants_location.PREDICT_DIR
    predict_csv = constants_location.PREDICTCSV_PATH
    predict_hdf5 = constants_location.PREDDATA_PATH
    predict_ds = constants_location.PREDDATASET_PATH     # chunk file path

    if args.bigru is True:
        d_spectra["graph"] = tf.Graph()
        with d_spectra["graph"].as_default():
            d_spectra["session"] = tf.compat.v1.Session()
            print("Tensorflow session created successfully.")

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
        print("BiGRU model loaded successfully.")
    else:
        d_spectra["model"], d_spectra["config"], d_spectra['weights_path'] = model_lib.load(
            model_dir,
            args.full,
            args.bigru,
            args.trained
        )
        print("Transformer model loaded successfully.")

    # create prediction list
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    print("Pediction result directory created: {}".format(predict_dir))

    if (not os.path.isfile(predict_csv)) or (not os.path.isfile(predict_hdf5)):
        if args.evaluate is True: 
            csvfile = constants_location.TESTCSV_PATH
            test_hdf5 = constants_location.TESTDATA_PATH
            if (os.path.isfile(csvfile) and os.path.isfile(test_hdf5)):
                predict_csv = csvfile
                predict_hdf5 = test_hdf5
                print("Reference CSV and HDF5 were provided. Move on to prediction.")
            else:
                temp_dir = constants_location.TEMP_DIR
                usimgffile = constants_location.REFORMAT_TEST_USITITLE_PATH
                reformatmgffile = constants_location.REFORMAT_TEST_PATH
                psmfile = constants_location.PSM_PATH
                if args.full is True:
                    # match peptide from PSM with spectra MGF to generate CSV with full spectra bins
                    dbsearch_df = rawfile2hdf_cospred.getPSM(psmfile)
                    dataset = rawfile2hdf_cospred.generateHDF5_transformer(
                            usimgffile, reformatmgffile, dbsearch_df,
                            predict_csv, None)
                    # transform full spectrum test peptides list to hdf5
                    # # Note: not practical since csv is not ideal for storing 3000 dimension full spectrum 
                    # dataset = rawfile2hdf_cospred.constructDataset_fullspectrum(csvfile, predict_csv)
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
            csvfile = constants_location.PREDICT_ORIGINAL
            # filter prediction list to remove non-amino acid and transform to dataset
            dataset = io_cospred.constructDataset_frompep(csvfile, predict_csv)
            io_cospred.to_hdf5(dataset, predict_hdf5)

    # check generated hdf
    io_cospred.read_hdf5(predict_hdf5)
    print('Generating HDF5 ... DONE!')

    # convert hdf5 to hugginface Dataset (Three array for predication only)
    predict_ds = io_cospred.genDataset(predict_hdf5, predict_ds, args.chunk)

    # prediction process
    print('MODEL LOADING finished. Start PREDICTION...')
    
    if predict_format == 'maxquant' or predict_format == 'msp':
        # Maxquant output
        predict(predict_csv, predict_dir, predict_format, predict_hdf5, predict_ds,
                args.bigru, args.full, args.evaluate, args.chunk)
    else:
        print('PREDICT_FORMAT could only be maxquant or msp')

    # disply elapsed time
    print('Elapsed time: {} seconds'.format(time.time()-start_time))
    
if __name__ == "__main__":
    main()
