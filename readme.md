# CoSpred

Complete MSMS spectrum prediction workflow.

This provide workflow to prepare your own training datasets from raw files and convert them into tensors.
These tensors are input for machine learning architecure. The architecture can be build using Tensorflow or Pytorch framework.

Here we are predicting full MSMS spectrum as set of (Mi,Ii) where Mi is the mass of the peak and Ii is the intenisty of the peak. For the BiGRU model, the MSMS spectrum is presented as b/y ion series discribed in original Prosit paper.

Two machine learning architectures were demonstrated.

* Transformer: Predicts full MSMS spectrum using transformer architecture  
* BiGRU: Predicts y-, b- ion intensities using BiGRU architecture 

## Installation

Install all the packages given in `conf/requirements_cospred_*.yml` in a virtual environment

```bash
conda env create -n cospred_gpu_py39 -f conf/requirements_cospred_gpu_py39.yml
```

Or use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies specified in the YAML files.

## Usage

### Configuration

* Main configuration regarding file location and preprocessing parameters could be found and modified in `params` folder. 

* For B/Y ion prediction, training parameters and model setup could be found in JSON files under `model_spectra` folder. Edit `model_construct.py` to modify architecture and generate compatible `model.json` file. Two examples `model_byion.json` and `model_fullspectrum.json` were provided.

```bash
python model_construct.py
```
* For transformer based model setup, users could directly modify `cospred_model/model/transformerEncoder.py`, as well `train_transformer` module in `training_cospred.py`. 

### Data Preprocessing

#### Database search and pair identification with spectra

* Create PSM file with identified peptides from softwares like Proteome Discoverer corresponding to each rawfiles from the experiment.

#### Spectrum file format conversion

* Convert rawfiles into mzml and mgf

Msconvert can be run using GUI version of the software on windows computer or can use Docker on linux machine. We recommend to run MSCovert in Windows GUI. At the end, assuming two files were generated, `example.mzML` and `example.mgf`. Keep these two files together with `example_PSMs.txt` got from Proteome Discoverer in the folder `data/example`.

* OPTION 1: The MGF file doesn't contain sequence information
    * Split the dataset into train and test set. (About 15mins for 300k spectra)

    5000 spectra will be randomly selected for test by default, which could be modified in the script. `example_train.mgf` and `example_test.mgf` will be generated from this step. `rawfile2hdf_prosit.py` (preparing dataset with b/y ion annotation) and `rawfile2hdf_cospred.py` (preparing dataset for full spectrum representation) are the scripts for this purpose. (About 2 minitues for the example dataset)

    ```bash
    python rawfile2hdf_cospred.py -w split
    ```
    
    * OPTION 1.1: Pair database search result with MGF spectrum, annotate B and Y ion for MSMS spectrum

    Pyteomics is used to parse annotations of y and b ions and their fragments charges from MZML and MGF, and report to annotated MGF files for downstream Prosit application. Note that to parse the input file correctly, you will likely need to adjust regex routine according to the specific MGF format you are using. (About 1 hour for the example dataset)

    ```bash
    python rawfile2hdf_prosit.py -w train
    python rawfile2hdf_prosit.py -w test
    ```

    * OPTION 1.2: Pair database search result with MGF spectrum, reformat to full MSMS using bins. (About 2 minitues for the example dataset)

    ```bash
    python rawfile2hdf_cospred.py -w train
    python rawfile2hdf_cospred.py -w test
    ```

* OPTION 2: For MGF file with assigned peptide sequence, reformat to full MSMS using bins. Note: you may need to reformat MGF so that peptide sequence representation is compatible for downstream. 

```bash
python mgf2hdf_cospred.py -w reformat       # Reformat the MGF for CoSpred workflow.
python mgf2hdf_cospred.py -w split_usi      # Split the dataset into train and test set.
python mgf2hdf_cospred.py -w train          # Convert training set into full spectrum bins
python mgf2hdf_cospred.py -w test           # Convert testing set into full spectrum bins
```

At the end, a few files will be generated. `train.hdf5` and `test.hdf5` are input files for the following ML modules.

### In-house training procedure

`training_cospred.py` is the script for customized training. Workflows could be selected by arguments, including 1) `-t`: fine-tuning / continue training the existing model; 2) `-f`: opt in full MS/MS spectrum model instead of B/Y ions; 3) `-c`: chucking the input dataset (to prevent memory overflow by large dataset); 4) `-p`: opt in for BiGRU model instead of Transformer.

#### Representative training workflows

```bash
python training_cospred.py -p   # Training B/Y ion spectrum prediction using BiGRU architecture.
python training_cospred.py      # Training B/Y ion spectrum prediction using Transformer architecture.
python training_cospred.py -pf   # Training full spectrum prediction using BiGRU architecture. 
python training_cospred.py -f   # Training full spectrum prediction using Transformer architecture. 
```

### Inference
Keep the best model under `model_spectra` folder. Some pre-trained model could be provided upon request. Naming will be like below,
* For B/Y ion, BiGRU model: `prosit_byion_[YYYYMMDD]_[HHMMSS]_epoch[integer]_loss[numeric].hdf5`
* For full spectrum, BiGRU model: `prosit_full_[YYYYMMDD]_[HHMMSS]_epoch[integer]_loss[numeric].hdf5`
* For B/Y ion, Transformer model: `transformer_byion_[YYYYMMDD]_[HHMMSS]_epoch[integer]_loss[numeric].pt`
* For full spectrum, Transformer model: `transformer_full_[YYYYMMDD]_[HHMMSS]_epoch[integer]_loss[numeric].pt`

With trained models, predict spectrum given peptide sequences from `peptidelist_test.csv`. Performance evaluation will be executed with `-e` argument, as long as ground truth `test.hdf5` is provided. Performance results will be stored under `prediction` folder, and the predicted spectra will be stored in `peptidelist_pred.msp`.

```bash
python prediction.py -p   # Predict B/Y ion spectrum prediction using BiGRU architecture.
python prediction.py      # Predict B/Y ion spectrum prediction using Transformer architecture.
python prediction.py -pf   # Predict full spectrum prediction using BiGRU architecture. 
python prediction.py -f   # Predict full spectrum prediction using Transformer architecture. 
```

### Plotting
Predicted spectrum and mirror plot for visual evaluation could be generated by `spectra_plot.py`. Plots will be stored in `prediction/plot` folder.

```bash
python spectra_plot.py
```

## License

[Apache license 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Contact
Liang.Xue@pfizer.com