# CoSpred

Complete MSMS spectrum prediction workflow.

This provide workflow to prepare your own training datasets from raw files and convert them into tensors.
These tensors are input for machine learning architecure. The architecture can be build using Tensorflow or Pytorch framework.

Here we are predicting full MSMS spectrum as set of (Mi,Ii) where Mi is the mass of the peak and Ii is the intenisty of the peak. For the BiGRU model, the MSMS spectrum is presented as b/y ion series discribed in original Prosit paper.

Two machine learning architectures were demonstrated.

* Transformer: Predicts full MSMS spectrum using transformer architecture  
* BiGRU: Predicts y-, b- ion intensities using BiGRU architecture 

## Reproducing results with Docker

To best test and experience usage of the software, we recommend to use docker environment at the beginning. All the software dependencies were pre-installed in the **docker image**, while model weight and example data were provided in **capsule** folder.

### 1. Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- [nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu) for code that leverages the GPU
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to use GPU in docker container.
- [GLIBCXX] Updated GCC compiler, version >= 3.4.29.

### 2. Setup the computing environment

- Git clone the repo, download the pre-trained model `pretrained_models.zip` and `example.zip` from [FigShare](https://figshare.com/s/8a60e7017cd82db9a1b7), create a data folder `CoSpred/data`, and store two zip files there.

#### 2.1 Configure environment using Docker

##### Option 1: Pull the pre-built docker image
```shell
docker pull xuel12pfizer/cospred:v0.3
```

##### Option 2: Build the computational environment locally

The provided dockerfile will allow you to reproduce the workflow and results published by the author on your local machine by following the instructions below.

> If there's any software requiring a license that needs to be run during the build stage, you'll need to make your license available. 

In your terminal, navigate to the `CoSpred` folder with the example data in `data` that you've extracted before, and execute the following command:

```shell
docker build . --tag cospred_docker -f conf/requirements_cospred_cuda12_gpu_py39.yml
```
> This step will recreate the environment (i.e., the Docker image) locally, fetching and installing any required dependencies in the process. If any external resources have become unavailable for any reason, the environment will fail to build. Note that in this example, the docker image name is `cospred_docker` which will be refered in the following session.

##### Option 3: Create virtual environment and package installation

Install all the packages given in `conf/requirements_cospred_*.yml` in a virtual environment. 

```bash
conda env create -n cospred_cuda12_gpu_py39 -f conf/requirements_cospred_cuda12_gpu_py39.yml
```

### 3. Reproduce the workflow and results in batch mode

Following is for the purpose of experiencing the end-to-end workflow. Presumably the data `example.zip` and model `pretrain_model.zip` have been kept in `data` folder that you've downloaded from FigShare. Execute the following command, run the docker container using the image just built in the previous session named `cospred_docker`, adjust parameters as needed (e.g. fot the machine that doesn't have GPU, remove the option flag `--gpus all`; for some system has restriction on shared memory for GPU, adjust `--shm-size` flag; or anything specific for your docker environment).

#### 3.1 Use Case I: Batch mode for training using Docker
```shell
docker run --platform linux/amd64 --rm --gpus all \
  --volume "$PWD/data":/data \
  --volume "$PWD/results":/results \
  cospred_docker bash "scripts/run_training.sh"
```

#### 3.2 Use Case II: Batch mode for prediction using Docker
```shell
docker run --platform linux/amd64 --rm --gpus all \
  --volume "$PWD/data":/data \
  --volume "$PWD/results":/results \
  cospred_docker bash "scripts/run_prediction.sh"
```

* When finished, the final results will be stored in `results` folder.

## Reproducing the workflow and results in interactive mode

For advanced usage, the following are the step-by-step guides for fine-grain control modular execution of CoSpred.

* OPTION 1: Docker with interactive mode
  
```shell
docker run --platform linux/amd64 --rm -it --gpus all \
  --volume "$PWD/data":/data \
  --volume "$PWD/results":/results \
  --shm-size=32g \
  cospred_docker
```

* OPTION 2: Viturtual environment
```shell
conda activate cospred_cuda12_gpu_py39
```

Once done the environment setup, navigate to `CoSpred` working directory, move forward to following steps.

### 1. Configuration

* Main configuration regarding file location and preprocessing parameters could be found and modified in `params` folder. 
* For all the following modules, log files could be found under `prediction` folder.
* For BiGRU based training parameters and model setup could be found in JSON files under `model_spectra` folder. Edit `model_construct.py` to modify architecture and generate compatible `model.json` file. Two examples `model_byion.json` and `model_fullspectrum.json` were provided.

```bash
python model_construct.py
```
* For transformer based model setup, users could directly modify `cospred_model/model/transformerEncoder.py`, as well `train_transformer` module in `training_cospred.py`. 

### 2. Data Preprocessing

#### Database search and pair identification with spectra

* Create PSM file with identified peptides from softwares like Proteome Discoverer corresponding to each rawfiles from the experiment.

#### Spectrum file format conversion

* Convert rawfiles into mzml and mgf

Msconvert can be run using GUI version of the software on windows computer or can use Docker on linux machine. We recommend to run MSCovert in Windows GUI. At the end, assuming files were generated, `*.mzML` and `*.mgf`. Keep these two type of files in folder `data/example/mzml` and `data/example/mgf` respectively, together with `example_PSMs.txt` and `example_InputFiles.txt` got from Proteome Discoverer in the folder `data/example`. Note that all file and folders names could be defined in the `params/constants_location.py`.

* OPTION 1: The MGF file doesn't contain sequence information
    * Split the dataset into train and test set. (About 15mins for 300k spectra)

    20% spectra will be randomly selected for test by default, which could be modified in the script. `example_train.mgf` and `example_test.mgf` will be generated from this step. `rawfile2hdf_byion.py` (preparing dataset with b/y ion annotation) and `rawfile2hdf_cospred.py` (preparing dataset for full spectrum representation) are the scripts for this purpose. (2019 Macbook Pro, about 2 minitues for the example dataset)

    ```bash
    python rawfile2hdf_cospred.py -w split
    ```
    
    * OPTION 1.1: Pair database search result with MGF spectrum, annotate B and Y ion for MSMS spectrum

    Pyteomics is used to parse annotations of y and b ions and their fragments charges from MZML and MGF, and report to annotated MGF files for downstream spectrum prediction/viewing application. Note that to parse the input file correctly, you will likely need to adjust regex routine (in the `reformatMGF` function within `io_cospred.py`) according to the specific MGF format you are using. (2019 Macbook Pro, about 1 hour for the example dataset)

    ```bash
    python rawfile2hdf_byion.py -w train
    python rawfile2hdf_byion.py -w test
    ```

    * OPTION 1.2: Pair database search result with MGF spectrum, reformat to full MSMS using bins. (2019 Macbook Pro, about 2 minitues for the example dataset)

    ```bash
    python rawfile2hdf_cospred.py -w train
    python rawfile2hdf_cospred.py -w test
    ```

* OPTION 2: For MGF file with assigned peptide sequence (e.g. `example.mgf`), reformat to full MSMS using bins. Note: you may need to reformat MGF so that peptide sequence representation is compatible for downstream.

```bash
python mgf2hdf_cospred.py -w reformat       # Reformat the MGF for CoSpred workflow.
python mgf2hdf_cospred.py -w split_usi      # Split the dataset into train and test set.
python mgf2hdf_cospred.py -w train          # Convert training set into full spectrum bins
python mgf2hdf_cospred.py -w test           # Convert testing set into full spectrum bins
```

At the end, a few files will be generated. `train.hdf5` and `test.hdf5` are input files for the following ML modules.

### 3. In-house training procedure

`training_cospred.py` is the script for customized training. Workflows could be selected by arguments, including 1) `-t`: fine-tuning / continue training the existing model; 2) `-f`: opt in full MS/MS spectrum model instead of B/Y ions; 3) `-c`: chunking the input dataset (to prevent memory overflow by large dataset); 4) `-b`: opt in for BiGRU model instead of Transformer.

#### Representative training workflows

```bash
python training_cospred.py -b   # Training B/Y ion spectrum prediction using BiGRU architecture.
python training_cospred.py      # Training B/Y ion spectrum prediction using Transformer architecture.
python training_cospred.py -bf   # Training full spectrum prediction using BiGRU architecture. 
python training_cospred.py -f   # Training full spectrum prediction using Transformer architecture. 
python training_cospred.py -bft   # Fine-tuning full spectrum prediction using BiGRU architecture with pre-trained weights.
```

During the training procedure under each epoch, model weights files will be auto-generated under the folder `model_spectra`. Naming of files will be like below,
* For B/Y ion, BiGRU model: `prosit_byion_[YYYYMMDD]_[HHMMSS]_epoch[integer]_loss[numeric].hdf5`
* For full spectrum, BiGRU model: `prosit_full_[YYYYMMDD]_[HHMMSS]_epoch[integer]_loss[numeric].hdf5`
* For B/Y ion, Transformer model: `transformer_byion_[YYYYMMDD]_[HHMMSS]_epoch[integer]_loss[numeric].pt`
* For full spectrum, Transformer model: `transformer_full_[YYYYMMDD]_[HHMMSS]_epoch[integer]_loss[numeric].pt`

#### Note: Usage for novel modification

To fine-tune the foundation model or re-train the model, following scripts and parameters needs to be modified. In this demo, we will use a non unimod chemical modification "Desthiobiotin" for example.

* Ensure that the related information of (DTBIA) is properly added in your `constants.py` file, as following.
```python
# add to alphabet
ALPHABET = {
    "C(DTBIA)": 26,  # Alphabet
}
# define the chemical composition
MODIFICATION_COMPOSITION = {
    'C(DTBIA)': {'H': 24, 'C': 14, 'O': 3, 'N': 4},     # Chemical composition
}
# annotate the novel modification with proforma, so that pyteomics library can parse
VARMOD_PROFORMA = {
    'C(DTBIA)': 'C[+296.185]',
}
```

### 4. Inference

Keep the best model under `model_spectra` folder as the trained model for inference phase. Some pre-trained model can be downloaded from [FigShare](https://figshare.com/s/8a60e7017cd82db9a1b7). 

#### 4.1 Spectrum library generation

With trained models, predict spectrum given peptide sequences from `peptidelist_test.csv`. All inference results including metrics, plots, and spectra library will be stored under `prediction` folder. Predicted spectra will be stored in `speclib_prediction.msp`.

Workflows could be selected by arguments, including 1) `-f`: opt in full MS/MS spectrum model instead of B/Y ions; 2) `-c`: chucking the input dataset (to prevent memory overflow by large dataset); 3) `-b`: opt in for BiGRU model instead of Transformer. Some examples like below.

```bash
python prediction.py -b   # Predict B/Y ion spectrum prediction using BiGRU architecture.
python prediction.py      # Predict B/Y ion spectrum prediction using Transformer architecture.
python prediction.py -bf   # Predict full spectrum prediction using BiGRU architecture. 
python prediction.py -f   # Predict full spectrum prediction using Transformer architecture. 
python prediction.py -fc   # Predict full spectrum prediction using Transformer architecture, with chunking for accomodating large peptide list. 
```

#### 4.2 Spectrum library prediction with reference for evaluation

Optionally, performance evaluation could be executed with `-e` argument, as long as ground truth a) `test.hdf5` or b) `example_PSMs.txt` with `test_usi.mgf` are provided, so that reference spectrum for the peptides could be extracted from database search result and the raw mass-spec data. Examples as below.

```bash
python prediction.py -be   # Predict B/Y ion spectrum prediction using BiGRU architecture.
python prediction.py -e    # Predict B/Y ion spectrum prediction using Transformer architecture.
python prediction.py -bfe   # Predict full spectrum prediction using BiGRU architecture. 
python prediction.py -fe   # Predict full spectrum prediction using Transformer architecture. 
python prediction.py -fce   # Predict full spectrum prediction using Transformer architecture, with chunking for accomodating large peptide list. 
```

The outputs of prediction will be generated under `prediction`, including predicted spectra library `speclib_prediction.msp` and `speclib_prediction.mgf`, plots and metrics under `prediction_library`, some other intermediate files for recording or diagnosis purpose.

### 5. Plotting
Predicted spectrum and mirror plot for visual evaluation could be separately generated by `spectra_plot.py`. By default, the required inputs are `peptidelist_predict.csv` (peptides list), `test_reformatted.mgf` (reference spectra), and `speclib_prediction.mgf` (predicited spectra). File names and location could be defined by `params/constants`. Plots will be stored in `prediction/plot` folder.

```bash
python spectra_plot.py
```

## License
[Apache license 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Contact
Liang.Xue@pfizer.com