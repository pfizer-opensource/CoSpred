# CoSpred
Complete MSMS spectrum prediction workflow.

This provide workflow to prepare your own training datasets from raw files and convert them into tensors.
These tensors are input for machine learning architecure. The architecture can be build using Keras or Pytorch framework.
Here we are predicting Complete MSMS spectrum as set of (Mi,Ii) where Mi is the mass of the peak and Ii is the intenisty of the peak.

## To run python script Install
 
*   Python
*   Msconvert    
*   Spectrum_utils (`pip install spectrum_utils`)
*   Pyteomics (`pip install pyteomics`)
*   Torch 
Install all the packages given in requirements.txt in a virtual environment.

## Machine learning architectures
* CoSpred: Transformer Architecture  
    * Predicts complete MSMS spectrum 
* Prosit: LSTM Architecture 
    * Predicts y-, b- ion intensities 
    
## How to run Prosit Model

### Download pre-trained models
To download the pre-trained models for Prosit use following link:
https://koina.proteomicsdb.org/

Move the model to workdir/model_spectra/

### Dataset preparation for training the Prosit model
* Requirements
    Using Conda:  
    ```
    conda env create -n prosit_cpu_py39 -f conf/requirements_prosit_cpu_py39.yml
    ```
    * msconvert to convert rawfiles into mzml and mgf
    * PSM file with identified peptides from softwares like Maxquant or Proteome Discoverer corresponding to each rawfiles from the experiment

### To run MSConvert 
Msconvert can be run using GUI version of the software on windows computer or can use Docker on linux machine. We recommend to run MSCovert in Windows GUI

### To run with command prompt 
* rawfile2csv_prosit.py: To parse annotations of y and b ions and there fragments charges Pyteomics is used by parsing the data from mzml, mgf and report to csv files for downstream Prosit application
```
python rawfile2csv_prosit.py -w train
```
* csv2hdf5tensor: Convert this merged csv into hdf5 file for training the prosit model
```
python csv2hdf5tensor.py -w train

```
* training_local.py: Training procedure using Prosit algorithm. 
```
python training_local.py -t
```
* prediction.py: Spectrum Prediction using peptidelist_test.csv as peptide list.
```
python prediction.py
```
* spectra_plot.py: Plotting for spectrum predicition results.
```
python spectra_plot.py
```

## How to run Transformer Model

### Dataset preparation for training the Transformer encoder architecture for complete spectrum prediction

* data4Transformannotated.py: For y- and b- fragment ion annotations use same step as prosit dataset preparation 
* Take the csv and convert the y- and b- ion fragments ions intensity in the format of 1D vector of length 
1600 and 20000 in case binsize in 0.1 
  
* data4Transformcomplete: For complete spectrum prediction take all the mz values and its corresponding intensity into a 1D array 
with size 1600 or 20000.
  
* Convert both the data into torch tensors to train transformer architecture

### To run with command prompt 
```
Python rawfile2csv_prosit.py
```
```
Python data4Transformcomplete.py -i <inputfile> -l <mzmlfile> -o <outputfile>
```
```
Python Datafrommgf.py -i <inputfile> -l <mgffile> -o <outputfile>
```

## Contact
shivani.tiwary@pfizer.com

Liang.Xue@pfizer.com