## How to use

### Download pre-trained models
To download the pre-trained models for Prosit use following link:
https://koina.proteomicsdb.org/

Move the model to workdir/model_spectra/

## Dataset preparation for training the Prosit model
* Requirements
    Using Conda:  
    ```
    conda env create -n prosit_cpu_py39 -f conf/requirements_prosit_cpu_py39.yml
    ```
    Using Pip
    *pip install spectrum_utils
    *pip install pyteomics
    *msconvert to convert rawfiles into mzml and mgf
    *csv file with identified peptides from softwares like Maxquant or Proteome Discoverer corresponding to each rawfiles from the experiment

## To run MSConvert using singularity 
Msconvert can be run using GUI version of the software on windows computer or can use Docker on linux machine.
To run HPC servers which do not support docker use singulairty.
```
singularity exec -B /pathtoyourfolder/:/data -B `mktemp -d /dev/shm/wineXXX`:/mywineprefix -w /pathtoyourfolder/pwiz_sandbox mywine msconvert /data/01640c_BE1-Thermo_SRM_Pool_5_01_01-2xIT_2xHCD-1h-R2.raw --32 --zlib --filter "peakPicking true 1-" --filter "zeroSamples removeExtra" -o  /pathtoyourfolder/ --mgf
```

Recommend to run MSCovert in Windows GUI

Scripts 
* rawfile2csv_prosit.py: To parse annotations of y and b ions and there fragments charges Pyteomics is used by parsing the 
data from mzml, mgf and report to csv files for downstream Prosit application
```
python rawfile2csv_prosit.py -w train
```
  
* Merge all the annotated csv files and then filter it for hcd if you want to train for hcd dataset and if 
not the filter it only for peptides with score >100 in Maxquant output and qvalue in PD.

* Merged and filtered csv is the output file

* csv2hdf5tensor: Convert this merged csv into hdf5 file for training the prosit model
```
python csv2hdf5tensor.py -w train

```
* Move hdf5 file to like data/abc.hdf, this is the input for Prosit.

* Switch to Prosit working directory. Run 
```
python training_local.py -t
```
* Plot
* Rename the abc.csv to peptidelist.csv, move it to examples/. Run Prediction.
```
python prediction.py
```
* Move reformattedMGF to data/. Run plotting.
```
python spectra_plot.py
```



## To run with command prompt 
```
Python rawfile2csv_prosit.py
```
```
Python csv2hdf5tensor.py
```
```
Python Datafrommgf.py -i <inputfile> -l <mgffile> -o <outputfile>
```

## Dataset preparation for training the Transformer encoder architecture for complete spectrum prediction

* data4Transformannotated.py: For y- and b- fragment ion annotations use same step as prosit dataset preparation 
* Take the csv and convert the y- and b- ion fragments ions intensity in the format of 1D vector of length 
1600 and 20000 in case binsize in 0.1 
  
* data4Transformcomplete: For complete spectrum prediction take all the mz values and its corresponding intensity into a 1D array 
with size 1600 or 20000.
  
* Convert both the data into torch tensors to train transformer architecture

## To run with command prompt 
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

