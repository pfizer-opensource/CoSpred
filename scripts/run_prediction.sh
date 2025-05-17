#!/usr/bin/env bash
set -ex

pwd

# Remove all files in the result folder interactively, but only if the folder exists and is not empty
if [ -d "/results" ] && [ "$(ls -A /results)" ]; then
    rm -rI /results/*
fi

# update repo
git pull https://github.com/pfizer-opensource/CoSpred.git --depth 1

# copy example data and models to the corresponding directory
mkdir -p model_spectra
mkdir -p prediction
mkdir -p data
unzip -o /data/pretrained_models.zip -d model_spectra/
unzip -o /data/example_processed.zip -d data/
mv data/example_processed data/example

# --- For the codes below, uncomment sessions based on the needs --- #
# # Split the dataset into train and test set. (about 3 mins)
# python rawfile2hdf_cospred.py -w split

# # Pair database search result with MGF spectrum, reformat to full MSMS using bins. (about 1 min)
# python rawfile2hdf_cospred.py -w train
# python rawfile2hdf_cospred.py -w test

# # training workflows
# python training_cospred.py -fc

## spectrum prediction from sequence
python prediction.py -fe

## plotting spectrum (only applicable to post prediction+evaluation workflow)
python spectra_plot.py
# --- END of main codes --- #

# store result and clean up
rm -rf /results/*
cp -rf data /results/                  # transformed data
cp -rf model_spectra /results/         # trained models
cp -rf prediction /results/            # prediction results
rm -rf data
rm -rf model_spectra
rm -rf prediction
# cd ..
# rm -rf CoSpred
