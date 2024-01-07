#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:50:49 2020

@author: xuel12
"""
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus
from pyteomics import mgf
import numpy as np
import pandas as pd
import re
from argparse import ArgumentParser

# try: 
#     os.chdir('/Users/xuel12/Documents/Projects/seq2spec/prosit/local_training_tf2')
#     print("Current directory is {}".format(os.getcwd()))
# except: 
#     print("Something wrong with specified directory. Exception- ", sys.exc_info())
    
import params.constants as constants
import params.constants_local as constants_local
import params.constants_gcp as constants_gcp
from prosit_model import msp_parser

fragment_tol_mass = constants.BIN_SIZE
fragment_tol_mode = constants.BIN_MODE
min_mz = 0
max_mz = constants.BIN_MAXMZ
min_intensity = 0.02

##### single plot
def singleplot(feature, predict_mgf, plot_dir):
    # Read the spectrum from an MGF file using Pyteomics.
    spectrum_dict = mgf.get_spectrum(predict_mgf, feature)
    # modifications = {8: 15.994915}
    modifications = {}
    
    identifier = spectrum_dict['params']['title']
    precursor_mz = spectrum_dict['params']['pepmass'][0]
    precursor_charge = spectrum_dict['params']['charge'][0]
    mz = spectrum_dict['m/z array']
    intensity = spectrum_dict['intensity array']
    retention_time = float(spectrum_dict['params']['rtinseconds'])
    peptide = spectrum_dict['params']['seq'].replace("(ox)", "[Oxidation]")\
                .replace("(ph)", "[Phospho]")
            
    # # if sorted mz is desired
    # pair = np.vstack([spectrum_dict['m/z array'], spectrum_dict['intensity array']]).T
    # sorted_pair = pair[np.argsort(pair[:, 0])]
    # mz, intensity = [sorted_pair[:,0], sorted_pair[:,1]]
    
    # Process the MS/MS spectrum.

    # Create the MS/MS spectrum.
    # sus.reset_modifications()
    spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge, mz, intensity,
                            retention_time=retention_time, 
                            # peptide=peptide,
                            # modifications=modifications
                            )    
    # Filter and clean up the MS/MS spectrum.    
    spectrum = spectrum.set_mz_range(min_mz=min_mz, max_mz=max_mz). \
        remove_precursor_peak(fragment_tol_mass, fragment_tol_mode). \
            filter_intensity(min_intensity=min_intensity, max_num_peaks=50)
    # Annotate the MS2 spectrum.
    spectrum = spectrum.annotate_proforma(peptide, 
                                fragment_tol_mass=fragment_tol_mass, 
                                fragment_tol_mode=fragment_tol_mode, 
                                ion_types="abcxyzImp"
                                # ion_types="by"
                                )                

    # # label the mz for all peaks
    # annotate_fragment_mz = sorted_pair[(sorted_pair[:,0]>min_mz) & (sorted_pair[:,1]>min_intensity), 0]
    # for fragment_mz in annotate_fragment_mz:
    #     spectrum.annotate_mz_fragment(fragment_mz, 1, fragment_tol_mass, fragment_tol_mode)
        
    # Plot the MS/MS spectrum.
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title(identifier)
    sup.spectrum(spectrum, ax=ax)
    singleplot_dir = plot_dir+'singleplot/'
    if not os.path.exists(singleplot_dir):
        os.makedirs(singleplot_dir)
    fig.savefig(singleplot_dir+'{}.png'.format(re.sub('/','_',identifier)))
    plt.close(fig)
    print('Single Peptide Plot Done!')

##### mirror plot for two different peptides
def mirroplot_twopeptides(peplist, predict_mgf, plot_dir):
    spectra = []            
    for spectrum_dict in mgf.read(predict_mgf):
        if peplist[0] in spectrum_dict['params']['title'] or peplist[1] in spectrum_dict['params']['title']:
            identifier = spectrum_dict['params']['title']
            precursor_mz = spectrum_dict['params']['pepmass'][0]
            precursor_charge = spectrum_dict['params']['charge'][0]
            mz = spectrum_dict['m/z array']
            intensity = spectrum_dict['intensity array']
            retention_time = float(spectrum_dict['params']['rtinseconds'])
            peptide = spectrum_dict['params']['seq'].replace("(ox)", "[Oxidation]")\
                        .replace("(ph)", "[Phospho]")
            # modifications = {6: 15.994915}
            modifications = {}
            
            # Create the MS/MS spectrum.
            spectrum = sus.MsmsSpectrum(identifier, precursor_mz, 
                                        precursor_charge, mz, intensity,
                                        retention_time=retention_time, 
                                        # peptide=peptide,
                                        # modifications=modifications
                                        )    
            # Filter and clean up the MS/MS spectrum.    
            spectrum = spectrum.set_mz_range(min_mz=min_mz, max_mz=max_mz). \
                remove_precursor_peak(fragment_tol_mass, fragment_tol_mode). \
                    filter_intensity(min_intensity=min_intensity, max_num_peaks=50)
                        
                    
            # Annotate the MS2 spectrum.
            spectrum = spectrum.annotate_proforma(peptide, 
                                        fragment_tol_mass=fragment_tol_mass, 
                                        fragment_tol_mode=fragment_tol_mode, 
                                        ion_types="abcxyzImp"
                                        # ion_types="by"
                                        )    
            
            spectra.append(spectrum)
                          
                
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    plt.title(re.sub('/','_',spectrum_top.identifier)+"_vs_"+re.sub('/','_',spectrum_bottom.identifier))
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    doubleplot_dir = plot_dir+'doubleplot/'
    if not os.path.exists(doubleplot_dir):
        os.makedirs(doubleplot_dir)
    fig.savefig(doubleplot_dir+'{}vs{}.png'.format(re.sub('/','_',spectrum_top.identifier),
                                              re.sub('/','_',spectrum_bottom.identifier)))
    plt.close(fig)
    print('Double Peptides Plot Done!')

##### mirror plot for two dataset
def mirroplot_twosets(peplist, predict_mgf, reference_spectra, plot_dir):
    # ## DEBUG
    # predict_mgf = example_dir+'peptidelist_pred.mgf'
    # reference_spectra = data_dir+'2023_10_02_HeLa_200ng_03_reformatted.mgf'

    # spectra = mgf.read(reference_spectra)
    # for spectrum in spectra:
    #     pass
    # ##
    if not os.path.isfile(predict_mgf):
        print('{} not found'.format(predict_mgf))
    elif not os.path.isfile(reference_spectra):
        print('{} not found'.format(reference_spectra))
    else:    
        pair = []
        for title in peplist:
            spectra = []
            try:
                pred_dict = mgf.get_spectrum(predict_mgf, title)
                ref_dict = mgf.get_spectrum(reference_spectra, title)
                if (ref_dict is None or pred_dict is None):
                    next
                pair = [pred_dict, ref_dict]
                
                for spectrum_dict in pair:
                    identifier = spectrum_dict['params']['title']
                    precursor_mz = spectrum_dict['params']['pepmass'][0]
                    precursor_charge = spectrum_dict['params']['charge'][0]
                    mz = spectrum_dict['m/z array']
                    intensity = spectrum_dict['intensity array']
                    retention_time = float(spectrum_dict['params']['rtinseconds'])
                    peptide = spectrum_dict['params']['seq'].replace("(ox)", "[Oxidation]")\
                                .replace("(ph)", "[Phospho]")
                    # modifications = {6: 15.994915}
                    modifications = {}
                    
                    # Create the MS/MS spectrum.
                    spectrum = sus.MsmsSpectrum(identifier, precursor_mz, 
                                                precursor_charge, mz, intensity,
                                                retention_time=retention_time, 
                                                # peptide=peptide,
                                                # modifications=modifications
                                                )    
                    # Filter and clean up the MS/MS spectrum.    
                    spectrum = spectrum.set_mz_range(min_mz=min_mz, max_mz=max_mz). \
                        remove_precursor_peak(fragment_tol_mass, fragment_tol_mode). \
                            filter_intensity(min_intensity=min_intensity, max_num_peaks=50)
                                
                            
                    # Annotate the MS2 spectrum.
                    spectrum = spectrum.annotate_proforma(peptide, 
                                                fragment_tol_mass=fragment_tol_mass, 
                                                fragment_tol_mode=fragment_tol_mode, 
                                                ion_types="abcxyzImp"
                                                # ion_types="by"
                                                )    
                            
                    spectra.append(spectrum)
                
                
                fig, ax = plt.subplots(figsize=(12, 6))
                plt.title(identifier)
                spectrum_top, spectrum_bottom = spectra
                sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
                mirrorplot_dir = plot_dir+'mirrorplot/'
                if not os.path.exists(mirrorplot_dir):
                    os.makedirs(mirrorplot_dir)
                fig.savefig(mirrorplot_dir+'/{}.png'.format(re.sub('/','_',identifier)))
                plt.close(fig)
                # plt.show()
                # plt.close()
            except:
                print('{} Not Found'.format(title))
        print('Mirror Plot Done!')


def peplist_from_csv(csvfile):
    peptidelist = []
    df = pd.read_csv(csvfile, sep = ',')
    # df['targetpep'] = df['sequence']+ '/' + df['precursor_charge'].astype(str) + '_' + df['collision_energy'].astype(str) + '_' + df['mod_num'].astype(str)
    df['targetpep'] = df['modified_sequence']+ '/' + df['precursor_charge'].astype(str) + '_' + df['collision_energy'].astype(str) + '_' + df['mod_num'].astype(str)
    peptidelist = df['targetpep'].tolist()
    # with open (csvfile, 'r') as f:
    #     f.readline()
    #     for line in f:
    #         seq, ce, charge = line.rstrip('\n').split(',')
    #         peptide = seq + '/' + charge + '_' + str(round(float(ce),1)) + '_' + '0'
    #         peptidelist.append(peptide)
    return (peptidelist)
    

def main():
    parser = ArgumentParser()
    parser.add_argument('-l', '--local', default=False, action='store_true',
                        help='execute in local computer')
    args = parser.parse_args()    

    if args.local is True:
        plot_dir = constants_local.PLOT_DIR
        predict_input = constants_local.PREDICT_INPUT
        predict_format = constants_local.PREDICT_FORMAT
        predict_dir = constants_local.PREDICT_DIR
        reference_spectra = constants_local.REFERENCE_SPECTRA
    else:
        # example_dir = constants_gcp.EXAMPLE_DIR
        plot_dir = constants_gcp.PLOT_DIR
        predict_input = constants_gcp.PREDICT_INPUT
        predict_format = constants_gcp.PREDICT_FORMAT
        predict_dir = constants_gcp.PREDICT_DIR
        reference_spectra = constants_gcp.REFERENCE_SPECTRA

    assert predict_format == 'msp', "PREDICT_FORMAT should be 'msp'"   
    # peptidelistfile = example_dir + 'peptidelist.csv'    
    peptidelistfile = predict_input
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    predict_msp = predict_dir + 'peptidelist_pred.msp'
    predict_mgf = predict_dir + 'peptidelist_pred.mgf'
    
    # get list of peptides for plotting
    peplist = peplist_from_csv(peptidelistfile)

    # store msp files to dictionary and convert to MGF from prosit prediction
    spectrum_prosit = msp_parser.from_msp_prosit(predict_msp)
    msp_parser.dict2mgf(spectrum_prosit, predict_mgf)
    
    # single spectra
    singleplot(peplist[0], predict_mgf, plot_dir)
    # compare two different peptides
    mirroplot_twopeptides(peplist[:2], predict_mgf, plot_dir)
    # compare same peptide from two methods
    mirroplot_twosets(peplist[:20], predict_mgf, reference_spectra, 
                      # data_dir+'human_synthetic_hcd_selected.mgf', 
                      plot_dir)
        
    
if __name__ == "__main__":
    main()

        