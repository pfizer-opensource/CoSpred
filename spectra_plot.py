#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:50:49 2020

@author: xuel12
"""
from prosit_model import msp_parser
import params.constants_location as constants_location
import params.constants as constants
from argparse import ArgumentParser
import re
import pandas as pd
from pyteomics import mgf
import spectrum_utils.spectrum as sus
import spectrum_utils.plot as sup
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


fragment_tol_mass = constants.BIN_SIZE
fragment_tol_mode = constants.BIN_MODE
min_mz = 0
max_mz = constants.BIN_MAXMZ
min_intensity = 0.02

# single plot


def singleplot(feature, predict_mgf, plot_dir):
    # Read the spectrum from an MGF file using Pyteomics.
    spectrum_dict = mgf.get_spectrum(predict_mgf, feature)

    identifier = spectrum_dict['params']['title']
    precursor_mz = spectrum_dict['params']['pepmass'][0]
    precursor_charge = spectrum_dict['params']['charge'][0]
    mz = spectrum_dict['m/z array']
    intensity = spectrum_dict['intensity array']
    retention_time = float(spectrum_dict['params']['rtinseconds'])
    peptide = spectrum_dict['params']['seq'].replace("(ox)", "[Oxidation]")\
        .replace("(ph)", "[Phospho]")

    # Create the MS/MS spectrum.
    spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge, mz, intensity,
                                retention_time=retention_time,
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
                                          )
    # Plot the MS/MS spectrum.
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title(identifier)
    sup.spectrum(spectrum, ax=ax)
    singleplot_dir = plot_dir+'singleplot/'
    if not os.path.exists(singleplot_dir):
        os.makedirs(singleplot_dir)
    fig.savefig(singleplot_dir+'{}.png'.format(re.sub('/', '_', identifier)))
    plt.close(fig)
    print('Single Peptide Plot Done!')

# mirror plot for two different peptides


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

            # Create the MS/MS spectrum.
            spectrum = sus.MsmsSpectrum(identifier, precursor_mz,
                                        precursor_charge, mz, intensity,
                                        retention_time=retention_time,
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
                                                  )
            spectra.append(spectrum)

    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    plt.title(re.sub('/', '_', spectrum_top.identifier)+"_vs_" +
              re.sub('/', '_', spectrum_bottom.identifier))
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    doubleplot_dir = plot_dir+'doubleplot/'
    if not os.path.exists(doubleplot_dir):
        os.makedirs(doubleplot_dir)
    fig.savefig(doubleplot_dir+'{}vs{}.png'.format(re.sub('/', '_', spectrum_top.identifier),
                                                   re.sub('/', '_', spectrum_bottom.identifier)))
    plt.close(fig)
    print('Double Peptides Plot Done!')

# mirror plot for two dataset


def mirroplot_twosets(peplist, predict_mgf, reference_spectra, plot_dir):
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
                    retention_time = float(
                        spectrum_dict['params']['rtinseconds'])
                    peptide = spectrum_dict['params']['seq'].replace("(ox)", "[Oxidation]")\
                        .replace("(ph)", "[Phospho]")

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
                        filter_intensity(
                            min_intensity=min_intensity, max_num_peaks=50)

                    # Annotate the MS2 spectrum.
                    spectrum = spectrum.annotate_proforma(peptide,
                                                          fragment_tol_mass=fragment_tol_mass,
                                                          fragment_tol_mode=fragment_tol_mode,
                                                          ion_types="abcxyzImp"
                                                          )

                    spectra.append(spectrum)

                fig, ax = plt.subplots(figsize=(12, 6))
                plt.title(identifier)
                spectrum_top, spectrum_bottom = spectra
                sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
                mirrorplot_dir = plot_dir+'mirrorplot/'
                if not os.path.exists(mirrorplot_dir):
                    os.makedirs(mirrorplot_dir)
                fig.savefig(mirrorplot_dir +
                            '/{}.png'.format(re.sub('/', '_', identifier)))
                plt.close(fig)
            except:
                print('{} Not Found'.format(title))
        print('Mirror Plot Done!')


def peplist_from_csv(csvfile):
    peptidelist = []
    df = pd.read_csv(csvfile, sep=',')
    df['targetpep'] = df['modified_sequence'] + '/' + df['precursor_charge'].astype(
        str) + '_' + df['collision_energy'].astype(str) + '_' + df['mod_num'].astype(str)
    peptidelist = df['targetpep'].tolist()
    return (peptidelist)


def main():
    parser = ArgumentParser()
    parser.parse_args()

    plot_dir = constants_location.PLOT_DIR
    predict_input = constants_location.PREDICT_INPUT
    predict_format = constants_location.PREDICT_FORMAT
    predict_dir = constants_location.PREDICT_DIR
    reference_spectra = constants_location.REFERENCE_SPECTRA

    assert predict_format == 'msp', "PREDICT_FORMAT should be 'msp'"
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
                      plot_dir)


if __name__ == "__main__":
    main()
