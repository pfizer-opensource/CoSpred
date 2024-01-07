#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:50:49 2020

@author: xuel12
"""
import os
#os.chdir('/Users/xuel12/Documents/MSdatascience/CS7180AI/project/prosit/local_training')
os.chdir('/Users/tiwars46/PycharmProjects/prosit_PfizerRD/local_training')
import matplotlib
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus
from pyteomics import mgf
import numpy as np
import constants
import msp_parser


def _charge_to_str(self):
    """
    Convert a numeric charge to a string representation.
    """
    if self.charge is None:
        return 'unknown'
    elif self.charge > 0:
        return '+' * self.charge
    elif self.charge < 0:
        return '-' * -self.charge
    else:
        return 'undefined'

def fragmentAnnotationmgftocsv(feature,rawfile,scan_number,peptide, file):
    spectrum_dict = mgf.get_spectrum(file, feature)
    modifications = {}
    identifier = spectrum_dict['params']['title']
    precursor_mz = spectrum_dict['params']['pepmass'][0]
    precursor_charge = spectrum_dict['params']['charge'][0]
    mz = spectrum_dict['m/z array']
    intensity = spectrum_dict['intensity array']
    retention_time = float(spectrum_dict['params']['rtinseconds'])
    #peptide = spectrum_dict['params']['seq']
    peptide = peptide
    raw_file= rawfile
    scannum=scan_number

    # Create the MS/MS spectrum.
    spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge, mz, intensity, \
                                retention_time=retention_time, peptide=peptide, \
                                modifications=modifications)

    # Process the MS/MS spectrum.
    fragment_tol_mass = 0.35
    fragment_tol_mode = 'Da'
    min_mz = 100
    min_intensity = 0.05
    spectrum = (spectrum.set_mz_range(min_mz=min_mz, max_mz=1400)
                .remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)
                .filter_intensity(min_intensity=min_intensity, max_num_peaks=50)
                # .scale_intensity('root')
                .annotate_peptide_fragments(fragment_tol_mass, fragment_tol_mode, max_ion_charge=3, ion_types='by'))
    #print("spectra")
    #print(spectrum.annotation)
    intensity_ann =spectrum.intensity.tolist()
    intensity_annotations = [str(element) for element in intensity_ann]
    intensity_annotations = ";".join(intensity_annotations)
    mz_ann= spectrum.mz.tolist()
    mz_annotations = [str(element) for element in mz_ann]
    mz_annotations = ";".join(mz_annotations)
    ion_ann = spectrum.annotation.tolist()
    ion_annotations = [str(element) for element in ion_ann]
    ion_annotations = ";".join(ion_annotations)
    import pandas as pd
    #dataset = {"intensities_raw": intensity_annotations, "masses_raw": mz_annotations, "matches_raw": ion_annotations}
    #dct = {k: [v] for k, v in dataset.items()}  # WORKAROUND
    #df = pd.DataFrame(dct)
    return raw_file,scannum,intensity_annotations, mz_annotations ,ion_annotations
    #print(df)
    #dataset.to_csv("data/frommgftocsv.csv", index = False)



##### single plot
def singleplot(feature, peptide, file):
    # Read the spectrum from an MGF file using Pyteomics.
    spectrum_dict = mgf.get_spectrum(file, feature)
    #dictdf=mgf.read(file,read_charges=True,convert_arrays=0)
    #print(dictdf.get_spectrum(feature))

    # modifications = {8: 15.994915}
    modifications = {}
    #peptide="ASASGVFCCPLCR"
    identifier = spectrum_dict['params']['title']
    print(identifier)
    precursor_mz = spectrum_dict['params']['pepmass'][0]
    precursor_charge = spectrum_dict['params']['charge'][0]
    mz = spectrum_dict['m/z array']
    intensity = spectrum_dict['intensity array']
    charged = spectrum_dict['charge array']
    retention_time = float(spectrum_dict['params']['rtinseconds'])
    #peptide = spectrum_dict['params']['seq']
    
    # # if sorted mz is desired
    # pair = np.vstack([spectrum_dict['m/z array'], spectrum_dict['intensity array']]).T
    # sorted_pair = pair[np.argsort(pair[:, 0])]
    # mz, intensity = [sorted_pair[:,0], sorted_pair[:,1]]
    
    # Create the MS/MS spectrum.
    spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge, mz, intensity, \
                              retention_time=retention_time, peptide=peptide, \
                                modifications=modifications)

    # Process the MS/MS spectrum.
    fragment_tol_mass = 0.35
    fragment_tol_mode = 'Da'
    min_mz = 100
    min_intensity = 0.05
    spectrum = (spectrum.set_mz_range(min_mz=min_mz, max_mz=1400)
                .remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)
                .filter_intensity(min_intensity=min_intensity, max_num_peaks=50)
                # .scale_intensity('root')
                .annotate_peptide_fragments(fragment_tol_mass, fragment_tol_mode, max_ion_charge=3, ion_types='by'))
    #intensity_ann = spectrum.intensity
    #mz_ann = spectrum.intensity
    #annotation_ann = spectrum.annotation
    #import pandas as pd
    #dataset= pd.DataFrame({"intensities_raw": intensity_ann})
    #print("spectra")
    #print(spectrum.intensity.tolist())
    #print(len(spectrum.intensity.tolist()))
    #print(spectrum.mz.tolist())
    #print(spectrum.annotation.tolist())


    #print(spectrum.annotate_peptide_fragments(fragment_tol_mass, fragment_tol_mode, ion_types='aby'))
    #print(sup.spectrum(spectrum))
    # # label the mz for all peaks
    # annotate_fragment_mz = sorted_pair[(sorted_pair[:,0]>min_mz) & (sorted_pair[:,1]>min_intensity), 0]
    # for fragment_mz in annotate_fragment_mz:
    #     spectrum.annotate_mz_fragment(fragment_mz, 1, fragment_tol_mass, fragment_tol_mode)
        
    # Plot the MS/MS spectrum.
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title(identifier)
    sup.spectrum(spectrum, ax=ax)
    plt.show()
    plt.close()


##### mirror plot
def mirroplot_twopeptides(peplist, file):
    fragment_tol_mass = 0.5
    fragment_tol_mode = 'Da'
    spectra = []
    for spectrum_dict in mgf.read(file):
        if peplist[0] in spectrum_dict['params']['title'] or peplist[1] in spectrum_dict['params']['title']:
            identifier = spectrum_dict['params']['title']
            precursor_mz = spectrum_dict['params']['pepmass'][0]
            precursor_charge = spectrum_dict['params']['charge'][0]
            mz = spectrum_dict['m/z array']
            intensity = spectrum_dict['intensity array']
            retention_time = float(spectrum_dict['params']['rtinseconds'])
            peptide = spectrum_dict['params']['seq']
            # modifications = {6: 15.994915}
            modifications = {}
    
            # Create the MS/MS spectrum.
            spectra.append(sus.MsmsSpectrum(identifier, precursor_mz,
                                            precursor_charge, mz, intensity,
                                            retention_time=retention_time,
                                            peptide=peptide,
                                            modifications=modifications)
                           .filter_intensity(0.01, 50)
                           # .scale_intensity('root')
                           .annotate_peptide_fragments(fragment_tol_mass, fragment_tol_mode, ion_types='aby'))
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()
    plt.close()


##### mirror plot for two dataset
def mirroplot_twosets(peplist, pred_file, ref_file, plot_dir):
    import re
    
    min_intensity = 0.01
    fragment_tol_mass = 0.5
    fragment_tol_mode = 'Da'
    title = peplist[4]
    for title in peplist:
        spectra = []
        pred_dict = mgf.get_spectrum(example_dir+'test.mgf', title)
        ref_dict = mgf.get_spectrum(data_dir+'human_synthetic_hcd_selected.mgf', title)
        if (ref_dict is None):
            break
        pair = [pred_dict, ref_dict]
        for spectrum_dict in pair:
            identifier = spectrum_dict['params']['title']
            precursor_mz = spectrum_dict['params']['pepmass'][0]
            precursor_charge = spectrum_dict['params']['charge'][0]
            mz = spectrum_dict['m/z array']
            intensity = spectrum_dict['intensity array']
            retention_time = float(spectrum_dict['params']['rtinseconds'])
            peptide = spectrum_dict['params']['seq']
            # modifications = {6: 15.994915}
            modifications = {}
            
            # Create the MS/MS spectrum.
            spectra.append(sus.MsmsSpectrum(identifier, precursor_mz,
                                            precursor_charge, mz, intensity,
                                            retention_time=retention_time,
                                            peptide=peptide,
                                            modifications=modifications)
                           .filter_intensity(min_intensity, 50)
                           # .scale_intensity('root')
                           .annotate_peptide_fragments(fragment_tol_mass, fragment_tol_mode, ion_types='aby'))
            
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.title(identifier)
        spectrum_top, spectrum_bottom = spectra
        sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
        fig.savefig(plot_dir+'/{}.png'.format(re.sub('/','_',identifier)))
        plt.close(fig)
        # plt.show()
        # plt.close()


def peplist_from_csv(csvfile):
    peptidelist = []
    with open (csvfile, 'r') as f:
        f.readline()
        for line in f:
            seq, ce, charge = line.rstrip('\n').split(',')
            peptide = seq + '/' + charge + '_' + str(round(float(ce),1)) + '_' + '0'
            peptidelist.append(peptide)
    return (peptidelist)

def readmzml(rawfilename,scannumber):
    from pyteomics import mzml
    import re
    f = mzml.MzML(data_dir + rawfilename+'.mzml')
    controller_str= 'controllerType=0 controllerNumber=1 '
    p = f.get_by_id(controller_str + "scan=" + str(scannumber))
    #p = f.get_by_id('controllerType=0 controllerNumber=1 scan=798')
    dfg = p.get('precursorList')
    fg= dfg['precursor']
    collision_energy = fg[0].get('activation').get('collision energy')
    charge_state = fg[0].get('selectedIonList').get('selectedIon')[0].get('charge state')
    filter_string = p.get('scanList').get('scan')[0].get('filter string')
    retention_time = p.get('scanList').get('scan')[0].get('scan start time')
    if re.search("hcd", filter_string):
        method = "HCD"
    if re.search("cid",filter_string):
        method = "CID"
    if re.search("etd",filter_string):
        method = "ETD"
    return collision_energy, charge_state, retention_time, method

def peplist_csv(csvfile):
    peptidelist = []
    with open (csvfile, 'r') as f:
        f.readline()
        for line in f:
            rawfile, scan_number, charge, seq = line.rstrip('\n').split(',')
            peptidelist.append(seq)

    return (peptidelist)
def rawfilelist_from_csv(csvfile):
    peptidelist = []
    with open (csvfile, 'r') as f:
        f.readline()
        #File: "01650b_BF3-TUM_first_pool_64_01_01-3xHCD-1h-R2.raw", NativeID: "controllerType=0 controllerNumber=1 scan=798"
        for line in f:
            rawfile, scan_number, charge, seq = line.rstrip('\n').split(',')
            peptide = rawfile + '.' + scan_number + '.' + scan_number + '.' + charge + ' File:"'+ rawfile + '.raw' + '"'+ ',' + ' NativeID:"controllerType=0 controllerNumber=1 ' + 'scan=' + scan_number+'"'
            peptidelist.append(str(peptide))
    return (peptidelist)
    
if __name__ == "__main__":
    os.chdir(constants.BASE_PATH + 'local_training')
    data_dir = constants.DATA_DIR
    example_dir = constants.EXAMPLE_DIR
    plot_dir = constants.PLOT_DIR
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # get list of peptides for plotting
    peplist = peplist_csv(example_dir + '/synpeptexample.csv')
    #print(peplist[0])
    import pandas as pd
    #rawfilelist =pd.read_csv(example_dir + '/synpeptexample.csv')
    rawfilelist = rawfilelist_from_csv(example_dir + '/synpeptexample.csv')
    print(len(rawfilelist))
    csvfile = example_dir + '/synpeptexample.csv'
    fd = []
    with open(csvfile, 'r') as f:
        f.readline()
        for line in f:
            rawfile, scan_number, charge, seq = line.rstrip('\n').split(',')
            rawfilels = rawfile + '.' + scan_number + '.' + scan_number + '.' + charge + ' File:"' + rawfile + '.raw' + '"' + ',' + ' NativeID:"controllerType=0 controllerNumber=1 ' + 'scan=' + scan_number + '"'
            #peptidelist.append(str(peptide))

        #for i in range(len(rawfilelist)):
            rawfiles,scan, intensity_annotations, mz_annotations, ion_annotations = fragmentAnnotationmgftocsv(rawfilels, rawfile, scan_number, seq, example_dir+'01650b_BF3-TUM_first_pool_64_01_01-3xHCD-1h-R2.mgf')
            collision_energy, charge_state, retention_time, method = readmzml(rawfile,scan_number)

            dataset = {"raw_files": rawfiles,"scan_number": scan,"intensities_raw": intensity_annotations, "masses_raw": mz_annotations, "matches_raw": ion_annotations, "collision_energy": collision_energy, "charge_state":charge_state, "retention_time": retention_time,"method":method}
            dct = {k: [v] for k, v in dataset.items()}
            fd.append(dct)
    df = pd.DataFrame(fd)
    df.to_csv(example_dir + '/annforspectr_utilex.csv',sep=',')



        

        