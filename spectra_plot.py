import logging
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
import warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


# single plot
def singleplot(feature, predict_mgf, plot_dir, min_mz=0, min_intensity=0.02):
    # Read the spectrum from an MGF file using Pyteomics.
    spectrum_dict = mgf.get_spectrum(predict_mgf, plot_sequence(feature))

    identifier = spectrum_dict['params']['title']
    precursor_mz = spectrum_dict['params']['pepmass'][0]
    precursor_charge = spectrum_dict['params']['charge'][0]
    mz = spectrum_dict['m/z array']
    intensity = spectrum_dict['intensity array']
    retention_time = float(spectrum_dict['params']['rtinseconds'])
    # peptide = spectrum_dict['params']['seq'].replace("(ox)", "[Oxidation]")\
    #     .replace("(ph)", "[Phospho]")
    peptide = spectrum_dict['params']['seq']

    # Create the MS/MS spectrum.
    spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge, mz, intensity,
                                retention_time=retention_time,
                                )
    # Filter and clean up the MS/MS spectrum.
    spectrum = spectrum.set_mz_range(min_mz=0, max_mz=constants.BIN_MAXMZ). \
        remove_precursor_peak(constants.BIN_SIZE, constants.BIN_MODE). \
        filter_intensity(min_intensity=min_intensity, max_num_peaks=50)
    # Annotate the MS2 spectrum.
    spectrum = spectrum.annotate_proforma(peptide,
                                          fragment_tol_mass=constants.BIN_SIZE,
                                          fragment_tol_mode=constants.BIN_MODE,
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
    logging.info('Single Peptide Plot Done!')

# mirror plot for two different peptides


def mirroplot_twopeptides(peplist, predict_mgf, plot_dir, min_mz=0, min_intensity=0.02):
    # convert peptide list to proforma
    peplist = [plot_sequence(x) for x in peplist]

    spectra = []
    for spectrum_dict in mgf.read(predict_mgf):
        if peplist[0] in spectrum_dict['params']['title'] or peplist[1] in spectrum_dict['params']['title']:
            identifier = spectrum_dict['params']['title']
            precursor_mz = spectrum_dict['params']['pepmass'][0]
            precursor_charge = spectrum_dict['params']['charge'][0]
            mz = spectrum_dict['m/z array']
            intensity = spectrum_dict['intensity array']
            retention_time = float(spectrum_dict['params']['rtinseconds'])
            # peptide = spectrum_dict['params']['seq'].replace("(ox)", "[Oxidation]")\
            #     .replace("(ph)", "[Phospho]")
            peptide = spectrum_dict['params']['seq']
            
            # Create the MS/MS spectrum.
            spectrum = sus.MsmsSpectrum(identifier, precursor_mz,
                                        precursor_charge, mz, intensity,
                                        retention_time=retention_time,
                                        )
            # Filter and clean up the MS/MS spectrum.
            spectrum = spectrum.set_mz_range(min_mz=min_mz, max_mz=constants.BIN_MAXMZ). \
                remove_precursor_peak(constants.BIN_SIZE, constants.BIN_MODE). \
                filter_intensity(min_intensity=min_intensity, max_num_peaks=50)

            # Annotate the MS2 spectrum.
            spectrum = spectrum.annotate_proforma(peptide,
                                                  fragment_tol_mass=constants.BIN_SIZE,
                                                  fragment_tol_mode=constants.BIN_MODE,
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
    logging.info('Double Peptides Plot Done!')

# mirror plot for two dataset
def mirroplot_twosets(peplist, predict_mgf, reference_spectra, plot_dir, min_mz=0, min_intensity=0.02):
    if not os.path.isfile(predict_mgf):
        logging.error('{} not found'.format(predict_mgf))
    elif not os.path.isfile(reference_spectra):
        logging.error('{} not found'.format(reference_spectra))
    else:
        pair = []
        for title in peplist:
            spectra = []
            try:
                pred_dict = mgf.get_spectrum(predict_mgf, plot_sequence(title))
                ref_dict = mgf.get_spectrum(reference_spectra, title)
                if (ref_dict is None or pred_dict is None):
                    next
                pair = [pred_dict, ref_dict]

                for spectrum_dict in pair:
                    identifier = plot_sequence(spectrum_dict['params']['title'])
                    precursor_mz = spectrum_dict['params']['pepmass'][0]
                    precursor_charge = spectrum_dict['params']['charge'][0]
                    mz = spectrum_dict['m/z array']
                    intensity = spectrum_dict['intensity array']
                    retention_time = float(
                        spectrum_dict['params']['rtinseconds'])
                    # peptide = spectrum_dict['params']['seq'].replace("(ox)", "[Oxidation]")\
                    #     .replace("(ph)", "[Phospho]")
                    peptide = plot_sequence(spectrum_dict['params']['seq'])

                    # Create the MS/MS spectrum.
                    spectrum = sus.MsmsSpectrum(identifier, precursor_mz,
                                                precursor_charge, mz, intensity,
                                                retention_time=retention_time,
                                                # peptide=peptide,
                                                # modifications=modifications
                                                )
                    # Filter and clean up the MS/MS spectrum.
                    spectrum = spectrum.set_mz_range(min_mz=min_mz, max_mz=constants.BIN_MAXMZ). \
                        remove_precursor_peak(constants.BIN_SIZE, constants.BIN_MODE). \
                        filter_intensity(
                            min_intensity=min_intensity, max_num_peaks=50)

                    # Annotate the MS2 spectrum.
                    spectrum = spectrum.annotate_proforma(peptide,
                                                          fragment_tol_mass=constants.BIN_SIZE,
                                                          fragment_tol_mode=constants.BIN_MODE,
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
                logging.error('{} Not Found'.format(title))
        logging.info('Mirror Plot Done!')


def peplist_from_csv(csvfile):
    peptidelist = []
    df = pd.read_csv(csvfile, sep=',')
    df['targetpep'] = df['modified_sequence'] + '/' + df['precursor_charge'].astype(
        str) + '_' + df['collision_energy'].astype(str) + '_' + df['mod_num'].astype(str)
    peptidelist = df['targetpep'].tolist()
    return (peptidelist)


def plot_sequence(sequence):
    """
    >>> plot_sequence("C(DTBIA)M(Oxidation)S(Phospho)T(Phospho)Y(Phospho)")
    'C[+296.185]M[Oxidation]S[Phospho]T[Phospho]Y[Phospho]'
    """
    # To plot byion, replace the modified amino acids with their corresponding proforma representation
    mod_dict = constants.VARMOD_PROFORMA
    for key, replacement in mod_dict.items():
        try:
            sequence = sequence.replace(key, replacement)
        except re.error as e:
            print(f"Error processing key '{key}': {e}")
    return sequence


# def plot_sequence(sequence):
#     mod_dict = constants.MODIFICATION
#     for modified_aa, _ in constants.MODIFICATION_COMPOSITION.items():
#         match = re.match(r"([A-Z])\(([A-Za-z]+.*)\)", modified_aa)
#         if match:
#             amino_acid = match.group(1)  # Extracts 'C'
#             modification = match.group(2)  # Extracts 'DTBIA'
#             # example: modified_aa = "C(DTBIA)" -> "C[+296.185]"
#             sequence = sequence.replace(modified_aa, f'{amino_acid}[+{mod_dict[modification.upper()]}]')
#     return sequence


def main():
    # Suppress warning message of tensorflow compatibility
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")

    # Configure logging
    log_file_plot = os.path.join(constants_location.LOGS_DIR, "cospred_plot.log")
    logging.basicConfig(
        filename=log_file_plot,
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

    parser = ArgumentParser()
    parser.parse_args()

    plot_dir = constants_location.PLOT_DIR
    predict_csv = constants_location.PREDICT_ORIGINAL
    predict_format = constants_location.PREDICT_FORMAT
    predict_dir = constants_location.PREDICT_DIR
    reference_spectra = constants_location.REFORMAT_TEST_PATH
    reference_spectra_usi = constants_location.REFORMAT_TEST_USITITLE_PATH
    predict_msp = predict_dir + constants_location.PREDICT_LIB_FILENAME + '.msp'
    predict_mgf = predict_dir + constants_location.PREDICT_LIB_FILENAME + '.mgf'
    min_mz = 0
    min_intensity = 0.02

    assert predict_format == 'msp', "PREDICT_FORMAT should be 'msp'"
    peptidelistfile = predict_csv
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # get list of peptides for plotting
    peplist = peplist_from_csv(peptidelistfile)
    # # convert peptide list to proforma
    # peplist = [plot_sequence(x) for x in peplist]
    # print("Proforma Peptide List: ", peplist)

    # store msp files to dictionary and convert to MGF from prosit prediction
    spectrum_prosit = msp_parser.from_msp_prosit(predict_msp)
    msp_parser.dict2mgf(spectrum_prosit, predict_mgf)

    # single spectra
    singleplot(peplist[0], predict_mgf, plot_dir, min_mz, min_intensity)
    # compare two different peptides
    mirroplot_twopeptides(peplist[:2], predict_mgf, plot_dir, min_mz, min_intensity)
    # compare same peptide from two methods
    if (os.path.exists(reference_spectra)):
        mirroplot_twosets(peplist[:20], predict_mgf, reference_spectra,
                        plot_dir, min_mz, min_intensity)
    elif (os.path.exists(reference_spectra_usi)):
        mirroplot_twosets(peplist[:20], predict_mgf, reference_spectra_usi,
                        plot_dir, min_mz, min_intensity)
    else:
        logging.error("No Reference Spectra for Mirror Plot")


if __name__ == "__main__":
    main()
