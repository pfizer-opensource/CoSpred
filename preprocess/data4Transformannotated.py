import functools
import math
import ast
import numpy as np
from Utils.constants import *
import pandas as pd
from Utils import utils

def get_sequence_integer(sequences):
    array = np.zeros([len(sequences), MAX_SEQUENCE], dtype=int)
    for i, sequence in enumerate(sequences):
        for j, s in enumerate(utils.peptide_parser(sequence)):
            array[i, j] = ALPHABET[s]
    return array

def get_numbers(vals, dtype=float):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])

def get_string(vals, dtype=bool):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])

def get_number(vals, dtype=int):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])

def get_precursor_charge_onehot(charges):
    array = np.zeros([len(charges), max(CHARGES)], dtype=int)
    for i, precursor_charge in enumerate(charges):
        array[i, int(precursor_charge) - 1] = 1
    return array

def get_method_onehot(methods):
    array = np.zeros([len(METHODS)], dtype=int)
    for i, method in enumerate(methods):
        array[i, int(method) - 1] = 1
    return array

def reshape_dims(array):
    n, dims = array.shape
    assert dims == 174
    nlosses = 1
    return array.reshape(
        [array.shape[0], MAX_SEQUENCE - 1, len(ION_TYPES), nlosses, MAX_FRAG_CHARGE]
    )


def reshape_flat(array):
    s = array.shape
    flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
    return array.reshape(flat_dim)

def mask_outofrange(array, lengths, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        array[i, lengths[i] - 1 :, :, :, :] = mask
    return array


def cap(array, nlosses=1, z=3):
    return array[:, :, :, :nlosses, :z]


def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        if int(charges[i]) > 3:
            array[i, :, :, :, int(charges[i]) :] = mask
    return array

def parse_ion(string):
    ion_type = ION_TYPES.index(string[0])
    if ("-") in string:
        ion_n, suffix = string[1:].split("-")
    else:
        ion_n = string[1:].split("+")[0]
        suffix = ""
    if ("+") in string:
        ion_frag=string.count("+")
    return ion_type, int(ion_n) - 1, NLOSSES.index(suffix), int(ion_frag)-1

def parse_ions(string):
    ion_type = ION_TYPES.index(string[0])
    if ("-") in string:
        ion_n, suffix = string[1:].split("-")
    else:
        ion_n = string[1:]
        suffix = ""
    return ion_type, int(ion_n) - 1, NLOSSES.index(suffix)

def get_intensity_target(df):

    df['sequence_integer'] = df['sequence_integer'].map(lambda x: ast.literal_eval(x))
    df['precursor_charge_onehot'] = df['precursor_charge_onehot'].map(lambda x: ast.literal_eval(x))
    df['collision_energy_aligned_normed'] = df['collision_energy_aligned_normed'].map(lambda x: ast.literal_eval(x))
    df['intensities_raw'] = df['intensities_raw'].map(lambda x: ast.literal_eval(x))
    df['masses_raw'] = df['masses_raw'].map(lambda x: ast.literal_eval(x))
    def calc_row(row):
        array = np.zeros([MAX_ION, len(ION_TYPES), len(NLOSSES), MAX_FRAG_CHARGE])
        lstions=row.matches_raw.split(";")
        lstintensities=row.intensities.split(";")
        import re
        for i in ION_TYPES:
            patternn = r"^"+i+"[0-9]+\++$"
            Ionsreg= re.compile(patternn)
            for index, ion in enumerate(lstions):
                if Ionsreg.match(ion):
                    it, _in, nloss,ion_frag = parse_ion(ion)
                    array[_in, it, nloss,ion_frag] = float(lstintensities[index])
        max_value = np.max(array)
        return [array/max_value]

    def calc_mass_int(row):
        lstions = row.Matches.split(";")
        lstintensities = row.Intensities.split(";")
        lstmasses = row.Masses.split(";")
        array1 = np.zeros(1600)
        for i in range(len(array1)):
            for index, ion in enumerate(lstmasses):
                if i == math.floor(float(ion)):
                    array1[i] = float(lstintensities[index])
        max_value = np.max(array1)
        return [array1/max_value]

    def calc_mass_int_from_hdf5(row):
        array1 = np.zeros(1600)
        #masses=row.masses_raw
        #print(row.masses_raw)
        masses = ';'.join(map(str, row.masses_raw))
        intensities = ';'.join(map(str, row.intensities_raw))
        lstintensities = intensities.split(";")
        lstmasses = masses.split(";")
        massbin=dict()
        for x in range(1600):
            massbin[x] = 0.0
        #print(massbin.keys())
        #massbin={key: None for key in array1}
        #print(massbin)
        for index, ion in enumerate(lstmasses):
            if math.floor(float(ion)) in massbin.keys():
                massbin[math.floor(float(ion))] = float(lstintensities[index])
        #for i in range(len(array1)):
        #    for index, ion in enumerate(lstmasses):
                #print(ion)
        #        if i == math.floor(float(ion)):
        #           array1[i] = float(lstintensities[index])
        #max_value = np.max(array1)
        values_list = list(massbin.values())
        return [values_list]
    intensity_series = df.apply(calc_mass_int_from_hdf5, 1)
    out = np.squeeze(np.stack(intensity_series))
    #print("outshape")
    #print(out.shape)
    #if len(out.shape) == 4:
    #    out = out.reshape([1] + list(out.shape))
    return out

def csvout(df):
    df.reset_index(drop=True, inplace=True)
    assert "sequence_integer" in df.columns
    assert "collision_energy_aligned_normed" in df.columns
    assert "precursor_charge_onehot" in df.columns
    data = {
        "sequence_integer": df.sequence_integer,
        "precursor_charge_onehot": df.precursor_charge_onehot,
        "collision_energy_aligned_normed": df.collision_energy_aligned_normed,
        "intensities_raw": get_intensity_target(df)
    }
    return data

df = pd.read_csv('C:/Users/tiwars46/PycharmProjects/data/prosit_data/train_tensor_for_transform.csv',sep=',')

dfy=csvout(df)
print(len(dfy))
print(dfy['collision_energy_aligned_normed'])


collision_energy_aligned_normed = dfy['collision_energy_aligned_normed'][:].tolist()
precursor_charge_onehot = dfy['precursor_charge_onehot'][:].tolist()
sequence_integer = dfy['sequence_integer'][:].tolist()
intensities_raw = dfy['intensities_raw'][:].tolist()
dataset = pd.DataFrame({"collision_energy_aligned_normed": collision_energy_aligned_normed, "precursor_charge_onehot": precursor_charge_onehot, "sequence_integer": sequence_integer, "intensities_raw": intensities_raw})
dataset.to_csv("C:/Users/tiwars46/PycharmProjects/data/train_annotated_for_transform.csv", index = False)


