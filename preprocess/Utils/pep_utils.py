# import torch
import numpy as np
import math
from pyteomics import mgf, mass
import argparse

SPECTRA_DIMENSION = 1600
BIN_SIZE = 1
MAX_PEPTIDE_LENGTH = 30
MAX_MZ = 1600
ALPHABET = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "M(ox)": 21,
    "M(O)": 21,
}
ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}
MAX_SEQUENCE = 30
CHARGES = [1, 2, 3, 4, 5, 6]
DEFAULT_MAX_CHARGE = len(CHARGES)
METHODS = ["CID", "HCD", "ETD"]


def asnp(x): return np.asarray(x)


def asnp32(x): return np.asarray(x, dtype='float32')


def get_precursor_charge_onehot(charges):
    array = np.zeros(max(CHARGES), dtype=int)
    for i, precursor_charge in enumerate(CHARGES):
        if int(precursor_charge) == charges:
            array[int(precursor_charge) - 1] = 1
    return array.tolist()


def getmod(pep):
    mod = np.zeros(len(pep))

    if pep.isalpha(): return pep, mod, 0

    seq = []
    nmod = 0

    i = -1
    while len(pep) > 0:
        if pep[0] == '(':
            if pep[:3] == '(O)':
                mod[i] = 1
                pep = pep[3:]
            elif pep[:4] == '(ox)':
                mod[i] = 1
                pep = pep[4:]
            elif pep[2] == ')' and pep[1] in 'ASDFGHJKLZXCVBNMQWERTYUIOP':
                mod[i] = -2
                pep = pep[3:]
            else:
                raise 'unknown mod: ' + pep

        elif pep[0] == '+' or pep[0] == '-':
            sign = 1 if pep[0] == '+' else -1

            for j in range(1, len(pep)):
                if pep[j] not in '.1234567890':
                    if i == -1:  # N-term mod
                        nmod += sign * float(pep[1:j])
                    else:
                        mod[i] += sign * float(pep[1:j])
                    pep = pep[j:]
                    break

            if j == len(pep) - 1 and pep[-1] in '.1234567890':  # till end
                mod[i] += sign * float(pep[1:])
                break
        else:
            seq += pep[0]
            pep = pep[1:]
            i = len(seq) - 1  # more realible

    return ''.join(seq), mod[:len(seq)], nmod


def peptide_parser(p):
    p = p.replace("_", "")
    if p[0] == "(":
        raise ValueError("sequence starts with '('")
    n = len(p)
    i = 0
    while i < n:
        if i < n - 3 and p[i + 1] == "(":
            j = p[i + 2:].index(")")
            offset = i + j + 3
            yield p[i:offset]
            i = offset
        else:
            yield p[i]
            i += 1


def get_sequence_integer(sequences):
    array = np.zeros(MAX_PEPTIDE_LENGTH, dtype=int)
    pep, mod, nterm_mod = getmod(sequences)
    # print(len(sequences))
    if nterm_mod != 0:
        # print("input", sequences, 'has N-term modification, ignored')
        pass
    elif np.any(mod != 0) and set(mod) != set([0, 1]):
        # print("Only Oxidation modification is supported, ignored", sequences)
        pass
    elif len(sequences) > MAX_PEPTIDE_LENGTH:
        # print("input", sequences, 'exceed max length of', MAX_PEPTIDE_LENGTH, ", ignored")
        pass
    else:
        for j, s in enumerate(peptide_parser(sequences)):
            array[j] = ALPHABET[s]
    return array.tolist()


def parse_spectra(sps):
    # ratio constants for NCE
    cr = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

    db = []

    for sp in sps:
        param = sp['params']

        c = int(str(param['charge'][0])[0])

        if 'seq' in param:
            pep = param['seq']
        else:
            pep = param['title']

        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])

        if 'hcd' in param:
            try:
                hcd = param['hcd']
                if hcd[-1] == '%':
                    hcd = float(hcd)
                elif hcd[-2:] == 'eV':
                    hcd = float(hcd[:-2])
                    hcd = hcd * 500 * cr[c] / mass
                else:
                    raise Exception("Invalid type!")
            except:
                hcd = 0
        else:
            hcd = 0.25

        mz = sp['m/z array']
        it = sp['intensity array']

        db.append({'pep': pep, 'charge': c,
                   'mass': mass, 'mz': mz, 'it': it, 'nce': hcd})

    return db


def readmgf(fn):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False,
                    dtype='float32', use_index=False)

    codes = parse_spectra(data)
    file.close()
    return codes


def spectrum2vector(mz_list, intensity_list, mass, bin_size, charge):
    intensity_list = intensity_list / np.max(intensity_list)
    vector = np.zeros(SPECTRA_DIMENSION, dtype='float32')

    mz_list = np.asarray(mz_list)
    indexes = np.around(mz_list).astype('int32')
    for i, index in enumerate(indexes):
        if index >= 1600:
            pass
        else:
            vector[index] += intensity_list[i]

    # normalize
    # vector = np.sqrt(vector)
    return vector


def spectrum2vectorn(mz_list, intensity_list, mass, bin_size, charge):
    intensity_list = intensity_list / np.max(intensity_list)
    vector = np.zeros(SPECTRA_DIMENSION, dtype='float32')
    massbin = dict()
    for x in range(SPECTRA_DIMENSION):
        massbin[x] = 0.0
    for index, ion in enumerate(mz_list):
        if math.floor(float(ion)) in massbin.keys():
            massbin[math.floor(float(ion))] = float(intensity_list[index])
    values_list = list(massbin.values())
    vector = np.sqrt(values_list)
    return values_list


def spectrum2vectorpf(mz_list, itensity_list, mass, bin_size, charge):
    itensity_list = itensity_list / np.max(itensity_list)

    vector = np.zeros(SPECTRA_DIMENSION, dtype='float32')

    mz_list = np.asarray(mz_list)

    indexes = mz_list / bin_size
    indexes = np.around(indexes).astype('int32')

    for i, index in enumerate(indexes):
        vector[index] += itensity_list[i]

    # normalize
    vector = np.sqrt(vector)

    # remove precursors, including isotropic peaks
    for delta in (0, 1, 2):
        precursor_mz = mass + delta / charge
        if precursor_mz > 0 and precursor_mz < 2000:
            vector[round(precursor_mz / bin_size)] = 0

    return vector

