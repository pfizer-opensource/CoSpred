import collections
import numpy as np

import constants
import utils
import match
import annotate
import sanitize
from constants import (
    CHARGES,
    MAX_SEQUENCE,
    ALPHABET,
    MAX_ION,
    NLOSSES,
    CHARGES,
    ION_TYPES,
    ION_OFFSET,
)


def stack(queue):
    listed = collections.defaultdict(list)
    for t in queue.values():
        if t is not None:
            for k, d in t.items():
                listed[k].append(d)
    stacked = {}
    for k, d in listed.items():
        if isinstance(d[0], list):
            stacked[k] = [item for sublist in d for item in sublist]
        else:
            stacked[k] = np.vstack(d)
    return stacked


def get_numbers(vals, dtype=float):
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_precursor_charge_onehot(charges):
    array = np.zeros([len(charges), max(CHARGES)], dtype=int)
    for i, precursor_charge in enumerate(charges):
        array[i, precursor_charge - 1] = 1
    return array


def get_sequence_integer(sequences):
    array = np.zeros([len(sequences), MAX_SEQUENCE], dtype=int)
    for i, sequence in enumerate(sequences):
        for j, s in enumerate(utils.peptide_parser(sequence)):
            array[i, j] = ALPHABET[s]
    return array


def parse_ion(string):
    ion_type = ION_TYPES.index(string[0])
    if ("-") in string:
        ion_n, suffix = string[1:].split("-")
    else:
        ion_n = string[1:]
        suffix = ""
    return ion_type, int(ion_n) - 1, NLOSSES.index(suffix)


def get_mz_applied(df, ion_types="yb"):
    ito = {it: ION_OFFSET[it] for it in ion_types}

    def calc_row(row):
        array = np.zeros([MAX_ION, len(ION_TYPES), len(NLOSSES), len(CHARGES)])
        fw, bw = match.get_forward_backward(row.modified_sequence)
        for z in range(row.precursor_charge):
            zpp = z + 1
            annotation = annotate.get_annotation(fw, bw, zpp, ito)
            for ion, mz in annotation.items():
                it, _in, nloss = parse_ion(ion)
                array[_in, it, nloss, z] = mz
        print("get mz")
        print(array)
        return [array]

    mzs_series = df.apply(calc_row, 1)
    out = np.squeeze(np.stack(mzs_series))
    if len(out.shape) == 4:
        out = out.reshape([1] + list(out.shape))
    return out

def get_array_from_df(df_column):
    import numpy
    m = df_column.str.split(';').str.join(',').to_numpy()
    arr = []
    for i, k in enumerate(m):
        p = list(map(float, k.split(',')))
        arr.append(p)
    b = np.zeros([len(arr), len(max(arr, key=lambda x: len(x)))])
    for i, j in enumerate(arr):
        b[i][0:len(j)] = j
    return b

def norm_intensity(intensities):
    import numpy
    m=intensities.str.split(',').str.join(',').to_numpy()
    arr = []
    for i, k in enumerate(m):
        p = list(map(float, k.split(',')))
        arr.append(p)
    b = np.zeros([len(arr), len(max(arr, key=lambda x: len(x)))])
    for i, j in enumerate(arr):
        b[i][0:len(j)] = j
    maxima = b.max(axis=1)
    b = b / maxima[:, numpy.newaxis]
    return b

def csv(df):
    df.reset_index(drop=True, inplace=True)
    assert "modified_sequence" in df.columns
    assert "collision_energy" in df.columns
    assert "precursor_charge" in df.columns
    assert "masses" in df.columns
    assert "collision_energy" in df.columns
    assert "intensities" in df.columns

    data = {
        "collision_energy_aligned_normed": get_numbers(df.collision_energy) / 100.0,
        "sequence_integer": get_sequence_integer(df.modified_sequence),
        "precursor_charge_onehot": get_precursor_charge_onehot(df.precursor_charge),
        "masses_pred": get_mz_applied(df),
        "masses_raw": get_array_from_df(df.masses),
       # "collision_energy": df.collision_energy,
        "intensities_raw": sanitize.normalize_base_peak(get_array_from_df(df.intensities)),
    }
    nlosses = 1
    z = 3
    lengths = (data["sequence_integer"] > 0).sum(1)
    #intensities_raw = sanitize.reshape_dims(get_array_from_df(df.intensities))
    #print(intensities_raw)
    masses_pred = get_mz_applied(df)
    masses_pred = sanitize.cap(masses_pred, nlosses, z)
    masses_pred = sanitize.mask_outofrange(masses_pred, lengths)
    masses_pred = sanitize.mask_outofcharge(masses_pred, df.precursor_charge)
    masses_pred = sanitize.reshape_flat(masses_pred)
    data["masses_pred"] = masses_pred
    print("massed")
    #data["masses_pred"].tofile("C:/Users/tiwars46/OneDrive-Pfizer/PycharmProjects/prosit_PfizerRD/data/outputmasses.txt",sep='\t')
    print(data["masses_pred"])
    print("raw masses")
    print(data["masses_raw"])
    print("intensities")
    print(data["intensities_raw"])
    return data

