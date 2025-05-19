VAL_SPLIT = 0.8

CHUNKSIZE = 2000
TRAIN_EPOCHS = 200
TRAIN_BATCH_SIZE = 1024
PRED_BATCH_SIZE = 2048
PRED_BAYES = False
PRED_N = 100

TOLERANCE_FTMS = 25
TOLERANCE_ITMS = 0.35
TOLERANCE_TRIPLETOF = 0.5

TOLERANCE = {"FTMS": (25, "ppm"), "ITMS": (0.35, "da"), "TripleTOF": (50, "ppm")}

BIN_MAXMZ = 1500
BIN_SIZE = 0.5
BIN_MODE = 'Da'
SPECTRA_DIMENSION = int(BIN_MAXMZ/BIN_SIZE)

# PLOT_FRAGMENT_TOLERANCE_MASS = BIN_SIZE
# PLOT_FRAGMENT_TOLERANCE_MODE = 'Da'

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
    "M(Oxidation)": 21,
    "S(Phospho)": 22,
    "T(Phospho)": 23,
    "Y(Phospho)": 24,
    "C(Carbamidomethyl)": 25,
    "C(DTBIA)": 26,
}
ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}
MAX_ALPHABETSIZE = 100

CHARGES = [1, 2, 3, 4, 5, 6]
DEFAULT_MAX_CHARGE = len(CHARGES)
MAX_FRAG_CHARGE = 3
MAX_SEQUENCE = 30
MAX_ION = MAX_SEQUENCE - 1
ION_TYPES = ["y", "b"]
NLOSSES = ["", "H2O", "NH3"]

FORWARD = {"a", "b", "c"}
BACKWARD = {"x", "y", "z"}

# Amino acids
# MODIFICATION_MASS = {
#     "Carbamidomethyl": 57.0214637236,  # Carbamidomethylation (CAM)
#     "Oxidation": 15.99491,  # Oxidation
#     "Phospho": 79.966331,    # Phosphorylation
#     "DTBIA": 296.185,       # Desthiobiotin
# }

MODIFICATION = {
    # 'Oxidation': 15.99491,
    # 'Phospho': 79.966331,
    # 'Carbamidomethyl': 57.0214637236,
    # 'DTBIA': 296.185,
    'OX': 15.99491,
    'PH': 79.966331,
    'CAM': 57.021464,
    'DTBIA': 296.185,
}

# Customed modification
MODIFICATION_COMPOSITION = {
    'M(Oxidation)': {'O': 1},  # Oxidation
    'S(Phospho)': {'H': 1, 'P': 1, 'O': 3},    # Phosphorylation
    'T(Phospho)': {'H': 1, 'P': 1, 'O': 3},    # Phosphorylation
    'Y(Phospho)': {'H': 1, 'P': 1, 'O': 3},    # Phosphorylation
    'C(Carbamidomethyl)': {'H': 3, 'C': 2, 'O': 1, 'N': 1},     # Carbamidomethylation (CAM)
    'C(DTBIA)': {'H': 24, 'C': 14, 'O': 3, 'N': 4},     # Desthiobiotin
    'C(+296.185)': {'H': 24, 'C': 14, 'O': 3, 'N': 4},     # Desthiobiotin
}

VARMOD_PROFORMA = {
    # Oxidation
    'M(Oxidation)': 'M[Oxidation]',
    'M+15.995': 'M[Oxidation]',

    # Phosphorylation
    'S(Phospho)': 'S[Phospho]',
    'T(Phospho)': 'T[Phospho]',
    'Y(Phospho)': 'Y[Phospho]',
    'S+79.966': 'S[Phospho]',
    'T+79.966': 'T[Phospho]',
    'Y+79.966': 'Y[Phospho]',

    # Carbamidomethylation
    'C(Carbamidomethyl)': 'C[Carbamidomethyl]',
    'C+57.021': 'C[Carbamidomethyl]',

    # Desthiobiotin
    'C(DTBIA)': 'C[+296.185]',
    'C+296.185': 'C[+296.185]',
}

FIXMOD_PROFORMA = {
    'C(Carbamidomethyl)': 'C[Carbamidomethyl]',
    'C+57.021': 'C[Carbamidomethyl]'
}

AMINO_ACID = {
    "G": 57.021464,
    "R": 156.101111,
    "V": 99.068414,
    "P": 97.052764,
    "S": 87.032028,
    "U": 150.95363,
    "L": 113.084064,
    "M": 131.040485,
    "Q": 128.058578,
    "N": 114.042927,
    "Y": 163.063329,
    "E": 129.042593,
    # "C": 103.009185 + MODIFICATION["CAM"],
    "C": 103.009185,
    "F": 147.068414,
    "I": 113.084064,
    "A": 71.037114,
    "T": 101.047679,
    "W": 186.079313,
    "H": 137.058912,
    "D": 115.026943,
    "K": 128.094963,
}
AMINO_ACID["M(Oxidation)"] = AMINO_ACID["M"] + MODIFICATION["OX"]
AMINO_ACID["S(Phospho)"] = AMINO_ACID["S"] + MODIFICATION["PH"]
AMINO_ACID["T(Phospho)"] = AMINO_ACID["T"] + MODIFICATION["PH"]
AMINO_ACID["Y(Phospho)"] = AMINO_ACID["Y"] + MODIFICATION["PH"]
AMINO_ACID["C(Carbamidomethyl)"] = AMINO_ACID["C"] + MODIFICATION["CAM"]
AMINO_ACID["C(DTBIA)"] = AMINO_ACID["C"] + MODIFICATION["DTBIA"]


# Atomic elements
PROTON = 1.007276467
ELECTRON = 0.00054858
H = 1.007825035
C = 12.0
O = 15.99491463
N = 14.003074

# Tiny molecules
N_TERMINUS = H
C_TERMINUS = O + H
CO = C + O
CHO = C + H + O
NH2 = N + H * 2
H2O = H * 2 + O
NH3 = N + H * 3

NEUTRAL_LOSS = {"NH3": NH3, "H2O": H2O}

ION_OFFSET = {
    "a": N_TERMINUS - CHO,
    "b": N_TERMINUS - H,
    "c": N_TERMINUS + NH2,
    "x": C_TERMINUS + CO - H,
    "y": C_TERMINUS + H,
    "z": C_TERMINUS - NH2,
}

METHODS = ["CID", "HCD", "ETD"]
