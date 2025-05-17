import numpy as np
import pyteomics
import re

import params.constants as constants
from prosit_model import utils


def preprocess_sequence(sequence):
    for modified_aa, _ in constants.MODIFICATION_COMPOSITION.items():
        # Extract the amino acid and modification
        match = re.match(r"([A-Z])\((.+)\)", modified_aa)
        if match:
            amino_acid = match.group(1)         # Extracts 'C'
            modification = match.group(2)       # Extracts 'DTBIA'
            # example: modified_aa = "C(DTBIA)" -> "dtbiaC"
            sequence = sequence.replace(modified_aa, f'{modification.lower()}{amino_acid}')
    return sequence


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
    
    # mod_dict = constants.MODIFICATION_MASS
    # for modified_aa, _ in constants.VARMOD_PROFORMA.items():
    #     # Extract the amino acid and modification
    #     match = re.match(r"([A-Z])\((.+)\)", modified_aa)
    #     if match:
    #         amino_acid = match.group(1)  # Extracts 'C'
    #         modification = match.group(2)  # Extracts 'DTBIA'
    #         # example: modified_aa = "C(DTBIA)" -> "C[+296.185]"
    #         sequence = sequence.replace(modified_aa, f'{amino_acid}[+{mod_dict[modification.upper()]}]')
    
    return sequence


def generate_aa_comp():
    """
    Generate the amino acid composition dictionary using definitions from constants.py.
    """
    # db = pyteomics.mass.Unimod()
    aa_comp = dict(pyteomics.mass.std_aa_comp)

    for mod_name, mod_composition in constants.MODIFICATION_COMPOSITION.items():
        # Extract the amino acid and modification
        match = re.match(r"([A-Z])\((.+)\)", mod_name)
        if match:
            amino_acid = match.group(1)  # Extracts 'C'
            modification = match.group(2)  # Extracts 'DTBIA'
            # example: modified_aa = "C(DTBIA)" -> "dtbiaC"
            mod_key = f"{modification.lower()}{amino_acid}"
            if mod_key not in aa_comp:
                aa_comp[mod_key] = aa_comp[amino_acid] + mod_composition

    return aa_comp


aa_comp = generate_aa_comp()


def get_ions():
    x = np.empty(
        [constants.MAX_ION, len(constants.ION_TYPES), constants.MAX_FRAG_CHARGE],
        dtype="|S6",
    )
    for fz in range(constants.MAX_FRAG_CHARGE):
        for fty_i, fty in enumerate(constants.ION_TYPES):
            for fi in range(constants.MAX_ION):
                ion = fty + str(fi + 1)
                if fz > 0:
                    ion += "({}+)".format(fz + 1)
                x[fi, fty_i, fz] = ion
    x.flatten()
    return x


# Don't see where this function is used
def calculate_mods(sequence_integer):
    """
    >>> x = np.array([2, 15, 4, 3, 0, 0])
    >>> calculate_mods(x)
    1
    >>> x = np.array([2, 15, 21, 3, 0, 0])
    >>> calculate_mods(x)
    2
    """
    # Get all modification values from constants.ALPHABET where the key contains "()"
    mod_ints = [value for key, value in constants.ALPHABET.items() if "(" in key and ")" in key]

    # Count the occurrences of any modification in sequence_integer
    return len(np.where(np.isin(sequence_integer, mod_ints))[0])


def generate_mods_string_tuples(sequence_integer):
    """
    Dynamically generate modification tuples using constants.ALPHABET.
    Only process keys in constants.ALPHABET that contain "()".
    """
    list_mods = []
    for mod_name, mod_int in constants.ALPHABET.items():
        if "(" in mod_name and ")" in mod_name:  # Process only keys with "()"
            for position in np.where(sequence_integer == mod_int)[0]:
                # Extract the amino acid and modification from the mod_name
                match = re.match(r"([A-Z])\((.+)\)", mod_name)
                if match:
                    amino_acid = match.group(1)  # Extracts the amino acid (e.g., 'C')
                    modification = match.group(2)  # Extracts the modification (e.g., 'DTBIA')
                    list_mods.append((position, amino_acid, modification))
                else:
                    # Handle cases where the mod_name does not follow the "A(mod)" format
                    list_mods.append((position, mod_name, ""))

    # Sort the list of modifications by position
    list_mods.sort(key=lambda tup: tup[0])  # Sort in-place by position
    return list_mods


# def generate_mods_string_tuples(sequence_integer):
#     list_mods = []
#     for mod in [ox_int, c_int, dtbiaC_int, s_int, t_int, y_int]:
#         for position in np.where(sequence_integer == mod)[0]:
#             if mod == c_int:
#                 # NOTE: don't count Carbamidomethyl as mod
#                 pass
#                 # list_mods.append((position, "C", "Carbamidomethyl"))
#             elif mod == ox_int:
#                 list_mods.append((position, "M", "Oxidation"))
#             elif mod == s_int:
#                 list_mods.append((position, "S", "Phospho"))
#             elif mod == t_int:
#                 list_mods.append((position, "T", "Phospho"))
#             elif mod == y_int:
#                 list_mods.append((position, "Y", "Phospho"))
#             elif mod == dtbiaC_int:
#                 list_mods.append((position, "C", "+296.185"))
#             # elif mod == dtbiaC_int:
#             #     list_mods.append((position, "C", "DBTIA"))
#             else:
#                 raise ValueError("cant be true")
#     list_mods.sort(key=lambda tup: tup[0])  # inplacewrite
#     # Example:
#     # list_mods = [(3, "C", "DTBIA"), (8, "M", "Oxidation"), (10, "S", "Phospho")]
#     return list_mods


def generate_mod_strings(sequence_integer):
    """
    The title on the MGF. Generate the modification strings for the given sequence integer array.
    >>> x = np.array([1,2,3,1,2,21,0])
    >>> y, z = generate_mod_strings(x)
    >>> y
    '3/1,C,Carbamidomethyl/4,C,Carbamidomethyl/5,M,Oxidation'
    >>> z
    'Carbamidomethyl@C2; Carbamidomethyl@C5; Oxidation@M6'
    """
    list_mods = generate_mods_string_tuples(sequence_integer)
    if len(list_mods) == 0:
        return "0", ""
    else:
        returnString_mods = ""
        returnString_modString = ""
        returnString_mods += str(len(list_mods))
        for i, mod_tuple in enumerate(list_mods):
            returnString_mods += (
                "/" + str(mod_tuple[0]) + "," + mod_tuple[1] + "," + mod_tuple[2]
            )
            if i == 0:
                returnString_modString += (
                    mod_tuple[2] + "@" + mod_tuple[1] + str(mod_tuple[0] + 1)
                )
            else:
                returnString_modString += (
                    "; " + mod_tuple[2] + "@" + mod_tuple[1] + str(mod_tuple[0] + 1)
                )

    return returnString_mods, returnString_modString


class Converter():
    def __init__(self, data, out_path, flag_fullspectrum):
        self.out_path = out_path
        self.data = data
        self.flag_fullspectrum = flag_fullspectrum
        
    def convert(self):
        if self.flag_fullspectrum is False:
            IONS = get_ions().reshape(174, -1).flatten()
        else:
            IONS = self.data['masses_pred'].reshape(self.data['masses_pred'].shape[1], -1).flatten().astype("|S6")

        with open(self.out_path, mode="w", encoding="utf-8") as f:
            first_spec = True
            # print(self.out_path)
            # print(self.data["intensities_pred"])
            # for i in range(self.data["iRT"].shape[0]):          # iRT prediction is not enabled
            
            # i = 0
            # for i in range(data["intensities_pred"].shape[0]):
            #     aIntensity = data["intensities_pred"][i]
            #     sel = np.where(aIntensity > 0)
            #     aIntensity = aIntensity[sel]
            #     collision_energy = data["collision_energy_aligned_normed"][i] * 100
            #     # iRT = self.data["iRT"][i]
            #     iRT = 0
            #     aMass = data["masses_pred"][i][sel]
            #     precursor_charge = data["precursor_charge_onehot"][i].argmax() + 1
            #     sequence_integer = data["sequence_integer"][i]
            #     aIons = IONS[sel]
            #     spec = Spectrum(
            #         aIntensity,
            #         collision_energy,
            #         iRT,
            #         aMass,
            #         precursor_charge,
            #         sequence_integer,
            #         aIons,
            #     )
                
            
            for i in range(self.data["intensities_pred"].shape[0]):
                aIntensity = self.data["intensities_pred"][i]
                sel = np.where(aIntensity > 0)
                aIntensity = aIntensity[sel]
                collision_energy = self.data["collision_energy_aligned_normed"][i] * 100
                # iRT = self.data["iRT"][i]
                iRT = 0
                aMass = self.data["masses_pred"][i][sel]
                precursor_charge = self.data["precursor_charge_onehot"][i].argmax() + 1
                sequence_integer = self.data["sequence_integer"][i]
                aIons = IONS[sel]
                spec = Spectrum(
                    aIntensity,
                    collision_energy,
                    iRT,
                    aMass,
                    precursor_charge,
                    sequence_integer,
                    aIons,
                )
                if not first_spec:
                    f.write("\n")
                first_spec = False
                f.write(str(spec))
            f.write("\n")
        return spec


class Spectrum(object):
    def __init__(
        self,
        aIntensity,
        collision_energy,
        iRT,
        aMass,
        precursor_charge,
        sequence_integer,
        aIons,
    ):
        self.aIntensity = aIntensity
        self.collision_energy = collision_energy
        self.iRT = iRT
        self.aMass = aMass
        self.precursor_charge = precursor_charge
        self.aIons = aIons
        self.mod, self.mod_string = generate_mod_strings(sequence_integer)

        self.sequence = utils.get_sequence(sequence_integer)

        # print(f"Mass_sequence: {preprocess_sequence(self.sequence)}")
        # print(f"Plot_sequence: {plot_sequence(self.sequence)}")

        # amino acid Z which is defined at the toplevel in generate_aa_comp
        self.precursor_mass = pyteomics.mass.calculate_mass(
            preprocess_sequence(self.sequence),
            aa_comp=aa_comp,
            ion_type="M",
            charge=int(self.precursor_charge),
        )

    def __str__(self):
        s = "Name: {sequence}/{charge}\nMW: {precursor_mass}\n"
        s += "Comment: Parent={precursor_mass} Collision_energy={collision_energy} "
        s += "Mods={mod} ModString={sequence}//{mod_string}/{charge}"
        s += "\nNum peaks: {num_peaks}"
        
        num_peaks = len(self.aIntensity)
        s = s.format(
            sequence=plot_sequence(self.sequence),
            charge=self.precursor_charge,
            precursor_mass=self.precursor_mass,
            collision_energy=np.round(self.collision_energy[0], 0),
            mod=self.mod,
            mod_string=self.mod_string,
            num_peaks=num_peaks,
        )
        for mz, intensity, ion in zip(self.aMass, self.aIntensity, self.aIons):
            s += "\n" + str(mz) + "\t" + str(intensity) + '\t"'
            s += ion.decode("UTF-8").replace("(", "^").replace("+", "") + '/0.0ppm"'
        return s
