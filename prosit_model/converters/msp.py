import numpy as np
import pyteomics
from pyteomics import mass

import params.constants as constants
from prosit_model import utils


def generate_aa_comp():
    """
    >>> aa_comp = generate_aa_comp()
    >>> aa_comp["M"]
    Composition({'H': 9, 'C': 5, 'S': 1, 'O': 1, 'N': 1})
    >>> aa_comp["Z"]
    Composition({'H': 9, 'C': 5, 'S': 1, 'O': 2, 'N': 1})
    """
    db = pyteomics.mass.Unimod()
    aa_comp = dict(pyteomics.mass.std_aa_comp)
    # oxidation
    s = db.by_title("Oxidation")["composition"]
    aa_comp["Z"] = aa_comp["M"] + s
    # Carbamidomethyl
    s = db.by_title("Carbamidomethyl")["composition"]
    aa_comp["C"] = aa_comp["C"] + s
    # phospho
    s = db.by_title('Phospho')['composition']
    aa_comp["pS"] = aa_comp["S"] + s
    aa_comp['pT'] = aa_comp["T"] + s
    aa_comp['pY'] = aa_comp["Y"] + s 
    # mass.calculate_mass('EAEEES(ph)DHN', aa_comp=aa_comp)   # () is not allowed in pyteomics
    # mass.calculate_mass('EAEEEpSDHN', aa_comp=aa_comp)
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


ox_int = constants.ALPHABET["M(ox)"]
c_int = constants.ALPHABET["C"]
s_int = constants.ALPHABET["S(ph)"]
t_int = constants.ALPHABET["T(ph)"]
y_int = constants.ALPHABET["Y(ph)"]


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
    return len(np.where((sequence_integer == ox_int) | (sequence_integer == c_int))[0])


def generate_mods_string_tuples(sequence_integer):
    list_mods = []
    for mod in [ox_int, c_int, s_int, t_int, y_int]:
        for position in np.where(sequence_integer == mod)[0]:
            if mod == c_int:
                # NOTE: don't count Carbamidomethyl as mod
                pass
                # list_mods.append((position, "C", "Carbamidomethyl"))
            elif mod == ox_int:
                list_mods.append((position, "M", "Oxidation"))
            elif mod == s_int:
                list_mods.append((position, "S", "Phospho"))
            elif mod == t_int:
                list_mods.append((position, "T", "Phospho"))
            elif mod == y_int:
                list_mods.append((position, "Y", "Phospho"))
            else:
                raise ValueError("cant be true")
    list_mods.sort(key=lambda tup: tup[0])  # inplacewrite
    return list_mods


def generate_mod_strings(sequence_integer):
    """
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
        # amino acid Z which is defined at the toplevel in generate_aa_comp
        self.precursor_mass = pyteomics.mass.calculate_mass(
            self.sequence.replace("M(ox)", "Z").replace("S(ph)", "pS")\
                .replace("T(ph)", "pT").replace("Y(ph)", "pY"),
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
            # sequence=self.sequence.replace("M(ox)", "M")\
            #     .replace("S(ph)", "S").replace("T(ph)", "T").replace("Y(ph)", "Y"),
            sequence=self.sequence,
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
