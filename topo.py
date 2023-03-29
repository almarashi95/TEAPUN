import numpy as np


class Mass:
    def __init__(self, type, mass):
        self.type = type
        self.mass = mass


def mass(raw):
    masses = []
    for i in raw:
        i = i.split("!")[0]
        if i.startswith("MASS"):
            element = i.split()[2:4]
            masses.append(Mass(*element))
    return masses


class Bond_type:
    def __init__(self, type1, type2, k, r_0):
        self.type1 = type1
        self.type2 = type2
        self.k = k
        self.r_0 = float(r_0) * 0.1


class Angle_type:
    def __init__(self, type1, type2, type3, k, r_0):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.k = k
        self.th_0 = r_0


class Dihedral_type:
    def __init__(self, type1, type2, type3, type4, k, multi, r_0):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.type4 = type4
        self.k = k
        self.multiplicty = multi
        self.th_0 = r_0


class Nonb:
    def __init__(self, type, _, epsilon, rm):
        self.type = type
        self.eps = epsilon
        self.rm = float(rm) * 0.1


class Atom:
    def __init__(self, name, type, charge):
        self.name = name
        self.type = type
        self.charge = charge
        self.coordinates = np.zeros((1, 3))


class Bond:
    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2


class Angle:
    def __init__(self, atom1, atom2, atom3):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3


class Dihedral:
    def __init__(self, atom1, atom2, atom3, atom4):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4


class Residue:
    def __init__(self, residue):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        for i in residue:
            if i.startswith("RESI"):
                self.name = i.split()[1]
                self.charge = i.split()[2]
            if i.startswith("ATOM"):
                i = i.split("!")[0]
                self.atoms.append(Atom(*i.split()[1::]))
            if i.startswith("BOND"):
                i = i.split("!")[0]
                self.bonds.append(Bond(*i.split()[1::]))
            if i.startswith("ANGL"):
                i = i.split("!")[0]
                self.angles.append(Angle(*i.split()[1::]))
            if i.startswith("DIHE"):
                i = i.split("!")[0]
                self.dihedrals.append(Dihedral(*i.split()[1::]))
            elif i.startswith("PRES"):
                break
