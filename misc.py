import random
import os
import numpy as np
from openmm import Vec3


def find_patch(m1, m2):
    patches = {
        "AMC,AMC": "AMC1",
        "AMC,STYR": "ASP",
        "STYR,AMC": "AMC2",
        "STYR,STYR": "ST2",
    }
    try:
        return patches[f"{m1},{m2}"]
    except KeyError:
        print(f"no Patch for {m1,m2} found")


def gen_box(psf, crd, enforce_cubic=False, octahedron=False):
    coords = crd.positions
    min_crds = [coords[0][0], coords[0][1], coords[0][2]]
    max_crds = [coords[0][0], coords[0][1], coords[0][2]]
    for coord in coords:
        min_crds[0] = min(min_crds[0], coord[0])
        min_crds[1] = min(min_crds[1], coord[1])
        min_crds[2] = min(min_crds[2], coord[2])
        max_crds[0] = max(max_crds[0], coord[0])
        max_crds[1] = max(max_crds[1], coord[1])
        max_crds[2] = max(max_crds[2], coord[2])
    boxlx = max_crds[0] - min_crds[0]
    boxly = max_crds[1] - min_crds[1]
    boxlz = max_crds[2] - min_crds[2]
    a = 1.02 * np.max([boxlx, boxly, boxlz])
    if enforce_cubic:
        print(f"Forced Cubic Box {a=}")
        psf.setBox(a, a, a)
    elif octahedron:
        print(f"Octahedral BoxDimensions: {a}")
        # psf.setBox(
        #     a,
        #     a,
        #     a,
        # )
        vectors = (
            Vec3(1, 0, 0),
            Vec3(1 / 3, 2 * np.sqrt(2) / 3, 0),
            Vec3(-1 / 3, np.sqrt(2) / 3, np.sqrt(6) / 3),
        )
        psf.boxVectors = [(a) * v for v in vectors]
        # psf.setBox(a, a, a)

    else:
        print("BoxDimensions:")
        print(f"{boxlx, boxly, boxlz}")
        psf.setBox(boxlx, boxly, boxlz)
    return psf


def pack_system(
    polymerchainpdb,
    n,
    solventres,
    solventpdb,
    n_s,
    verbose=False,
    salt=False,
    cat="",
    c_n=0,
    an="",
    a_n=0,
    boxsize=1500,
):
    print("PACKing System\nTEA_PUN powered by PACKMOL\n")
    ###########################################################################################################################################################
    # write PACKMOL input#######################################################################################################################################
    ###########################################################################################################################################################
    f = open("tmp_packmol.inp", "w")
    f.write(
        f"tolerance 2.0\ndiscale 1.5\nmaxit 5\nmovebadrandom\nsidemax {2*boxsize}\nstructure {polymerchainpdb}\nnumber {n}\ncenter\nfixed {boxsize/2} {boxsize/2} {boxsize/2} 0. 0. 0. \nradius 5.2\nend structure\n"
    )
    f.write(
        f"structure {solventpdb}\nnumber {n_s}\ninside box 0. 0. 0. {boxsize*1} {boxsize*.9} {boxsize*.8}\nradius 5.1\n"
    )
    f.write("end structure\n")
    if salt:
        f.write(
            f"structure {cat}\nnumber {c_n}\ninside box 0. 0. 0. {boxsize*1} {boxsize*.9} {boxsize*.8}\nradius 1.0\nend structure\n"
        )
        f.write(
            f"structure {an}\nnumber {a_n}\ninside box 0. 0. 0. {boxsize*1} {boxsize*.9} {boxsize*.8}\nradius 1.0\nend structure\n"
        )
    f.write(
        f"output {polymerchainpdb.split('.')[0].split('_')[0]}_{solventres.lower()}.pdb"
    )
    f.close()
    os.system("packmol <tmp_packmol.inp> packmol.out")
    if not verbose:
        os.remove("tmp_packmol.inp")
        os.remove("packmol.out")


def get_possible_directions(point, dist):
    directions = [
        [point[0], point[1], point[2] + dist],  # up
        [point[0], point[1], point[2] - dist],  # down
        [point[0] + dist, point[1], point[2]],  # left
        [point[0] - dist, point[1], point[2]],  # right
        [point[0], point[1] + dist, point[2]],  # front
        [point[0], point[1] - dist, point[2]],
    ]  # back
    return directions


def randomwalk(steps, dist):
    print("RANDom Walk")
    try:
        visited_points = []
        xyz = [0, 0, 0]
        for i in range(steps):
            visited_points.append(xyz)
            all_directions = get_possible_directions(xyz, dist)
            new_points = [
                direction
                for direction in all_directions
                if direction not in visited_points
            ]
            xyz = random.choice(new_points)
            future_directions = get_possible_directions(xyz, dist)
            future_points = [
                direction
                for direction in future_directions
                if direction not in visited_points
            ]
            if not future_points:
                xyz = random.choice(new_points)
                # print(xyz)
        return visited_points
    except IndexError:
        print("Dead End")
        return randomwalk(steps, dist)


def write_xml(PolymerChain, out_file="output.xml"):
    import xml.etree.ElementTree as ET
    from datetime import date
    import xml.dom.minidom as md

    # system = ET.Element(f"{PolymerChain.id}")
    forcefield = ET.Element("ForceField")
    info = ET.SubElement(forcefield, "Info")
    DateGenerated = ET.SubElement(info, "DateGenerated")
    DateGenerated.text = f"{date.today()}"
    masses = ET.SubElement(forcefield, "AtomTypes")
    t = []
    for k in PolymerChain.masses:
        k_element = ET.SubElement(masses, "Type")
        k_element.set("class", str(k.type))
        k_element.set("element", "C")
        k_element.set("mass", f"{k.mass}")
        k_element.set("name", f"{k.type}")
    residues = ET.SubElement(forcefield, "Residues")
    for resid in PolymerChain._allresidues:
        residue = ET.SubElement(residues, "Residue", name=resid.name)
        for atom in resid.atoms:
            atm = ET.SubElement(
                residue,
                "Atom",
                charge=atom.charge,
                name=atom.name,
                type=atom.type,
            )
        for bond in resid.bonds:
            bnd = ET.SubElement(
                residue,
                "Bond",
                atomName1=bond.atom1,
                atomName2=bond.atom2,
            )
    BondTypes = ET.SubElement(forcefield, "HarmonicBondForce")
    for bonds in PolymerChain.bonds:
        bnd_types = ET.SubElement(
            BondTypes,
            "Bond",
            k=bonds.k,
            length=f"{bonds.r_0}",
            type1=bonds.type1,
            type2=bonds.type2,
        )
    AngleTypes = ET.SubElement(forcefield, "HarmonicAngleForce")
    for angles in PolymerChain.angles:
        angl_types = ET.SubElement(
            AngleTypes,
            "Angle",
            angle=angles.th_0,
            k=angles.k,
            type1=angles.type1,
            type2=angles.type2,
            type3=angles.type3,
        )
    DihedralTypes = ET.SubElement(forcefield, "PeriodicTorsionForce")
    for dihes in PolymerChain.dihedrals:
        dihe_types = ET.SubElement(
            AngleTypes,
            "Proper",
            k=dihes.k,
            periodicity=dihes.multiplicty,
            phase1=dihes.th_0,
            type1=dihes.type1,
            type2=dihes.type2,
            type3=dihes.type3,
            type4=dihes.type4,
        )
    NonBonded = ET.SubElement(
        forcefield,
        "NonbondedForce",
        coulomb14scale="1.0",
        lj14scale="1.0",
        useDispersionCorrection="False",
    )
    use_charge = ET.SubElement(NonBonded, "UseAttributeFromResidue", name="charge")
    for atm in PolymerChain.nonb:
        atm_types = ET.SubElement(
            NonBonded, "Atom", epsilon="0.0", sigma="1.0", type=atm.type
        )
    LJForce = ET.SubElement(
        forcefield, "LennardJonesFore", lj14scale="1.0", useDispersionCorrection="False"
    )
    for at in PolymerChain.nonb:
        at_types = ET.SubElement(
            LJForce, "Atom", epsilon=at.eps, sigma=f"{at.rm}", type=at.type
        )
    tree = ET.ElementTree(forcefield)
    xml_str = ET.tostring(forcefield)
    xml_str_pretty = md.parseString(xml_str).toprettyxml(indent="\t")
    # tree.write(f"test.xml", encoding="UTF-8", xml_declaration=True, pretty_print=True)
    with open(out_file, "w") as f:
        f.write(xml_str_pretty)
