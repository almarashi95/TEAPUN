import random
import os
import numpy as np


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
        a = 1.5 * a
        print(f"Octahedral BoxDimensions: {a}")
        psf.setBox(
            a,
            a,
            a,
        )
        # vectors = (
        #     openmm.Vec3(1, 0, 0),
        #     openmm.Vec3(1 / 3, 2 * sqrt(2) / 3, 0),
        #     openmm.Vec3(-1 / 3, sqrt(2) / 3, sqrt(6) / 3),
        # )
        # psf.boxVectors = [(a) * v for v in vectors]
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
        f"tolerance 0.5\nsidemax {2*boxsize}\nstructure {polymerchainpdb}\nnumber {n}\ncenter\nfixed {boxsize/2} {boxsize/2} {boxsize/2} 0. 0. 0. \nradius 2.0\nend structure\n"
    )
    f.write(
        f"structure {solventpdb}\nnumber {n_s}\ninside box 0. 0. 0. {boxsize*1} {boxsize*.9} {boxsize*.8}\nradius 2.0\n"
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
