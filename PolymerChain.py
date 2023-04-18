import numpy as np
from openmm.app import *
from openmm import *
import openmm.unit as unit
from sys import stdout
from itertools import product
import os
import topo
import misc
import simulation_details as sd


class PolymerChain:
    def __init__(self, id, monomers, n_of_monomers, toppar, charmm="charmm"):
        self.charmm = charmm
        self.id = id
        self.monomers = monomers
        self.n_of_monomers = n_of_monomers
        self.backbone_atoms = []
        self.patches = []
        self.totmon = len(self.monomers)
        self.chain_length = np.sum(n_of_monomers)
        # generate the monomer geometries
        header = "RESI"
        residues = []
        current_atoms = []
        self.toppar = toppar
        # done =[]
        with open(toppar, "r") as t:
            for line in t:
                # line= line.split('!')[0]
                if line.startswith("!"):
                    continue
                if line.startswith(header) and current_atoms:
                    residues.append(current_atoms[:])
                    current_atoms = []
                if line.startswith("END"):
                    residues.append(current_atoms[:])
                    break
                # print(line)
                current_atoms.append(line)
            for i in residues:
                for j in i:
                    if j == "\n" or j.startswith("!"):
                        i.remove(j)
            self.masses = topo.mass(residues[0])
            residues_in_chain = []
            for i in self.monomers:
                for k in residues:
                    if k[0].split()[1] == i:
                        residues_in_chain.append(k)
            bonds = []
            angles = []
            dihedrals = []
            nonb = []
            section = None
            for line in t:
                if line.startswith("END"):
                    break
                if (
                    line.startswith("!")
                    or line.startswith("\n")
                    or line.startswith(" ")
                ):
                    continue
                elif line.startswith("BOND"):
                    section = "BOND"
                    continue
                elif line.startswith("ANGL"):
                    section = "ANGLE"
                    continue
                elif line.startswith("DIHE"):
                    section = "DIHE"
                    continue
                elif line.startswith("NONB"):
                    section = "NONB"
                    continue
                elif line.startswith("NBFIX"):
                    section = "NBFIX"
                    continue
                line = line.split("!")[0]
                if section == "BOND":
                    bonds.append(line)
                if section == "ANGLE":
                    angles.append(line)
                if section == "DIHE":
                    dihedrals.append(line)
                if section == "NONB":
                    nonb.append(line)
        self.residues = []
        self.angles = []
        self.bonds = []
        self.dihedrals = []
        self.nonb = []
        for i in angles:
            # print(i.split())
            self.angles.append(topo.Angle_type(*i.split()))
        for i in bonds:
            self.bonds.append(topo.Bond_type(*i.split()))
        for i in dihedrals:
            # print(i)
            self.dihedrals.append(topo.Dihedral_type(*i.split()))
        for i in nonb[1:]:
            # print(i)
            self.nonb.append(topo.Nonb(*i.split()))
        for k in residues_in_chain:
            # try:
            self.residues.append(topo.Residue(k))
        # except:
        #     print(k)
        #     quit()
        ##all residues in toppar submitted
        self._allresidues = []
        for l in residues[1:]:
            self._allresidues.append(topo.Residue(l))

    def build_Monomers(self, verbose=False):
        # Building Monomer psf/crd using internal coordinates
        for k in self.residues:
            if len(k.atoms) >= 3:
                f = open("tmp.inp", "w")
                f.write(f"ioformat extended \nstream {self.toppar}\n\n")
                f.write(f"read sequence {k.name} 1 \n")
                f.write(f"generate {k.name} first none last none\n")
                f.write("\nic gene\nic param\n")
                f.write(
                    f"ic seed {k.name} 1 {k.atoms[2].name}  {k.name} 1 {k.atoms[1].name}  {k.name} 1 {k.atoms[0].name}\n"
                )
                f.write("ic build\ncoor shake\nenergy\nmini sd nstep 600\n")
                f.write(f"write coor card name {k.name}.crd\n")
                f.write(f"write psf card xplor name {k.name}.psf\n")
                f.write("stop")
                f.close()
                os.system(f"{self.charmm}  -i tmp.inp >tmp.out")
                if not verbose:
                    os.system("rm tmp.inp")
                    os.system("rm tmp.out")
            else:
                print(f"{k.name} not enough atoms for ic")
                continue

    def build_simpleChain(
        self, dyna=False, dynastep=500, verbose=False, random=False, iter=3
    ):
        print(
            f"BUILDing Chain\nTEA_PUN powered by CHARMM\nmake sure you have the correct patches and residues in the toppar before running this step!\nIC blocks may be needed for complex monomers"
        )
        names = []
        segid = []
        segid_num = []
        seg_atoms = []
        atoms = []
        ids = []
        f = open("tmp.inp", "w")
        f.write(
            f"dimension chsize 450000\nioformat extended \nstream {self.toppar}\n\n"
        )
        ##################################################################################################################################################
        ##combining the blocks for reading them into charmm###############################################################################################
        ##################################################################################################################################################
        for k in range(len(self.residues)):
            if self.residues[k].name not in segid:
                segid.append(self.residues[k].name)
                seg_atoms.append(self.residues[k].atoms)
                segid_num.append(self.n_of_monomers[k])
            else:
                segid_num[
                    segid.index(self.residues[k].name)
                ] += self.n_of_monomers[k]
        for segment in range(len(segid)):
            f.write(f"read sequence {segid[segment]} {segid_num[segment]}\n")
            f.write(f"\ngenerate {segid[segment]} first none last none\n")
        #################################################################################################################################################
        ##here the atoms are assigned to correct residues for IC seed####################################################################################
        #################################################################################################################################################
        for res in range(len(self.residues)):
            for i in range(self.n_of_monomers[res]):
                names.append(self.residues[res].name)
                atoms.append(self.residues[res].atoms)
        f.write("AUTOGENerate OFF\n")
        f.write("ic gene\n")
        ctr = 1
        #################################################################################################################################################
        # parameters for displaceing monomers#############################################################################################################
        #################################################################################################################################################
        z_vector = 0.4
        r = (z_vector * self.chain_length) / 2
        f.write("bomblev -1\n")

        for m in range(len(names)):
            c = names[:m].count(names[m])
            ids.append(c + 1)
        ####################################################
        ##Random Walk
        ####################################################
        coords = misc.randomwalk(self.chain_length, 4)

        if random:
            import random

            c = list(zip(names, atoms, ids))
            random.shuffle(c)
            names, atoms, ids = zip(*c)
        ######################################################
        # writing the charmm input
        ######################################################
        for num in range(len(names)):
            n = names[num]
            try:
                f.write(
                    f"ic param\nic seed {n} {ids[num]} {atoms[num][1].name} {n} {ids[num]} {atoms[num][0].name} {n} {ids[num]} {atoms[num][3].name}\n"
                )
            except:
                f.write(
                    f"coor set sele (segid {n} .and. resid {ids[num]}) end xdir 0 ydir 0 zdir 0\n"
                )
            f.write(
                f"coor trans sele (segid {n} .and. resid {ids[num]}) end xdir {coords[num][0]} ydir {coords[num][1]} zdir {coords[num][2]}\n"
            )  # add function to find real bondlenght
            if ctr > 1 and ctr <= self.chain_length:
                f.write(
                    f"patch {misc.find_patch(prev_seg,n)} {prev_seg} {prev} {n} {ids[num]}\n"
                )
            prev = ids[num]
            ctr += 1
            prev_seg = n

        f.write("ic build\n")
        f.write(f"rename segid {self.id} sele all end\ncoor shake\n")
        f.write("autogen angl dihe \n")
        f.write("bomblev -5\n")
        f.write("energy\n")
        for i in range(iter):
            f.write(f"mini sd nstep 10000\n")
            f.write(f"write coor card name {self.id.lower()}.crd\n")

        f.write("mini abnr nstep 300\n")
        if dyna:
            f.write(f"DYNA nstep {dynastep}\n")
        f.write(f"write psf card xplor name {self.id.lower()}.psf\n")
        f.write(f"write coor card name {self.id.lower()}.crd\n")
        f.write("stop")
        f.close()
        print("invoking charmm-process")
        os.system(f"{self.charmm}  -i tmp.inp >build.out")
        if verbose == False:
            os.system("rm tmp.inp")
            os.system("rm build.out")

    def relax_chain(self, nstep=150_000, useBMH=False):
        #####################################################################################################################################################
        # Compressing the chain in a Force Cage and relaxing afterwards to get usable starting structures for MD
        #####################################################################################################################################################
        # setting up openMM
        print(
            f"RELAxing Chain\nTEA_PUN powered by openMM\nmake sure you have a psf and crd before running this step!(example:chainID.psf/chainID.crd)"
        )
        psf = CharmmPsfFile(f"{self.id.lower()}.psf")
        crd = CharmmCrdFile(f"{self.id.lower()}.crd")
        toppar = CharmmParameterSet(f"{self.toppar}")
        DEFAULT_PLATFORMS = "CUDA", "OpenCL", "CPU"
        enabled_platforms = [
            Platform.getPlatform(i).getName()
            for i in range(Platform.getNumPlatforms())
        ]
        for platform in DEFAULT_PLATFORMS:
            if platform in enabled_platforms:
                platform = Platform.getPlatformByName(platform)
                break
        print("Using platform:", platform.getName())
        prop = (
            dict(CudaPrecision="single")
            if platform.getName() == "CUDA"
            else dict()
        )
        print("BUILDing SYSTem")
        #######################################################################################
        psf = misc.gen_box(psf, crd, enforce_cubic=True)
        ########################################################################################################################################################
        system = psf.createSystem(
            toppar,
            nonbondedMethod=PME,
            ewaldErrorTolerance=0.005,
            nonbondedCutoff=1.5 * unit.nanometer,
            solventDielectric=60,
            constraints=None,
        )

        # #########################################################################################################################################################
        # Initialize Simulation
        ################################################################################################
        integrator = NoseHooverIntegrator(
            300 * unit.kelvin, 50 / unit.picosecond, 0.0001 * unit.picoseconds
        )
        # system, epsilons, sigm = eliminate_LJ(psf)
        # system = usemodBMH(self, psf, epsilons, sigm, NBfix=False)
        simulation = Simulation(
            psf.topology, system, integrator, platform, prop
        )
        simulation.context.setPositions(crd.positions)
        simulation.reporters.append(
            StateDataReporter(
                stdout, 1_000, step=True, totalEnergy=True, separator="\t"
            )
        )
        if useBMH:
            system, epsilons, sigm = sd.eliminate_LJ(psf)
            # print(np.sum(epsilons))
            # system = eliminate_elec(psf)
            system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)
        # system.addForce(MonteCarloBarostat(1 * unit.atmospheres, 300 * unit.kelvin, 10))
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
        ###############################################################################################
        # Compress System
        ###############################################################################################
        print("MINImizing ENERgy")
        simulation.minimizeEnergy(maxIterations=1_500)
        print("\nINITial SYSTem ENERgy")
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        simulation.reporters.append(
            DCDReporter(f"{self.id.lower()}_relax.dcd", int(nstep / 100))
        )
        simulation.reporters.append(
            PDBReporter(f"{self.id.lower()}_relax.pdb", nstep)
        )
        print("STARting DYNAmics")

        simulation.step(nstep)
        state = simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        with open(f"{self.id.lower()}_relax.rst", "w") as f:
            f.write(XmlSerializer.serialize(state))

    def solvate_charmm(
        self,
        num_of_Polymers,
        solvent_res,
        solvent_num,
        pack=False,
        build=True,
        verbose=False,
        pdb_file="default",
        solvent_pdb="default",
        crd="default",
        boxsize=1500,
        salt=False,
        c="",
        c_pdb="",
        c_n=0,
        a="",
        a_pdb="",
        a_n=0,
        square=False,
        read=False,
        n_proc=16,
    ):
        ######################################################################################################
        ##Set pdbfile for packing to default if not defined
        ######################################################################################################
        if pdb_file == "default":
            pdb_file = f"{self.id.lower()}_relax.pdb"
        if crd == "default":
            crd = f"{self.id.lower()}.crd"
        ######################################################################################################
        # checking toppar for Solvent residue
        ######################################################################################################
        found = False
        for i in self._allresidues:
            try:
                if solvent_res == i.name:
                    print("Solvent found in toppar")
                    atoms = i.atoms
                    found = True
                    break
            except AttributeError:
                pass
        if salt:
            for i in self._allresidues:
                try:
                    if c == i.name:
                        print("cation found in toppar")
                        found = True
                        break
                except AttributeError:
                    pass
            for i in self._allresidues:
                try:
                    if a == i.name:
                        print("anion found in toppar")
                        found = True
                        break
                except AttributeError:
                    pass
        spacing = 5
        if square:
            x = np.linspace(0, int(boxsize), int(boxsize // spacing))
            y = np.linspace(0, int(boxsize), int(boxsize // spacing))
            z = np.linspace(0, int(boxsize), int(boxsize // spacing))
        else:
            x = np.linspace(0, int(boxsize), int(boxsize // spacing))
            y = np.linspace(
                0, int(boxsize * 0.9), int(boxsize * 0.9 // spacing)
            )
            z = np.linspace(
                0, int(boxsize * 0.8), int(boxsize * 0.8 // spacing)
            )
        coords = product(x, y, z)
        if build:
            solvent = misc.solvation_coordinates(x, y, z, solvent_res, atoms)
        elif read:
            solvent = "slvnt.crd"
        if solvent_num == "auto":
            solvent_num = len(list(coords))
            coords = product(x, y, z)
        if not found:
            print(f"ERRor something not in toppar!")
            return
        #######################################################################################################
        #######################################################################################################
        print("GENErating PSF\nTEA_PUN powered by CHARMM\n")
        #######################################################################################################
        f = open("tmp.inp", "w")
        f.write(
            f"dimension chsize 50000000\nioformat extended\nstream {self.toppar}\nset parall {n_proc}\n"
        )
        f.write(f"open unit 1 card name {self.id.lower()}.psf\n")
        f.write(f"read psf card unit 1 \n")
        f.write("close unit 1\n")
        if build or read:
            f.write(f"open unit 2 card name {crd}\n")  #
            f.write(f"read coor card unit 2 \n")
            f.write("close unit 2\n")
            f.write(
                f"coor trans sele segid {self.id} end xdir {boxsize/2} ydir {boxsize/2} zdir {boxsize/2}\n"
            )
        if num_of_Polymers != 1:
            f.write(
                f"coor trans sele segid {self.id} end xdir -50 ydir 0 zdir 0\n"
            )
            f.write(f"rename segid {self.id}_1 sele all end\n")
            f.write(f"")
            for i in range(num_of_Polymers - 1):
                f.write(f"generate {self.id}_{i+2} duplicate {self.id}_1\n")
                f.write(
                    f"coor dupl sele segid {self.id}_{i+1} end sele segid  {self.id}_{i+2} end\n"
                )
                f.write(
                    f"coor trans sele segid {self.id}_{i+2} end xdir 50 ydir 0 zdir 0\n"
                )
        f.write(
            f"read sequence {solvent_res} {solvent_num}\ngenerate {solvent_res} first none last none noangle nodihedral setup warn\n"
        )
        if build or read:
            f.write(f"open unit 15 card name {solvent}\n")  #
            f.write(f"read coor append card unit 15 \n")
            f.write(
                f"bomblev -1\ndelete atom sele .byres. (segid {solvent_res} .and. segid {self.id} .around. 3.5) end\n"
            )
            if num_of_Polymers != 1:
                for i in range(num_of_Polymers):
                    f.write(
                        f"bomblev -1\ndelete atom sele .byres. (segid {solvent_res} .and. segid {self.id}_{i+1} .around. 3.5) end\n"
                    )
        if salt:
            f.write(
                f"read sequence {c} {c_n}\ngenerate {c} first none last none setup warn\n"
            )
            f.write(
                f"read sequence {a} {a_n}\ngenerate {a} first none last none setup warn\n"
            )
        # f.write("\nenergy\nmini sd nstep 500\n")
        f.write(
            f"open unit 10 write form name ./{self.id.lower()}_in_{solvent_res.lower()}.psf\nwrite unit 10 psf xplor card\nclose unit 10\n"
        )
        f.write(
            f"write coor card name {self.id.lower()}_in_{solvent_res.lower()}.crd\n\nstop"
        )
        f.close()
        print("invoking charmm-process")
        os.system(f"{self.charmm}  -i tmp.inp >solvate.out")
        print(f"check {self.id.lower()}_in_{solvent_res.lower()}.psf")
        #######################################################################################################
        #######################################################################################################
        if verbose == False:
            os.system("rm tmp.inp")
            os.system("rm solvate.out")
            os.system("rm slvnt.crd")
        #######################################################################################################
        # refering to packmol!
        #######################################################################################################
        if pack:
            if solvent_pdb == "default":
                solvent_pdb = f"{solvent_res.lower()}.pdb"

            misc.pack_system(
                pdb_file,
                num_of_Polymers,
                solvent_res,
                solvent_pdb,
                solvent_num,
                verbose=verbose,
                boxsize=boxsize,
                salt=salt,
                cat=c_pdb,
                c_n=c_n,
                an=a_pdb,
                a_n=a_n,
            )

    def solvate(
        self,
        num_of_Polymers,
        solvent_res,
        solvent_num,
        pack=False,
        build=True,
        verbose=False,
        pdb_file="default",
        solvent_pdb="default",
        crd="default",
        boxsize=1500,
        salt=False,
        c="",
        c_pdb="",
        c_n=0,
        a="",
        a_pdb="",
        a_n=0,
        square=False,
        read=False,
        n_proc=16,
    ):
        ######################################################################################################
        ##Set pdbfile for packing to default if not defined
        ######################################################################################################
        if pdb_file == "default":
            pdb_file = f"{self.id.lower()}_relax.pdb"
        if crd == "default":
            crd = f"{self.id.lower()}.crd"
        ######################################################################################################
        # checking toppar for Solvent residue
        ######################################################################################################
        found = False
        for i in self._allresidues:
            try:
                if solvent_res == i.name:
                    print("Solvent found in toppar")
                    atoms = i.atoms
                    found = True
                    break
            except AttributeError:
                pass
        if salt:
            for i in self._allresidues:
                try:
                    if c == i.name:
                        print("cation found in toppar")
                        found = True
                        break
                except AttributeError:
                    pass
            for i in self._allresidues:
                try:
                    if a == i.name:
                        print("anion found in toppar")
                        found = True
                        break
                except AttributeError:
                    pass
        if not found:
            print(f"ERRor something not in toppar!")
            return
        spacing = 6
        if square:
            x = np.linspace(0, int(boxsize), int(boxsize // spacing))
            y = np.linspace(0, int(boxsize), int(boxsize // spacing))
            z = np.linspace(0, int(boxsize), int(boxsize // spacing))
        else:
            x = np.linspace(0, int(boxsize), int(boxsize // spacing))
            y = np.linspace(
                0, int(boxsize * 0.9), int(boxsize * 0.9 // spacing)
            )
            z = np.linspace(
                0, int(boxsize * 0.8), int(boxsize * 0.8 // spacing)
            )
        coords = product(x, y, z)
        if build:
            solvent = misc.solvation_coordinates(x, y, z, solvent_res, atoms)
        elif read:
            solvent = "slvnt.crd"
        if solvent_num == "auto":
            solvent_num = len(list(coords))
            coords = product(x, y, z)

        #######################################################################################################
        #######################################################################################################
        print("GENErating PSF\nTEA_PUN powered by CHARMM\n")
        #######################################################################################################
        f = open("tmp.inp", "w")
        f.write(
            f"dimension chsize 50000000\nioformat extended\nstream {self.toppar}\nset parall {n_proc}\n"
        )
        f.write(f"open unit 1 card name {self.id.lower()}.psf\n")
        f.write(f"read psf card unit 1 \n")
        f.write("close unit 1\n")
        f.write(f"open unit 2 card name {crd}\n")  #
        f.write(f"read coor card unit 2 \n")
        f.write("close unit 2\n")
        f.write(
            f"coor trans sele segid {self.id} end xdir {boxsize/2} ydir {boxsize/2} zdir {boxsize/2}\n"
        )
        if num_of_Polymers != 1:
            f.write(
                f"coor trans sele segid {self.id} end xdir -50 ydir 0 zdir 0\n"
            )
            f.write(f"rename segid {self.id}_1 sele all end\n")
            f.write(f"\n")
            for i in range(num_of_Polymers - 1):
                f.write(f"generate {self.id}_{i+2} duplicate {self.id}_1\n")
                f.write(
                    f"coor dupl sele segid {self.id}_{i+1} end sele segid  {self.id}_{i+2} end\n"
                )
                f.write(
                    f"coor trans sele segid {self.id}_{i+2} end xdir 70 ydir 0 zdir 0\n"
                )
        f.write(f"write coor card name {self.id.lower()}_n.crd\n")
        f.write(
            f"read sequence {solvent_res} 1\ngenerate {solvent_res} first none last none noangle nodihedral setup warn\n"
        )

        f.write(
            f"open unit 10 write form name ./{self.id.lower()}_in_{solvent_res.lower()}.psf\nwrite unit 10 psf xplor card\nclose unit 10\n"
        )
        f.write(
            f"write coor card name {self.id.lower()}_in_{solvent_res.lower()}.crd\n\nstop"
        )
        f.close()
        print("invoking charmm-process")
        os.system(f"{self.charmm}  -i tmp.inp >solvate.out")
        print("Switching to Parmed")
        misc.make_psf(
            f"{self.id.lower()}_in_{solvent_res.lower()}.psf", solvent_num
        )
        print("invoking charmm-process")
        f = open("tmp.inp", "w")
        f.write(
            f"dimension chsize 50000000\nioformat extended\nstream {self.toppar}\nset parall {n_proc}\n"
        )
        f.write(
            f"open unit 1 card name {self.id.lower()}_in_{solvent_res.lower()}.psf\n"
        )
        f.write(f"read psf card unit 1 \n")
        f.write("close unit 1\n")

        f.write(f"open unit 2 card name {self.id.lower()}_n.crd\n")  #
        f.write(f"read coor card unit 2 \n")
        f.write("close unit 2\n")

        f.write(f"open unit 15 card name {solvent}\n")  #
        f.write(f"read coor append card unit 15 \n")
        f.write(
            f"bomblev -1\ndelete atom sele .byres. (segid {solvent_res} .and. segid {self.id} .around. 3.5) end\n"
        )
        if num_of_Polymers != 1:
            for i in range(num_of_Polymers):
                f.write(
                    f"bomblev -1\ndelete atom sele .byres. (segid {solvent_res} .and. segid {self.id}_{i+1} .around. 3.5) end\n"
                )
        f.write(
            f"open unit 10 write form name ./{self.id.lower()}_in_{solvent_res.lower()}.psf\nwrite unit 10 psf xplor card\nclose unit 10\n"
        )
        f.write(
            f"write coor card name {self.id.lower()}_in_{solvent_res.lower()}.crd\n\nstop"
        )
        f.close()
        os.system(f"{self.charmm}  -i tmp.inp >solvate_2.out")
        print(f"check {self.id.lower()}_in_{solvent_res.lower()}.psf")

        #######################################################################################################
        #######################################################################################################
        if verbose == False:
            os.system("rm tmp.inp")
            os.system("rm solvate.out")
            os.system("rm solvate_2.out")
            os.system("rm slvnt.crd")

    def equilibrate(
        self,
        psf,
        pdb,
        nstep=2_000_000,
        iter=50,
        dt=0.0001,
        p=10,
        T=300,
        useBMH=False,
        freeze=False,
        multiplicator=1.1,
        octahedron=False,
    ):
        print(f"EQUIlibrate Chain\nTEA_PUN powered by openMM")
        psf = CharmmPsfFile(psf)
        try:
            pdb = PDBFile(pdb)
        except:
            pdb = CharmmCrdFile(pdb)
        params = CharmmParameterSet(self.toppar)
        DEFAULT_PLATFORMS = "CUDA", "OpenCL", "CPU"
        enabled_platforms = [
            Platform.getPlatform(i).getName()
            for i in range(Platform.getNumPlatforms())
        ]
        for platform in DEFAULT_PLATFORMS:
            if platform in enabled_platforms:
                platform = Platform.getPlatformByName(platform)
                break
        print(f"Using {platform.getName()}")
        prop = (
            dict(CudaPrecision="single")
            if platform.getName() == "CUDA"
            else dict()
        )
        psf = misc.gen_box(psf, pdb)
        system = psf.createSystem(
            params,
            nonbondedMethod=CutoffPeriodic,
            nonbondedCutoff=1.5 * unit.nanometer,
            constraints=None,
        )
        integrator = NoseHooverIntegrator(
            T * unit.kelvin, 50 / unit.picosecond, 0.00005 * unit.picoseconds
        )
        if useBMH:
            system, epsilons, sigm = sd.eliminate_LJ(psf)
            # print(np.sum(epsilons))
            system = sd.usemodBMH(
                PolymerChain=self,
                psf=psf,
                epsilons=epsilons,
                sigm=sigm,
                NBfix=True,
            )
        if freeze:
            system = sd.freeze_polymer(psf, self)
        simulation = Simulation(
            psf.topology, system, integrator, platform, prop
        )
        simulation.context.setPositions(pdb.positions)
        if octahedron:
            try:
                boxVectors = sd.set_octahedron(psf, multiplicator)
                print(f"setting {boxVectors}")
                simulation.context.setPeriodicBoxVectors(*boxVectors)
            except:
                print("failed to Transform into octahedron")
        print("MINImizing ENERgy")
        # simulation.minimizeEnergy(maxIterations=2_000_000)
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        simulation.context.setVelocitiesToTemperature(5 * unit.kelvin)
        simulation.reporters.append(
            StateDataReporter(
                stdout,
                2000,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator="\t",
            )
        )
        simulation.reporters.append(DCDReporter(f"pre_comp.dcd", 10_000))
        integrator.setTemperature(5 * unit.kelvin)
        integrator.setStepSize(0.000001 * unit.picoseconds)
        simulation.context.reinitialize(preserveState=True)
        simulation.step(100_000)
        integrator.setStepSize(0.000002 * unit.picoseconds)
        simulation.context.reinitialize(preserveState=True)
        simulation.step(100_000)
        integrator.setStepSize(0.000003 * unit.picoseconds)
        simulation.context.reinitialize(preserveState=True)
        simulation.step(100_000)
        integrator.setStepSize(0.00004 * unit.picoseconds)
        simulation.context.reinitialize(preserveState=True)
        simulation.step(100_000)
        integrator.setStepSize(0.00005 * unit.picoseconds)
        simulation.context.reinitialize(preserveState=True)
        simulation.step(300_000)
        # simulation.minimizeEnergy(maxIterations=2_000)
        state = simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        with open("init.rst", "w") as f:
            f.write(XmlSerializer.serialize(state))
        # simulation.minimizeEnergy(maxIterations=200_000_000)
        integrator.setTemperature(T * unit.kelvin)
        integrator.setStepSize(dt * unit.picoseconds)
        simulation.context.reinitialize(preserveState=True)
        # simulation.minimizeEnergy(maxIterations=200_000_000)
        simulation.step(200_000)
        print("\nINITial SYSTem ENERgy")
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        pdbfile.PDBFile.writeModel(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            file=open("init.pdb", "w"),
        )
        ctr = 0
        boxv = simulation.context.getState().getPeriodicBoxVectors()
        # psf.boxVectors = boxv
        for i in range(iter):
            print(f"TimeStep set to {dt}ps")
            state = simulation.context.getState(
                getPositions=True, getVelocities=True
            )
            rst = f"{self.id.lower()}_temp.rst"
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            system = psf.createSystem(
                params,
                nonbondedMethod=CutoffPeriodic,
                nonbondedCutoff=1.3 * unit.nanometer,
                rigidWater=False,
                solventDielectric=60,
            )
            integrator = NoseHooverIntegrator(
                T * unit.kelvin, 50 / unit.picosecond, dt * unit.picoseconds
            )
            # barostat = system.addForce(
            # MonteCarloBarostat(p * unit.atmospheres, 50 * unit.kelvin, 10)
            # )
            if useBMH:
                system, epsilons, sigm = sd.eliminate_LJ(psf)

                system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)
            simulation = Simulation(
                psf.topology, system, integrator, platform, prop
            )
            with open(rst, "r") as f:
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.reporters.append(
                StateDataReporter(
                    stdout,
                    10_000,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                    speed=True,
                    separator="\t",
                )
            )
            if ctr == 0:
                try:
                    simulation.context.setPeriodicBoxVectors(*boxv)
                    print(
                        f"setting {simulation.context.getState().getPeriodicBoxVectors()}"
                    )
                    simulation.minimizeEnergy(maxIterations=2_000_000)
                except:
                    print("failed\nusing init box")
            if ctr > 0:
                print(f"Using Dimensions: {boxv}")
                simulation.context.setPeriodicBoxVectors(*boxv)

            simulation.reporters.append(
                DCDReporter(
                    f"{self.id.lower()}_equilibration_{dt}_{p}_{ctr}.dcd",
                    20_000,
                )
            )
            # integrator.setTemperature(50 * unit.kelvin)
            # simulation.context.reinitialize(preserveState=True)

            # simulation.step(1000)
            # system.removeForce(barostat)
            # integrator.setTemperature(T * unit.kelvin)
            # simulation.context.setVelocitiesToTemperature(T * unit.kelvin)
            # simulation.context.reinitialize(preserveState=True)
            simulation.step(500_000)
            boxv = simulation.context.getState().getPeriodicBoxVectors()
            state = simulation.context.getState(
                getPositions=True, getVelocities=True
            )
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            ctr += 1
        system = psf.createSystem(
            params,
            nonbondedMethod=CutoffPeriodic,
            nonbondedCutoff=1.5 * unit.nanometer,
            constraints=None,
        )
        integrator = NoseHooverIntegrator(
            300 * unit.kelvin, 50 / unit.picosecond, 0.004 * unit.picoseconds
        )
        if useBMH:
            system, epsilons, sigm = sd.eliminate_LJ(psf)
            print(np.sum(epsilons))
            # system = eliminate_elec(psf)
            system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)
        simulation = Simulation(
            psf.topology, system, integrator, platform, prop
        )
        with open(rst, "r") as f:
            simulation.context.setState(XmlSerializer.deserialize(f.read()))
        simulation.step(nstep)
        self.boxv = simulation.context.getState().getPeriodicBoxVectors()
        state = simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        with open(f"{self.id.lower()}.rst", "w") as f:
            f.write(XmlSerializer.serialize(state))

    def restart(
        self,
        psf,
        rst,
        nstep=2_000_000,
        dt=0.0005,
        p=10,
        T=300,
        iter=50,
        useBMH=False,
        freeze=False,
        uPME=False,
    ):
        if uPME:
            nb_meth = PME
        else:
            nb_meth = CutoffPeriodic
        print(f"EQUIlibrate Chain\nTEA_PUN powered by openMM")
        psf = CharmmPsfFile(psf)
        pdb = PDBFile("init.pdb")
        params = CharmmParameterSet(self.toppar)
        DEFAULT_PLATFORMS = "CUDA", "OpenCL", "CPU"
        enabled_platforms = [
            Platform.getPlatform(i).getName()
            for i in range(Platform.getNumPlatforms())
        ]
        for platform in DEFAULT_PLATFORMS:
            if platform in enabled_platforms:
                platform = Platform.getPlatformByName(platform)
                break
        print(f"Using {platform.getName()}")
        prop = (
            dict(CudaPrecision="single")
            if platform.getName() == "CUDA"
            else dict()
        )
        psf = misc.gen_box(
            psf,
            pdb,
        )
        system = psf.createSystem(
            params,
            nonbondedMethod=nb_meth,
            nonbondedCutoff=1.5 * unit.nanometer,
            constraints=None,
        )
        integrator = NoseHooverIntegrator(
            T * unit.kelvin, 50 / unit.picosecond, 0.0001 * unit.picoseconds
        )

        if useBMH:
            system, epsilons, sigm = sd.eliminate_LJ(psf)

            system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)
        if freeze:
            system = sd.freeze_polymer(psf, self)

        simulation = Simulation(
            psf.topology, system, integrator, platform, prop
        )

        with open(rst, "r") as f:
            simulation.context.setState(XmlSerializer.deserialize(f.read()))

        print("MINImizing ENERgy")
        simulation.minimizeEnergy(maxIterations=200_000_000)
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        simulation.context.reinitialize(preserveState=True)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
        simulation.reporters.append(
            StateDataReporter(
                stdout,
                2000,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator="\t",
            )
        )
        simulation.reporters.append(DCDReporter(f"pre_comp.dcd", 10_000))
        simulation.step(10_000)
        simulation.minimizeEnergy(maxIterations=200_000_000)
        simulation.step(nstep)
        # simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
        print("\nINITial SYSTem ENERgy")
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        pdbfile.PDBFile.writeModel(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            file=open("init.pdb", "w"),
        )
        ctr = 0
        boxv = simulation.context.getState().getPeriodicBoxVectors()
        # psf.boxVectors = boxv
        for i in range(iter):
            # print(f"TimeStep set to {dt}ps")
            state = simulation.context.getState(
                getPositions=True, getVelocities=True
            )
            rst = f"{self.id.lower()}_temp.rst"
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            system = psf.createSystem(
                params,
                nonbondedMethod=nb_meth,
                nonbondedCutoff=1.5 * unit.nanometer,
                rigidWater=False,
                solventDielectric=60,
            )
            integrator = NoseHooverIntegrator(
                T * unit.kelvin, 50 / unit.picosecond, dt * unit.picoseconds
            )
            # barostat = system.addForce(
            #     MonteCarloBarostat(p * unit.atmospheres, 100 * unit.kelvin, 10)
            # )
            if useBMH:
                system, epsilons, sigm = sd.eliminate_LJ(psf)

                system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)

            simulation = Simulation(
                psf.topology, system, integrator, platform, prop
            )
            with open(rst, "r") as f:
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.reporters.append(
                StateDataReporter(
                    stdout,
                    10_000,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                    speed=True,
                    separator="\t",
                )
            )
            simulation.reporters.append(
                DCDReporter(
                    f"{self.id.lower()}_equilibration_{dt}_{p}_{ctr}_rst.dcd",
                    20_000,
                )
            )
            # print("STARting DYNAmics")
            # integrator.setTemperature(50 * unit.kelvin)
            # integrator.setStepSize(0.00005 * unit.picoseconds)
            # simulation.context.reinitialize(preserveState=True)
            # simulation.step(1000)
            # system.removeForce(barostat)
            # integrator.setTemperature(T * unit.kelvin)
            # integrator.setStepSize(dt * unit.picoseconds)
            # simulation.context.setVelocitiesToTemperature(T * unit.kelvin)
            # simulation.context.reinitialize(preserveState=True)
            simulation.step(500_000)
            boxv = simulation.context.getState().getPeriodicBoxVectors()
            state = simulation.context.getState(
                getPositions=True, getVelocities=True
            )
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            ctr += 1
        system = psf.createSystem(
            params,
            nonbondedMethod=nb_meth,
            nonbondedCutoff=1.5 * unit.nanometer,
            constraints=None,
        )
        integrator = NoseHooverIntegrator(
            300 * unit.kelvin, 50 / unit.picosecond, 0.004 * unit.picoseconds
        )
        print(useBMH)
        if useBMH:
            system, epsilons, sigm = sd.eliminate_LJ(psf)
            system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)
        simulation = Simulation(
            psf.topology, system, integrator, platform, prop
        )
        simulation.reporters.append(
            StateDataReporter(
                stdout,
                10_000,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator="\t",
            )
        )
        simulation.reporters.append(
            DCDReporter(
                f"{self.id.lower()}_equilibration_{dt}_{p}_{ctr}_rst.dcd",
                20_000,
            )
        )
        simulation.context.setState(state)
        simulation.step(nstep)
        self.boxv = simulation.context.getState().getPeriodicBoxVectors()
        state = simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        with open(f"{self.id.lower()}.rst", "w") as f:
            f.write(XmlSerializer.serialize(state))

    def rest(
        self,
        psf,
        rst,
        nstep=2_000_000,
        dt=0.0005,
        p=10,
        T=200,
        iter=50,
        useBMH=False,
        freeze=False,
        co=1.3,
        uPME=False,
    ):
        if uPME:
            nb_meth = PME
        else:
            nb_meth = CutoffPeriodic
        print(f"EQUIlibrate Chain\nTEA_PUN powered by openMM")
        psf = CharmmPsfFile(psf)
        pdb = PDBFile("init.pdb")
        params = CharmmParameterSet(self.toppar)
        DEFAULT_PLATFORMS = "CUDA", "OpenCL", "CPU"
        enabled_platforms = [
            Platform.getPlatform(i).getName()
            for i in range(Platform.getNumPlatforms())
        ]
        for platform in DEFAULT_PLATFORMS:
            if platform in enabled_platforms:
                platform = Platform.getPlatformByName(platform)
                break
        print(f"Using {platform.getName()}")
        prop = (
            dict(CudaPrecision="single")
            if platform.getName() == "CUDA"
            else dict()
        )
        psf = misc.gen_box(psf, pdb, enforce_cubic=True)
        system = psf.createSystem(
            params,
            nonbondedMethod=nb_meth,
            nonbondedCutoff=co * unit.nanometer,
            constraints=None,
        )
        integrator = NoseHooverIntegrator(
            T * unit.kelvin, 50 / unit.picosecond, 0.0001 * unit.picoseconds
        )
        # barostat = system.addForce(
        #     MonteCarloBarostat(1 * unit.atmospheres, 300 * unit.kelvin, 10)
        # )
        if useBMH:
            system, epsilons, sigm = sd.eliminate_LJ(psf)
            # system = eliminate_elec(psf)
            system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)
        if freeze:
            system = sd.freeze_polymer(psf, self)

        simulation = Simulation(
            psf.topology, system, integrator, platform, prop
        )

        with open(rst, "r") as f:
            simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.minimizeEnergy(maxIterations=20_000_000)
            integrator.setStepSize(0.000005)
            simulation.context.reinitialize(preserveState=True)
            simulation.step(10_000)
            integrator.setStepSize(0.0001)
            simulation.context.reinitialize(preserveState=True)
        print("MINImizing ENERgy")
        simulation.minimizeEnergy(maxIterations=200_000_000)
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        simulation.context.reinitialize(preserveState=True)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
        simulation.reporters.append(
            StateDataReporter(
                stdout,
                2000,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator="\t",
            )
        )
        simulation.reporters.append(DCDReporter(f"pre_comp.dcd", 10_000))
        simulation.step(10_000)
        simulation.minimizeEnergy(maxIterations=200_000_000)
        simulation.step(200_000)
        # simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
        print("\nINITial SYSTem ENERgy")
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        pdbfile.PDBFile.writeModel(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            file=open("init.pdb", "w"),
        )
        ctr = 0
        boxv = simulation.context.getState().getPeriodicBoxVectors()
        # psf.boxVectors = boxv
        for i in range(iter):
            state = simulation.context.getState(
                getPositions=True, getVelocities=True
            )
            rst = f"{self.id.lower()}_temp.rst"
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            system = psf.createSystem(
                params,
                nonbondedMethod=nb_meth,
                nonbondedCutoff=1.3 * unit.nanometer,
                rigidWater=False,
                solventDielectric=60,
            )
            integrator = NoseHooverIntegrator(
                T * unit.kelvin, 50 / unit.picosecond, dt * unit.picoseconds
            )
            barostat = system.addForce(
                MonteCarloBarostat(p * unit.atmospheres, 100 * unit.kelvin, 10)
            )
            if useBMH:
                system, epsilons, sigm = sd.eliminate_LJ(psf)

                system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)
            simulation = Simulation(
                psf.topology, system, integrator, platform, prop
            )
            with open(rst, "r") as f:
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.reporters.append(
                StateDataReporter(
                    stdout,
                    10_000,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                    speed=True,
                    separator="\t",
                )
            )
            if ctr == 0:
                try:
                    simulation.context.setPeriodicBoxVectors(*boxv)
                    print(
                        f"setting {simulation.context.getState().getPeriodicBoxVectors()}"
                    )
                    simulation.minimizeEnergy(maxIterations=2_000_000)
                except:
                    print("failed\nusing init box")
            if ctr > 0:
                print(f"Using Dimensions: {boxv}")
                simulation.context.setPeriodicBoxVectors(*boxv)

            simulation.reporters.append(
                DCDReporter(
                    f"{self.id.lower()}_equilibration_{dt}_{p}_{ctr}_rst.dcd",
                    20_000,
                )
            )
            # print("STARting DYNAmics")
            simulation.step(1000)
            system.removeForce(barostat)
            simulation.context.reinitialize(preserveState=True)
            simulation.step(500_000)
            boxv = simulation.context.getState().getPeriodicBoxVectors()
            state = simulation.context.getState(
                getPositions=True, getVelocities=True
            )
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            ctr += 1
        system = psf.createSystem(
            params,
            nonbondedMethod=nb_meth,
            nonbondedCutoff=1.5 * unit.nanometer,
            constraints=None,
        )
        integrator = NoseHooverIntegrator(
            300 * unit.kelvin, 50 / unit.picosecond, 0.004 * unit.picoseconds
        )
        if useBMH:
            system, epsilons, sigm = sd.eliminate_LJ(psf)
            # print(np.sum(epsilons))
            # system = eliminate_elec(psf)
            system = sd.usemodBMH(self, psf, epsilons, sigm, NBfix=True)
        simulation = Simulation(
            psf.topology, system, integrator, platform, prop
        )
        simulation.reporters.append(
            StateDataReporter(
                stdout,
                10_000,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator="\t",
            )
        )
        simulation.reporters.append(
            DCDReporter(
                f"{self.id.lower()}_equilibration_{dt}_{p}_{ctr}_rst.dcd",
                20_000,
            )
        )

        simulation.step(nstep)
        self.boxv = simulation.context.getState().getPeriodicBoxVectors()
        state = simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        with open(f"{self.id.lower()}.rst", "w") as f:
            f.write(XmlSerializer.serialize(state))
