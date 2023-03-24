import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import random
import os
import topo
from misc import *
from simulation_details import *


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
            self.masses = residues[0]
            residues_in_chain = []
            for i in self.monomers:
                for k in residues:
                    if k[0].split()[1] == i:
                        residues_in_chain.append(k)
            bonds = []
            angles = []
            dihedrals = []
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
                line = line.split("!")[0]
                if section == "BOND":
                    bonds.append(line)
                if section == "ANGLE":
                    angles.append(line)
                if section == "DIHE":
                    dihedrals.append(line)
                if section == "NONB":
                    pass
        self.residues = []
        self.angles = []
        self.bonds = []
        self.dihedrals = []
        for i in angles:
            # print(i.split())
            self.angles.append(topo.Angle_type(*i.split()))
        for i in bonds:
            self.bonds.append(topo.Bond_type(*i.split()))
        for i in dihedrals:
            # print(i)
            self.dihedrals.append(topo.Dihedral_type(*i.split()))
        for k in residues_in_chain:
            # try:
            self.residues.append(topo.Residue(k))
        # except:
        #     print(k)
        #     quit()
        ##all residues in toppar submitted
        self._allresidues = []
        for l in residues:
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
                segid_num[segid.index(self.residues[k].name)] += self.n_of_monomers[k]
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
        coords = randomwalk(self.chain_length, 4)

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
                    f"patch {find_patch(prev_seg,n)} {prev_seg} {prev} {n} {ids[num]}\n"
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

    def relax_chain(self, nstep=150_000):
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
            Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
        ]
        for platform in DEFAULT_PLATFORMS:
            if platform in enabled_platforms:
                platform = Platform.getPlatformByName(platform)
                break
        print("Using platform:", platform.getName())
        prop = dict(CudaPrecision="single") if platform.getName() == "CUDA" else dict()
        print("BUILDing SYSTem")
        #######################################################################################
        psf = gen_box(psf, crd, enforce_cubic=True)
        ########################################################################################################################################################
        system = psf.createSystem(
            toppar,
            nonbondedMethod=PME,
            ewaldErrorTolerance=0.005,
            nonbondedCutoff=1.5 * nanometer,
            solventDielectric=60,
            constraints=None,
        )

        # #########################################################################################################################################################
        # Initialize Simulation
        ################################################################################################
        integrator = NoseHooverIntegrator(
            300 * kelvin, 50 / picosecond, 0.0001 * picoseconds
        )
        # system, epsilons, sigm = eliminate_LJ(psf)
        # system = usemodBMH(self, psf, epsilons, sigm, NBfix=False)
        simulation = Simulation(psf.topology, system, integrator, platform, prop)
        simulation.context.setPositions(crd.positions)
        simulation.reporters.append(
            StateDataReporter(
                stdout, 1_000, step=True, totalEnergy=True, separator="\t"
            )
        )
        system.addForce(MonteCarloBarostat(1 * atmospheres, 300 * kelvin, 10))
        simulation.context.setVelocitiesToTemperature(300 * kelvin)
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
        simulation.reporters.append(PDBReporter(f"{self.id.lower()}_relax.pdb", nstep))
        print("STARting DYNAmics")

        simulation.step(nstep)
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        with open(f"{self.id.lower()}_relax.rst", "w") as f:
            f.write(XmlSerializer.serialize(state))

    def solvate_old(
        self,
        num_of_Polymers,
        solvent_res,
        solvent_num,
        pack=True,
        verbose=False,
        pdb_file="default",
        solvent_pdb="default",
        boxsize=1500,
        salt=False,
        c="",
        c_pdb="",
        c_n=0,
        a="",
        a_pdb="",
        a_n=0,
    ):
        ######################################################################################################
        ##Set pdbfile for packing to default if not defined
        ######################################################################################################
        if pdb_file == "default":
            pdb_file = f"{self.id.lower()}_relax.pdb"
        ######################################################################################################
        # checking toppar for Solvent residue
        ######################################################################################################
        found = False
        for i in self._allresidues:
            try:
                if solvent_res == i.name:
                    print("Solvent found in toppar")
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
        #######################################################################################################
        #######################################################################################################
        print("GENErating PSF\nTEA_PUN powered by CHARMM\n")
        #######################################################################################################
        f = open("tmp.inp", "w")
        f.write(f"dimension chsize 4500000\nioformat extended\nstream {self.toppar}\n")
        f.write(f"open unit 1 card name {self.id.lower()}.psf\n")
        f.write(f"read psf card unit 1 \n")
        f.write("close unit 1\n")
        if num_of_Polymers != 1:
            f.write(f"rename segid {self.id}_1 sele all end\n")
            for i in range(num_of_Polymers - 1):
                f.write(f"generate {self.id}_{i+2} duplicate {self.id}_1\n")
        f.write(
            f"read sequence {solvent_res} {solvent_num}\ngenerate {solvent_res} first none last none noangle nodihedral setup warn\n"
        )
        if salt:
            f.write(
                f"read sequence {c} {c_n}\ngenerate {c} first none last none setup warn\n"
            )
            f.write(
                f"read sequence {a} {a_n}\ngenerate {a} first none last none setup warn\n"
            )
        f.write(
            f"open unit 10 write form name ./{self.id.lower()}_in_{solvent_res.lower()}.psf\nwrite unit 10 psf xplor card\nclose unit 10\nstop"
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
        #######################################################################################################
        # refering to packmol!
        #######################################################################################################
        if pack:
            if solvent_pdb == "default":
                solvent_pdb = f"{solvent_res.lower()}.pdb"

            pack_system(
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

    def equilibrate(
        self,
        psf,
        pdb,
        nstep=2_000_000,
    ):
        print(f"EQUIlibrate Chain\nTEA_PUN powered by openMM")
        psf = CharmmPsfFile(psf)
        pdb = PDBFile(pdb)
        params = CharmmParameterSet(self.toppar)
        DEFAULT_PLATFORMS = "CUDA", "OpenCL", "CPU"
        enabled_platforms = [
            Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
        ]
        for platform in DEFAULT_PLATFORMS:
            if platform in enabled_platforms:
                platform = Platform.getPlatformByName(platform)
                break
        print(f"Using {platform.getName()}")
        prop = dict(CudaPrecision="single") if platform.getName() == "CUDA" else dict()
        psf = gen_box(psf, pdb, enforce_cubic=True)
        system = psf.createSystem(
            params,
            nonbondedMethod=CutoffPeriodic,
            nonbondedCutoff=1.5 * nanometer,
            constraints=None,
        )
        integrator = NoseHooverIntegrator(
            300 * kelvin, 50 / picosecond, 0.0001 * picoseconds
        )
        barostat = system.addForce(
            MonteCarloBarostat(1 * atmospheres, 300 * kelvin, 10)
        )

        # system, epsilons, sigm = eliminate_LJ(psf)
        # # system = eliminate_elec(psf)
        # system = usemodBMH(self, psf, epsilons, sigm, NBfix=True)
        system = freeze_polymer(psf, self)
        simulation = Simulation(psf.topology, system, integrator, platform, prop)
        simulation.context.setPositions(pdb.positions)
        try:
            boxVectors = set_octahedron(psf)
            print(f"setting {boxVectors}")
            simulation.context.setPeriodicBoxVectors(*boxVectors)
        except:
            print("failed to Transform into octahedron")
        print("MINImizing ENERgy")
        simulation.minimizeEnergy(maxIterations=20_000_000)
        simulation.context.reinitialize(preserveState=True)
        simulation.context.setVelocitiesToTemperature(300 * kelvin)
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
        for t in [1, 2, 4, 4, 2, 1]:
            print(t)
            simulation.context.setParameter(system.getForce(barostat).Pressure(), t)
            simulation.step(200_000)
            simulation.minimizeEnergy(maxIterations=200_000_000)
            # simulation.context.setVelocitiesToTemperature(300 * kelvin)
        print("\nINITial SYSTem ENERgy")
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
        pdbfile.PDBFile.writeModel(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            file=open("init.pdb", "w"),
        )
        ctr = 0
        boxv = simulation.context.getState().getPeriodicBoxVectors()
        psf.boxVectors = boxv
        for dt in [0.0001, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.005]:
            dt2p = [1, 1, 1, 1, 1, 1, 1]
            print(f"TimeStep set to {dt}ps")
            state = simulation.context.getState(getPositions=True, getVelocities=True)
            rst = f"{self.id.lower()}_temp.rst"
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            system = psf.createSystem(
                params,
                nonbondedMethod=CutoffPeriodic,
                nonbondedCutoff=1.3 * nanometer,
                rigidWater=False,
                solventDielectric=60,
            )
            integrator = NoseHooverIntegrator(
                300 * kelvin, 50 / picosecond, dt * picoseconds
            )
            barostat = system.addForce(
                MonteCarloBarostat(dt2p[ctr] * atmospheres, 300 * kelvin, 10)
            )
            system, epsilons, sigm = eliminate_LJ(psf)
            system = system = usemodBMH(self, psf, epsilons, sigm, NBfix=True)
            simulation = Simulation(psf.topology, system, integrator, platform, prop)
            with open(rst, "r") as f:
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            simulation.reporters.append(
                StateDataReporter(
                    stdout,
                    15_000,
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
                    simulation.minimizeEnergy(maxIterations=20_000_000)
                except:
                    print("failed\nusing init box")
            if ctr > 0:
                print(f"Using Dimensions: {boxv}")
                simulation.context.setPeriodicBoxVectors(*boxv)

            simulation.reporters.append(
                DCDReporter(
                    f"{self.id.lower()}_equilibration_{dt}_{dt2p[ctr]}_{ctr}.dcd",
                    20_000,
                )
            )
            print("STARting DYNAmics")
            simulation.step(nstep // 8)
            boxv = simulation.context.getState().getPeriodicBoxVectors()
            state = simulation.context.getState(getPositions=True, getVelocities=True)
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            ctr += 1
        self.boxv = simulation.context.getState().getPeriodicBoxVectors()
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        with open(f"{self.id.lower()}.rst", "w") as f:
            f.write(XmlSerializer.serialize(state))

    def restart(self, psf, rst, nstep=2_000_000, dt2p=[1, 1, 1, 1, 1, 1], boxl="None"):
        psf = CharmmPsfFile(psf)
        # psf = gen_box(psf, PDBFile("init.pdb"))
        params = CharmmParameterSet(self.toppar)
        DEFAULT_PLATFORMS = "CUDA", "OpenCL", "CPU"
        enabled_platforms = [
            Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
        ]
        for platform in DEFAULT_PLATFORMS:
            if platform in enabled_platforms:
                platform = Platform.getPlatformByName(platform)
                break
        prop = dict(CudaPrecision="single") if platform.getName() == "CUDA" else dict()
        ctr = 0
        psf = gen_box(psf, PDBFile("init.pdb"), octahedron=True)
        for dt in [0.0001, 0.0001, 0.0002, 0.0005, 0.0005, 0.001]:
            print(f"TimeStep set to {dt}ps")
            rst = f"{self.id.lower()}_temp.rst"
            system = psf.createSystem(
                params,
                nonbondedMethod=CutoffPeriodic,
                nonbondedCutoff=1.5 * nanometer,
                rigidWater=False,
                soluteDielectric=80,
                solventDielectric=80,
            )
            integrator = NoseHooverIntegrator(
                300 * kelvin, 50 / picosecond, dt * picoseconds
            )
            barostat = system.addForce(
                MonteCarloBarostat(dt2p[ctr] * atmospheres, 300 * kelvin, 10)
            )
            simulation = Simulation(psf.topology, system, integrator, platform, prop)
            with open(rst, "r") as f:
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
                if ctr == 0:
                    simulation.context.setPeriodicBoxVectors(
                        *(simulation.context.getState().getPeriodicBoxVectors() * 1.1)
                    )
                    simulation.minimizeEnergy(maxIterations=20_000_000)
                    integrator.setStepSize(0.000005)
                    simulation.context.reinitialize(preserveState=True)
                    simulation.step(10_000)
                    integrator.setStepSize(dt)
                    simulation.context.reinitialize(preserveState=True)
            if ctr > 0:
                simulation.context.setPeriodicBoxVectors(*boxv)
            simulation.reporters.append(
                StateDataReporter(
                    stdout,
                    15_000,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    temperature=True,
                    volume=True,
                    speed=True,
                    separator="\t",
                )
            )
            simulation.reporters.append(
                DCDReporter(
                    f"{self.id.lower()}_equilibration_{dt}_{dt2p[ctr]}_{ctr}_rst.dcd",
                    20_000,
                )
            )
            # simulation.reporters.append(
            #     EnergyReporter(
            #         f"{self.id.lower()}_equilibration_energy_{dt}_{dt2p[ctr]}_{ctr}_rst.out",
            #         20_000,
            #     )
            # )
            print("STARting DYNAmics")
            simulation.step(nstep)
            state = simulation.context.getState(getPositions=True, getVelocities=True)
            with open(rst, "w") as f:
                f.write(XmlSerializer.serialize(state))
            ctr += 1
            boxv = simulation.context.getState().getPeriodicBoxVectors()
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        with open(f"{self.id.lower()}.rst", "w") as f:
            f.write(XmlSerializer.serialize(state))
