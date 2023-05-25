from openmm.app import *
from openmm import *
from openmm.unit import nanometers
import numpy as np


def freeze_polymer(psf, PolymerChain):
    system = psf.system
    for i in psf.topology.chains():
        if i.id.startswith(PolymerChain.id.upper()):
            print(i.id)
            for j in i.atoms():
                system.setParticleMass(j.index, 0.0)
    return system


def eliminate_LJ(psf):
    system = psf.system
    epsilons = []
    sigm = []
    for i in range(system.getNumForces()):
        j = system.getForce(i)
        if type(j) == NonbondedForce:
            for k in range(system.getNumParticles()):
                param = j.getParticleParameters(k)
                epsilons.append(param[2])
                sigm.append(param[1])
                j.setParticleParameters(
                    k, charge=param[0], sigma=0.01 * nanometers, epsilon=0
                )
            break
    return system, epsilons, sigm


def eliminate_elec(psf):
    system = psf.system
    for i in range(system.getNumForces()):
        j = system.getForce(i)
        if type(j) == NonbondedForce:
            for k in range(system.getNumParticles()):
                param = j.getParticleParameters(k)
                j.setParticleParameters(k, charge=0, sigma=param[1], epsilon=param[2])
            break
    return system


def usemodBMH(PolymerChain, psf, epsilons, sigm, NBfix=False, a=7.953, f_param=5.871):
    system = psf.system
    group = []
    if NBfix:
        for i in psf.topology.chains():
            print(f"Implementing NBFIX for {i.id}")
            if i.id.startswith(PolymerChain.id.upper()):
                print('as Polymer')
                for f in i.atoms():
                    if f.name != "STYR":
                        group.append(1)
                    else:
                        group.append(2)
            else:
                for f in i.atoms():
                    group.append(-1)
    else:
        for i in range(psf.system.getNumParticles()):
            group.append(-1)
    for i in range(system.getNumForces()):
        j = system.getForce(i)
        if type(j) == NonbondedForce:
            break
    # Modified BMH-Potentail with added scaling for solvent solute interaction; Zhe Wu, Qiang Cui, and Arun Yethiraj J. Phys. Chem. B 2010, 114, 10524–10529
    energy = "((scale*eps)/((1-(f/a))-((6-f)/12)))"
    energy += "*((((6-f)/12)*((rm/r)^12))-((rm/r)^6)"
    energy += "+((f/a)*exp(a*(1-(r/rm)))))"
    energy += ";scale=select(group1+group2,1,1.1);"  # sclaling for solvent interaction
    energy += "eps=sqrt(eps1*eps2);rm=0.5*(rm1+rm2)"  # Lorentz-Berthelot rules
    nb_force = CustomNonbondedForce(energy)
    nb_force.addGlobalParameter("a", a)
    nb_force.addGlobalParameter("f", f_param)
    nb_force.addPerParticleParameter("eps")
    nb_force.addPerParticleParameter("rm")
    nb_force.addPerParticleParameter("group")
    
    for i in range(j.getNumParticles()):
        nb_force.addParticle([epsilons[i], sigm[i], group[i]])
    nb_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    nb_force.setCutoffDistance(1.5 * nanometers)
    system.addForce(nb_force)
    print(f"Number of Particles in BMH-Forcegroup {nb_force.getNumParticles()}")
    for index in range(j.getNumExceptions()):
        l, k, chargeprod, sigma, epsilon = j.getExceptionParameters(index)
        nb_force.addExclusion(l, k)
    return system


def set_octahedron(psf, multiplicator):
    vectors = (
        Vec3(1, 0, 0),
        Vec3(1 / 3, 2 * np.sqrt(2) / 3, 0),
        Vec3(-1 / 3, np.sqrt(2) / 3, np.sqrt(6) / 3),
    )
    a = psf.boxLengths[0] * multiplicator
    boxVectors = [a * v for v in vectors]
    return boxVectors
