from openmm.app import *
from openmm import *
from openmm.unit import nanometers


def freeze_polymer(psf, PolymerChain):
    system = psf.system
    for i in psf.topology.chains():
        if i.id.startswith(PolymerChain.id.upper()):
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
                    k, charge=param[0], sigma=0.1 * nanometers, epsilon=0
                )
            break
    return system, epsilons, sigm


def useBMH(PolymerChain, psf, epsilons, sigm, NBFix=False, a=7.953, f=5.871):
    system = psf.system
    group = []
    if NBFix:
        for i in psf.topology.chains():
            print(i.id)
            if i.id.startswith(PolymerChain.id.upper()):
                for f in i.atoms():
                    if f.name != "STYR":
                        group.append(1)
                    else:
                        group.append(2)
            else:
                for f in i.atoms():
                    group.append(-1)
    else:
        for i in psf.system.getNumParticles:
            group.append(-1)
    for i in range(system.getNumForces()):
        j = system.getForce(i)
        if type(j) == NonbondedForce:
            break
    # Modified BMH-Potentail with added scaling for solvent solute interaction; Zhe Wu, Qiang Cui, and Arun Yethiraj J. Phys. Chem. B 2010, 114, 10524–10529
    energy = "((scale*eps)/((1-(f/a))-((6-f)/12)))"
    energy += "*((((6-f)/12)*((rm/r)^12))-((rm/r)^6)"
    energy += "+((f/a)*exp(a*(1-(r/rm)))))"
    energy += ";scale=select(group1+group2,1,1.2);"
    energy += "eps=sqrt(eps1*eps2);rm=0.5*(rm1+rm2)"
    nb_force = CustomNonbondedForce(energy)
    nb_force.addGlobalParameter("a", a)
    nb_force.addGlobalParameter("f", f)
    nb_force.addPerParticleParameter("eps")
    nb_force.addPerParticleParameter("rm")
    nb_force.addPerParticleParameter("group")
    for i in range(j.getNumParticles()):
        nb_force.addParticle([epsilons[i], sigm[i], group[i]])
    nb_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    nb_force.setCutoffDistance(1.5 * nanometers)
    system.addForce(nb_force)
    print(nb_force.getNumParticles())
    for index in range(j.getNumExceptions()):
        l, k, chargeprod, sigma, epsilon = j.getExceptionParameters(index)
        nb_force.addExclusion(l, k)
    return system


def set_octahedron(psf):
    vectors = (
        Vec3(1, 0, 0),
        Vec3(1 / 3, 2 * sqrt(2) / 3, 0),
        Vec3(-1 / 3, sqrt(2) / 3, sqrt(6) / 3),
    )
    a = 1.01 * psf.boxLengths[0]
    boxVectors = [a * v for v in vectors]
    return boxVectors
