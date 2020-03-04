'''
Created on Oct 28, 2013

@author: J.Akeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy
from schwimmbad import MPIPool

# If mpi4py is installed, import it.
try:
    from mpi4py import MPI

    MPI = MPI
except ImportError:
    MPI = None


from mpipso.pso import ParticleSwarmOptimizer


class MpiParticleSwarmOptimizer(ParticleSwarmOptimizer):
    """
    PSO with support for MPI to distribute the workload over multiple nodes
    """

    def __init__(self, func, low, high, particleCount=25, threads=1):
        self.threads = threads
        pool = MPIPool()
        super(MpiParticleSwarmOptimizer, self).__init__(func, low, high, particleCount=particleCount, pool=pool)

    def _converged(self, it, p, m, n):

        if self.isMaster():
            converged = super(MpiParticleSwarmOptimizer, self)._converged(it, p, m, n)
        else:
            converged = False

        converged = mpiBCast(converged)
        return converged

    def _get_fitness(self, swarm):

        mpiSwarm = mpiBCast(swarm)

        pos = numpy.array([part.position for part in mpiSwarm])
        results = self.pool.map(self.func, pos)
        lnprob = numpy.array([l[0] for l in results])
        for i, particle in enumerate(swarm):
            particle.fitness = lnprob[i]
            particle.position = pos[i]

    def isMaster(self):
        return self.pool.isMaster()


def mpiBCast(value):
    """
    Mpi bcasts the value and Returns the value from the master (rank = 0).
    """
    #getLogger().debug("Rank: %s, pid: %s MpiPool: bcast", MPI.COMM_WORLD.Get_rank(), os.getpid())
    return MPI.COMM_WORLD.bcast(value)
