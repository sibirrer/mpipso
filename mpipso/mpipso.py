'''
Created on Oct 28, 2013

@author: J.Akeret, S. Birrer
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy
#import schwimmbad
from schwimmbad import MPIPool
#from emcee.mpi_pool import MPIPool


from mpipso.pso import ParticleSwarmOptimizer
import sys


class MpiParticleSwarmOptimizer(ParticleSwarmOptimizer):
    """
    PSO with support for MPI to distribute the workload over multiple nodes
    """

    def __init__(self, func, low, high, particleCount=25, threads=1):
        self.threads = threads
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        ParticleSwarmOptimizer.__init__(self, func, low, high, particleCount=particleCount, pool=pool)

    def _converged(self, it, p, m, n):

        if self.is_master():
            converged = super(MpiParticleSwarmOptimizer, self)._converged(it, p, m, n)
        else:
            converged = False
        converged = self.mpiBCast(converged)
        return converged

    def _get_fitness(self, swarm):

        mpiSwarm = self.mpiBCast(swarm)
        pos = numpy.array([part.position for part in mpiSwarm])
        results = self.pool.map(self.func, pos)
        lnprob = numpy.array([l[0] for l in results])
        for i, particle in enumerate(swarm):
            particle.fitness = lnprob[i]
            particle.position = pos[i]

    def is_master(self):
        return self.pool.is_master()

    def isMaster(self):
        return self.is_master()

    def mpiBCast(self, value):
        """
        Mpi bcasts the value and Returns the value from the master (rank = 0).
        """
        #getLogger().debug("Rank: %s, pid: %s MpiPool: bcast", MPI.COMM_WORLD.Get_rank(), os.getpid())

        return self.pool.comm.bcast(value)
        #return MPI.COMM_WORLD.bcast(value)
