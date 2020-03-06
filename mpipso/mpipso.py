"""
Created on Oct 28, 2013

@author: J.Akeret, S. Birrer, A. Shajib
"""

import sys
import numpy as np
from schwimmbad import MPIPool

from .pso import ParticleSwarmOptimizer


class MpiParticleSwarmOptimizer(ParticleSwarmOptimizer):
    """
    PSO with support for MPI to distribute the workload over multiple nodes
    """

    def __init__(self, func, low, high, particle_count=25, threads=1):
        self.threads = threads
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        ParticleSwarmOptimizer.__init__(self, func, low, high,
                                        particle_count=particle_count,
                                        pool=pool)

    def _converged(self, it, p, m, n):

        if self.is_master():
            converged = super(MpiParticleSwarmOptimizer,
                              self)._converged(it, p, m, n)
        else:
            converged = False
        converged = self.mpi_broadcast(converged)
        return converged

    def _get_fitness(self, swarm):

        mpi_swarm = self.mpi_broadcast(swarm)
        pos = np.array([part.position for part in mpi_swarm])
        results = self.pool.map(self.func, pos)
        ln_probability = np.array([l[0] for l in results])
        for i, particle in enumerate(swarm):
            particle.fitness = ln_probability[i]
            particle.position = pos[i]

    def is_master(self):
        return self.pool.is_master()

    def mpi_broadcast(self, value):
        """
        Mpi broadcasts the value and Returns the value from the master (rank
        = 0).
        """
        #getLogger().debug("Rank: %s, pid: %s MpiPool: bcast",
        # MPI.COMM_WORLD.Get_rank(), os.getpid())

        return self.pool.comm.bcast(value)
        #return MPI.COMM_WORLD.bcast(value)
