import numpy as np
import time
import sys
from schwimmbad import MPIPool

n_particle = 20
n_iterations = 10


def lnprop(x):
    return - x ** 2, None


from mpipso.mpipso import MpiParticleSwarmOptimizer
#from mpipso.pso import ParticleSwarmOptimizer

#pool = MPIPool()
#if not pool.is_master():
#    print('test not master')
#    pool.wait()
#    sys.exit(0)

pso = MpiParticleSwarmOptimizer(func=lnprop, low=[-10], high=[10], particle_count=n_particle, threads=1)
#pso = ParticleSwarmOptimizer(func=lnprop, low=[-10], high=[10], particleCount=n_particle, threads=1, pool=pool)

init_pos = np.array([0])
pso.global_best.position = init_pos
pso.global_best.velocity = [0] * len(init_pos)
pso.global_best.fitness, _ = lnprop(init_pos)
X2_list = []
vel_list = []
pos_list = []
time_start = time.time()
if pso.is_master():
    print('Computing the PSO...')
num_iter = 0

for swarm in pso.sample(n_iterations):
    X2_list.append(pso.global_best.fitness * 2)
    vel_list.append(pso.global_best.velocity)
    pos_list.append(pso.global_best.position)
    num_iter += 1
    if pso.is_master():
        if num_iter % 10 == 0:
            print(num_iter)

if pso.is_master():
    result = pso.mpi_broadcast(pso.global_best.position)
    time_end = time.time()
    print(time_end - time_start)
    print(result)
