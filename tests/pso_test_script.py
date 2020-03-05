import numpy as np
import time

n_particle = 100
n_iterations = 100


def lnprop(x):
    return - x ** 2, None


from mpipso.pso import ParticleSwarmOptimizer
pso = ParticleSwarmOptimizer(func=lnprop, low=[-10], high=[10], particleCount=n_particle, threads=1)

init_pos = np.array([1])
pso.gbest.position = init_pos
pso.gbest.velocity = [0] * len(init_pos)
pso.gbest.fitness, _ = lnprop(init_pos)
X2_list = []
vel_list = []
pos_list = []
time_start = time.time()
if pso.is_master():
    print('Computing the PSO...')
num_iter = 0
for swarm in pso.sample(n_iterations):
    X2_list.append(pso.gbest.fitness * 2)
    vel_list.append(pso.gbest.velocity)
    pos_list.append(pso.gbest.position)
    num_iter += 1
    if pso.is_master():
        if num_iter % 10 == 0:
            print(num_iter)

result = pso.gbest.position

time_end = time.time()
print(time_end - time_start)
print(result)
