"""
Test the PSO module.

Execute with py.test -v

"""
import numpy as np
import time
import numpy.testing as npt
from mpipso.pso import ParticleSwarmOptimizer, Particle


class TestPSO(object):
    ctx = None
    params = np.array([[1, 2, 3], [4, 5, 6]])

    def setup(self):
        pass

    def test_Particle(self):
        particle = Particle.create(2)
        assert particle.fitness == -np.inf

        assert particle == particle.pbest

        particle2 = particle.copy()
        assert particle.fitness == particle2.fitness
        assert particle.paramCount == particle2.paramCount
        assert (particle.position == particle2.position).all()
        assert (particle.velocity == particle2.velocity).all()

        particle.fitness = 1
        particle.updatePBest()

        assert particle.pbest.fitness == 1

    def test_setup(self):
        low = np.zeros(2)
        high = np.ones(2)
        pso = ParticleSwarmOptimizer(None, low, high, 10)

        assert pso.swarm is not None
        assert len(pso.swarm) == 10

        position = [part.position for part in pso.swarm]

        assert (position >= low).all()
        assert (position <= high).all()

        velocity = [part.velocity for part in pso.swarm]
        assert (velocity == np.zeros(2)).all()

        fitness = [part.fitness == 0 for part in pso.swarm]
        assert all(fitness)

        assert pso.gbest.fitness == -np.inf

    def test_optimize(self):
        low = np.zeros(2)
        high = np.ones(2)
        func = lambda p: (-np.random.rand(), None)
        pso = ParticleSwarmOptimizer(func, low, high, 10)

        maxIter = 10
        swarms, gbests = pso.optimize(maxIter)
        assert swarms is not None
        assert gbests is not None
        assert len(swarms) == maxIter
        assert len(gbests) == maxIter

        fitness = [part.fitness != 0 for part in pso.swarm]
        assert all(fitness)

        assert pso.gbest.fitness != -np.inf

    def test_sample(self):

        np.random.seed(42)
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
        npt.assert_almost_equal(result[0], 0, decimal=6)
