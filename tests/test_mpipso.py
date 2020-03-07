#!/usr/bin/env python

"""Tests for `mpipso` package."""

import pytest
import numpy as np
from schwimmbad import MPIPool

from mpipso.mpipso import MpiParticleSwarmOptimizer


class TestMpiParticleSwarmOptimizer(object):
    """

    """
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_optimize(self):
        """

        :return:
        :rtype:
        """
        try:
            MPIPool()
        except ValueError:
            # MPI is not initiated, so skip test
            pass
        else:
            low = np.zeros(2)
            high = np.ones(2)

            pso = MpiParticleSwarmOptimizer(func, low, high, 10)

            max_iter = 10
            swarms, global_bests = pso.optimize(max_iter)
            assert swarms is not None
            assert global_bests is not None
            assert len(swarms) == max_iter
            assert len(global_bests) == max_iter

            fitness = [part.fitness != 0 for part in pso.swarm]
            assert all(fitness)

            assert pso.global_best.fitness != -np.inf


def func(p):
    """
    Function used for testing.
    :return:
    :rtype:
    """
    return -np.random.rand(), None


if __name__ == '__main__':
    pytest.main()
