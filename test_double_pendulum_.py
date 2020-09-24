from double_pendulum import DoublePendulum
import numpy as np
def test_double_pendulum_initial_sol():
    double_pendulumtest = DoublePendulum()
    T = 100
    dt = 0.1
    eps = 1e-15
    t_values = np.linspace(0,T,int(T/dt))
    double_pendulumtest.solve((0,0,0,0),T,dt)
    t = double_pendulumtest.t
    theta1 = double_pendulumtest.theta1; theta2 = double_pendulumtest.theta2
    omega1 = double_pendulumtest.omega1; omega2 = double_pendulumtest.omega2
    #Tests:
    for i in range(len(t)):
        assert t[i] == t_values[i]
    for i,j,k,l in zip(theta1,omega1,theta2,omega2):
        assert abs(i) < eps
        assert abs(j) < eps
        assert abs(k) < eps
        assert abs(l) < eps
def test_double_pendulum_total_energy():
    double_pendulumtest = DoublePendulum()
    theta1 = np.pi/3; theta2 = 3*np.pi/2
    omega1 = 0.15; omega2 = 0.075
    T = 40
    dt = 1e-3
    eps = 0.2
    double_pendulumtest.solve((theta1,omega1,theta2,omega2),T,dt)
    kinetic = double_pendulumtest.kinetic
    potential = double_pendulumtest.potential
    total_energy = kinetic[0] + potential[0]
    for i,j in zip(kinetic,potential):
        assert abs(i+j-total_energy) < eps
    """
    Vi testet dette uten method = 'Radau', med samme T og dt, da var den laveste epsilonen vi
    fant lik 10, så med method = 'Radau' gikk feilen til energibevarelsen merkbart ned.
    Fra eps = 10 til eps = 0.2
    """
def test_double_pendulum_angles():
    L = 2.4
    theta1 = np.pi/4; theta2 = np.pi
    omega1 = 0.2; omega2 = 0.01
    T = 50
    dt = 1e-3

    example1 = DoublePendulum(L)
    example1.solve((theta1,omega1,theta2,omega2),T,dt)
    theta1 = 45; theta2 = 180

    example2 = DoublePendulum(L)
    example2.solve((theta1,omega1,theta2,omega2),T,dt,angles = 'deg')

    kinetic1 = example1.kinetic; kinetic2 = example2.kinetic
    potential1 = example1.potential; potential2 = example2.potential

    for i,j,k,l in zip(kinetic1,potential1,kinetic2,potential2):
        assert i == k
        assert j == l
