import numpy as np
from matplotlib.pyplot import *
import scipy.integrate
from matplotlib.animation import FuncAnimation

class DoublePendulum:
    def __init__(self,M1=1,L1=1,M2=1,L2=1):
        self.M1 = M1
        self.L1 = L1
        self.M2 = M2
        self.L2 = L2
        self.g = 9.81
    def __call__(self,t,y):
        M1 = self.M1; L1 = self.L1
        M2 = self.M2; L2 = self.L2
        g = self.g

        theta1, omega1,theta2,omega2 = y
        theta1_dt,theta2_dt = omega1,omega2
        d = theta2-theta1

        domega1 = M2*L1*omega1**2*np.sin(d)*np.cos(d)+M2*g*np.sin(theta2)*np.cos(d)+M2*L2*omega2**2*np.sin(d)-(M1+M2)*g*np.sin(theta1)
        domega2 = -M2*L2*omega2**2*np.sin(d)*np.cos(d)+(M1+M2)*g*np.sin(theta1)*np.cos(d)-(M1+M2)*omega1**2*np.sin(d)-(M1+M2)*g*np.sin(theta2)
        dt = (M1+M2)*L1-M2*L1*np.cos(d)**2

        domega1_dt = domega1/dt
        domega2_dt = domega2/dt
        return (theta1_dt, domega1_dt, theta2_dt, domega2_dt)
    def solve(self,y0,T,dt,angles = 'rad',method = 'Radau'):
        self.dt = dt
        N = round(T/dt)
        t_span = [0,T]
        t_range = np.linspace(0,T,N)
        if angles == 'deg':
            y = list(y0)
            y[0] = y[0]*(np.pi/180)
            y[2] = y[2]*(np.pi/180)
            y0 = tuple(y)
        sol = scipy.integrate.solve_ivp(self.__call__,t_span, y0, t_eval=t_range,method = method)
        self.time_points = sol.t
        self.theta1_solved, self.omega1_solved, self.theta2_solved, self.omega2_solved = sol.y
        self.solve_called = True
    @property
    def t(self):
        if self.solve_called:
            return self.time_points
        else:
            raise AttributeError("solve method has not been called")
    @property
    def theta1(self):
        if self.solve_called:
            return self.theta1_solved
        else:
            raise AttributeError("solve method has not been called")
    @property
    def theta2(self):
        if self.solve_called:
            return self.theta2_solved
        else:
            raise AttributeError("solve method has not been called")
    @property
    def omega1(self):
        if self.solve_called:
            return self.omega1_solved
        else:
            raise AttributeError("solve method has not been called")
    @property
    def omega2(self):
        if self.solve_called:
            return self.omega2_solved
        else:
            raise AttributeError("solve method has not been called")
    @property
    def x1(self):
        if self.solve_called:
            return self.L1*np.sin(self.theta1)
    @property
    def y1(self):
        if self.solve_called:
            return -self.L1*np.cos(self.theta1)
        else:
            raise AttributeError("solve method has not been called")
    @property
    def x2(self):
        if self.solve_called:
            return self.x1 + self.L2*np.sin(self.theta2)
        else:
            raise AttributeError("solve method has not been called")
    @property
    def y2(self):
        if self.solve_called:
            return self.y1 - self.L2*np.cos(self.theta2)
        else:
            raise AttributeError("solve method has not been called")
    @property
    def potential(self):
        if self.solve_called:
            P1 = self.M1*self.g*(self.y1 + self.L1)
            P2 = self.M2*self.g*(self.y2 + self.L1 + self.L2)
            return P1 + P2
        else:
            raise AttributeError("solve method has not been called")
    @property
    def vx1(self):
        if self.solve_called:
            return np.gradient(self.x1,self.dt)
        else:
            raise AttributeError("solve method has not been called")
    @property
    def vy1(self):
        if self.solve_called:
            return np.gradient(self.y1,self.dt)
        else:
            raise AttributeError("solve method has not been called")
    @property
    def vx2(self):
        if self.solve_called:
            return np.gradient(self.x2,self.dt)
        else:
            raise AttributeError("solve method has not been called")
    @property
    def vy2(self):
        if self.solve_called:
            return np.gradient(self.y2,self.dt)
        else:
            raise AttributeError("solve method has not been called")
    @property
    def kinetic(self):
        if self.solve_called:
            """
            vx1_2 = self.vx1**2; vy1_2 = self.vy1**2
            vx2_2 = self.vx2**2; vy2_2 = self.vy2**2
            K1 = 0.5*self.M1*(np.add(vx1_2,vy1_2))
            K2 = 0.5*self.M2*(np.add(vx2_2,vy2_2))
            """
            K1 = 0.5*self.M1*(self.vx1**2 + self.vy1**2)
            K2 = 0.5*self.M2*(self.vx2**2 + self.vy2**2)
            return K1 + K2
        else:
            raise AttributeError("solve method has not been called")


if __name__ == '__main__':
    L1 = 3; L2 = 6
    theta1 = np.pi/3; theta2 = 3*np.pi/2
    omega1 = 0.15; omega2 = 0.075
    T = 40
    dt = 1e-3
    example = DoublePendulum(L1 = L1, L2 = L2)
    example.solve((theta1,omega1,theta2,omega2),T,dt)

    t = example.t
    potential = example.potential
    kinetic = example.kinetic
    total_energy = np.add(kinetic,potential)

    plot(t,kinetic, color = 'r', label = "Kinetic energy")
    plot(t,potential, color = 'b', label = "Potential energy")
    plot(t,total_energy, color = 'g', label = "Total energy")
    xlabel("t")
    ylabel("E")
    legend()
    show()
    example.create_animation()
