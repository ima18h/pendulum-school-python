import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

class ODEsNotSolve(AssertionError):
    pass

# Oppgave 3a)
class DoublePendulum:
    def __init__(self, L1=1, L2=1, g=9.81):
        self.L1 = L1
        self.L2 = L2
        self.g = g
        self._t = None
        self._theta1, self._omega1 = None, None
        self._theta2, self._omega2 = None, None

    def __call__(self, t, y):
        theta1, omega1, theta2, omega2 = y
        dtheta1 = omega1
        dtheta2 = omega2
        delta_t = theta2 - theta1

        domega1 = (
            (self.L1 * omega1 ** 2 * np.sin(delta_t) * np.cos(delta_t))
            + (self.g * np.sin(theta2) * np.cos(delta_t))
            + (self.L2 * omega2 ** 2 * np.sin(delta_t))
            - (2 * self.g * np.sin(theta1))
        ) / ((2 * self.L1) - (self.L1 * np.cos(delta_t) ** 2))

        domega2 = (
            -(self.L2 * omega2 ** 2 * np.sin(delta_t) * np.cos(delta_t))
            + (2 * self.g * np.sin(theta1) * np.cos(delta_t))
            - (2 * self.L1 * omega1 ** 2 * np.sin(delta_t))
            - (2 * self.g * np.sin(theta2))
        ) / ((2 * self.L2) - (self.L2 * np.cos(delta_t) ** 2))

        return dtheta1, domega1, dtheta2, domega2

# Oppgave 3c)
    def solve(self, y0, T, dt, angle="rad"):
        self.dt = dt
        """ Beregner løsninger av ODE-systemet for 0 <= t <= T """
        if angle == "deg":
            y0[0] = np.radians(y0[0])
            y0[2] = np.radians(y0[2])

        sol = solve_ivp(self, (0, T), y0, max_step=dt, method="LSODA")
        self._t = sol.t
        self._theta1, self._omega1 = sol.y[0], sol.y[1]
        self._theta2, self._omega2 = sol.y[2], sol.y[3]

# Oppgave 3d)
    @property
    def t(self):
        if self._t is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._t

    @property
    def theta1(self):
        if self._theta1 is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._theta1

    @property
    def theta2(self):
        if self._theta2 is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._theta2

    @property
    def omega1(self):
        if self._omega1 is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._omega1

    @property
    def omega2(self):
        if self._omega2 is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._omega2

    @property
    def x1(self):
        return self.L1 * np.sin(self.theta1)

    @property
    def y1(self):
        return -self.L1 * np.cos(self.theta1)

    @property
    def x2(self):
        return self.x1 + self.L2 * np.sin(self.theta2)

    @property
    def y2(self):
        return self.y1 - self.L2 * np.cos(self.theta2)

    @property
    def potential(self):
        # masses are 1
        p1 = self.g * (self.y1 + self.L1)
        p2 = self.g * (self.y2 + self.L2 + self.L1)
        return p1 + p2

    @property
    def vx1(self):
        return np.gradient(self.x1, self.t)

    @property
    def vy1(self):
        return np.gradient(self.y1, self.t)

    @property
    def vx2(self):
        return np.gradient(self.x2, self.t)

    @property
    def vy2(self):
        return np.gradient(self.y2, self.t)

# Oppgave 3e)
    @property
    def kinetic(self):
        # masses are 1
        k1 = 1/2 * (self.vx1**2 + self.vy1**2)
        k2 = 1/2 * (self.vx2**2 + self.vy2**2)
        return k1 + k2


# Oppgave 4a)
    def create_animation(self):
        # Create empty figure
        fig = plt.figure()
            
        # Configure figure
        plt.axis('equal')
        plt.axis('off')
        plt.axis((-3, 3, -3, 3))
            
        # Make an "empty" plot object to be updated throughout the animation
        self.pendulums, = plt.plot([], [], 'o-', lw=2)
            
        # Call FuncAnimation
        self.animation = animation.FuncAnimation(fig,
                                                self._next_frame,
                                                frames=range(len(self.x1)), 
                                                repeat=None,
                                                interval=1000*self.dt, 
                                                blit=True)

    def _next_frame(self, i):
        self.pendulums.set_data((0, self.x1[i], self.x2[i]),
                                (0, self.y1[i], self.y2[i]))
        return self.pendulums,

    def show_animation(self):
        plt.show()

    def save_animation(self, filename):
        self.animation.save(filename, fps=60)


if __name__ == '__main__': 
    # plotter dobbel pendulum graf fra opppgave 3
    pend = DoublePendulum()
    pend.solve((3 * np.pi / 7, 1, 3 * np.pi / 4, 1), 10, 0.01)

    plt.plot(pend.t, pend.potential, label="potential energy")
    plt.plot(pend.t, pend.kinetic, label="kinetic energy")
    plt.plot(pend.t, pend.kinetic + pend.potential, label="kinetic + potential")
    plt.legend()
    plt.title("Graphs of energy conservation")
    plt.show()

    # Plotter animasjon fra oppgave 4
    model = DoublePendulum()
    model.solve((3 * np.pi / 7, 1, 3 * np.pi / 4, 1), 6, 0.01)
    model.create_animation()
    plt.title("Animation of the double pendulum")
    model.show_animation()
    model.save_animation("example_simulation.mp4")
