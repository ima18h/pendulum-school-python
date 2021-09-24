import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

class ODEsNotSolve(AssertionError):
    """
    Lager en klasse som brukes til å kaste en feil dersom de private 
    attributtene t, theta1, theta2, omega1 og omega2 prøver å hentes før man 
    kaller på solve-metoden."""
    pass

# Oppgave 3a)
class DoublePendulum:
    """
    En klasse som tar inn to pendulumer, der den andre pendulumen er festet 
    i den første pendulumen."""
    def __init__(self, L1=1, L2=1, g=9.81):
        """
        Tar inn parameterne L1, L2, g. Om noe annet ikke er gitt så er 
        L = 1 m, L2 = 1 m og g = 9.81 m/s^2 som standardverdi. Tar også inn 
        parameterne t, theta1, theta2, omega1, omega2 som private attributter.
        """
        self.L1 = L1
        self.L2 = L2
        self.g = g
        self._t = None
        self._theta1, self._omega1 = None, None
        self._theta2, self._omega2 = None, None

    def __call__(self, t, y):
        """
        Tar inn parameterne t og y, og beregner og returnerer de deriverte av 
        theta1, theta2, omega1, omega2 (altså h.s av ODE-systemet).
        """
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
        """
        Beregner løsninger av ODE-systemet for 0 <= t <= T ved å bruke scipys
        innebygde metoden solve_ivp. Solve tar også inn en vinkel som enten gis
        i radianer eller grader. Har satt til radianer som standard, og dersom
        vinkelen blir gitt som grader, gjøres den om til radianer.
        """
        self.dt = dt
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
        """
        Lager t som en privat attributt og kaster en AssertionError om man 
        prøver å hente den før metoden solve blir kalt på.
        """
        if self._t is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._t

    @property
    def theta1(self):
        """
        Lagrer theta1 som en privat attributt og kaster en AssertionError om 
        man prøver å hente den før metoden solve blir kalt på.
        """
        if self._theta1 is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._theta1

    @property
    def theta2(self):
        """
        Lagrer theta2 som en privat attributt og kaster en AssertionError om 
        man prøver å hente den før metoden solve blir kalt på.
        """
        if self._theta2 is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._theta2

    @property
    def omega1(self):
        """
        Lagrer omega1 som en privat attributt og kaster en AssertionError om 
        man prøver å hente den før metoden solve blir kalt på.
        """
        if self._omega1 is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._omega1

    @property
    def omega2(self):
        """
        Lagrer omega2 som en privat attributt og kaster en AssertionError om 
        man prøver å hente den før metoden solve blir kalt på.
        """
        if self._omega2 is None: 
            raise ODEsNotSolve(
                "No solution found. Did you remember to call solve?")
        return self._omega2

    @property
    def x1(self):
        """
        Returnerer en array av horisontale verdier i kartetiske koordinater lik
        som den i pendulum.py fila. Plasseres i midten av koordinatsystemet i 
        festepunktet til pendelen.
        """
        return self.L1 * np.sin(self.theta1)

    @property
    def y1(self):
        """
        Returnerer en array av vertikale verdier i kartetiske koordinater lik
        som den i pendulum.py fila. Plasseres i midten av koordinatsystemet i 
        festepunktet til pendelen.
        """
        return -self.L1 * np.cos(self.theta1)

    @property
    def x2(self):
        """
        Returnerer en array av vertikale verdier i kartetiske koordinater som 
        bruker x1 i beregningene. Festes på enden av den første pendulumen.
        """
        return self.x1 + self.L2 * np.sin(self.theta2)

    @property
    def y2(self):
        """
        Returnerer en array av horisontale verdier i kartetiske koordinater som 
        bruker x1 i beregningene. Festes på enden av den første pendulumen.
        """
        return self.y1 - self.L2 * np.cos(self.theta2)
    
# Oppgave 3e)
    @property
    def potential(self):
        """
        Beregner den potensielle energien, som er summen av de potensielle 
        energiene av de to pendulumene.
        """
        p1 = self.g * (self.y1 + self.L1)
        p2 = self.g * (self.y2 + self.L2 + self.L1)
        return p1 + p2

    @property
    def vx1(self):
        """Farten, v_x1, til x-verdiene i pendulum 1."""
        return np.gradient(self.x1, self.t)

    @property
    def vy1(self):
        """Farten, v_y1, til y-verdiene i pendulum 1."""
        return np.gradient(self.y1, self.t)

    @property
    def vx2(self):
        """Farten, v_x2, til x-verdiene i pendulum 2."""
        return np.gradient(self.x2, self.t)

    @property
    def vy2(self):
        """Farten, v_y2, til y-verdiene i pendulum 2."""
        return np.gradient(self.y2, self.t)

    @property
    def kinetic(self):
        """
        Beregner den kinetiske energien, som er summen av de kinetiske
        energiene av de to pendulumene.
        """
        k1 = 1/2 * (self.vx1**2 + self.vy1**2)
        k2 = 1/2 * (self.vx2**2 + self.vy2**2)
        return k1 + k2


# Oppgave 4a)
    def create_animation(self):
        """
        Setter opp en figur og setter sammen figuren ved hjelp av 
        akser, navngivning, tittel og lignende.
        """
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
        """
        Tar inn et tall, i, og oppdaterer figuren for hvert bilde i 
        animasjonen.
        """
        self.pendulums.set_data((0, self.x1[i], self.x2[i]),
                                (0, self.y1[i], self.y2[i]))
        return self.pendulums,

    def show_animation(self):
        """Viser animasjonen."""
        plt.show()

    def save_animation(self, filename):
        """Lagrer animasjonen."""
        self.animation.save(filename, fps=60)


if __name__ == '__main__': 
    """Plotter den dobble pendulumen fra oppg. 3 og animasjonen fra oppg. 4."""
    # Plotter den dobble pendulumen fra oppgave 3
    pend = DoublePendulum()
    pend.solve((3 * np.pi / 7, 1, 3 * np.pi / 4, 1), 10, 0.01)

    plt.plot(pend.t, pend.potential, label="potential energy")
    plt.plot(pend.t, pend.kinetic, label="kinetic energy")
    plt.plot(pend.t, pend.kinetic + pend.potential, label="kinetic + potential")
    plt.legend()
    plt.title("Graphs of energy conservation")
    plt.show()

    # Plotter animasjonen fra oppgave 4
    model = DoublePendulum()
    model.solve((3 * np.pi / 7, 1, 3 * np.pi / 4, 1), 6, 0.01)
    model.create_animation()
    plt.title("Animation of the double pendulum")
    model.show_animation()
    model.save_animation("example_simulation.mp4")
