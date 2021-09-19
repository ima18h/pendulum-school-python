import numpy as np

class Pendulum:
    def __init__(self, L1=1, L2=1, g=9.81):
        self._L1 = L1
        self._L2 = L2
        self._g = g

    def __call__(self, t, y):
        pass