import math
import numpy as np

class Ref_wave(object):
    def __init__(self, T):
        self.T = T

    def fun(self, k, amplitude):
        if k % 30 == 0:
            amplitude = np.random.uniform(-30 * np.pi / 180, 30 * np.pi / 180)
        x = np.sin((2 * np.pi / self.T) * 0.1 * k)
        y = amplitude * math.copysign(1, x)

        return y


