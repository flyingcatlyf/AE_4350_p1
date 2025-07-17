import numpy as np


class RK4(object):
    def __init__(self, step):
        self.step = step

    def solver(self, t, x0, u0):  # the input x0,u0 are required to be np.arrary. #function solver solves x1 and reward
        k1 = self.system(t, x0, u0)
        k2 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k1) / 2.0, u0)
        k3 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k2) / 2.0, u0)
        k4 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k3), u0)

        x1 = x0 + self.step * np.transpose(k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        #reward = 100* x1[0]*x1[0] + x1[1]*x1[1] + u0*u0
        #reward = -1 * x1[0] * x1[0] - 1 * x1[1] * x1[1] - 1 * x1[3] * x1[3] - 0.1 * u0[0] * u0[0] - 0.1 * u0[1] * u0[1]
        #reward = -1 * x1[0] * x1[0] - 1 * x1[1] * x1[1] - 1 * x1[3] * x1[3] #score=-33
        #reward = -57.3 * x1[0] * x1[0]
        terminated = False
        return np.array(x1, dtype=np.float32), terminated, False, {}

    def system(self, t, x0, u0):
        phi = x0[0]
        p = x0[1]
        beta = x0[2]
        r = x0[3]

        deltaa=u0[0]
        deltar=u0[1]

        Lp=-1.699
        Lr=0.172
        Lb=-4.546

        Np = -0.0654
        Nr = -0.0893
        Nb = 3.382

        Yp = 0
        Yphi = 0.0488
        Yr = 0
        Yb = -0.0829


        Ldeltaa=27.276
        Ldeltar=0.576

        Ndeltaa=0.395
        Ndeltar=-1.362

        Ydeltaa=0
        Ydeltar=0.0116


        dphi = p #bank angle
        dp = Lp * p +Lr * r +Lb * beta + Ldeltaa * deltaa + Ldeltar * deltar #roll rate
        dbeta = Yp * p + Yphi * phi +(Yr-1) * r + Yb * beta + Ydeltaa * deltaa + Ydeltar * deltar #sideslip angle
        dr = Np * p + Nr * r + Nb * beta + Ndeltaa * deltaa + Ndeltar * deltar #yaw rate

        dxt = np.array([dphi, dp, dbeta, dr])

        return dxt



