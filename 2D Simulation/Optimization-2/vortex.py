from timeit import default_timer

start_time = default_timer()

import src as wings
from scipy.linalg import lu_factor, lu_solve
import numpy as np
import os
import logging
import src.globals as g

# Initialize logging and create missing directories
log_file = "output.txt"
fig = "fig/"
if not os.path.exists(g.fig):
    os.makedirs(g.fig)

if not os.path.exists(f"{g.fig}wake/"):
    os.makedirs(f"{g.fig}wake/")

if not os.path.exists(f"{g.fig}velocity/"):
    os.makedirs(f"{g.fig}velocity/")

logging.basicConfig(filename=log_file, filemode="w",
                    force=True, level=logging.INFO, format="%(message)s")

np.set_printoptions(precision=4)

l_ = 0.5 * 5.0  # Change this number.
n = 101
atmp_ = 0.8
x_ = np.linspace(-atmp_, atmp_, n, endpoint=True)
camber = 0.0
y_ = camber * (atmp_ ** 2 - x_ ** 2)
c_ = x_[n - 1] - x_[0]
m = 5
phiT_ = 45
phiB_ = -45
a_ = 0
beta_ = -30
f_ = 30
gMax_ = 30
p = 5
rtOff = 0.0
rho_ = 0.001225
U_ = 100.0
V_ = 0.0
itinc = 1
svInc = 0.025
svMax = 2.5
g.svCont = np.arange(0.0, svMax + 1e-10, svInc)
wvInc = 0.1
wvMax = 7.0
g.wvCont = np.arange(0.0, wvMax + 1e-10, wvInc)
q = 1.0
dt = 0.025
nstep = 81

v_, t_, d_, e, c, x, y, a, beta, gMax, U, V = wings.in_data(
    l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_)

g.delta = 0.5 * c / (m - 1) * q

if itinc == 1:
    nperiod = 1
    dt = min(c / (m - 1), 0.1 * (4 / p))
    nstep = int(nperiod * np.ceil(2/dt))

air = np.sqrt(U_ ** 2 + V_ ** 2)
fk = 2 * f_ * d_ / air
r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
k = fk * r

if air <= 1e-03:
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)

xv, yv, xc, yc, dfc, m = wings.mesh_r(c, x, y, n, m)

GAMAw = np.zeros(2 * nstep)
GAMAf = np.zeros(2 * nstep)
sGAMAw = 0.0
iGAMAw = 0
iGAMAf = 0
ZF = np.zeros(2 * nstep, dtype=complex)
ZW = np.zeros(2 * nstep, dtype=complex)
impulseLb = np.zeros(nstep, dtype=complex)
impulseLw = np.zeros(nstep, dtype=complex)
impulseAb = np.zeros(nstep)
impulseAw = np.zeros(nstep)
LDOT = np.zeros(nstep)
HDOT = np.zeros(nstep)

MVN = wings.matrix_coef(xv, yv, xc, yc, dfc, m)
MVN_lu = lu_factor(MVN)

ZETA = 0
if g.vfplot == 1:
    if camber == 0.0:
        ZETA = wings.c_mesh(c_, d_)
    else:
        ZETA = wings.camber_mesh(c_, d_, camber)

for istep in range(nstep):
    t = istep * dt

    alp, l, h, dalp, dl, dh = wings.airfoil_m(t, e, beta, gMax, p, rtOff, U, V)

    LDOT[istep] = dl
    HDOT[istep] = dh

    NC, ZV, ZC, ZVt, ZCt, ZWt = wings.wing_global(
        istep, t, a, alp, l, h, xv, yv, xc, yc, dfc, ZW, U, V)

    VN = wings.airfoil_v(ZC, ZCt, NC, t, dl, dh, dalp)
    VNW = wings.velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw)

    GAMA = VN - VNW
    GAMA = np.append(GAMA, -sGAMAw)
    GAMA = lu_solve(MVN_lu, GAMA)

    impulseLb[istep] = -1j * np.sum(GAMA * ZVt)
    impulseAb[istep] = 0.5 * np.sum(GAMA * np.abs(ZVt) ** 2)
    impulseLw[istep] = -1j * np.sum(GAMAw[0:iGAMAw] * ZWt[0:iGAMAw])
    impulseAw[istep] = 0.5 * \
        np.sum(GAMAw[0:iGAMAw] * np.abs(ZWt[0:iGAMAw]) ** 2)

    iGAMAf = 2 * (istep + 1)

    ZF[iGAMAf - 2] = ZV[0]
    ZF[iGAMAf - 1] = ZV[m - 1]

    VELF = wings.velocity_improved(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw)

    ZW[0:iGAMAf] = ZF[0:iGAMAf] + VELF * dt

    iGAMAw = iGAMAw + 2
    GAMAw[iGAMAf - 2] = GAMA[0]
    GAMAw[iGAMAf - 1] = GAMA[m - 1]
    sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]

    ZF = ZW

end_time = default_timer()

print(f"Time Elapsed: {end_time - start_time}")