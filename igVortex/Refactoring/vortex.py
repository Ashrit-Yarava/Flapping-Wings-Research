import numpy as np
import logging
import matplotlib.pyplot as plt

from src import *

import src.globals as g

np.set_printoptions(precision=4)

l0_ = 5.0
l_ = 0.5 * l0_
n = 101
atmp_ = 0.8
x_ = np.linspace(-atmp_, atmp_ + 1e-10, n)
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
g.tau = 0.0
rho_ = 0.001225
U_ = 100.0
V_ = 0.0
itinc = 1
eps = 0.5e-6
g.ibios = 1
svInc = 0.025
svMax = 2.5
svCont = np.arange(0.0, svMax + 1e-10, svInc)
wvInc = 0.1
wvMax = 7.0
wvCont = np.arange(0.0, svMax + 1e-10, wvInc)
ivCont = 0
vpFreq = 1

v_, t_, d_, e, c, x, y, a, beta, gMax, U, V = in_data(
    l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_)
g.delta = 0.5 * c / (m - 1)
q = 1.0
g.delta *= q

if itinc == 0:
    dt = 0.025
    nstep = 81
else:
    nperiod = 1
    dt = np.min((c / (m - 1), 0.1 * (4 / p)))
    nstep = int(nperiod * np.ceil(2 / dt))

air = np.sqrt(U_ ** 2 + V_ ** 2)
if air > 1e-03:
    fk = 2 * f_ * d_ / air
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
    k = fk * r
else:
    fk = None
    k = None
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)

xv, yv, xc, yc, dfc, m = mesh_r(c, x, y, n, m)

GAMAw = np.zeros((2 * nstep))
sGAMAw = 0.0
iGAMAw = 0
iGAMAf = 0

ZF = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))
ZW = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))

LDOT = np.zeros((nstep))
HDOT = np.zeros((nstep))

MVN = matrix_coef(xv, yv, xc, yc, dfc, m)

if g.vfplot == 1:
    if camber == 0.0:
        ZETA = c_mesh(c_, d_)
    else:
        ZETA = camber_mesh(c_, d_, camber)

impulseLb = np.zeros((nstep)) + 1j * np.zeros((nstep))
impulseAb = np.zeros((nstep))
impulseLw = np.zeros((nstep)) + 1j * np.zeros((nstep))
impulseAw = np.zeros((nstep))


logging.info(f"air speed = {air}")
logging.info(f"flapping/air: speed ratio = {fk}")
logging.info(f"pitching/flapping: speed ratio = {r}")
logging.info(f"pitch/air: speed ratio = {k}")
logging.info(f"c = {c}\nv_ = {v_}")
logging.info(f"mpath = {g.mpath}")
logging.info(f"ibios = {g.ibios}")
logging.info(
    f"l_ = {l_}, phiT_ = {phiT_}, phiB = {phiB_}, a = {a_}, " +
    "beta = {beta_}, f_ = {f_}")
logging.info(f"gMax_ = {gMax_}, p = {p}, rtOff = {rtOff}, tau = {g.tau}")
logging.info(f"U_ = {U_}, V_ = {V_}, m = {m}, n = {n}")
logging.info(f"nstep = {nstep}, dt = {dt}")

logging.info(f"========================")
logging.info(f" Start Of Time Marching ")
logging.info(f"========================")

ip = None
for istep in range(1, nstep + 1):
    t = (istep - 1) * dt
    alp, l, h, dalp, dl, dh = air_foil_m(t, e, beta, gMax, p, rtOff, U, V)
    LDOT[istep - 1] = dl
    HDOT[istep - 1] = dh

    NC, ZV, ZC, ZVt, ZCt, ZWt = wing_global(istep, t, a,
                                            alp, l, h,
                                            xv, yv, xc, yc,
                                            dfc, ZW, U, V)

    VN = air_foil_v(ZC, ZCt, NC, t, dl, dh, dalp)

    VNW, eps = velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, eps)

    GAMA, MVN, ip = solution(m, VN, VNW, istep, sGAMAw, MVN, ip)

    Lb, Ab, Lw, Aw = impulses(istep,
                              ZVt, ZWt,
                              a, GAMA,
                              m, GAMAw,
                              iGAMAw)

    impulseLb[istep - 1] = Lb
    impulseAb[istep - 1] = Ab
    impulseLw[istep - 1] = Lw
    impulseAw[istep - 1] = Aw

    iGAMAf = 2 * istep

    if istep == 1:
        ZF[2 * istep - 2] = ZV[0]
        ZF[2 * istep - 1] = ZV[m - 1]
    else:
        ZF = np.concatenate((ZF, np.array([ZV[0]])))
        ZF = np.concatenate((ZF, np.array([ZV[m - 1]])))

    VELF, eps = velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, eps)

    ZW = ZF[0:iGAMAf] + VELF * dt

    iGAMAw = iGAMAw + 2
    GAMAw[2 * istep - 2] = GAMA[0]
    GAMAw[2 * istep - 1] = GAMA[m - 1]

    sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]

    ZF = ZW
