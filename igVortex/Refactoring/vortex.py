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

print(MVN)

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


for istep in range(1, nstep + 1):
    t = (istep - 1) * dt
    alp, l, h, dalp, dl, dh = air_foil_m(t, e, beta, gMax, p, rtOff, U, V)
