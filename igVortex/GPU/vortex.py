import jax.numpy as jnp
import numpy as np
import logging
import os
from src import *
from src.time_march import time_march

log_file = "output.txt"
logging.basicConfig(filename=log_file, filemode="w",
                    force=True, level=logging.INFO, format="%(message)s")
folder = "fig/"
if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(f"{folder}wake/"):
    os.makedirs(f"{folder}wake/")

if not os.path.exists(f"{folder}velocity/"):
    os.makedirs(f"{folder}velocity/")

# Debugging Parameters
mplot = 1
vplot = 0
wplot = 1
zavoid = 0
vfplot = 1

# Input Variables
l0_ = 5.0
l_ = 0.5 * l0_
n = 101
atmp_ = 0.8
x_ = np.linspace(-atmp_, atmp_, n, endpoint=True)
camber = 0.0
y_ = camber * (atmp_ ** 2 - x_ ** 2)
c_ = x_[n - 1] - x_[0]
m = 5

# Wing Motion Parameters
phiT_ = 45
phiB_ = -45
a_ = 0
beta_ = -30
f_ = 30
gMax_ = 30
p = 5
rtOff = 0.0
tau = 0.0
mpath = 0

# Fluid Parameters
rho_ = 0.001225
U_ = 100.0
V_ = 0.0
itinc = 1
eps = 0.5e-6
ibios = 1
svInc = 0.025
svMax = 2.5
svCont = np.linspace(0.0, svMax, svInc, endpoint=True)
wvInc = 0.1
wvMax = 7.0
wvCont = np.linspace(0.0, svMax, wvInc, endpoint=True)
ivCont = 0
vpFreq = 1

v_, t_, d_, e, c, x, y, a, beta, gMax, U, V, T_ = in_data(
    l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_)

delta = 0.5 * c / (m - 1)
q = 1.0
delta = delta * q

if itinc == 0:
    dt = 0.025
    nstep = 81
else:
    nperiod = 1
    dt = min(c / (m - 1), 0.1 * (4 / p))
    nstep = nperiod * np.ceil(2 / dt)

# Comparison of flapping, pitching and air speeds
air = np.sqrt(U_ ** 2 + V_ ** 2)

if air > 1e-03:
    fk = 2 * f_ * d_ / air
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
    k = fk * r  # Pitch/Air Speed Ratio
else:
    fk = None
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
    k = None

xv, yv, xc, yc, dfc, m = mesh_r(c, x, y, n, m, mplot, folder)

GAMAw = np.zeros((2 * nstep))
sGAMAw = 0.0
iGAMAw = 0
iGAMAf = 0
ZF = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))
ZW = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))

LDOT = np.zeros((nstep))
HDOT = np.zeros((nstep))

ZETA = None
if vfplot == 1:
    if camber == 0.0:
        ZETA = c_mesh(c_, d_)
    else:
        ZETA = camber_mesh(c_, d_, camber)

impulseLb = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))
impulseAb = np.zeros((2 * nstep))
impulseLw = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))
impulseAw = np.zeros((2 * nstep))

# Debugging Parameters
logging.info(f"mpath = {mpath}")
logging.info(f"ibios = {ibios}")
logging.info(
    f"l_ = {l_}, phiT_ = {phiT_}, phiB = {phiB_}, a = {a_}, " +
    "beta = {beta_}, f_ = {f_}")
logging.info(f"gMax_ = {gMax_}, p = {p}, rtOff = {rtOff}, tau = {tau}")
logging.info(f"U_ = {U_}, V_ = {V_}, m = {m}, n = {n}")
logging.info(f"T_ = {T_}")
logging.info(f"d_ = {d_}")
logging.info(f"c = {c}")
logging.info(f"v_ = {v_}")
logging.info(f"U = {U}, V = {V}")
logging.info(f"nstep = {nstep}, dt = {dt}")
logging.info(f"air speed = {air}")
logging.info(f"flapping/air: speed ratio = {fk}")
logging.info(f"pitching/flapping: speed ratio = {r}")
logging.info(f"pitch/air: speed ratio = {k}")

LDOT = jnp.array(LDOT)
HDOT = jnp.array(HDOT)

# Time Marching
for istep in range(1, nstep + 1):
    t = (istep - 1) * dt
    alp, l, h, dalp, dl, dh = air_foil_m(t, e, beta, gMax, p, rtOff, U, V)

    LDOT = LDOT.at[istep - 1].set(dl)
    HDOT = HDOT.at[istep - 1].set(dh)

    NC, ZV, ZC, ZVt, ZCt, ZWt = wing_global(
        istep, t, a, alp, l, h, xv, yv, xc, yc, dfc, ZW, U, V)
    VN = air_foil_v(ZC, ZCt, NC, t, dl, dh, dalp)

    VNW, eps = n_velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, ibios, eps, delta)

    GAMA, MVN, ip = solution(m, VN, VNW, istep, sGAMAw, MVN, ip)

    impulseLb, impulseAb, impulseLw, impulseAw = impulses(
        istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw, impulseLb, impulseAb, impulseLw, impulseAw)

    iGAMAf = 2 * istep

    ZF = lax.cond(istep == 1, lambda zf,
                  zv: zf.at[2 * istep - 2].set(zv[0]), lambda zf, zv: jnp.concatenate(zf, zv[0]), ZF, ZV)

    ZF = lax.cond(istep == 1, lambda zf,
                  zv: zf.at[2 * istep - 1].set(zv[m - 1]), lambda zf, zv: jnp.concatenate(zf, zv[m - 1]), ZF, ZV)

    VELF, eps = velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, eps)

    ZW = ZF * VELF * dt

    iGAMAw = iGAMAw + 2
    GAMAw = GAMAw.at[2 * istep - 2].set(GAMA[0])
    GAMAw = GAMAw.at[2 * istep - 1].set(GAMA[m - 1])

    sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]
