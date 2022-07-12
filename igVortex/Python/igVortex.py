import seaborn as sns
import jax.numpy as jnp
import numpy as np
import math
import logging
import src.globals as g
from src.velocityPlot.igplotVelocity import igplotVelocity
from src.iginData import iginData
from src.meshes.igmeshR import igmeshR
from src.igmatrixCoef import igmatrixCoef
from src.meshes.igcMESH import igcMESH
from src.meshes.igcamberMESH import igcamberMESH
from src.mPath.igwing2global import igwing2global, igwing2global_plot
from src.igvelocity import igvelocity
from src.igsolution import igsolution
from src.igplotVortexw import igplotVortexw
from src.igplotMVortexw import igplotMVortexw
from src.ignVelocityw2 import ignVelocityw2
from src.igconvect import igconvect
from src.airofil import igairfoilM, igairfoilV, igairfoilVplot
from src.force.igimpulses import igimpulses
from src.force.igforceMoment import igforceMoment
import os
import matplotlib.pyplot as plt


from timeit import default_timer
starting_time = default_timer()
sns.set_theme()
plt.ioff()
jnp.arange(5)
np.set_printoptions(precision=4)

g.log_file = "output.txt"
g.folder = "fig/"

if not os.path.exists(g.folder):
    os.makedirs(g.folder)

if not os.path.exists(f"{g.folder}wake/"):
    os.makedirs(f"{g.folder}wake/")

if not os.path.exists(f"{g.folder}velocity/"):
    os.makedirs(f"{g.folder}velocity/")

logging.basicConfig(filename=g.log_file, filemode="w",
                    force=True, level=logging.INFO, format="%(message)s")
logging.info("-------------------------------------------")
logging.info("igVortex")
logging.info("-------------------------------------------")

g.mplot = 1
g.vplot = 0
g.wplot = 1
g.zavoid = 0
vfplot = 1

l0_ = 5.0
l_ = 0.5 * l0_
n = 101

atmp_ = 0.8
x_ = np.linspace(-atmp_, atmp_ + 1e-10, n)

camber = 0.0
y_ = camber * (atmp_ ** 2 - x_ ** 2)
c_ = x_[n - 1] - x_[0]

m = 15

phiT_ = 45
phiB_ = -45

a_ = 0
beta_ = -30
f_ = 30
gMax_ = 30
p = 5

rtOff = 0.0

g.tau = 0.0
g.mpath = 0

rho_ = 0.001225
U_ = 100.0
V_ = 0.0
itinc = 1
g.eps = 0.5e-6
g.ibios = 1

svInc = 0.025
svMax = 2.5
g.svCont = np.arange(0.0, svMax + 1e-10, svInc)
wvInc = 0.1
wvMax = 7.0
g.wvCont = np.arange(0.0, svMax + 1e-10, wvInc)
g.ivCont = 0
g.vpFreq = 1

v_, t_, d_, e, c, x, y, a, beta, gMax, U, V = iginData(
    l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_)
g.delta = 0.5 * c / (m - 1)
q = 1.0
g.delta *= q

if itinc == 0:
    dt = 0.025
    nstep = 81
else:
    nperiod = 1
    dt = min(c / (m - 1), 0.1 * (4 / p))
    nstep = nperiod * math.ceil(2 / dt)

air = math.sqrt(U_ ** 2 + V_ ** 2)
logging.info(f"air speed = {air}")
if air > 1e-03:
    fk = 2 * f_ * d_ / air
    logging.info(f"flapping/air: speed ratio = {fk}")
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
    logging.info(f"pitching/flapping: speed ratio = {r}")
    k = fk * r
    logging.info(f"pitch/air: speed ratio = {k}")
else:
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
    logging.info(f"pitch/flapping: speed ratio = {r}")

xv, yv, xc, yc, dfc, m = igmeshR(c, x, y, n, m)

GAMAw = np.zeros((2 * nstep))
sGAMAw = 0.0
iGAMAw = 0
iGAMAf = 0

ZF = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))
ZW = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))

g.LDOT = np.zeros((nstep))
g.HDOT = np.zeros((nstep))

igmatrixCoef(xv, yv, xc, yc, dfc, m)

if vfplot == 1:
    if camber == 0.0:
        g.ZETA = igcMESH(c_, d_)
    else:
        g.ZETA = igcamberMESH(c_, d_, camber)

g.impulseLb = np.zeros((nstep), dtype=np.complex64)
g.impulseAb = np.zeros((nstep), dtype=np.complex64)
g.impulseLw = np.zeros((nstep), dtype=np.complex64)
g.impulseAw = np.zeros((nstep), dtype=np.complex64)

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

    logging.info(f"------------------------")
    logging.info(f"istep = {istep}\tt = {round(t, 8)}")
    logging.info(f"------------------------")

    alp, l, h, dalp, dl, dh = igairfoilM(t, e, beta, gMax, p, rtOff, U, V)

    g.LDOT[istep - 1] = dl
    g.HDOT[istep - 1] = dh

    NC, ZV, ZC, ZVt, ZCt, ZWt = igwing2global(
        istep, a, alp, l, h, xv, yv, xc, yc, dfc, ZW)

    VN = igairfoilV(ZC, ZCt, NC, t, dl, dh, dalp)
    VNW, g.eps = ignVelocityw2(m, ZC, NC, ZF, GAMAw, iGAMAw, g.eps)

    GAMA, g.MVN, g.ip = igsolution(m, VN, VNW, istep, sGAMAw, g.MVN, g.ip)

    g.impulseLb, g.impulseAb, g.impulseLw, g.impulseAw = igimpulses(
        istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw, g.impulseLb, g.impulseAb, g.impulseLw, g.impulseAw)

    iGAMAf = 2 * istep

    if istep == 1:
        ZF[2 * istep - 2] = ZV[0]
    else:
        ZF = np.concatenate((ZF, np.array([ZV[0]])))

    if istep == 1:
        ZF[2 * istep - 1] = ZV[m - 1]
    else:
        ZF = np.concatenate((ZF, np.array([ZV[m - 1]])))

    VELF, g.eps = igvelocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, g.eps)

    ZW = igconvect(ZF, VELF, dt, iGAMAf)

    iGAMAw = iGAMAw + 2
    GAMAw[2 * istep - 2] = GAMA[0]
    GAMAw[2 * istep - 1] = GAMA[m - 1]

    sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]

    ZF = ZW

    # All the plotting stuff.
    igairfoilVplot(ZC, NC, VN, t)
    igwing2global_plot(ZC, NC, t)
    igplotVortexw(iGAMAw, ZV, ZW, istep)
    if vfplot == 1:
        igplotVelocity(istep, ZV, ZW, a, GAMA, m, GAMAw,
                       iGAMAw, U, V, alp, l, h, dalp, dl, dh)

    logging.info(f"alp = {alp}\nl = {l}\nh = {h}\nimpulseLb = {g.impulseLb}\nimpulseLw = {g.impulseLw}\nimpulseAb = {g.impulseAb}\nimpulseAw = {g.impulseAw}\nsGAMAw = {sGAMAw}\nZF = {ZF}")

logging.info("-----------------")
logging.info("End Of Time March")
logging.info("-----------------")

igforceMoment(rho_, v_, d_, nstep, dt, U, V)
igplotMVortexw(v_, d_, GAMAw, nstep)

ending_time = default_timer()

logging.info(f"TIME ELAPSED: {ending_time - starting_time}")
