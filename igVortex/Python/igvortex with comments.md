```python
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

```



```python
from timeit import default_timer

starting_time = default_timer()
sns.set_theme()
```



Turn off interactive mode. (There's no point in using it.)

```python
plt.ioff()
```



Load jax for the first time before starting logging. Any warnings printed out will not be shown in the log.

```python
jnp.arange(5)
```



Print only 4 decimal places, similar to MATLAB

```python
np.set_printoptions(precision=4)
```



```python
g.log_file = "output.txt"
g.folder = "fig/"
```



Directory Creation

Makes the directories that need to be created.

```python
if not os.path.exists(g.folder):
    os.makedirs(g.folder)

if not os.path.exists(f"{g.folder}wake/"):
    os.makedirs(f"{g.folder}wake/")

if not os.path.exists(f"{g.folder}velocity/"):
    os.makedirs(f"{g.folder}velocity/")
```



### LOGGING

```python
logging.basicConfig(filename=g.log_file, filemode="w",
                    force=True, level=logging.INFO, format="%(message)s")
logging.info("-------------------------------------------")
logging.info("igVortex")
logging.info("-------------------------------------------")
```



### DEBUGGING PARAMETERS

Airfoil Mesh Plot: mplot mesh plot 0 (no) 1 (yes) 2 (Compare equal arc and equal abscissa mesh points)

```python
g.mplot = 1
```

Airfoil normal velocity plot: vplot 0 (no) 1 (yes)

```python
g.vplot = 0
```

Wake Vortex plot: 0 (no) 1 (yes)

```python
g.wplot = 1
```

Velocity plot by avoid source and observation points coincidence.

Zavoid 0 (no, faster) 1 (yes, slower)

```python
g.zavoid = 0
```

Velocity field plot: 0 (no) 1 (yes)

```python
vfplot = 1
```



### INPUT VARIABLES

#### Wing Geometry

Wing Span (cm)

```python
l0_ = 5.0
```

Reduce the wing span by half to be used for 2D Modeling

```python
l_ = 0.5 * l0_
```



c_ = chord length (cm)

calculated while specifying the airfoil shape

 \# of data points that define the airfoil shape

```python
n = 101
```

Read airfoil shape data to determine the chord length

Here use a formula to specify the airfoil shape



```python
atmp_ = 0.8
x_ = np.linspace(-atmp_, atmp_ + 1e-10, n)
```



Camber options are not elaborated yet.

```python
camber = 0.0
y_ = camber * (atmp_ ** 2 - x_ ** 2)
c_ = x_[n - 1] - x_[0]
```



\# of vortex points on the airfoil

```
m = 15
```



#### Wing Motion Parameters



Stroke angles (degrees)

```python
phiT_ = 45
phiB_ = -45
```



rotation axis offset (cm)

```python
a_ = 0
```

stroke plane angle (degrees)

```python
beta_ = -30
```

flapping frequency (1/sec)

```python
f_ = 30
```

max rotation (degrees)

```python
gMax_ = 30
```

Rotation Speed Parameter [p >= 4]

```python
p = 5
```



rotation timing offset (nondimentional)

rtOff < 0 (advanced), rtoff = 0 (symmetric), rtOff > 0 (delayed)       -0.5 < rtOff < 0.5

```python
rtOff = 0.0
```



phase shift for the time [0 <= tau < 2]

0 (start from TOP), 0 < tau < 1 (in between, start with DOWN stroke)

1 (BOTTOM), 1 < tau < 2 (in between, start with UP stroke), 2 (TOP)

```python
g.tau = 0.0
```



Motion path parameter

```python
g.mpath = 0
```

0 (no tail), 1 (DUTail 2 periods), 2 (UDTail 2 periods)

3 (DUDUTail 4 periods), 4 (UDUDTail 4 periods)



#### Fluid Parameters

g/cm^3 (air density)

```python
rho_ = 0.001225
```

Ambient velocity (cm/sec, assume constant) can be interpreted as the flight velocity when the wind is calm

```python
U_ = 100.0
V_ = 0.0
```



Time increment and # of time steps option

```python
itinc = 1
```

0 (manually specified) 1 (automatic)

Specify nperiod ( # of periods ) below



Distance between the source and the observation

```python
g.eps = 0.5e-6
```

point to judged as zero.



Vortex core model (Modified Biot-Savart equation)

```python
g.ibios = 1
```

0 (no) 1 (yes)



#### velocity contour plots in space-fixed system

used as the 4th argument of countour to control the range of velocity.



#### space-fixed velocity plot: svInc (increment) svMax (max velocity)

```python
svInc = 0.025
svMax = 2.5
```

add a tiny number to include endpoint.

```python
g.svCont = np.arange(0.0, svMax + 1e-10, svInc)
```



#### wing-fixed velocity plot: wvInc (increment), wvMax (max velocity)

```python
wvInc = 0.1
wvMax = 7.0
g.wvCont = np.arange(0.0, svMax + 1e-10, wvInc)
```



Use of svCont and wvCont 0 (no) 1 (yes)

The velocity range varies widely depending on the input parameters

It is recommended to respecify this when input parameters are changed.

```python
g.ivCont = 0
```

Frequency of velocity plots

```python
g.vpFreq = 1
```



#### Nondimentionalize the input variables.

```python
v_, t_, d_, e, c, x, y, a, beta, gMax, U, V = iginData(
    l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_)
```



#### Threshold radius for modified Biot-Savart equation

Multiplier 0 < q <= 1

```python
g.delta = 0.5 * c / (m - 1)
q = 1.0  
g.delta *= q
```



Manual

```python
if itinc == 0:
    dt = 0.025
    nstep = 81
```

Automatic

```python
else:
    nperiod = 1
    dt = min(c / (m - 1), 0.1 * (4 / p)) 
    nstep = nperiod * math.ceil(2 / dt)
```

4 / p = duration of pitch



#### Comparison of flapping, pitching and air speeds.

```python
air = math.sqrt(U_ ** 2 + V_ ** 2)
logging.info(f"air speed = {air}")
if air > 1e-03:
```

Flapping/Air Speed Ratio

```python
    fk = 2 * f_ * d_ / air
```

```python
    logging.info(f"flapping/air: speed ratio = {fk}")
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
```

Pitch/Flapping Speed Ratio

```python
    logging.info(f"pitching/flapping: speed ratio = {r}")
```

Pitch/Air Speed Ratio

```python
    k = fk * r
```

```python
    logging.info(f"pitch/air: speed ratio = {k}")
else:
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
    logging.info(f"pitch/flapping: speed ratio = {r}")
```



Generate the vortex and collocation points on the airfoil.

```python
xv, yv, xc, yc, dfc, m = igmeshR(c, x, y, n, m)
```



Time Marching

initialize the wake vortex magnitude array

GAMAw(1:2) step 1, GAMAw(3:4) step 2, GAMAw(5:6) step 3, ...

Leading Edge: odd components, Trailing edge: even components

```python
GAMAw = np.zeros((2 * nstep))
```

Initialize the free vortex magnitude array. This is the vortex to be shed or convected

GAMAf = np.zeros((2 * nstep))

Initialize the total wake vortex sum

```python
sGAMAw = 0.0
```

Initialize the total wake vortex number

```python
iGAMAw = 0
```

Initialize the # of vortices to be convected or shed.

```python
iGAMAf = 0
```

Initialize teh free+wake vortex location array (before convection)

ZF(1:2) step 1, ZF(3:4) step 2, ZF(5:6) step 3, ...

Leading Edge: odd components, Trailing edge: even components

```python
ZF = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))
```

Initialize teh wake vortex location array (after convection)

ZW(1:2) step 1, ZW(3:4) step 2, ZW(5:6) step 3, ...

Leading edge: odd components, Trailing Edge: even components

```python
ZW = np.zeros((2 * nstep)) + 1j * np.zeros((2 * nstep))
```

This is further transformed into a new body-fixed coordinate system



Initialize the linear and angular impuse arrays

```python
g.LDOT = np.zeros((nstep))
g.HDOT = np.zeros((nstep))
```



Vortex convection Time history sample

step 1: iGAMAw_1=0, iGAMAf_1=2

GAMAw_1 = [0         , 0          ]; no wake vortex

GAMAf_1 = [GAMA_1(1) , GAMA_1(m)  ]; vortex to be convected or shed

ZF_1  = [ZV_1(1), ZV_1(m) ] = [ZF(1), ZF(2)]; leading and trailing edges

ZW_1  = [ ZW_1(1)  ,  ZW_1(2)   ]; Convect ZF_1



step 2: iGAMAw_2=2, iGAMAf_2=4

GAMAw_2=GAMAf_1=[GAMA_1(1) , GAMA_1(m) ]; wake vortex

GAMAf_2=[GAMA_1(1) , GAMA_1(m), GAMA_2(1) , GAMA_2(m) ]; vortex to be

convected or shed

ZF_2  = [ZW_1(1)  , ZW_1(2) , ZV_1(1), ZV_1(m) ]

​      = [ ZF_2(1)  ,  ZF_2(2) ,  ZF_2(3)  ,  ZF_2(4)  ]

ZW_2  = [ ZW_2(1)  ,  ZW_2(2) ,  ZW_2(3)  ,  ZW_2(4)  ]; Convect ZF_2 in the

current coord system



step 3: iGAMAw_3=4, iGAMAf_3=6

GAMAw_3=GAMAf_2=[GAMA_1(1) , GAMA_1(m), GAMA_2(1) , GAMA_2(m) ]; wake vortex

GAMAf_3=[GAMA_1(1) , GAMA_1(m), GAMA_2(1) , GAMA_2(m), GAMA_3(1) , GAMA_3(m) ]; vortex to be convected or shed

ZF_3  = [ZW_2(1)  , ZW_2(2) , ZW_2(3)  , ZW_2(4) , ZV_1(1), ZV_1(m) ]

​      = [ ZF_3(1)  ,  ZF_3(2) ,  ZF_3(3)  ,  ZF_3(4) ,  ZF_3(5)  ,  ZF_3(6)  ]

ZW_3  = [ ZW_3(1)  ,  ZW_3(2) ,  ZW_3(3)  ,  ZW_3(4) ,  ZW_3(5)  ,  ZW_3(6)  ]; Convect ZF_3 in the current coord system



```python
igmatrixCoef(xv, yv, xc, yc, dfc, m)

if vfplot == 1:
    if camber == 0.0:
        g.ZETA = igcMESH(c_, d_)
    else:
        g.ZETA = igcamberMESH(c_, d_, camber)
```



Initialize the impulses.

```python
g.impulseLb = np.zeros((nstep), dtype=np.complex64)
g.impulseAb = np.zeros((nstep), dtype=np.complex64)
g.impulseLw = np.zeros((nstep), dtype=np.complex64)
g.impulseAw = np.zeros((nstep), dtype=np.complex64)
```



```python
logging.info(f"mpath = {g.mpath}")
logging.info(f"ibios = {g.ibios}")
logging.info(
    f"l_ = {l_}, phiT_ = {phiT_}, phiB = {phiB_}, a = {a_}, " +
    "beta = {beta_}, f_ = {f_}")
logging.info(f"gMax_ = {gMax_}, p = {p}, rtOff = {rtOff}, tau = {g.tau}")
logging.info(f"U_ = {U_}, V_ = {V_}, m = {m}, n = {n}")
logging.info(f"nstep = {nstep}, dt = {dt}")
```



Start time marching

```python
logging.info(f"========================")
logging.info(f" Start Of Time Marching ")
logging.info(f"========================")
```





```
for istep in range(1, nstep + 1):

    t = (istep - 1) * dt
```

Log the timestep

```python
    logging.info(f"------------------------")
    logging.info(f"istep = {istep}\tt = {round(t, 8)}")
    logging.info(f"------------------------")
```

Get airfoil motion parameters

```python
    alp, l, h, dalp, dl, dh = igairfoilM(t, e, beta, gMax, p, rtOff, U, V)
```

Subtraction is necessary when finding index.

```python
    g.LDOT[istep - 1] = dl
    g.HDOT[istep - 1] = dh
```



Get the global coordinates of the votex and collocation points on the wing

ZV,ZC      vortex and collocation points on the wing (global system)

ZVt,ZCt    vortex and collocation points on the wing (translatingsystem)

NC         unit normal of the wing at the collocation points (global)

ZW,ZWt     wake vortex in the global and translational systems

​           ZW in istep=1 is assigned zero (or null) by initialization

​           ZW=ZF for istep >=2 (see the last command of the time marching loop)



```python
    NC, ZV, ZC, ZVt, ZCt, ZWt = igwing2global(
        istep, a, alp, l, h, xv, yv, xc, yc, dfc, ZW)
```



Normal velocity on the airfoil due to the bound vortex.

```python
    VN = igairfoilV(ZC, ZCt, NC, t, dl, dh, dalp)
    VNW, g.eps = ignVelocityw2(m, ZC, NC, ZF, GAMAw, iGAMAw, g.eps)
```



Solve the system of equations

MVN (coefficient matrix) has m-1 components so far; need to add mth component.

```python
    GAMA, g.MVN, g.ip = igsolution(m, VN, VNW, istep, sGAMAw, g.MVN, g.ip)
```



```python
    g.impulseLb, g.impulseAb, g.impulseLw, g.impulseAw = igimpulses(
        istep, ZVt, ZWt, a, GAMA, m, GAMAw, iGAMAw, g.impulseLb, g.impulseAb, g.impulseLw, g.impulseAw)
```



#### iGAMAf = 2 * istep



Calculate at the velocity at the free and wake vortices to be shed or convected.

```python
iGAMAf = 2 * istep
```

Append the coordinate of the leading edge.



Check for first index because the original starting array is much larger.

```python
    if istep == 1:
        ZF[2 * istep - 2] = ZV[0]
    else:
        ZF = np.concatenate((ZF, np.array([ZV[0]])))
```



Append the coordinate of the trainling edge.



Check for first index because the original starting array is much larger.

```python
    if istep == 1:
        ZF[2 * istep - 1] = ZV[m - 1]
    else:
        ZF = np.concatenate((ZF, np.array([ZV[m - 1]])))

    VELF, g.eps = igvelocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, g.eps)
```



Convect GAMAf from ZF to ZW

```python
    ZW = igconvect(ZF, VELF, dt, iGAMAf)
```



Increment the number of wake vortices

```python
    iGAMAw = iGAMAw + 2
    GAMAw[2 * istep - 2] = GAMA[0]
    GAMAw[2 * istep - 1] = GAMA[m - 1]
```



Add the new wake vortices from the current step

```python
    sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]
```



#### iGAMAw = 2 * istep

All the connected vortices become wake vortices. Set these wake vortex to be the free vortices in the next step. Where two more free vortices (leading and trailing edge vortices) will be added before convection.



```python
ZF = ZW
```



#### All the plotting stuff

```python
    igairfoilVplot(ZC, NC, VN, t)
    igwing2global_plot(ZC, NC, t)
    igplotVortexw(iGAMAw, ZV, ZW, istep)
    if vfplot == 1:
        igplotVelocity(istep, ZV, ZW, a, GAMA, m, GAMAw,
                       iGAMAw, U, V, alp, l, h, dalp, dl, dh)

    logging.info(f"alp = {alp}\nl = {l}\nh = {h}\nimpulseLb = {g.impulseLb}\nimpulseLw = {g.impulseLw}\nimpulseAb = {g.impulseAb}\nimpulseAw = {g.impulseAw}\nsGAMAw = {sGAMAw}\nZF = {ZF}")
```



PRINT OUT ALL THE VARIABLES AT THE END OF THE LOOP



```python
logging.info(f"========================")
logging.info(f"  End Of Time Marching  ")
logging.info(f"========================")
```



Calculate the dimensional force and moment on the airfoil.

The force and moment are per unit length (cm) in out-of-plane direction

```python
igforceMoment(rho_, v_, d_, nstep, dt, U, V)
```



Print and plot the magnitudes of the dimensional wake vortex.

```python
igplotMVortexw(v_, d_, GAMAw, nstep)

ending_time = default_timer()

logging.info(f"TIME ELAPSED: {ending_time - starting_time}")
```

