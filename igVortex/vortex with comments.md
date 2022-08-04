

```python
import numpy as np
import logging
import matplotlib.pyplot as plt

from src import *

import src.globals as g
```



```python
np.set_printoptions(precision=4)
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

```python
m = 5
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



#### space-fixed velocity plot: svInc (increment) svMax (max velocity)

```python
svInc = 0.025
svMax = 2.5
```

add a tiny number to include endpoint.

```python
svCont = np.arange(0.0, svMax + 1e-10, svInc)
```



#### \# wing-fixed velocity plot: wvInc (increment), wvMax (max velocity)

```python
wvInc = 0.1
wvMax = 7.0
wvCont = np.arange(0.0, svMax + 1e-10, wvInc)
```



Use of svCont and wvCont 0 (no) 1 (yes)

The velocity range varies widely depending on the input parameters

It is recommended to respecify this when input parameters are changed.

```python
ivCont = 0
```

Frequency of velocity plots

```python
vpFreq = 1
```



#### Nondimentionalize the input variables.

```python
v_, t_, d_, e, c, x, y, a, beta, gMax, U, V = in_data(
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
air = np.sqrt(U_ ** 2 + V_ ** 2)
if air > 1e-03:
```

Flapping/Air Speed Ratio

```python
    fk = 2 * f_ * d_ / air
```

Pitch/Flapping Speed Ratio

```python
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
```

Pitch/Air Speed Ratio

```python
    k = fk * r
```

```python
else:
    fk = None
    k = None
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
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



```python
LDOT = np.zeros((nstep))
HDOT = np.zeros((nstep))
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
MVN = matrix_coef(xv, yv, xc, yc, dfc, m)

if g.vfplot == 1:
    if camber == 0.0:
        ZETA = c_mesh(c_, d_)
    else:
        ZETA = camber_mesh(c_, d_, camber)
```



Initialize the impulses

```
impulseLb = np.zeros((nstep)) + 1j * np.zeros((nstep))
impulseAb = np.zeros((nstep))
impulseLw = np.zeros((nstep)) + 1j * np.zeros((nstep))
impulseAw = np.zeros((nstep))
```



log

```
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
```



Start time marching

```python
logging.info(f"========================")
logging.info(f" Start Of Time Marching ")
logging.info(f"========================")
```



    for istep in range(1, nstep + 1):
        t = (istep - 1) * dt

Get airfoil motion parameters

```python
    alp, l, h, dalp, dl, dh = air_foil_m(t, e, beta, gMax, p, rtOff, U, V)
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
    NC, ZV, ZC, ZVt, ZCt, ZWt = wing_global(istep, t, a,
                                            alp, l, h,
                                            xv, yv, xc, yc,
                                            dfc, ZW, U, V)
```



Normal velocity on the airfoil due to the bound vortex.

```python
    VN = air_foil_v(ZC, ZCt, NC, t, dl, dh, dalp)
```



####  iGAMAw = 2 * (istep - 1)

Normal velocity on the air foil due to the wake vortex.

```python
    VNW, eps = velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, eps)
```



Solve the system of equations

MVN (coefficient matrix) has m-1 components so far; need to add mth component.

```python
    GAMA, MVN, ip = solution(m, VN, VNW, istep, sGAMAw, MVN, ip)
```



```
    Lb, Ab, Lw, Aw = impulses(istep,
                              ZVt, ZWt,
                              a, GAMA,
                              m, GAMAw,
                              iGAMAw)

    impulseLb[istep - 1] = Lb
    impulseAb[istep - 1] = Ab
    impulseLw[istep - 1] = Lw
    impulseAw[istep - 1] = Aw
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
        ZF[2 * istep - 1] = ZV[m - 1]
    else:
        ZF = np.concatenate((ZF, np.array([ZV[0]])))
        ZF = np.concatenate((ZF, np.array([ZV[m - 1]])))
```



Append the coordinate of the trainling edge.

Check for first index because the original starting array is much larger.

```python
    VELF, eps = velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw, eps)
```



Convect GAMAf from ZF to ZW

```python
    ZW = ZF[0:iGAMAf] + VELF * dt
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

