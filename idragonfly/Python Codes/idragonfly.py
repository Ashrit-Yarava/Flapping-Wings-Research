import sys
import math
import logging
import jax.numpy as jnp

from functions import *

# Global Variables

mplot = 0
vplot = 0
fid = 0
eps = 0
folder = 0
wplot = 0
zavoid = 0
DELta = 0
ibios = 0
svCont = 0
wvCont = 0
ivCont = 0
vpFreq = 0

## Impulses wrt body-translating system

Limpulseb = 0
Aimpulseb = 0
Limpulsew = 0
Aimpulsew = 0

LDOT = 0
HDOT = 0
ZETA = 0

## Impulses wrt forward wing translating system

LimpulsebF = 0
AimpulsebF = 0
LimpulsewF = 0
AimpulsewF = 0

## Impulses wrt rear wing translating system

LimpulsebR = 0
AimpulsebR = 0
LimpulsewR = 0
AimpulsewR = 0

# FILE MANIPULATION

output_folder = 'fig/'
# Set the logging file.
logging.basicConfig(filename="output.txt", filemode="w", level=logging.INFO)

# -----------------------------------------------
# Debugging Parameters

# Airfoil Mesh plot: mplot mesh plot 0 (no), 1(yes), 2(Compare equal arc and equal abscissa mesh points)
mplot = 0 # `0` in production phase.

# Airfoil normal velocity plot: vplot 0(no), 1(yes)
vplot = 0 # `0` in production phase.

# Wake Vortex plot: 0(no), 1(yes)
wplot = 1 # `1` for production run.

# Zavoid: 0 (no, faster), 1 (yes, slower)
zavoid = 0; # `1` if vllocity plot shows blowup somewhere.

# Velocity field plot: 0 (no), 1 (yes)
vfplot = 0 
# `0` production phase to save memory.
# `1` selective cases shown to the audience.

# Which wing to plot velocity field.
vplotFR = 1 # 1: Forewing, 2: Rear Wing.
# 1: The original meshing is done in the local wing-fixed system for the forward wing
# 2: The original meshing is done in the local wing-fixed system for the rear wing    

# Velocity Contour plots in space-fixed system
# used as 4th argument of contourf to control the range of velocity.

## Space fixed velocity plot: svInc (increment) svMax (max velocity)
svInc = 0.025
svMax = 2.5
# Generate the list index.
# jnp.arange doesn't include the last index so append it.
svCont = jnp.append(jnp.arange(0, svMax, svInc, dtype=jnp.float32), jnp.array([[svMax]]))

## Wing-fixed velocity plot: wvInc (increment) wvMax (max velocity)
wvInc = 0.1
wvMax = 7
# Similar method as with svCont.
wvCont = jnp.append(jnp.arange(0, wvMax, wvInc, dtype=jnp.float32), jnp.array([[wvMax]]))

# use of svCont and wvCont: 0 (no) 1 (yes)
# The velocity range varies widely depending on the input parameters
# It is recommended to respecify this when input parameters are changed
ivCont = 0
# Frequency of velocity plots: vpFreq
vpFreq = 1

# -----------------------------------------------
# Input Variables.

## Body Goemetry
nwing = 2 # Number of wings.
delta_ = 0 # Body Angle
b_ = jnp.array([
    -0.5, # b_1
    0.5, # b_2
])
# Requirements:
# b_1 <= 0, b_2 >= 0.
# b_1 >= 0 and b_2 <= 0 will place fore-wing
# back and hind-wing front. Tis trick can be
# used to give a reference quantities to the
# wing in the back, which still remain as the
# wing 1 that now rests in the back.
# AKA the front wing will be in the back and the back wing
# witll be in the front.

# b_ = [
#     -0,
#     0
# ] # Use this for clap and fling.

## Wing Geometry

# l_ => wing span (cm) for the actual wings in 3D.
l0_ = jnp.array([
    5.0,
    5.0,
])
l_ = l0_ * 0.5 # Reduce in half, used for 2D modeling.

# c_ = chord length (cm)
n = 100 # 101? # number of data points defining the airfoil shape.
# Here, use a formula to specify the airfoil shape
atmp_ = [0.4, 0.4]
camber = [0, 0]

x_ = []
y_ = []
c_ = []

for i in range(0, nwing):
    x_.append( jnp.linspace(-atmp_[i], atmp_[i], n) )
    y_.append( camber[i] * atmp_[i] ** 2 - jnp.power(x_[i], 2) )
    c_.append( x_[i][n] - x_[i][0] )

m = 5 # Vortex points on the tinfoil.

## Wing Motion

# Stroke angles (degrees)
phiT_ = [45, 45]
phiB_ = [-45, -45]

# Rotation axis offset (cm)
a_ = [0, 0]

# Stroke plan angle (degrees) wrt the body axis
beta_ = [30, 30]

# Flapping frequency (1/sec)
f_ = [30, 30]

# max rotation (degrees) amplitude: actual rotation is 2*gMax
gMax_ = [30, 30]
# gMax_ = [15, 60]

# Rotation Speed paramter (nondimensional): p[i] = p_[i] / (0.5 * T_[i])
# (Note p[0]=p_0/t_ref, but p[1] ~=(not equal)p_[1]/t_ref, wherere t_ref is the rference time)
# p >= 4

p = [5, 5] # Both must be greater than or equal to 4.

if p[0] < 4 or p[1] < 4:
    print("Values of p must be >= 4.")
    sys.exit()

# rtOff = rotation timing offset (nondimentional): rtOff(i)=rtOff_(i)/(0.5*T_(i))
# (Note rtOff(0)=rtOff_(0)/t_ref, but rtOff(1)~=rtOff_(1)/t_ref, wherere t_ref is the rference time)
# rtOff<0(advanced), rtOff=0 (symmetric), rtOff>0(delayed)

# -0.5<rtOff<0.5
rtOff = [ 0.0, 0.0 ]
# rtOff = [ 0.0, 0.2 ]

if abs(rtOff[0]) > 0.5 or abs(rtOff[1]) > 0.5:
    print("Values of rtOff not greater than 0.5")
    sys.exit()

# tau = phase shift for the time: tau(i)=tau_(i)/(0.5*T_(i))
# (Note tau(0)=tau_(0)/t_ref, but tau(1)~=tau_(1)/t_ref, wherere t_ref is the rference time)
# 0(start from TOP and down), 0<tau<1(in between, start with DOWN STROKE),
# 1(BOTTOM and up), 1<tau<2(in between, start with UP STROKE), 2(TOP): 
# 0 <= tau < 2

tau = [0.0, 0.0]

if (not 0 <= tau[0] <= 2) and (not 0 <= tau[1] <= 2):
    print("Values of tau are not within the valid range. [0, 2]")

# Motion path parameter: mpath 0(no tail), 1 (DUTail; 2eriods), 2(UDTail; 2 periods),
# 3(DUDUTail; 4 periods), 4(UDUDTail; 4 periods)

mpath = [0, 0] # Other options not implemented yet.
logging.info(f"mpath[0], mpath[1] = {mpath[0]}, {mpath[1]}")

## Fluid Parameters

# Air density
rho_ = 0.001225 # g/cm^3

# Ambient velocity (cm/sec, assume it is constant)
# Can be interpreted as the flight velocity when the wind is calm.
U_ = 100.0
V_ = 0.0

# Time increment and # of time steps option 0(manually specify),%1(automatic, recommended)
# itinc = 0 # manually specify the time increment.
itinc = 1 # Specify nperiod (# of periods) below.

# Distance between the house and the observation point to be judged as zero.
eps = 0.5e-6

# Vortex Core model (Modified Biot-Savart equation): 0 (no) 1 (yes)
ibios = 1
logging.info(f"ibios = {ibios}")

# -----------------------------------------------
# Print Input Data

for i in range(nwing):
    logging.info(f"iwing = {i}, l_[i] = {l_[i]}, " +
     f"phiB_[i] = {phiB_[i]}, a_[i] = {a_[i]}, beta_[i] = {beta_[i]}" +
     f"f_[i] = {f_[i]}")
    logging.info(f"iwing = {i}, gMax_[i] = {gMax_[i]}, p[i] = {p[i]}, rtOff[i] = {rtOff[i]}, tau = {tau[i]}")

logging.info(f"U_ = {U_}, V_ = {V_}, m = {m}, n = {n}")
# -----------------------------------------------

# Nondimentionalize the input variables

rT, v_, t_, d_, e, c, x, y, a, b, beta, delta, gMax, U, V = dfinData(l_, phiT_, phiB_, c_, x_, y_, a_, b_, beta_, delta_, f_, gMax_, U_, V_)

# Period Ratio: rt[i] = T_[0] / T_[i]
rt = [1.0, rT] # rT = T_[0] / T_[1]



# Threshold raidus for modified Biot-Savart Equation
DELTA = 0.5 * c/(m - 1)
q = 1.0 # Multiplier 0 < q <= 1
DELTA = q * DELTA
DELta = jnp.min(DELTA)
logging.info(f"q = {q}, delta = {DELta}")

# Time increment

# Default (0)
dt = 0.1 # 0.025 (m = 21)
nstep = 21 # 81 (m = 21)


# Automatic (1)
if itinc == 1:
    nperiod = 1 # # of period to calculate (default = 1)
    dt = min(min(c/(m-1), 0.1*(4./p))) # 4/p = duration of pitch
    nstep = nperiod * math.ceil(2 / dt) # One period = 2 (nondimensional)

logging.info(f"nstep = {nstep}, dt = {dt}")


# Comparison of flapping, pitching and air speeds
air = jnp.sqrt(math.pow(U_, 2) + math.pow(V_, 2))
logging.info(f"Air Speed = {air}")
if air > 1e-3:
    # Flapping/Air Seed Ratio
    fk = 2 * jnp.multiply(f_, d_ / jnp.sqrt(math.pow(U_, 2) + math.pow(V_, 2)) )
    logging.info(f"Flapping/Air: Speed Ratio = {fk}")

    # Pitch/Flapping Speed Ratio
    r = 0.25 * jnp.multiply(jnp.multiply(jnp.divide(c_, d_), p / t_), jnp.divide(gMax, f_) )
    logging.info(f"Pitch/Flapping: Speed Ratio = {r}")

    # Pitch/Air Speed Ratio
    k = jnp.multiply(fk, r)
    logging.info(f"Pitch/Air: Speed Ratio = {k}")

# Generate the vortex and collocation points on the airfoil
xv = []
for i in range(0, nwing):
    xv_i, yv_i, xc_i, yc_i, dfc_i, nNew = ``