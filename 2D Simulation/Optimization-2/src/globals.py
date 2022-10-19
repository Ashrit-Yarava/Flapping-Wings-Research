"""Globals File

Contains all the global variables found in the 2D Simulation Program.

* tau:
Phase shift for the time
Start from TOP (0), Between: Start with down stroke (0 < tau < 1),
Start from BOTTOM (1), Between: Start with up stroke (1 < tau < 2),
0 <= tau < 2

* mplot:
Airfoil mesh plot: yes (1), no (0), Compare equal arc and equal abscissa
mesh pointts (2)

* vplot: Airfoil normal velocity plot: yes (1), no (0)

* eps: Used for Modified Biot-Savart Equation

* wplot: Wake vortex plot: yes (1), no (0)

* zavoid: Zavoid: yes (1), no (0)

* mpath:
Motion path parameter: No tail (0), DUTail; 2 periods (1),
                       UDTail; 2 periods (2), DUDUTail; 4 periods (3),
                       UDUDTail; 4 periods (4)

* delta: Distance between the collocation point and the vortex point on the wing

* ibios: Vortex core model (Modified Biot-Savart Equation): yes (1), no (0)

* svCont: Space-Fixed Velocity Plot

* wvCont: Wing-Fixed Velocity Plot

* ivCont:
Use of svCont and ivCont: yes (1), no (0)
The velocity range varies widely depending on the input parameters
It is recommended to respecify this when input parameters are changed.

* vpFreq: Frequency of the velocity plots

* vfplot: TODO

* fig: Figure Folder
"""

tau = 0
mplot = 1
vplot = 1
eps = 0.5e-6
wplot = 1
zavoid = 0
mpath = 0
delta = 0
svCont = 0
wvCont = 0
ivCont = 0
vpFreq = 0
vfplot = 1
fig = "fig/"
log_file = "output.txt"
