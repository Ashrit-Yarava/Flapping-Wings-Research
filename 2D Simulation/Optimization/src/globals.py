import logging
import os


log_file = "output.txt"
folder = "fig/"
#
# if not os.path.exists(folder):
#     os.makedirs(folder)
#
# if not os.path.exists(f"{folder}wake/"):
#     os.makedirs(f"{folder}wake/")
#
# if not os.path.exists(f"{folder}velocity/"):
#     os.makedirs(f"{folder}velocity/")
#
# logging.basicConfig(filename=log_file, filemode="w",
#                     force=True, level=logging.INFO, format="%(message)s")
# logging.info("-------------------------------------------")
# logging.info("igVortex")
# logging.info("-------------------------------------------")

mplot = 1
nplot = None
vplot = 1
wplot = 1
zavoid = 0
vfplot = 1
tau = 0

mpath = 0
ibios = 1

ivCont = 0
vpFreq = 1

delta = 0
iterations = None

impulseAb = None
impulseLb = None
impulseAw = None
impulseLw = None

LDOT = None
HDOT = None


