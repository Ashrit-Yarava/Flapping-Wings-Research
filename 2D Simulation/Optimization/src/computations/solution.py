import numpy as np
from scipy.linalg import lu_factor, lu_solve


def solution(m, VN, VNW, istep, sGAMAw, MVN, ip):
    GAMA = VN - VNW
    GAMA = np.append(GAMA, -sGAMAw)

    if istep == 1:

        ip, MVN = decomp(m, MVN)
        ip = ip.astype('int32')
    return SOLVER(m, MVN, GAMA, ip), MVN, ip

# 173017
