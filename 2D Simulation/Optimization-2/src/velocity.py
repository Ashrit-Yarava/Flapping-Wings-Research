import numpy as np
from src.vel_vortex import vel_vortex, vel_vortex_improved


def velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw):
    VEL = np.zeros((iGAMAf), dtype=complex)
    for i in range(iGAMAf):
        for j in range(m):
            VELF = vel_vortex(GAMA[j], ZF[i], ZV[j])
            VEL[i] = VEL[i] + VELF
        for j in range(iGAMAw):
            VELF = vel_vortex(GAMAw[j], ZF[i], ZF[j])
            VEL[i] = VEL[i] + VELF
        # Air velocity
        # VEL[i] += complex(U - dl, V - dh)
    # VELF, eps = vel_vortex(GAMA, ZF, ZV)
    return VEL * -1


def velocity_improved(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw):
    v1 = np.sum(vel_vortex_improved(GAMA[0:m], ZF[0:iGAMAf], ZV[0:m]), axis=1)
    v2 = np.sum(vel_vortex_improved(
        GAMAw[0:iGAMAw], ZF[0:iGAMAf], ZF[0:iGAMAw]), axis=1)
    vs1 = v1.shape[0]
    vs2 = v2.shape[0]

    v1_final = np.pad(v1, (0, max(vs2 - vs1, 0)), mode="constant")
    v2_final = np.pad(v2, (0, max(vs1 - vs2, 0)), mode="constant")

    return ((v1_final + v2_final) * -1)[0:iGAMAf]
