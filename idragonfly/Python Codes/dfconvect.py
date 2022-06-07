import jax.numpy as jnp

def dfconvect(ZF, VELF, dt, iGAMAf):
    
    ZW = jnp.zeros((2, iGAMAf[0]))
    
    for iwing in range(0, 1):
        for i in range(iGAMAf[0]):
            ZW = ZW.at[iwing, i].set(ZF[iwing, i] + VELF[iwing, i] * dt)

    return ZW