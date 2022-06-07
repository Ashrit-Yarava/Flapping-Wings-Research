import jax.numpy as jnp
from numpy import imag

def dfnVelocitym(m, ZC, NC, nwing, ZF, GAMAw, iGAMAw, eps, DELta, ibios):
    VNW = jnp.zeros((m - 1))

    if ibios == 0:
        eps *= 1000

        for i in range(m - 1):
            VNW[i] = 0.0

            for iwing in range(nwing):
                for j in range(iGAMAw[iwing]):
                    r = jnp.abs(ZC[i] - ZF[iwing, j])
                    GF = complex(0.0, 0.0)

                    if r > eps:
                        GF = 1.0 / ( ZC[i] - ZF[iwing, j] )
                    
                    VNW[i] += GAMAw[iwing, j] * jnp.imag(NC[i] * GF) / (2.0 * jnp.pi)
    
    elif ibios == 1:
        for i in range(m - 1):
            VNW[i] = 0.0

            for iwing in range(nwing):
                for j in range(iGAMAw[iwing]):
                    r = jnp.abs(ZC[i] - ZF[iwing, j])
                    if r < eps:
                        GF = complex(0.0, 0.0)
                    else:
                        GF = 1.0 / (ZC[i] - ZF[iwing, j])
                        if r < DELta:
                            GF = GF * ((r / DELta) ** 2)
                    VNW[i] += GAMAw[iwing, j] * jnp.imag(NC[i] * GF) / (2.0 * jnp.pi)

    
    return VNW