import jax.numpy as jnp


def n_velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, ibios, eps, delta):
    VNW = jnp.zeros((m - 1))

    if ibios == 0:
        eps = eps * 1000

        for i in range(1, m):
            for j in range(1, iGAMAw):
                r = jnp.abs(ZC[i - 1] - ZF[j - 1])
                GF = complex(0.0, 0.0)
                if r > eps:
                    GF = 1.0 / (ZC[i - 1] - ZF[j - 1])
                VNW = VNW.at[i - 1].set(VNW[i - 1] + GAMAw[j - 1] *
                                        jnp.imag(NC[i - 1] * GF) / (2.0 * jnp.pi))

    elif ibios == 1:
        for i in range(1, m):
            for j in range(1, iGAMAw + 1):
                r = jnp.abs(ZC[i - 1] - ZF[j - 1])
                if r < eps:
                    GF = complex(0.0, 0.0)
                else:
                    GF = 1.0 / (ZC[i - 1] - ZF[j - 1])
                    if r < delta:
                        GF = GF * (r / delta) ** 2
                VNW = VNW.at[i - 1].set(VNW[i - 1] + GAMAw[j - 1] *
                                        jnp.imag(NC[i - 1] * GF) / (2.0 * jnp.pi))

    return VNW, eps
