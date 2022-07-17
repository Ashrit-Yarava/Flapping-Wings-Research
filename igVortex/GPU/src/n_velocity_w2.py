import jax.numpy as jnp
import jax
from jax import lax


# def n_velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, ibios, eps, delta):
#     VNW = jnp.zeros((m - 1))

#     if ibios == 0:
#         eps = eps * 1000

#         for i in range(1, m):
#             for j in range(1, iGAMAw):
#                 r = jnp.abs(ZC[i - 1] - ZF[j - 1])
#                 GF = complex(0.0, 0.0)
#                 if r > eps:
#                     GF = 1.0 / (ZC[i - 1] - ZF[j - 1])
#                 VNW = VNW.at[i - 1].set(VNW[i - 1] + GAMAw[j - 1] *
#                                         jnp.imag(NC[i - 1] * GF) / (2.0 * jnp.pi))

#     elif ibios == 1:
#         for i in range(1, m):
#             for j in range(1, iGAMAw + 1):
#                 r = jnp.abs(ZC[i - 1] - ZF[j - 1])
#                 if r < eps:
#                     GF = complex(0.0, 0.0)
#                 else:
#                     GF = 1.0 / (ZC[i - 1] - ZF[j - 1])
#                     if r < delta:
#                         GF = GF * (r / delta) ** 2
#                 VNW = VNW.at[i - 1].set(VNW[i - 1] + GAMAw[j - 1] *
#                                         jnp.imag(NC[i - 1] * GF) / (2.0 * jnp.pi))

#     return VNW, eps


def velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw, ibios, eps, delta):

    def ibios0(iGAMAw, ZC, NC, ZF, GAMAw, eps, delta):
        VNW = jnp.zeros((m - 1))
        # for i in jnp.arange(1, m):
        #     for j in jnp.arange(1, iGAMAw):
        #         r = jnp.abs(ZC[i - 1] - ZF[j - 1])
        #         GF = lax.cond(r > eps, lambda x, y: 0.0 + 1j * 0.0, lambda x, y: 1.0 / (x - y),
        #                 ZC[i - 1], ZF[j - 1])
        #         VNW = VNW.at[i - 1].set(VNW[i - 1] + GAMAw[j - 1] *
        #                 jnp.imag( NC[i - 1] * GF ) / (2.0 * jnp.pi))

        for i in jnp.arange(1, m):
            r = jnp.abs(ZC[0:i - 1] - ZF)
            GF = jnp.where(r > eps) * (1.0 / (ZC[i - 1] - ZF[0:iGAMAw - 1]))
            VNW = VNW.at[i - 1].add(GAMAw[0:iGAMAw - 1] * jnp.imag(
                NC[i - 1] * GF / (2.0 * jnp.pi)))
        return VNW

    def ibios1(iGAMAw, ZC, NC, ZF, GAMAw, eps, delta):
        VNW = jnp.zeros((m - 1))
        for i in jnp.arange(1, m):
            r = jnp.abs(ZC[0:i - 1] - ZF)
            GF = jnp.where(r > eps) * (1.0 / (ZC[i - 1] - ZF))
            GF = lax.cond(r > delta, lambda gf, r, d: gf * (r / d) ** 2,
                    lambda gf, r, d: gf, GF, r, delta)
            VNW = VNW.at[i - 1].add(GAMAw * jnp.imag(
                NC[i - 1] * GF / (2.0 * jnp.pi)))
        return VNW

    ibios0_jit = jax.jit(ibios0, static_argnums=(0))
    ibios1_jit = jax.jit(ibios1, static_argnums=(0))
    eps = lax.cond(ibios == 0, lambda e: e * 1000, lambda e: e, eps)
    VNW = lax.cond(ibios == 0, ibios0_jit, ibios1_jit, jnp.array(iGAMAw), 
            ZC, NC, ZF, GAMAw, eps, delta) 

    return VNW, eps
