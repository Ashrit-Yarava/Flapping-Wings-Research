import jax.numpy as jnp


def dfairfoilM(mpath, t, rt, tau, d, e, b, beta, delta, gMax, p, rtOff, U, V):
    gbeta = beta - delta

    if mpath == 0:
        
        # Translational Motion
        l = -U * t + b * jnp.cos(delta) + 0.5 * ( d * jnp.cos( jnp.pi * ( t * rt * tau ) ) + e ) * jnp.cos(gbeta)
        h = -V * t - b * jnp.sin(delta) + 0.5 * ( d * jnp.cos( jnp.pi * ( t * rt * tau ) ) + e ) * jnp.sin(gbeta)
        dl = -U - 0.5 * jnp.pi * rt * d * jnp.sin( jnp.pi * (t * rt + tau) ) * jnp.cos(gbeta)
        dh = -V - 0.5 * jnp.pi * rt * d * jnp.sin( jnp.pi * (t * rt + tau) ) * jnp.sin(gbeta)

        # Rotational Motion

        gam = gMax * dftableG(t, rt, tau, p, rtOff)
        alp = 0.5 * jnp.pi - gbeta + gam

        dalp = gMax * dfDtableG(t, rt, tau, p, rtOff)
        
    return alp, l, h, dalp, dl, dh