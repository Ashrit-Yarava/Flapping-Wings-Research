import jax.numpy as jnp


def vel_vortex(GAM, z, z0, eps, ibios, delta):
    r = abs(z - z0)

    if(ibios == 0):
        eps = eps * 1000
        v = complex(0.0, 0.0)

        if r > eps:
            v = -1j * GAM / (z - z0) / (2.0 * jnp.pi)

    elif(ibios == 1):
        if r < eps:
            v = complex(0.0, 0.0)
        else:
            v = 1j * GAM / (z - z0) / (2.0 * jnp.pi)
            if r < delta:
                v = v * (r / delta) ** 2

    # Convert the complex velocity v = v_x - i * v_y to the true velocity v = v_x + i * v_y
    return jnp.conjugate(v), eps
