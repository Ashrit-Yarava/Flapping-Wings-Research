import logging
import jax.numpy as jnp

def iginData(l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_):
    # Period
    T_ = 1.0 / f_
    logging.info(f"T_ = {T_}")

    # Convert angles to radians
    fac = jnp.pi / 180.0
    phiT = fac * phiT_
    phiB = fac * phiB_
    beta = fac * beta_
    gMax = fac * gMax_

    

