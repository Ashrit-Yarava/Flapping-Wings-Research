import logging
import jax.numpy as jnp

from src.igndData import igndData

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
    
    v_, t_, d_, e, c, x, y, a, U, V = igndData(l_, phiT, phiB, c_, x_, y_, a_, U_, V_, T_)

    return v_, t_, d_, e, c, x, y, a, beta, gMax, U, V

