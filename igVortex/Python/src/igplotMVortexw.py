import numpy as np
import logging
import src.globals as g

import matplotlib.pyplot as plt


def igplotMVortexw(v_, d_, GAMAw, nstep):
    """
    Print and plot the magnitudes of the wake vortex.

    Input:
    * GAMAw: wake vortex
    * nstep: # of time steps
    """

    # Reference value for circulation
    gama_ = v_ * d_
    logging.info(f"gama_ * GAMAw = {gama_ * GAMAw}")

    # Dimensional alues of the circulation
    GAMAwo = gama_ * GAMAw[0::2]
    GAMAwe = gama_ * GAMAw[1::2]
    it = list(range(1, nstep + 1))

    plt.plot(it, GAMAwo, 'o-k', it, GAMAwe, 'o-r')
    plt.savefig(f"{g.folder}GAMAw.tif")
    plt.clf()
