import py_compile
import src.globals as g
import matplotlib.pyplot as plt
import logging


def plot_m_vortex(v_, d_, GAMAw, nstep):

    gama_ = v_ * d_
    logging.info(f"gama_ * GAMAw = {gama_ * GAMAw}")

    # Dimensional alues of the circulation
    GAMAwo = gama_ * GAMAw[0::2]
    GAMAwe = gama_ * GAMAw[1::2]
    it = list(range(1, nstep + 1))

    plt.plot(it, GAMAwo, 'o-k', it, GAMAwe, 'o-r')
    plt.savefig(f"{g.folder}GAMAw.tif")
    plt.clf()
