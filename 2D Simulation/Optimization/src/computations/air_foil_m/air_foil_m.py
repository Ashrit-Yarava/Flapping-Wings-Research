import numpy as np
import src.globals as g


from src.mpath import *


def air_foil_m(t, e, beta, gMax, p, rtOff, U, V):
    if(g.mpath == 0):
        l = -U * t + 0.5 * (np.cos(np.pi * (t + g.tau)) + e) * np.cos(beta)
        h = -V * t + 0.5 * (np.cos(np.pi * (t + g.tau)) + e) * np.sin(beta)
        dl = -U - 0.5 * np.pi * np.sin(np.pi * (t + g.tau)) * np.cos(beta)
        dh = -V - 0.5 * np.pi * np.sin(np.pi * (t + g.tau)) * np.sin(beta)

        gam = table_g(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.pi - beta + gam
        dgam = dtable_g(t, p, rtOff)
        dalp = gMax * dgam
    elif(g.mpath == 1):
        dl = -U + 0.5 * d_cos_tail_g(t + g.tau) * np.cos(beta)
        dh = -V + 0.5 * d_cos_tail_g2(t + g.tau) * np.sin(beta)
        l = -U * t + 0.5 * cos_tail_g(t + g.tau, e) * np.cos(beta)
        h = -V * t + 0.5 * cos_tail_g2(t + g.tau, e) * np.sin(beta)

        gam = table_s_tail_g2(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.pi - beta + gam
        dgam = d_table_s_tail_g2(t, p, rtOff)
        dalp = gMax * dgam
    elif(g.mpath == 2):
        # Translational Motion
        dl = -U * 0.5 * d_cos_up_tail_g2(t + g.tau) * np.cos(beta)
        dh = -V + 0.5 * d_cos_up_tail_g2(t + g.tau) * np.sin(beta)
        l = -U * t + 0.5 * cos_up_tail_g2(t + g.tau, e) * np.cos(beta)
        h = -V * t + 0.5 * cos_up_tail_g2(t + g.tau, e) * np.sin(beta)

        # Rotational Motion
        gam = table_up_s_tail_g2(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.i - beta + gam
        dgam = d_table_up_s_tail_g2(t, p, rtOff)
        dalp = gMax * dgam
    elif(g.mpath == 3):
        # Translational Motion
        dl = -U * 0.5 * d_cos_tail_g(t + g.tau) * np.cos(beta)
        dh = -V + 0.5 * d_cos_tail_g(t + g.tau) * np.sin(beta)
        l = -U * t + 0.5 * cos_tail_g(t + g.tau, e) * np.cos(beta)
        h = -V * t + 0.5 * cos_tail_g(t + g.tau, e) * np.sin(beta)

        # Rotational Motion
        gam = table_s_tail_g(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.i - beta + gam
        dgam = d_table_s_tail_g(t, p, rtOff)
        dalp = gMax * dgam
    elif(g.mpath == 4):
        # Translational Motion
        dl = -U * 0.5 * d_cos_up_tail_g(t + g.tau) * np.cos(beta)
        dh = -V + 0.5 * d_cos_up_tail_g(t + g.tau) * np.sin(beta)
        l = -U * t + 0.5 * cos_up_tail_g(t + g.tau, e) * np.cos(beta)
        h = -V * t + 0.5 * cos_up_tail_g(t + g.tau, e) * np.sin(beta)

        # Rotational Motion
        gam = table_up_s_tail_g(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * np.i - beta + gam
        dgam = d_table_up_s_tail_g(t, p, rtOff)
        dalp = gMax * dgam
    return alp, l, h, dalp, dl, dh
