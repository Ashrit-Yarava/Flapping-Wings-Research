import jax.numpy as jnp

def assembleMatrix(m, MVNs, MVNs_12, MVNs_21):
    """
    Assemblle 4 sub-matrices 
    INPUT
    MVNs      diagonal
    MVNs_12   off-diagonal
    MVNs_21   off-diagonal
    """

    MVN = jnp.zeros((2*m, 2*m))

    MVN = MVN.at[:m, :m].set(MVNs[:m, :m, 0])
    MVN = MVN.at[:m, m:2*m].set(MVNs_12[:m, :m])
    MVN = MVN.at[m:2*m, :m].set(MVNs_21[:m, :m])
    MVN = MVN.at[m:2*m, m:2*m].set(MVNs[:m, :m, 1])

    return MVN