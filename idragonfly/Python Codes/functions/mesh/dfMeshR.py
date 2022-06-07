def dfMeshR(c, x, y, n, m):
    """
    ==========================================================================
    Refined mesh: vortex points near the end point are refined
    Given an airfoil, identify n+1 points (including the end points)
    that divide the airfoil at an equal n interval
    INPUT VARIABLES
    x,y   data points on the airfoil
    c     chord length (nondimentionalized by d=stroke length)

    n     # of data points to define the airfoil shape (n > m)
    m     # of vortex points (# of collocation points = m-1)
    OUTPUT VAIABLES
    xv, yv    coordinates of the vortex points
    xc, yc    coordinates of the collocation points
    dfc       slope at the collocation points  
    mNew      use m+4 for the number of refined vortex points
    ==========================================================================
    """

    