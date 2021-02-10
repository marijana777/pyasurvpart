"""
  coeff and bhk codes from http://astrostatistics.psu.edu/statcodes/asurv "translated" into Python 
  
"""

import numpy as np
import scipy.special as special


def coeff(i,x,ip,ia,ib,ic,iid,ie,ig):
    """

    Parameters:
    -----------
    i : int
      index in the array x
    x : array of floats
    ip : int (0,1,2,3,4, -1,-2,-3,-4)
      censoring information of x-array, see bhk function below

    Return:
    -------
    icoeff : array of int
      concordance information of x[i]

    """

    ntot = len(x)
    icoeff = np.zeros(ntot)
    for j in range(ntot):
        icoeffj = 0
        if (x[i] < x[j]):
            if ((ip[i] != ia) & (ip[i] != ib) & (ip[i] != ic) &
                    (ip[j] != iid) & (ip[j] != ie)  & (ip[j] != ig) ):
                icoeffj = 1
        if (x[i] > x[j]):
            if ((ip[i] != iid) & (ip[i] != ie) & (ip[i] != ig) &
                    (ip[j] != ia) & (ip[j] != ib)  & (ip[j] != ic) ):
                icoeffj = -1
        icoeff[j] = icoeffj

    return icoeff



def bhk(ind,xx,yy):
    """

    Parameters:
    -----------
    ind : float array (0,1,2,3,4,-1,-2,-3,-4)
       censoring information of x-array and y-array
      (0 - detected; 1 - lolim in y; 2 - lolim in x; 3 - lolim in x and y; 4 - x lolim and y uplim)
      (minus sign: the same, just uplim <--> lolim)
    xx : float array
    yy : float array

    Return:
    -------
    tau : float
      tau-value
    prob : float
      significance level that x and y are not correlated under the Gaussian distribution (as in asurv)
    z : float

    """

    ntot = len(xx)
    sis = 0.
    asum = 0.
    bsum = 0.
    aasum = 0.
    bbsum = 0.

    x = xx
    y = yy
    ip = ind


    for i in range(ntot):

        ia = 2
        ib = 3
        ic = 4
        iid = -2
        ie = -3
        ig = -4
        iaa = coeff(i, x, ip, ia, ib, ic, iid, ie, ig)

        ia = 1
        ib = 3
        ic = -4
        iid = -1
        ie = -3
        ig = 4
        ibb = coeff(i, y, ip, ia, ib, ic, iid, ie, ig)


        for j in range(ntot):
            ijaa = iaa[j]
            ijbb = ibb[j]
            #
            sis = sis + ijaa*ijbb
            asum = asum+ijaa**2
            bsum = bsum+ijbb**2

            for k in range(ntot):
                ikaa = iaa[k]
                ikbb = ibb[k]
                if (ijaa != 0):
                    if (ikaa != 0):
                        aasum = aasum + ijaa*ikaa
                if (ijbb != 0):
                    if (ikbb != 0):
                        bbsum = bbsum + ijbb*ikbb

    d1 = ntot*(ntot - 1)
    d2 = d1*(ntot - 2)
    alp = 2.*(asum*bsum)/d1
    gam = 4.*((aasum-asum)*(bbsum-bsum))/d2
    var = alp + gam
    tau = 2*sis/alp
    sigma = var**0.5
    z = sis / sigma
    prob = special.erfc(np.abs(z) / 1.4142136)

    return tau,prob,z

