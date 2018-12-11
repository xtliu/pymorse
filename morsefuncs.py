"""
helper functions to generate morse wavelets
"""

import numpy as np
from scipy.special import gamma, gammaln, comb

def morsefreq(ga, be, nout=1):
    """
    Calculate important frequencies for generalized Morse wavelets

    to use
    ------
        fm, fe, fi, cf = morsefreq(gamma, beta)
    parameters
    ----------
        ga: matrix or a scalar
            gamma
        be: matrix of the same size as gamma or a scalar
            beta
        nout: integer
            number of outputs
    returns
    -------
        fm: radian
            the modal or peak frequency
        fe: radian
            the "energy" frequency
        fi: radian
            the instantaneous frequency at the wavelet center
        cf: radian
            curvature of fi
    ___________________________________________________________________
    note
    ----
        for beta=0, the "wavelet" becomes an analytic lowpass filter, 
        and FM is not defined in the usual way.  
        Instead, FM is defined as the point at which the filter has decayed to 
        one-half of its peak power. 
    source
    ------
    - Lilly and Olhede (2009).  Higher-order properties of analytic wavelets.  
    IEEE Trans. Sig. Proc., 57 (1), 146--160.
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsefreq.m
    
    """
    # make sure inputs are of type numpy array
    ga = np.array(ga)
    be = np.array(be)

    fm = np.exp(np.multiply(np.divide(1,ga), (np.log(be)-np.log(ga))))
    if np.size(be)==1:
        if be==0:
            fm = np.power((np.log(2)), np.divide(1,ga)) 
    else:
        fm[be==0] = np.power((np.log(2)), np.divide(1,ga[be==0])) # Half-power point

    fe = np.multiply(np.divide(1,2**np.divide(1,ga)), np.divide(gamma(np.divide(2**be+2,ga)),gamma(np.divide(2**be+1,ga))))
    fi = np.divide(gamma(np.divide(be+2,ga)),gamma(np.divide(be+1,ga)))
    
    if nout==1:
        return fm
    elif nout==2:
        return fm, fe 
    elif nout==3:
        return fm, fe, fi
    elif nout == 4:
        m2, n2, k2 = morsemom(2, ga, be, 3)
        m3, n3, k3 = morsemom(3, ga, be, 3)
        cf = -np.divide(k3,k2**(3/2))
        return fm, fe, fi, cf
    else:
        print("number of outputs from function 'morsefreq' can only be 1, 2, 3, or 4")
        return


def morseafun(ga, be, k=1, nmlz='bandpass'):

    if nmlz=='bandpass':
        om = morsefreq(ga, be)    
        a = np.divide(2, np.exp(np.multiply(be,np.log(om)) - np.power(om,ga)))
        if np.size(be) > 1:
            a[be==0] = 2
    elif nmlz=='test':
        a = np.sqrt(np.divide(2*np.pi*np.multiply(ga,2**np.divide(2*be+1,ga)), gamma(np.divide(2*be+1,ga))))
    elif nmlz=='energy':
        r = np.divide(2*be+1,ga)
        a = np.sqrt(2*np.pi*np.multiply(np.multiply(ga, (2**r)),np.exp(gammaln(k)-gammaln(k+r-1))))
    
    return a


def morsemom(p, ga, be, nargout):
    """
    MORSEMOM: frequency-domain moments of generalized Morse wavelets

    morsemom is a low-level function called by several other Morse wavelet functions.

    [mp, np, kp, lp] = morsemom(p, gamma, beta) computes the pth order frequency-
    domain moment M and energy moment N of the lower-order generalized
    Morse wavelet specified by parameters gamma and beta. it also returns the pth order
    cumulant kp and the pth order energy cumulant lp.

    the pth moment and energy moment are defined as:
        mp = 1/(2pi) int omega^p psi(omega)     d omega
        np = 1/(2pi) int omega^p |psi(omega)|.^2    d omega
    respectively, where omega is the radian frequency. these are evaluated
    using the bandpass normalization, which has max(abs(psi(omega)))=2.

    input parameters must be either matrices of the same size of some may be matrices
    and the others scalars

    if you just want mp, just mp and np, or just mp, np, and kp, you can input a different
    value of nargout specifying the number of outputs you want. we would suggest doing all four.

    modified from J.M. Lilly, above comments taken from J.M. Lilly
    (for details, see Lilly and Olhede (2009) Higher-order properties of analytic wavelets)
    """

    m = morsemom1(p, ga, be)
    p = np.array(p)
    # there are two separate calculations of n, neither of which
    # is commented out so I'm going with the latter calcuation
    n = np.multiply(np.divide(2, 2**(np.divide(1+p, ga))), morsemom1(p, ga, 2*be))

    mcell = []
    for i in range(maxmax(p)+1):
        mcell.append(np.array(morsemom1(i, ga, be)))
    kcell = mom2cum(mcell)

    if np.size(p)==1:
        k = kcell[p]
    else:
        k = np.zeros(np.shape(m))
        for i in range(m.size):
            k[i] = kcell[i]

    ncell = []
    for i in range(maxmax(p)+1):
        ncell.append(np.array(np.multiply(np.divide(2, 2**(np.divide(1+i, ga))), morsemom1(i, ga, 2*be))))
    lcell = mom2cum(ncell)
    if np.size(p)==1:
        l = lcell[p]
    else:
        l = np.zeros(np.shape(m))
        for i in range(len(m)):
            l[i] = lcell[i]

    if nargout == 1:
        return m

    if nargout == 2:
        return m, n

    if nargout == 3:
        return m, n, k

    if nargout == 4:
        return m, n, k, l

def maxmax(x):
    # find the maximum finite component 
    x = np.array(x)
    x[x==np.inf] = 0
    m = np.max(x)
    return m


def morsemom1(p, ga, be):
    m = np.multiply(morseafun(ga, be), morsef(ga, be+p))
    m = np.array(m)
    return m

def morsef(ga, be):
    f_part1 = np.divide(1, 2*np.pi*ga)
    f_part2 = gamma(np.divide(be+1, ga))
    f = np.multiply(f_part1, f_part2)
    return f

def mom2cum(cell):
    """
    mom2cum converts moments to cumulants. the unfortunate function name and following
    variable names were not our doing.

    [k0, k1, ..., kn] = mom2cum(m0, m1, ..., mn) converts the first N moments
    m0, m1, ..., mn into the first N cumulants k0, k1, ..., kn

    mn and kn are all scalars or arrays of the same size

    you can also input a list of lists (which is what morsemom does) and get out
    a similar list of lists (we think)

    usage: kcell = mom2cum(mcell)

    modified from J. M. Lilly
    (for details, see Lilly and Olhede (2009) Higher-order properties of analytic wavelets)
    """
    mom = cell
    cum = np.zeros(np.shape(mom))
    cum[0] = np.log(mom[0])

    for n in range(1, len(mom)):
        coeff = np.zeros(np.shape(cum[0]))
        for k in range(1, n):
            coeff = coeff + np.multiply(comb(n-1, k-1), np.multiply(cum[k], np.divide(mom[n-k], mom[0])))
        cum[n] = np.divide(mom[n], mom[0]) - coeff

    return cum

def mom2cum_test():
    m = np.random.normal(size=4)
    m = np.array([1,2,3,4])
    k = np.zeros((4,1))
    k[0] = m[0]
    k[1] = m[1] - m[0]**2
    k[2] = 2*m[0]**3 - 3*m[0] * m[1] + m[2]
    k[3] = -6*m[0]**4 + 12*m[0]**2 * m[1] - 3*m[1]**2 - 4*m[0] * m[2] + m[3]
    
    k_test = mom2cum(m)
    print(k_test, k)
    return None
    