import numpy as np
from scipy.special import gamma
import moresemom

def morsefreq(ga, be):
    """
    MORSEFREQ  Frequency measures for generalized Morse wavelets. [with F. Rekibi]

    for beta=0, the "wavelet" becomes an analytic lowpass filter, and FM 
    is not defined in the usual way.  Instead, FM is defined as the point
    at which the filter has decayed to one-half of its peak power. 

    to use
    ------
        fm, fe, fi, cf = morsefreq(gamma, beta)

    parameters
    ----------
        ga: matrix or a scalar
            gamma
        be: matrix of the same size as gamma or a scalar
            beta

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
    source
    ------
    - Lilly and Olhede (2009).  Higher-order properties of analytic wavelets.  
    IEEE Trans. Sig. Proc., 57 (1), 146--160.
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    """

    if be == 0:
        fm = np.multiply((np.log(2)), np.divide(1,ga)) #%Half-power point
    else:
        fm = np.exp(np.multiply(np.divide(1,ga), (np.log(be)-np.log(ga))))

    fe = np.multiply(np.divide(1,2**np.divide(1,ga)), np.divide(gamma(np.divide(2**be+2,ga)),gamma(np.divide(2**be+1,ga))))

    fi = np.divide(gamma(np.divide(be+2,ga)),gamma(np.divide(be+1,ga)))

    m2, n2, k2 = morsemom(2, ga, be)
    m3, n3, k3 = morsemom(3, ga, be)
    cf = -np.divide(k3,k2**(3/2))

    return fm, fe, fi, cf
