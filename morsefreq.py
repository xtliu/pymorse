import numpy as np
import moresemom

def morsefreq(gamma, beta):
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
        gamma: matrix or a scalar
        beta: matrix of the same size as gamma or a scalar

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

    if beta == 0:
        fm = np.multiply((np.log(2)), np.divide(1,gamma)) #%Half-power point
    else:
        fm = np.exp(np.multiply(np.divide(1,gamma), (np.log(beta)-np.log(gamma))))

    fe = np.multiply(np.divide(1,2**np.divide(1,gamma)), np.divide(gamma(np.divide(2**beta+2,gamma)),gamma(np.divide(2**beta+1,gamma))))

    fi = np.divide(gamma(np.divide(beta+2,gamma)),gamma(np.divide(beta+1,gamma)))

    m2, n2, k2 = morsemom(2, gamma, beta)
    m3, n3, k3 = morsemom(3, gamma, beta)
    cf = -np.divide(k3,k2**(3/2))

    return fm, fe, fi, cf

