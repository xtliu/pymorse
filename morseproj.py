# MORSEPROJ projection coefficient for two generalized Morse wavelets
# MORSEPROJ is a low-level function called by MORSEWAVE
#
# B = MORSEPROJ(GA, BE1, BE2) returns the projection coefficient B for the
# projection of one generalized Morse wavelet onto another.
#
# The first generalized Morse has parameters GA and BE1, and the second
# has parameters GA and BE2.
#
# 'morseproj --t' runs a test
#
# Usage: b = moresproj(ga, be1, be2)
#
# adapted from J.M. Lilly
# (for details, see Lilly and Olhede (2009) Higher-order properties of analytic wavelets)
import numpy as np
from scipy.special import gamma

def morseproj(ga, be1, be2):
    numerator = np.multiply(np.array(morseafun(ga, be1, 'energy')), np.array(morseafun(ga, be2, 'energy')))
    denominator = 2**(np.divide(be1+be2+1, ga))
    c = np.divide(numerator, denominator)
    b_intermediate = np.multiply(c, np.divide(1, 2*np.pi*ga))
    b_gamma = gamma(np.divide(be1+be2+1, ga))
    b = np.multiply(b_intermediate, b_gamma)
    return b