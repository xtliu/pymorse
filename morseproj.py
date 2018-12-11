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

# test code lol maybe finish later
#def morseproj_test():
#    ga1 = np.linspace(2,9,8)
#    be1 = np.linspace(1,10,10)
#    [ga, be] = np.meshgrid(ga1,be1)
#    om = morsefreq(ga, be)
#
#    dom = 0.01;
#    om = np.linspace(0,20,2001)
#    om2 = om[..., np.newaxis, np.newaxis].T
#
#a = np.linspace(2,9,8)
#b = np.linspace(1,10,10)
#[a2,b2] = np.meshgrid(a,b)
#om = np.linspace(0,20,2001)
#om2 = om[..., np.newaxis, np.newaxis].T
#print(om2.shape)