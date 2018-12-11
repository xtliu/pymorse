"""
morse wavelet transformation
"""

import numpy as np
from morsewave import morse_wave
from scipy.fftpack import fft, ifft

def morse_transform(x, ga, be, fs, K=1, nmlz='bandpass', fam='primary'):
    """
    continuous morse wavelet transform
    generate wavelets based on the input parameters

    """
    N = np.shape(x)
    psi, psif = morse_wave(N, ga, be, fs, K=1, nmlz='bandpass', fam='primary')
    w = cont_transform(x, psif)

    return w


def cont_transform(x, psif):
    """
    computes continuous wavelet transform based on the given signal and wavelets
    parameters
    ----------
        x: array
            time series
        psi: array
            morse wavelets in time domain
    return
    ------
    
    """

    # process the input data 
    # Unitary transform normalization
    if sum(np.isreal(x))==np.size(x):
        x /= np.sqrt(2)
<<<<<<< Updated upstream
        
=======
>>>>>>> Stashed changes
    X = fft(x)
    T = np.multiply(X, psif)
    t = ifft(T)

    return t



