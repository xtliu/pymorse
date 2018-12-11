import numpy as np
from scipy.special import gamma, comb
def morsemom(p, ga, be, nargout):
    """
    MORSEMOM: frequency-domain moments of generalized Morse wavelets

    morsemom is a low-level function called by several other Morse wavelet functions.

    [mp, np, kp, lp] = morsemom(p, gamma, beta) computes the pth order frequency-
    domain moment M and enerfy moment N of the lower-order generalized
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
    # there are two separate calculations of n in Lilly's code, neither of which
    # is commented out so I'm going with the second calcuation
    n = np.multiply(np.divide(2, 2**(np.divide(1+p, ga))), morsemom1(p, ga, 2*be))

    mcell = []
    for i in range(maxmax(p)):
        mcell.append(np.array(morsemom1(i, ga, be)))
    kcell = mom2cum(mcell)
    if len(p) == 1:
        k = kcell[p]
    else:
        k = np.zeros(np.shape(m))
        for i in range(len(m)):
            k[i] = kcell[i]

    ncell = []
    for i in range(maxmax(p)):
        ncell.append(np.array(np.multiply(np.divide(2, 2**(np.divide(1+i, ga))), morsemom1(i, ga, 2*be))))
    lcell = mom2cum(ncell)
    if len(p) == 1:
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
    # find the max of the finite terms in x
    # does the same thing as Lilly's function maxmax
    y = np.isfinite(x)
    idx = []
    count = 0
    for val in y:
        if val == True:
            idx.append(count)
        count += 1
    new_x = []
    for i in idx:
        new_x.append(x[i])
    b = max(new_x)
    return b

def morsemom1(p, ga, be):
    # extra calculation made into separate function bc things
    # were getting messy
    m = np.multiply(morseafun(ga, be), morsef(ga, be+p))
    return m

def morsef(ga, be):
    # returns the generalized Morse wavelet first moment "f"
    # of the lower-order generalized Morse wavelet specified
    # by params gamma and beta.
    # modified from J. M. Lilly
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
    cum = []
    mom = cell
    cum.append(np.log(mom[0]))

    for n in range(1, len(mom)):
        coeff = np.zeros(np.shape(cum[0]))
        for k in range(1, n):
            coeff = coeff + np.multiply(comb(n-1, k-1), np.multiply(cum[k], np.divide(mom[n-k], mom[0])))
        cum[n] = np.divide(mom[n], mom[0]) - coeff

    return cum