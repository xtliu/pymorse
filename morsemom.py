import numpy as np
from scipy.special import gamma, comb

def morsemom(p, ga, be, nargout):
    m = morsemom1(p, ga, be)
    # there are two separate calculations of n, neither of which
    # is commented out so I'm going with the latter calcuation

    n = np.multiply(np.divide(2, 2**(np.divide(1+p, ga))), morsemom1(p, ga, 2*be))

    mcell = []
    for i = 0:maxmax(p):
        mcell.append(np.array(morsemom1(i, ga, be)))
    kcell = mom2cum(mcell)
    if len(p) == 1:
        k = kcell[p]
    else:
        k = np.zeros(np.shape(m))
        for i in range(len(m)):
            k[i] = kcell[i]

    ncell = []
    for i = 0:maxmax(p):
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
    m = np.multiply(morseafun(ga, be), morsef(ga, be+p))
    return m

def morsef(ga, be):
    f_part1 = np.divide(1, 2*np.pi*ga)
    f_part2 = gamma(np.divide(be+1, ga))
    f = np.multiply(f_part1, f_part2)
    return f

def mom2cum(cell):
    cum = []
    mom = cell
    cum.append(np.log(mom[0]))

    for n in range(1, len(mom)):
        coeff = np.zeros(np.shape(cum[0]))
        for k in range(1, n):
            coeff = coeff + np.multiply(comb(n-1, k-1), np.multiply(cum[k], np.divide(mom[n-k], mom[0])))
        cum[n] = np.divide(mom[n], mom[0]) - coeff

    return cum