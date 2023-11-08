#-f -a --compile-args=-DCYTHON_TRACE_NOGIL=1 
cimport cython
cython: infer_types=True
cython: profile=True
import warnings
warnings.filterwarnings("ignore")

from REVEALER.ccal import * 
import numpy as np
cimport numpy as np
import math
cdef np.double_t EPS = np.finfo(float).eps
import sys

import pandas as pd

from libc.float cimport DBL_EPSILON
from libc.limits cimport INT_MAX
from libc.math cimport exp, log, sqrt, M_PI, fabs, isnan, fmin, fmax, round, isnan
from libc.stdlib cimport rand, malloc, free, srand
from libc.stdio cimport printf
from cython.parallel import prange
from scipy import stats


cdef extern from "limits.h":
    int RAND_MAX
cdef int RANDOM_SEED = 20121020

cdef struct density:
    np.double_t d
    np.uint32_t* cnt
    
cdef np.double_t DELMAX=1000

import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm.notebook import tqdm
import time
import matplotlib.colors as clr
import multiprocessing as mp
import warnings
import pickle
import random
import os
import tarfile

plt.switch_backend('agg')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline density VR_den_bin(
    int n,              #Size of example
    int nb,             #nb to calculate bcv
    np.double_t* x,     #Pointer to list x
    np.double_t x_min,  #Minumum value of list x
    np.double_t x_max   #Maximum value of list x
    ) nogil:

    '''
    Function to calculate bin dencity
    '''

    cdef int i, j, ii, jj, iij
    cdef density ret
    ret.cnt = <np.uint32_t*> malloc(nb * sizeof(np.uint32_t))
    for i in range(nb):
        ret.cnt[i] = 0
    
    cdef np.double_t rang = (x_max - x_min) * 1.01
    ret.d = rang / <np.double_t>nb

    cdef np.double_t x_i, x_j
    for i in range(1,n):
        ii = (int)(x[i] / ret.d)
        for j in range(i):
            jj = (int)(x[j] / ret.d)
            iij = (ii - jj)
            iij = iij if iij >= 0 else -iij
            if(ret.cnt[iij] == INT_MAX):
                printf("maximum count exceeded in pairwise distance binning\n")
            ret.cnt[iij] += 1
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline np.double_t VR_bcv_bin(
    int n,          #N value
    int nb,         #nb value to calculate bcv
    density den,    #density
    np.double_t h   #h value
    ) nogil:

    '''
    Funnction to calculate bcv
    '''

    cdef np.double_t total_sum = 0.0
    cdef np.double_t delta, term
    cdef np.double_t hh = h / 4.0
    cdef int i
    for i in range(nb):
        delta = (i * den.d / hh) 
        delta *= delta
        if delta >= DELMAX:
            break
        term = exp(-1.0 * delta / 4.0) * (delta * delta - 12.0 * delta + 12.0)
        total_sum += term * den.cnt[i]
    return 1.0 / (2.0 * n * hh * sqrt(M_PI)) + total_sum / (64.0 * n * n * hh * sqrt(M_PI))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline np.double_t r_var(
    np.double_t* x,     #Pointer to list
    int n               #N value
    ) nogil:
    
    '''
    Function to calculate variance
    '''

    cdef np.double_t mean = 0.0
    cdef int i
    for i in range(n):
        mean += x[i]
    mean = mean / n
    
    cdef np.double_t out = 0.0
    
    for i in range(n):
        out += (x[i] - mean) ** 2
    return out / (n - 1.0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline np.double_t sign(
    np.double_t x   #Value to be get sign
    ) nogil:

    '''
    Function to get sign of a double value
    '''

    if x < 0:
        return -1.0
    elif x == 0.0:
        return 0.0
    else:
        return 1.0
    
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int imin(
    int x,  #X value 
    int y   #Y value
    ) nogil:

    '''
    Function to compare two value and return smaller one
    '''

    if x < y:
        return x
    else: 
        return y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline int imax(
    int x,  #X value 
    int y   #Y value
    ) nogil:

    '''
    Function to compare two value and return larger one
    '''

    if x > y:
        return x
    else: 
        return y
    
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int rint(
    np.double_t x   #Value to be converted to int
    ) nogil:

    '''
    Functionn to make double to int
    '''

    return <int> round(x)

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.double_t findmin(
    np.double_t[:] arr,     #List to find minimum value
    int size                #Size of list
    ) nogil:

    '''
    Function to find minimum value in a list
    '''

    cdef np.double_t minVal = arr[0]
    cdef int i = 0
    for i in range(size):
        if arr[i] < minVal:
            minVal = arr[i]
    return minVal

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef np.double_t findmax(
    np.double_t[:] arr,     #List to find minimum value 
    int size                #Size of list
    ) nogil:

    '''
    Function to find maximum value in a list
    '''

    cdef np.double_t minVal = arr[0]
    cdef int i = 0
    for i in range(size):
        if arr[i] > minVal:
            minVal = arr[i]
    return minVal

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef np.double_t calc_bandwidth(
    np.double_t[:] x_view,  #List of double
    int n,                  #Size of list
    int nb=1000             #nb value to calculate bcv
    ) nogil:

    '''
    Function to calculate bandwidth
    '''

    cdef int i = 0
    cdef np.double_t x_min = findmin(x_view,n)
    cdef np.double_t x_max = findmax(x_view,n)
    cdef np.double_t* x_ptr = &x_view[0]
    cdef np.double_t ret = 0.0
    
    ret = py_bcv(x_ptr, n, x_min, x_max, nb)
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline np.double_t py_bcv(
    np.double_t* x,     #Pointer to list x
    int n,              #Size of sample
    np.double_t x_min,  #Minimum value in list x
    np.double_t x_max,  #Maximum value in list x
    int nb=1000         #nb to calculate bcv
    ) nogil:

    '''
    Cython version of bcv
    '''

    cdef np.double_t h, lower, upper
    cdef density den
    if(n == 0):
        printf("'x' has length zero\n")
        return -1
    hmax = 1.144 * sqrt(r_var(x, n)) * (n**(-1.0/5.0)) * 4 #Why *4?
    lower = 0.1 * hmax
    upper = hmax
    den = VR_den_bin(n, nb, x, x_min, x_max)
    
    # Optimize
    cdef np.double_t x1 = lower, x2 = upper, xatol = 0.1*lower
    cdef int maxfun = 500, num
    cdef np.double_t sqrt_eps = sqrt(2.2e-16), golden_mean = 0.5 * (3.0 - sqrt(5.0))
    cdef np.double_t a, b, nfc, xf, fulc, 
    cdef np.double_t rat, e, fx, ffulc, fnfc, xm, tol1, tol2, r, p, q, golden, si, fu

    a = x1; b = x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    h = xf
    
    fx = VR_bcv_bin(n, nb, den, h)
    num = 1

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * fabs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    while (fabs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1.0
        # Check for parabolic fit
        if fabs(e) > tol1:
            golden = 0.0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = fabs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((fabs(p) < fabs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                h = xf + rat

                if ((h - a) < tol2) or ((b - h) < tol2):
                    si = sign(xm - xf) + ((xm - xf) == 0.0)
                    rat = tol1 * si
            else:      # do a golden section step
                golden = 1.0

        if golden:  # Do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = sign(rat) + (rat == 0.0)
        h = xf + si * fmax(fabs(rat), tol1)
        fu = VR_bcv_bin(n, nb, den, h)
        num += 1

        if fu <= fx:
            if h >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = h, fu
        else:
            if h < xf:
                a = h
            else:
                b = h
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = h, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = h, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * fabs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            break
    h = xf
    free(den.cnt)

    return h

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef np.double_t local_pearsonr(
    np.double_t[:] x,   #list x
    np.int_t[:] y,      #list y
    int n               #size of list above
    ) nogil:

    '''
    Local function to calculater perason correlation without gil 
    '''

    cdef np.double_t mx, my, r_num = 0, r_den, r
    cdef np.double_t sum_x = 0, sum_y = 0
    cdef np.double_t xm, ym
    cdef int i
    mx = x[0]
    my = y[0]
    for i in range(1, n):
        mx += x[i]
        my += y[i]
    mx = mx / n
    my = my / n
    
    for i in range(n):
        xm = x[i] - mx
        ym = y[i] - my
        r_num += xm * ym
        sum_x += xm ** 2
        sum_y += ym ** 2
    r_den = sqrt(sum_x * sum_y)
    r = r_num / r_den
    return r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef np.double_t binaryInformationCoefficient_cython(
    np.double_t[:] y,           #Target
    np.int_t[:] x,              #Features to calculate correlation
    int k,                      #K value for kernel
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    int neighborhood,           #Multiplication for bandwidth
    np.double_t deltay,
    np.double_t miny,
    np.double_t jitter=1E-10    #jitter to be added
    ) nogil:


    """ Calculate IC between continuous 'y' and binary 'x' """
    cdef np.double_t  cor, y_bandwidth, sigma_y, deltak
    cdef np.double_t  term_2_pi_sigma_y, integral, mutual_information
    cdef np.double_t  pX_0, pX_1, pysum, py0sum, py1sum, p_y_xsum
    cdef int y_d, i, j, sumx, grid
    cdef np.double_t* normy
    cdef np.double_t* kernal
    cdef np.double_t* p_y_total
    cdef np.double_t* p_y_1
    cdef np.double_t* p_y_0

    sumx = 0
    i = 0
    for i in range(size):
        sumx += <int>x[i]
    
    # Calculate bandwidth
    cor = local_pearsonr(y, x, size)
    
    y_bandwidth = bandwidth * (bandwidth_mult * (1 + (bandwidth_adj) * fabs(cor)))

    deltak = y_bandwidth * neighborhood
    grid = <int>(round(deltay/deltak*k))
    
    if grid < 2*k+1:
        grid = 2*k+1
    
    normy = <np.double_t*> malloc(size * sizeof(np.double_t))
    for i in range(size):
        normy[i] = (y[i] - miny) / deltay * (grid - 1)

    # Prepare grids
    p_y_total = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_y_0 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_y_1 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    
    sigma_y = (grid * y_bandwidth) / deltay
    term_2_pi_sigma_y = 2.0 * M_PI * sigma_y
    kernel = <np.double_t*> malloc(2 * k * sizeof(np.double_t))
    i = 0
    for i in range(2*k):
         kernel[i] = exp(-0.5*(((<np.double_t>i-<np.double_t>k)/sigma_y)**2))/term_2_pi_sigma_y
    
    i = 0
    for i in range(grid):
        p_y_total[i] = 0.0
        p_y_1[i] = 0.0
        p_y_0[i] = 0.0
    
    i = 0
    for i in range(size):
        y_d = <int>(round(normy[i]))
        for j in range(2*k):
            index = y_d - k + j
            if index < 0:
                continue
            if index >= grid:
                continue
            p_y_total[index] += kernel[j]
            if 0 == x[i]:
                p_y_0[index] += kernel[j]
            else:
                p_y_1[index] += kernel[j]

    pysum = 0.0
    p_y_xsum = 0.0
    
    for i in range(grid):
        p_y_total[i] = p_y_total[i] + EPS
        p_y_0[i] = p_y_0[i] + EPS
        p_y_1[i] = p_y_1[i] + EPS
        pysum += p_y_total[i]
        p_y_xsum += p_y_0[i] +  p_y_1[i]

    for i in range(grid):
        p_y_total[i] = p_y_total[i]/pysum
        p_y_0[i] = p_y_0[i]/p_y_xsum
        p_y_1[i] = p_y_1[i]/p_y_xsum    
        
    mutual_information = 0
    
    pX_1 = <double>sumx/<double>size
    pX_0 = <double>(size-sumx)/<double>size

    mutual_information = 0.0
    
    integral = 0.0
    for i in range(grid):    
        integral = integral + (p_y_0[i] * log(p_y_0[i]/(p_y_total[i] * pX_0)))
      
    mutual_information +=  integral
    
    integral = 0.0    
    for i in range(grid):    
        integral = integral + (p_y_1[i] * log(p_y_1[i]/(p_y_total[i] * pX_1)))

    mutual_information +=  integral
    
    free(p_y_total)
    free(p_y_0)
    free(p_y_1)
    free(kernel)
    free(normy)

    return (cor/fabs(cor)) * sqrt(1.0 - exp(-2.0 * mutual_information))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef np.double_t ConditionalInformationCoefficient_cython(
    np.double_t[:] y,           #Target
    np.int_t[:] x,              #Feature that correlation is calculated
    np.int_t[:] z,              #Feature that CIC depend on
    int k,                      #K value for kernel
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    int neighborhood,           #Multiplication for bandwidth
    np.double_t deltay,
    np.double_t miny,    
    np.double_t jitter=1E-10    #jitter to be added
    ) nogil:
    """ 
    Calculate IC between continuous 'y' and binary 'x' given condition 'z' 
    """

    cdef np.double_t  cor, sigma_y, term_2_pi_sigma_y, y_bandwidth, integral, deltak
    cdef np.double_t  conditional_mutual_information, pX0Z0, pX0Z1, pX1Z0, pX1Z1, p_Z1, p_Z0
    cdef np.double_t  pysum, pyx0z0sum, pyx0z1sum, pyx1z0sum ,pyx1z1sum ,pyz0sum ,pyz1sum, p_y_xsum, pyxzsum, pyxsum, pyzsum
    cdef np.double_t* kernel
    cdef np.double_t* normy
    cdef np.double_t* p_y_total
    cdef np.double_t* p_x0z0
    cdef np.double_t* p_x0z1
    cdef np.double_t* p_x1z0
    cdef np.double_t* p_x1z1
    cdef np.double_t* p_z0
    cdef np.double_t* p_z1
    cdef np.double_t* p_x0
    cdef np.double_t* p_x1
    cdef int* saver
    cdef int y_d, i, j, sumx, x0z0Count, x1z0Count, x0z1Count, x1z1Count, grid,sumz

    cor = local_pearsonr(y, x, size)
    x0z0Count = 0
    x1z0Count = 0
    x0z1Count = 0
    x1z1Count = 0

    i = 0
    for i in range(size):
        if x[i] == 0 and z[i] == 0:
            x0z0Count += 1
        elif x[i] == 0 and z[i] == 1:
            x0z1Count += 1
        elif x[i] == 1 and z[i] == 0:
            x1z0Count += 1
        elif x[i] == 1 and z[i] == 1:
            x1z1Count += 1

    sumz = 0
    for i in range(size):
        sumz += <int>z[i]

    saver = <int*> malloc(4 * sizeof(int))
    for i in range(4):
        saver[i] = 0

    if x0z0Count > 1:
        saver[0] = 1
    if x0z1Count > 1:
        saver[1] = 1
    if x1z0Count > 1:
        saver[2] = 1
    if x1z1Count > 1:
        saver[3] = 1

    y_bandwidth = bandwidth * (bandwidth_mult * (1 + (bandwidth_adj) * fabs(cor)))
    deltak = y_bandwidth * neighborhood
    grid = <int>(round(deltay/deltak*k))
    #printf("grid is %.d\n",grid)
    
    if grid < 2*k+1:
        grid = 2*k+1

    # Prepare grids
    p_y_total = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_z0 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_z1 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_x0 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_x1 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_x0z0 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_x0z1 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_x1z0 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    p_x1z1 = <np.double_t*> malloc(grid * sizeof(np.double_t))
    for i in range(grid):
        p_y_total[i] = 0.0
        p_z0[i] = 0.0
        p_z1[i] = 0.0
        p_x0[i] = 0.0
        p_x1[i] = 0.0
        p_x0z0[i] = 0.0
        p_x0z1[i] = 0.0
        p_x1z0[i] = 0.0
        p_x1z1[i] = 0.0

    normy = <np.double_t*> malloc(size * sizeof(np.double_t))
    for i in range(size):
        normy[i] = (y[i] - miny) / deltay * (grid - 1)

    sigma_y = (grid * y_bandwidth) / deltay
    term_2_pi_sigma_y = 2.0 * M_PI * sigma_y
    kernel = <np.double_t*> malloc(2 * k * sizeof(np.double_t))                          
    
    i = 0
    for i in range(2*k):
        kernel[i] = exp(-0.5*(((i-k)/sigma_y)**2))/term_2_pi_sigma_y

    i = 0
    for i in range(size):
        y_d = <int>(round(normy[i]))
        for j in range(2*k):
            index = y_d - k + j
            if index < 0:
                continue
            if index >= grid:
                continue
            p_y_total[index] += kernel[j]
            if 0 == x[i] and 0 == z[i] and saver[0] == 1:
                p_x0z0[index] += kernel[j]
                p_z0[index] += kernel[j]
                p_x0[index] += kernel[j]
            elif 0 == x[i] and 1 == z[i] and saver[1] == 1:
                p_x0z1[index] += kernel[j]
                p_z1[index] += kernel[j]
                p_x0[index] += kernel[j]
            elif 1 == x[i] and 0 == z[i] and saver[2] == 1:
                p_x1z0[index] += kernel[j]
                p_z0[index] += kernel[j]
                p_x1[index] += kernel[j]
            elif 1 == x[i] and 1 == z[i] and saver[3] == 1:
                p_x1z1[index] += kernel[j]
                p_z1[index] += kernel[j]
                p_x1[index] += kernel[j]

    pZ_1 = <double>sumz/<double>size
    pZ_0 = <double>(size - sumz)/<double>size
    conditional_mutual_information = 0.0

    pysum = 0.0
    pyx0z0sum = 0.0
    pyx0z1sum = 0.0
    pyx1z0sum = 0.0
    pyx1z1sum = 0.0
    pyz0sum = 0.0
    pyz1sum = 0.0
    p_y_zsum = 0.0
    pyxsum = 0.0
    pyzsum = 0.0
    pyxzsum = 0.0

    i = 0
    for i in range(grid):
        p_y_total[i] = p_y_total[i] + EPS
        pysum += p_y_total[i]

        p_x0[i] = p_x0[i] + EPS
        p_x1[i] = p_x1[i] + EPS
        pyxsum += p_x0[i] + p_x1[i]

        p_z0[i] = p_z0[i] + EPS
        p_z1[i] = p_z1[i] + EPS
        pyzsum += p_z0[i] + p_z1[i]
        if saver[0] != 0:
            p_x0z0[i] = p_x0z0[i] + EPS
            pyxzsum += p_x0z0[i]
        if saver[1] != 0:
            p_x0z1[i] = p_x0z1[i] + EPS
            pyxzsum += p_x0z1[i]
        if saver[2] != 0:
            p_x1z0[i] = p_x1z0[i] + EPS
            pyxzsum += p_x1z0[i]
        if saver[3] != 0:
            p_x1z1[i] = p_x1z1[i] + EPS
            pyxzsum += p_x1z1[i]        


        #pyxzsum += p_x0z0[i] + p_x0z1[i] + p_x1z0[i] + p_x1z1[i]

    i = 0
    for i in range(grid):
        p_y_total[i] = p_y_total[i]/pysum
        p_z0[i] = p_z0[i]/pyxzsum 
        p_z1[i] = p_z1[i]/pyxzsum 
        p_x0[i] = p_x0[i]/pyxzsum 
        p_x1[i] = p_x1[i]/pyxzsum 

    if saver[0] != 0:
        i = 0
        for i in range(grid):
            p_x0z0[i] = p_x0z0[i]/pyxzsum
        pX0Z0 = <double>x0z0Count/<double>size
        integral = 0.0
        i = 0
        for i in range(grid):
            integral = integral + (p_x0z0[i] * log((p_x0z0[i] * pZ_0)/(p_z0[i] * pX0Z0)))
            
        conditional_mutual_information += integral

    if saver[1] != 0:
        i = 0
        for i in range(grid):
            p_x0z1[i] = p_x0z1[i]/pyxzsum

        pX0Z1 = <double>x0z1Count/<double>size
        integral = 0.0
        i = 0
        for i in range(grid):
            integral = integral + (p_x0z1[i] * log((p_x0z1[i] * pZ_1)/(p_z1[i] * pX0Z1)))

        conditional_mutual_information += integral

    if saver[2] != 0:
        i = 0
        for i in range(grid):
            p_x1z0[i] = p_x1z0[i]/pyxzsum

        pX1Z0 = <double>x1z0Count/<double>size
        integral = 0.0
        i = 0
        for i in range(grid):
            integral = integral + (p_x1z0[i] * log((p_x1z0[i] * pZ_0)/(p_z0[i] * pX1Z0)))
        conditional_mutual_information += integral

    if saver[3] != 0:
        i = 0
        for i in range(grid):
            p_x1z1[i] = p_x1z1[i]/pyxzsum

        pX1Z1 = <double>x1z1Count/<double>size
        integral = 0.0
        i = 0
        for i in range(grid):
            integral = integral + (p_x1z1[i] * log((p_x1z1[i] * pZ_1)/(p_z1[i] * pX1Z1)))
        conditional_mutual_information += integral
        
    free(p_y_total)
    free(p_z0)
    free(p_z1)
    free(p_x0)
    free(p_x1)
    free(p_x0z0)
    free(p_x0z1)
    free(p_x1z0)
    free(p_x1z1)
    free(kernel)
    free(saver)
    free(normy)

    return (cor/fabs(cor)) * sqrt(1.0 - exp(-2.0 * conditional_mutual_information))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef (np.int_t,np.double_t,np.double_t) findBestCIC(
    np.double_t[:] y_in,        #Target
    np.int_t[:] z_in,           #Feature that CIC depend on
    np.int_t[:,:] xs_in,        #List of features
    int k,                      #K value for kernel
    int n,                      #Number of result
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    direction,                  #Direction that feature should match with target
    int grid,                   #Size of grid
    int thread_number,          #Number of thread
    int neighborhood
    ):

    '''
    Function to find feature with best CIC among features
    '''

    cdef np.double_t[:] res = np.zeros(n)
    cdef np.double_t[:] y = np.asarray(y_in)
    cdef np.int_t[:,:] x = np.asarray(xs_in)
    cdef np.int_t[:] z = np.asarray(z_in)
    cdef int i
    cdef int maxseedid = 0
    cdef np.double_t maxCIC = 0.0
    cdef np.double_t miny = findmin(y, size)
    cdef np.double_t deltay = findmax(y, size) - findmin(y, size)
    cdef np.double_t substart = time.time()

    for i in prange(n, nogil=True,num_threads=thread_number):       
        res[i] = ConditionalInformationCoefficient_cython(y, x[i,:], z, k, size, bandwidth, 
                                                          bandwidth_mult, bandwidth_adj, 
                                                          neighborhood, deltay, miny)
    cdef np.double_t spenttime = time.time() - substart
    i = 0
    
    if direction == 'neg':
        maxCIC = 10000
        for i in range(n):
            if res[i] <= maxCIC:
                maxCIC = res[i]
                maxseedid = i + 1
    else:
        maxCIC = -10000
        for i in range(n):
            if res[i] >= maxCIC:
                maxCIC = res[i]
                maxseedid = i + 1
                
    return maxseedid,maxCIC,spenttime

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef np.double_t[:] rankCIC(
    np.double_t[:] y_in,        #Target
    np.int_t[:] z_in,           #Feature that CIC depend on
    np.int_t[:,:] xs_in,        #List of features
    int k,                      #K value for kernel
    int n,                      #Number of result
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    int grid,                   #Size of grid
    int thread_number,          #Number of thread
    int neighborhood
    ):

    '''
    Function to calculate bunch of CIC parallelly
    '''

    cdef int i
    cdef np.double_t[:] res = np.zeros(n)
    cdef np.double_t[:] y = np.asarray(y_in)
    cdef np.int_t[:,:] x = np.asarray(xs_in)
    cdef np.int_t[:] z = np.asarray(z_in)
    cdef np.double_t miny = findmin(y, size)
    cdef np.double_t deltay = findmax(y, size) - findmin(y, size)

    for i in prange(n, nogil=True,num_threads=thread_number):       
        res[i] = ConditionalInformationCoefficient_cython(y, x[i,:], z, k, size, bandwidth, 
                                                          bandwidth_mult, bandwidth_adj, 
                                                          neighborhood, deltay, miny)

    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef (np.int_t,np.double_t) findBestIC(
    np.double_t[:] y_in,        #Target
    np.int_t[:,:] xs_in,        #List of features
    int k,                      #K value for kernel
    int n,                      #Number of result
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    direction,                  #Direction that feature should match with target
    int grid,                   #Size of grid
    int thread_number,          #Number of thread
    int neighborhood
    ):

    '''
    Function to find feature with best IC among all features
    '''

    cdef np.double_t[:] res = np.zeros(n)
    cdef np.double_t[:] y = np.asarray(y_in)
    cdef np.int_t[:,:] x = np.asarray(xs_in)

    cdef int i
    cdef int maxseedid = 0
    cdef np.double_t maxCIC = 0.0
    cdef np.double_t miny = findmin(y, size)
    cdef np.double_t deltay = findmax(y, size) - findmin(y, size)
    
    for i in prange(n, nogil=True,num_threads=thread_number):
        res[i] = binaryInformationCoefficient_cython(y, x[i,:],k,size, bandwidth,bandwidth_mult,
                                                     bandwidth_adj, neighborhood, deltay, miny)
        
    i = 0
    
    if direction == 'neg':
        maxCIC = 10000
        for i in range(n):
            if res[i] <= maxCIC:
                maxCIC = res[i]
                maxseedid = i + 1
    else:
        maxCIC = -10000
        for i in range(n):
            if res[i] >= maxCIC:
                maxCIC = res[i]
                maxseedid = i + 1
    return maxseedid,maxCIC

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef np.double_t[:] rankIC(
    np.double_t[:] y_in,        #Target
    np.int_t[:,:] xs_in,        #List of features
    int k,                      #K value for kernel
    int n,                      #Number of result
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    int grid,                   #Size of grid
    int thread_number,          #Number of thread
    int neighborhood
    ):

    '''
    Function to run multiple IC in parallel
    '''

    cdef np.double_t[:] res = np.zeros(n)
    cdef np.double_t[:] y = np.asarray(y_in)
    cdef np.int_t[:,:] x = np.asarray(xs_in)
    cdef int i
    cdef np.double_t miny = findmin(y, size)
    cdef np.double_t deltay = findmax(y, size) - findmin(y, size)
    
    for i in prange(n, nogil=True,num_threads=thread_number):
        res[i] = binaryInformationCoefficient_cython(y, x[i,:],k,size, bandwidth,bandwidth_mult,
                                                     bandwidth_adj, neighborhood, deltay, miny)
        
    return res

def drawTarget(
    comb    #Combined dataframe 
    ):
    
    '''
    Function to plot target
    '''

    f, ax = plt.subplots(figsize=(10, 1))
    ax = sns.heatmap(comb.iloc[[0]].to_numpy(), cmap='bwr', annot=False, yticklabels=False,
                     xticklabels=False, cbar=False, center=comb.iloc[0].mean())
    return f

def drawFeature(
    comb,           #Combined dataframe
    featurecmap,    #Cmap to plot feature
    seed_name=None, #Seed name to plot
    seedID=None     #Seed id to plot
    ):

    '''
    Function to plot feature
    '''

    if seed_name != None:
        f, ax = plt.subplots(figsize=(10, 1))
        ax = sns.heatmap(comb.loc[[seed_name]].to_numpy(), cmap=featurecmap, annot=False,
                         yticklabels=False, xticklabels=False, cbar=False)
        return f
    else:
        f, ax = plt.subplots(figsize=(10, 1))
        ax = sns.heatmap(comb.iloc[[seedID]].to_numpy(), cmap=featurecmap, annot=False,
                         yticklabels=False, xticklabels=False, cbar=False)
        return f
    
def drawSeed(
    seed,       #Seed list
    seedcmap    #Cmap to draw seed
    ):

    '''
    Function to draw seed
    '''

    f, ax = plt.subplots(figsize=(10, 1))
    ax = sns.heatmap([seed], cmap=seedcmap, annot=False, yticklabels=False, 
                     xticklabels=False, cbar=False)
    return f

def seedCombine(
    currentseed,    #Current seed
    newseed         #New seed to be added
    ):

    '''
    Function to combine two seed
    '''

    seed = []
    for i in range(len(currentseed)):
        if currentseed[i] == 1 or newseed[i] == 1:
            seed.append(1)
        else:
            seed.append(0)

    return seed

cpdef topmatch(
    comb,           #Combined dataframe
    currentseed,    #Current seed to run IC/CIC
    y,              #Target value
    grid,           #Size of grid
    k,              #K value for kernel
    size,           #Number of sample
    bandwidth,      #Value of bandwidth
    bandwidth_mult, #Multiplier of bandwidth
    bandwidth_adj,  #Adjust of bandwidth
    thread_number,  #Number of thread to run
    direction,      #Direction that feature should match with target
    num_top,        #Number of top feature
    prefix,         #Prefix for generated files
    figure_format,  #Format for figures
    ifSeed,         #Indicate if seed exist
    if_pval,        #Indicate if p-values are calculated
    if_bootstrap,   #Indicate if variances are calculated
    if_cluster,     #Indicate if features are clustered
    collapse,       #Indicate if same features are collapsed together for intermediate files
    seedcmap,       #Cmap for seed heatmap
    featurecmap,    #Cmap for feature heatmap
    nitr,           #Itration number for this figure
    out_folder,     #Folder name to put results
    seed_name,      #List of seed names
    target_name,    #Name of target
    gmt,
    locusdic,
    neighborhood
    ):

    '''
    Function to create intermediate file for each run to report top features in each run
    '''

    CICs = []
    pVals = []
    bootstraps = []
    cdef np.double_t miny = findmin(y, size)
    cdef np.double_t deltay = findmax(y, size) - findmin(y, size)
    
    if ifSeed == True:
        cythonseed = np.asarray(currentseed).astype(int)
        rank = topMatches(comb=comb, y=y, grid=grid, k=k, size=size, bandwidth=bandwidth,
                          bandwidth_mult=bandwidth_mult, bandwidth_adj=bandwidth_adj,
                          thread_number=thread_number, neighborhood = neighborhood,seed=cythonseed)

        IC = binaryInformationCoefficient_cython(y, cythonseed, k, size, bandwidth, bandwidth_mult,
                                                     bandwidth_adj, neighborhood, deltay, miny)

        CICs.append(IC)
        if if_bootstrap == True:
            bootstrap = calcBootstrapIC(subcomb = pd.DataFrame([comb.iloc[0].tolist(),currentseed]),
                                        size = size, bandwidth = bandwidth, k = k, 
                                        bandwidth_mult = bandwidth_mult, 
                                        bandwidth_adj = bandwidth_adj, grid = grid, 
                                        thread_number = thread_number, IC = IC, neighborhood = neighborhood)
            bootstraps.append(bootstrap)

        if if_pval == True:
            pVal = calcIndivisualPvalIC(subcomb = pd.DataFrame([comb.iloc[0].tolist(),currentseed]),
                                        size = size, bandwidth = bandwidth, k = k, 
                                        bandwidth_mult = bandwidth_mult, 
                                        bandwidth_adj = bandwidth_adj, grid = grid, 
                                        thread_number =thread_number, IC = IC, neighborhood = neighborhood)
            pVals.append(pVal)

    else:
        currentseed = [0]*len(comb.columns)
        cythonseed = np.asarray(currentseed).astype(int)
        rank = topMatches(comb=comb,y = y,grid=grid,k=k,size = size,bandwidth = bandwidth,
                          bandwidth_mult=bandwidth_mult,bandwidth_adj=bandwidth_adj,
                          thread_number=thread_number, neighborhood = neighborhood)
        CICs.append('')
        if if_bootstrap == True:
            bootstraps.append('')
        if if_pval == True:
            pVals.append('')

    rankdf = pd.DataFrame(rank)
    if direction == 'pos':
        rankdf.sort_values(by=1, ascending=False, inplace=True)
        rank = rankdf.to_numpy()
    else:
        rankdf.sort_values(by=1, ascending=True, inplace=True)
        rank = rankdf.to_numpy()

        
    pilesize = []
    if collapse == True:
        pileres,ICs = pileRank(rank,num_top,prefix,out_folder,nitr)
        pilename = []

        for i in range(len(pileres)):
            pilename.append(pileres[i][0][0])
            pilesize.append(len(pileres[i]))
        rankfeature = pd.concat([comb.iloc[[0]],comb.loc[pilename]])
        CICs = CICs + list(ICs)
    else:
        rankfeature = pd.concat([comb.iloc[[0]],comb.loc[rank[:,0][:num_top]]])
        CICs = CICs + list(rank[:,1][:num_top])
    
    if ifSeed == True:
        if if_pval==True:
            results = prepPvalCIC(comb = comb, cythonseed = cythonseed, size = size, 
                                  bandwidth = bandwidth, k = k, bandwidth_mult = bandwidth_mult, 
                                  bandwidth_adj = bandwidth_adj, grid = grid, 
                                  thread_number =thread_number, neighborhood = neighborhood)
            for i in range(1,len(rankfeature.index.tolist())):
                if i != 1 and CICs[i] == CICs[i-1]:
                    pVals.append(pVals[-1])
                else:
                    #print('CIC:'+str(CICs[i]))
                    pVals.append(calcPvalCIC(results = results, IC = CICs[i], direction = direction))

        if if_bootstrap==True:
            for i in range(1,len(rankfeature.index.tolist())):
                if i != 1 and CICs[i] == CICs[i-1]:
                    bootstraps.append(bootstraps[-1])
                else:
                    bootstrap = calcBootstrap(subcomb = rankfeature.iloc[[0,i],],  
                                              cythonseed = cythonseed, size = size, 
                                              bandwidth = bandwidth, k = k, 
                                              bandwidth_mult = bandwidth_mult, 
                                              bandwidth_adj = bandwidth_adj, grid = grid,
                                              thread_number = thread_number, IC = IC, 
                                              neighborhood = neighborhood)
                    bootstraps.append(bootstrap)
    else:
        if if_pval==True:
            results = prepPvalIC(comb = comb, size = size, 
                                 bandwidth = bandwidth, k = k, bandwidth_mult = bandwidth_mult, 
                                 bandwidth_adj = bandwidth_adj, grid = grid,
                                 thread_number = thread_number, neighborhood = neighborhood)
            for i in range(1,len(rankfeature.index.tolist())):
                if i != 1 and CICs[i] == CICs[i-1]:
                    pVals.append(pVals[-1])
                else:
                    #print('CIC:'+str(CICs[i]))
                    pVals.append(calcPvalIC(results = results, IC = CICs[i], direction = direction))

        if if_bootstrap==True:
            for i in range(1,len(rankfeature.index.tolist())):
                if i != 1 and CICs[i] == CICs[i-1]:
                    bootstraps.append(bootstraps[-1])
                else:
                    bootstrap = calcBootstrapIC(subcomb = rankfeature.iloc[[0,i],], size = size, 
                                                bandwidth = bandwidth, k = k, 
                                                bandwidth_mult = bandwidth_mult, 
                                                bandwidth_adj = bandwidth_adj, 
                                                grid = grid, thread_number = thread_number, 
                                                IC = CICs[i], neighborhood = neighborhood)
                    bootstraps.append(bootstrap)

    savetopfig(plotcomb = rankfeature.copy(), CICs = CICs, cythonseed = cythonseed, 
               seedcmap = seedcmap, featurecmap = featurecmap, prefix = prefix, 
               figure_format = figure_format, pVals = pVals, num_top = num_top, 
               bootstraps = bootstraps.copy(), pilesize = pilesize, nitr = nitr, gmt = gmt,
               out_folder = out_folder, target_name = target_name, seed_name = seed_name,
               locusdic = locusdic)

    if if_cluster == True and collapse == False:
        plotCluster(plotcomb = rankfeature.copy(), CICs = CICs, size = size, 
                    cythonseed = cythonseed, seedcmap = seedcmap, featurecmap = featurecmap,
                    prefix = prefix, figure_format = figure_format, pVals = pVals, 
                    num_top = num_top, bootstraps = bootstraps, nitr = nitr, gmt = gmt,
                    out_folder = out_folder, target_name = target_name, seed_name = seed_name,
                    locusdic = locusdic)

    return comb,rank,rankfeature

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef runBenchmark(target_file='no', # gct file for target(continuous or binary
                  feature_file='no', # gct file for features(binary)
                  seed_file='no', # file for seed, if not provided, feature file is used directly 
                  target_df='no', # dataframe for target
                  feature_df='no', # dataframe for features
                  seed_df='no', # dataframe for seed
                  prefix='test', # prefix for result files 
                  seed_name=None, # names for seed, should be a list of string indicating the name of seed
                  grid=34, # number of grid, default is 34
                  target_name=0, # name/index of target in target file. can be int n for nth row, or string s for row with index s
                  k=5, # size of kernel for k standard deviation away
                  bandwidth_mult=0.65, # multiplication for bandwidth
                  bandwidth_adj=-0.95, # adjustion value for bandwidth
                  direction='pos', # direction that feature should match with target
                  mode='single', # indicate if multiple parameter set is passes. if True, then prefix, k, grid, bandwidth_mult, and bandwidth_adj has to be a list
                  num_top=30, # number of top matches shown in intermediate file
                  low_threshold=3, # lowest threshold that feature with less than this value occurence will be removed
                  high_threshold=100, # highest threshold that feature with less than this value occurence will be removed
                  collapse=True, # indicate if same features are collapsed together for intermediate files
                  normalize='zerobase', # normalize method for target
                  gene_locus='None', # gene_locus file indicating gene name and location of that gene
                  verbose=1, # verbose level(if 0, no report)
                  max_iteration=-1, # maximum of iteration for best CIC discovery, automatic detection by IC value if -1 
                  thread_number=1, # number of core used for parallel computing.
                  figure_format='pdf', # format for result figure
                  subset='no', # if list of string passed, only columns in this list is picked for calculation
                  if_pval=True, # if True, p-values are calculated for all result
                  if_bootstrap=True, # if True, variance by bootstrap is calculated for all result
                  if_cluster=False, # if True, features in intermediate files are clustered with NMF 
                  if_intermediate=False, # if True, intermediate result with top CIC value features are reported
                  out_folder='.', # folder to put output files inside
                  gene_set=None,
                  gene_separator='_',
                  gmt_file = None,
                  alpha = 1,
                  neighborhood = 4,
                  tissue_file = None,
                  gzip = True
                 ):
    
    report = ''

    if verbose != 0:
        print(prefix+' start!')
    report = report + 'Run name: ' + str(prefix) + '\n'
    
    # if / is not at the end of output folder, add it.
    if out_folder[-1] != '/':
        out_folder = out_folder + '/'
    
    # if output folder does not exist, make it
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    start = time.time()
    if (isinstance(target_df,str) or isinstance(feature_df,str)) and \
       (target_file == 'no' or feature_file == 'no') :
        raise Exception("Please put input!")
    if mp.cpu_count() <thread_number:
        if verbose != 0:
            print('Indicated Core Number exceed maximum available core. All core is used')
        thread_number = mp.cpu_count()
    if thread_number == -1:
        thread_number = mp.cpu_count()
        if verbose != 0:
            print('All core('+str(thread_number)+') Used')
    report = report + 'Number of thread used: ' + str(thread_number) + '\n'
    
    if seed_file == 'no':
        seed_file = feature_file
    if isinstance(feature_df,str):
        comb,locusdic = readInput(target_file = target_file, feature_file = feature_file, 
                         seed_file = seed_file, seed_name = seed_name, target_name = target_name,
                         direction = direction, gene_locus = gene_locus,
                         low_threshold = low_threshold, high_threshold = high_threshold,
                         subset = subset, gene_set = gene_set, 
                         gene_separator = gene_separator, thread_number = thread_number,
                         tissue_file = tissue_file)
    else:
        if isinstance(seed_df,str):
            comb = pd.concat([target_df,feature_df], join='inner')
        else:
            comb = pd.concat([target_df,seed_df,feature_df], join='inner')
        rmrow=[]
        for i in comb.index.tolist()[2:]:
            if sum(comb.loc[i].tolist()) <= low_threshold or \
               sum(comb.loc[i].tolist()) >= high_threshold:
                rmrow.append(i)
        comb = comb.drop(index=rmrow)
        comb = comb.rename(index={seed_name[0]: 'firstseed'})
        origcomb = comb.copy()
        #seed_name = ['firstseed']
        locusdic = None
    if verbose != 0:
        print('Time used to read input: %s second(s)'%(int(time.time() - start)))
    
    if gmt_file != None:
        gmt = pd.read_csv(gmt_file,sep='\t',header=None,index_col=0)
        gmt = gmt.drop(columns=gmt.columns[0])
        gmt.index = gmt.index.str.replace("-", "_")
        if (len(list(set(comb.index.tolist())&set(gmt.index.tolist()))) == 0) and verbose != 0:
            print('Names in gmt file is not matching with input file. Setting gmt to None.')
            gmt = None
    else:
        gmt = None

    if mode == 'multi':
        for ind in range(len(prefix)):
            target_df, seed_df, featuresdf = REVEALERInner(prefix = prefix[ind], comb = comb, 
                                                           grid = grid[ind], k = k[ind], 
                                                           bandwidth_mult = bandwidth_mult[ind], 
                                                           bandwidth_adj = bandwidth_adj[ind], 
                                                           direction = direction, num_top = num_top,
                                                           verbose = verbose, 
                                                           max_iteration = max_iteration, 
                                                           collapse = collapse,
                                                           thread_number =thread_number, 
                                                           figure_format = figure_format, 
                                                           if_pval = if_pval, seed_name = seed_name,
                                                           if_bootstrap = if_bootstrap, 
                                                           if_cluster = if_cluster, 
                                                           if_intermediate = if_intermediate, 
                                                           out_folder = out_folder, 
                                                           normalize = normalize, 
                                                           target_name = target_name,
                                                           report = report,
                                                           gmt = gmt,
                                                           locusdic = locusdic,
                                                           alpha = alpha,
                                                           neighborhood = neighborhood,
                                                           tissue_file = tissue_file,
                                                           gzip = gzip)
        
    elif mode == 'single':
        #target_df, seed_df, featuresdf 
        seed, CICs = REVEALERInner(prefix = prefix, comb = comb, grid = grid, 
                                                       k = k, bandwidth_mult = bandwidth_mult, 
                                                       normalize = normalize, 
                                                       bandwidth_adj = bandwidth_adj, 
                                                       direction = direction, num_top = num_top, 
                                                       verbose = verbose, 
                                                       max_iteration = max_iteration,
                                                       thread_number =thread_number, 
                                                       figure_format = figure_format, 
                                                       if_pval = if_pval, seed_name = seed_name, 
                                                       if_bootstrap = if_bootstrap, 
                                                       if_cluster = if_cluster, 
                                                       if_intermediate = if_intermediate, 
                                                       out_folder = out_folder,
                                                       collapse = collapse, 
                                                       target_name = target_name,
                                                       report = report,
                                                       gmt = gmt,
                                                       locusdic = locusdic,
                                                       alpha = alpha,
                                                       neighborhood = neighborhood,
                                                       tissue_file = tissue_file,
                                                       gzip = gzip)
        
    return seed, CICs#target_df, seed_df, featuresdf

@cython.boundscheck(False)
@cython.cdivision(True)   
cpdef REVEALERInner(
    prefix,             #Prefix for generated files
    comb,               #Combined dataframe
    grid,               #Size of grid
    k,                  #K value for kernel
    bandwidth_mult,     #Multiplier of bandwidth
    bandwidth_adj,      #Adjust of bandwidth
    direction,          #Direction that feature should match with target
    num_top,            #Number of top feature
    verbose,            #Verbose state
    max_iteration,      #Maximum of iteration for best CIC discovery
    collapse,           #Indicate if same features are collapsed together for intermediate files
    thread_number,      #Number of thread to run
    figure_format,      #Format for figures
    if_pval,            #Indicate if p-values are calculated
    if_bootstrap,       #Indicate if variances are calculated
    if_cluster,         #Indicate if features are clustered
    if_intermediate,    #Indicate if intermediate files are generated
    out_folder,         #Folder name to put results
    seed_name,          #List of seed names
    normalize,          #Normalization method
    target_name,        #Name of target
    report,
    gmt,
    locusdic,
    alpha,
    neighborhood,
    tissue_file,
    gzip
    ):

    seedcmap = clr.LinearSegmentedColormap.from_list('custom greys', [(.9,.9,.9),(0.5,0.5,0.5)], 
                                                     N=256)
    featurecmap = clr.LinearSegmentedColormap.from_list('custom greys', 
                                                        [(<double>(176)/<double>(255),
                                                        <double>(196)/<double>(255),
                                                        <double>(222)/<double>(255)),
                                                        (0,0,<double>(139)/<double>(255))], N=256)


    newpheno = []
    target = comb.iloc[0].tolist()
    targetmean = np.mean(target)
    targetstd = np.std(target)
    comb.iloc[0] = (target - targetmean)/targetstd
    for i in comb.iloc[0].tolist():
        newpheno.append(np.sign(i)*(abs(i)**alpha))
    comb.iloc[0] = newpheno
    
    if tissue_file:
        patient = pd.read_csv(tissue_file,skiprows=[0,1],sep='\t',index_col=0)
        patient = patient.drop(columns=patient.columns[0])
        patient.columns = patient.columns.str.replace("-", "_")
        patient = patient[comb.columns]
        combpatient = pd.concat([comb.iloc[[0]],patient])
        corr = []
        for i in combpatient.index.tolist()[1:]:
            if combpatient.loc[i].sum() == 0:
                corr.append(-100)
            else:
                corr.append(stats.pearsonr(combpatient.iloc[0],combpatient.loc[i])[0])
        combpatient['corr'] = [1]+corr
        combpatient = combpatient.sort_values('corr',ascending=False).drop(columns= ['corr'])

        patientfig(df = combpatient, seedcmap = seedcmap, featurecmap = featurecmap, 
                   prefix = prefix, figure_format = figure_format, out_folder = out_folder, 
                   target_name = target_name)

    plt.plot(comb.iloc[0].tolist())
    plt.savefig(out_folder + prefix+'_PhenotypeDistribution.'+figure_format,format=figure_format)
    plt.close()

    if verbose != 0:
        if seed_name != None:
            print("Number of features that pass the threshold is: "+str(len(comb.index)-2))
        else:
            print("Number of features that pass the threshold is: "+str(len(comb.index)-1))

    if seed_name != None:
        report = report + 'Number of features passing threshold is: ' + str(len(comb.index)-2)+'\n'
    else:
        report = report + 'Number of features passing threshold is: ' + str(len(comb.index)-1)+'\n'
    

    start= time.time()
    cdef int size = len(comb.iloc[0])
    cdef np.double_t[:] y = np.ascontiguousarray(np.asarray(comb.iloc[0].tolist()))
    cdef np.double_t miny = findmin(y, size)
    cdef np.double_t deltay = findmax(y, size) - findmin(y, size)
    i = 0
    cdef np.double_t bandwidth = calc_bandwidth(y,size)/2
                                                
    cdef np.int_t[:] cythonseed
    
    seedids = []
    CICs = []
    seedIC = []
    pVals = []
    bootstraps = []
    addseed = 0
    
    if grid == None:
        grid = len(comb.columns.tolist())+1
    if verbose != 0:
        print('grid size: '+str(grid))
    if verbose != 0:
        print('bandwidth: '+str(bandwidth))

    report = report + 'Grid size is: ' + str(grid) + '\n'
    report = report + 'Universal bandwidth is: ' + str(bandwidth) + '\n'

    savecomb = comb.copy()
    seedlists = []
    if seed_name == None:
        seedlists.append([0]*len(comb.columns.tolist()))
        CICs.append('')
        if verbose != 0:
            print('seed Search...')
        seedid,IC = findBestIC(y, np.array(comb.iloc[1:].values.astype(int).tolist()),
                             k, int(len(comb.index)-1), size, bandwidth,
                             bandwidth_mult, bandwidth_adj, direction,grid,thread_number, neighborhood)
        if if_pval == True:
            pVals.append('')
            results = prepPvalIC(comb = comb, size = size, bandwidth = bandwidth, k = k, 
                                 bandwidth_mult = bandwidth_mult, bandwidth_adj = bandwidth_adj, 
                                 grid = grid, thread_number = thread_number, neighborhood = neighborhood)
            pVal = calcPvalIC(results = results, IC = IC, direction = direction)
            pVals.append(pVal)
            pVals.append(pVal)
            
        if if_bootstrap == True:
            bootstraps.append('')
            bootstrap = calcBootstrapIC(subcomb = comb.iloc[[0,seedid],], size = size, 
                                        bandwidth = bandwidth, k = k, 
                                        bandwidth_mult = bandwidth_mult, 
                                        bandwidth_adj = bandwidth_adj, grid = grid,
                                        thread_number = thread_number,IC = IC, neighborhood = neighborhood)
            bootstraps.append(bootstrap)
            bootstraps.append(bootstrap)

        CICs.append(IC)
        CICs.append(IC)
        
        if if_intermediate == True:
            topmatch(comb = comb, currentseed = [0]*len(comb.columns.tolist()), y = y, grid = grid, 
                     k = k, size = size, bandwidth = bandwidth, bandwidth_mult = bandwidth_mult, 
                     bandwidth_adj = bandwidth_adj,thread_number =thread_number, 
                     direction = direction, num_top = num_top, prefix = prefix, locusdic = locusdic,
                     figure_format = figure_format, ifSeed = False, if_pval = if_pval, 
                     if_bootstrap = if_bootstrap, if_cluster = if_cluster, collapse = collapse, 
                     seedcmap = seedcmap, featurecmap = featurecmap, nitr = 0, gmt = gmt,
                     out_folder = out_folder, seed_name = seed_name, target_name = target_name, 
                     neighborhood = neighborhood)
        
        currentseed = comb.iloc[seedid].tolist()
        seedlists.append(currentseed)
        addseed = currentseed
        seedids.append(comb.index[seedid])
        if verbose != 0:
            print('Picked seed is: '+ comb.index[seedid])
        report = report + 'Picked seed is: '+ comb.index[seedid] + '\n'
        comb = comb.drop(index = comb.index[seedid])
    else:
        currentseed = comb.loc['firstseed'].tolist()
        seedlists.append(currentseed)
        cythonseed = np.asarray(currentseed).astype(int)
        IC = binaryInformationCoefficient_cython(y, cythonseed,k,size,bandwidth,
                                                 bandwidth_mult,bandwidth_adj, neighborhood, deltay, miny)
        if if_bootstrap == True:
            bootstrap = calcBootstrapIC(subcomb = pd.DataFrame([comb.iloc[0].tolist(),currentseed]), 
                                        size = size, bandwidth = bandwidth, k = k, 
                                        bandwidth_mult = bandwidth_mult, 
                                        bandwidth_adj = bandwidth_adj, grid = grid, 
                                        thread_number = thread_number, IC = IC, neighborhood = neighborhood)
            bootstraps.append(bootstrap)
        if if_pval == True:
            subcomb = comb.copy()
            subcomb.iloc[1] = currentseed
            pVal = calcIndivisualPvalIC(subcomb = pd.DataFrame([comb.iloc[0].tolist(),currentseed]), 
                                        size = size, bandwidth = bandwidth, k = k, 
                                        bandwidth_mult = bandwidth_mult, 
                                        bandwidth_adj = bandwidth_adj, grid = grid,
                                        thread_number = thread_number,IC = IC, neighborhood = neighborhood)
            pVals.append(pVal)

        CICs.append(IC)
        addseed = currentseed
        comb = comb.drop(index=['firstseed'])
        
    
    n=1
    while (max_iteration == -1 or n <= max_iteration) and (len(comb.index.tolist()) > 1):
        
        if verbose != 0:
            print("Iteration" + str(n) + ':')
        cythonseed = np.asarray(currentseed).astype(int)
        #substart = time.time()
        seedid,CIC,spenttime = findBestCIC(y, cythonseed, np.array(comb.iloc[1:].values.astype(int).tolist()),
                         k, int(len(comb.index)-1), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, direction, grid,thread_number, neighborhood)
        
#         if verbose != 0:
#             print('Time used to run one loop: %s second(s)'%(int(time.time() - substart)))
#         report = report + 'Time used to run one loop: %s second(s)'%(int(time.time() - substart)) + '\n'
        
        if verbose != 0:
            print('Time used to run one loop: %s second(s)'%(int(spenttime)))
        report = report + 'Time used to run one loop: %s second(s)'%(int(spenttime)) + '\n'
        
        newseed = comb.iloc[seedid].tolist()
        
        if if_intermediate == True:
            topmatch(comb = comb, currentseed = currentseed, y = y, grid = grid, k = k, size = size,
                     bandwidth = bandwidth, bandwidth_mult = bandwidth_mult, locusdic = locusdic,
                     bandwidth_adj = bandwidth_adj,thread_number =thread_number, 
                     direction = direction, num_top = num_top, prefix = prefix, 
                     figure_format = figure_format, ifSeed = True, if_pval=if_pval, 
                     if_bootstrap=if_bootstrap, if_cluster=if_cluster, collapse = collapse, 
                     seedcmap = seedcmap, featurecmap = featurecmap, nitr = n, gmt = gmt,
                     out_folder = out_folder, seed_name = seed_name, target_name = target_name,
                     neighborhood = neighborhood)  
        
        currentseed = seedCombine(currentseed,newseed)
        seedids.append(comb.index[seedid])
        CICs.append(CIC)
        seedlists.append(currentseed)
        if if_pval == True:
            results = prepPvalCIC(comb = comb, cythonseed = cythonseed, size = size, 
                                  bandwidth = bandwidth, k = k, bandwidth_mult = bandwidth_mult, 
                                  bandwidth_adj = bandwidth_adj, grid = grid,
                                  thread_number =thread_number, neighborhood = neighborhood)
            pVals.append(calcPvalCIC(results = results, IC = CIC, direction = direction))
                                        
        if if_bootstrap == True:
            bootstrap = calcBootstrap(subcomb = comb.iloc[[0,seedid],], cythonseed = cythonseed, 
                                      size = size, bandwidth = bandwidth, k = k, 
                                      bandwidth_mult = bandwidth_mult, 
                                      bandwidth_adj = bandwidth_adj, grid = grid,
                                      thread_number = thread_number, IC = IC, neighborhood = neighborhood)
            bootstraps.append(bootstrap)
            
  
        # Update cythonseed after calculating bootstrap and p-value
        cythonseed = np.asarray(currentseed).astype(int)
        if verbose != 0:
            print("CIC calculated in this round is: " + str(CIC))
            print("Best feature choosen in this round is: "+comb.index[seedid])
        report = report + "Round " + str(n) + '\n'
        report = report + "CIC calculated in this round is: " + str(CIC) + '\n'
        report = report + "Best feature choosen in this round is: "+comb.index[seedid] + '\n'
        newIC = binaryInformationCoefficient_cython(y, cythonseed,k,size,bandwidth,
                                                    bandwidth_mult,bandwidth_adj, neighborhood, deltay, miny)
        if if_bootstrap == True:
            bootstrap = calcBootstrapIC(subcomb = pd.DataFrame([comb.iloc[0].tolist(),currentseed]), 
                                        size = size, bandwidth = bandwidth, k = k, 
                                        bandwidth_mult = bandwidth_mult, 
                                        bandwidth_adj = bandwidth_adj, grid = grid,
                                        thread_number =thread_number, IC = newIC, neighborhood = neighborhood)
            bootstraps.append(bootstrap)

        if if_pval == True:
            pVals.append(0)
        
        if verbose != 0:
            print("IC of new seed is: "+str(newIC))
        report = report + "IC of new seed is: "+str(newIC) + '\n'
        seedIC.append(newIC)
        CICs.append(newIC)
        comb = comb.drop(index = [comb.index[seedid]])
        

        if max_iteration == -1:
            if direction == 'pos':
                if len(seedIC) >= 2 and newIC <= seedIC[-2]:
                    seedids = seedids[:-1]
                    CICs = CICs[:-2]
                    pVals = pVals[:-2]
                    seedlists = seedlists[:-1]
                    bootstraps = bootstraps[:-2]
                    break
            elif direction == 'neg':
                if len(seedIC) >= 2 and newIC >= seedIC[-2]:
                    seedids = seedids[:-1]
                    CICs = CICs[:-2]
                    pVals = pVals[:-2]
                    seedlists = seedlists[:-1]
                    bootstraps = bootstraps[:-2]
                    break
        n=n+1
    
    if len(comb.index.tolist()) == 1:
        'Loop ended before it reach requirement because no unused feature left!'

    if if_pval == True:
        pVal = updatePval(pVals=pVals, CICs=CICs, seedlists=seedlists, 
                          target = comb.iloc[0].tolist(), size = size, bandwidth = bandwidth, k = k, 
                          bandwidth_mult = bandwidth_mult, bandwidth_adj = bandwidth_adj, 
                          grid = grid, thread_number =thread_number, seed_name = seed_name,
                          direction = direction, neighborhood = neighborhood)
    if if_pval == True:
        saveresfigWithPval(savecomb, seedids, CICs, seedlists, seedcmap, featurecmap, prefix,
                           figure_format, pVals, bootstraps, out_folder, target_name, seed_name,
                           locusdic, gmt)
    else:
        saveresfig(savecomb, seedids, CICs, seedlists, seedcmap, featurecmap, prefix,
                   figure_format, bootstraps, out_folder, target_name, seed_name, locusdic, gmt)

    if verbose != 0:
        print('Time used to run loops: %s second(s)'%(int(time.time() - start)))
    report = report + 'Time used to run loops: %s second(s)'%(int(time.time() - start)) + '\n'

    with open(out_folder + prefix + 'report.txt','w') as f:
        f.write(report)

#    if gzip == True:
#        with tarfile.open(prefix+'.tar.gz', "w:gz") as tar:
#            tar.add(out_folder, arcname=os.path.basename(out_folder))

    if seed_name == None:
        return savecomb.iloc[[0]], None, savecomb.iloc[1:]
    else:
        return seedids, CICs#savecomb.iloc[[0]], savecomb.loc[['firstseed']], savecomb.iloc[1:].drop(index=['firstseed'])

cpdef updatePval(
    pVals,          #List of p-values
    CICs,           #List of CIC/IC
    seedlists,      #List of name of feature picked
    target,         #Target to be aligned
    size,           #Number of sample
    bandwidth,      #Value of bandwidth
    k,              #K value for kernel
    bandwidth_mult, #Multiplier of bandwidth
    bandwidth_adj,  #Adjust of bandwidth
    grid,           #Size of grid
    thread_number,  #Number of thread
    seed_name,      #List of seed name
    direction,      #Direction that feature should match with target
    neighborhood
    ):
    
    '''
    Function to update p-value based on seeds and originally calculated CICs
    '''
    
    results = []
    if seed_name != None:
        seedcomb = pd.DataFrame(seedlists)
    else:
        seedcomb = pd.DataFrame(seedlists[1:])

    #Calculate ICs using seeds
    results = prepPvalICInd(seedcomb, target, size, bandwidth, k, bandwidth_mult,bandwidth_adj, 
                            grid, thread_number, neighborhood)

    #Calculate each pVals based on CIC/IC at that position
    if direction == 'pos':
        if seed_name != None:
            for i in range(int((len(pVals)+1)/2)):
                count = 0
                for j in results:
                    if CICs[i*2] <= j:
                        count += 1
                if count == 0:
                    pVals[i*2] = len(results)
                else:
                    pVals[i*2] = float(count)/float(len(results))
        else:
            for i in range(1,int((len(pVals)+1)/2)):
                count = 0
                for j in results:
                    if CICs[i*2] <= j:
                        count += 1
                if count == 0:
                    pVals[i*2] = len(results)
                else:
                    pVals[i*2] = float(count)/float(len(results))
    else:
        if seed_name != None:
            for i in range(int((len(pVals)+1)/2)):
                count = 0
                for j in results:
                    if CICs[i*2] >= j:
                        count += 1
                if count == 0:
                    pVals[i*2] = len(results)
                else:
                    pVals[i*2] = float(count)/float(len(results))
        else:
            for i in range(1,int((len(pVals)+1)/2)):
                count = 0
                for j in results:
                    if CICs[i*2] >= j:
                        count += 1
                if count == 0:
                    pVals[i*2] = len(results)
                else:
                    pVals[i*2] = float(count)/float(len(results))
                
    return pVals

def patientfig(
    df,       #Dataframe used to plot 
    seedcmap,       #Cmap for seed heatmap
    featurecmap,    #Cmap for feature heatmap
    prefix,         #Prefix for file generated
    figure_format,  #Format for result figure
    out_folder,     #Name of output folder
    target_name
    ):

    '''
    Function to save result summary figure without p-values. 
    '''

    fig = plt.figure()
    fig.set_figheight(len(df.index)/2+0.5)
    fig.set_figwidth(11)
    
    #Target name at left of target heatmap
    ax = plt.subplot2grid(shape=(5*(len(df.index)+1),110), loc=(0, 0), colspan=20,rowspan=5)
    ax.set_axis_off()
    ax.text(0.9,0.5, 'Target:'+str(target_name), ha='right', va='center')

    #Plot heatmap for target
    ax = plt.subplot2grid(shape=(5*(len(df.index)+1),110), loc=(0, 20), colspan=90,rowspan=5)
    ax = sns.heatmap(df.iloc[[0]].to_numpy(), cmap='bwr', annot=False, yticklabels=False,
                     xticklabels=False, cbar=False, center=df.iloc[0].mean())

    #Loop to plot best feature and new seeds.
    for i in range(1,len(df.index)):

        #Label name of picked feature in this round
        ax = plt.subplot2grid(shape=(5*(len(df.index)+1),110), loc=(((i))*5, 0), 
                              colspan=20,rowspan=5)
        ax.set_axis_off()
        ax.text(0.9,0.5,df.index.tolist()[i] + '(' + str(df.iloc[i].tolist().count(1)) + ')', ha='right', va='center')

        #Plot heatmap for best feature
        ax = plt.subplot2grid(shape=(5*(len(df.index)+1),110), loc=(((i))*5, 20), 
                              colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([df.iloc[i].tolist()]).astype(int),
                         cmap=featurecmap, annot=False, yticklabels=False, xticklabels=False, cbar=False)

    #Save figure 
    plt.savefig(out_folder + prefix+'_tissueType.'+figure_format,format=figure_format)
    plt.close()


cpdef prepPvalICInd(
    seedcomb,       #Combined dataframe
    target,         #target to be compared
    size,           #Number of sample
    bandwidth,      #Value of bandwidth
    k,              #K value for kernel
    bandwidth_mult, #Multiplier of bandwidth
    bandwidth_adj,  #Adjust of bandwidth
    grid,           #Size of grid
    thread_number,  #Number of thread
    neighborhood
    ):

    '''
    Function to prepare IC values for update
    '''

    #Get original target
    origtarget = target
    targets = []
    targets.append(origtarget)
    
    #Create 1000 randomized target
    for i in range(10000):
        targets.append(random.sample(origtarget, len(origtarget)))
        
    cdef np.double_t[:] y
    results = []
        
    #Make list of results with all the IC calculated with each randomized target
    for i in range(len(targets)):
        y = np.ascontiguousarray(np.asarray(targets[i]))
        
        results = results + list(np.asarray(rankIC(y, np.array(seedcomb.values.astype(int).tolist()),
                         k, int(len(seedcomb.index)), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid, thread_number, neighborhood)))

    return results
    

cpdef calcPvalCIC(
    results,    #List of IC to be compared
    IC,         #IC to be compared
    direction   #Direction to be compared
    ):

    '''
    Calculate p-value using CIC value and results calculate using prepPvalIC
    '''
    
    pval = 0
    count = 0
    if direction == 'pos':
        for i in results:
            if i > IC:
                count = count + 1
    else:
        for i in results:
            if i < IC:
                count = count + 1
    if count == 0:
        return len(results)

    return count/len(results)
    
cpdef prepPvalCIC(
    comb,           #Combined dataframe
    cythonseed,     #Seed to be used to calculate CIC
    k,              #K value for kernel
    size,           #Number of sample
    bandwidth,      #Value of bandwidth
    bandwidth_mult, #Multiplier of bandwidth
    bandwidth_adj,  #Adjust of bandwidth
    grid,           #Size of grid
    thread_number,  #Number of thread
    neighborhood
    ):

    """
    Function to prepare all CIC calculate with 100 randomized target. 
    Return value can be used as results parameter for calcPvalCIC.
    """

    #Get original target
    origtarget = comb.iloc[0].tolist()
    targets = []
    targets.append(origtarget)

    #Create 100 randomized target
    for i in range(1000):
        targets.append(random.sample(origtarget, len(origtarget)))
        
    cdef np.double_t[:] y
    results = []
    
    #Make list of results with all the CIC calculated with each randomized target
    for i in range(len(targets)):
        y = np.ascontiguousarray(np.asarray(targets[i]))
        
        results = results + list(np.asarray(rankCIC(y, cythonseed, np.array(comb.iloc[1:].values.astype(int).tolist()),
                         k, int(len(comb.index)-1), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid,thread_number, neighborhood)))

    return results

cpdef calcPvalIC(
    results,    #List of IC to be compared
    IC,         #IC to be compared
    direction   #Direction to be compared
    ):

    '''
    Calculate p-value using IC value and results calculate using prepPvalIC
    '''

    pval = 0
    count = 0
    if direction == 'pos':
        for i in results:
            if i > IC:
                count = count + 1
    else:
        for i in results:
            if i < IC:
                count = count + 1
    if count == 0:
        return len(results)

    return count/len(results)
    

cpdef prepPvalIC(
    comb,           #Combined dataframe
    k,              #K value for kernel
    size,           #Number of sample
    bandwidth,      #Value of bandwidth
    bandwidth_mult, #Multiplier of bandwidth
    bandwidth_adj,  #Adjust of bandwidth
    grid,           #Size of grid
    thread_number,  #Number of thread
    neighborhood
    ):

    """
    Function to prepare all IC calculate with 100 randomized target. Return value can be used as results parameter
    for calcPvalIC
    """

    #Get original target
    origtarget = comb.iloc[0].tolist()
    targets = []
    targets.append(origtarget)

    #Create 100 randomized target
    for i in range(1000):
        targets.append(random.sample(origtarget, len(origtarget)))
        
    cdef np.double_t[:] y
    results = []

    #Make list of results with all the IC calculated with each randomized target
    for i in range(len(targets)):
        y = np.ascontiguousarray(np.asarray(targets[i]))
        
        results = results + list(np.asarray(rankIC(y, np.array(comb.iloc[1:].values.astype(int).tolist()),
                         k, int(len(comb.index)-1), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid,thread_number, neighborhood)))

    return results


def calcIndivisualPvalIC(
    subcomb,        #Combined dataframe
    k,              #K value for kernel
    size,           #Number of sample
    bandwidth,      #Value of bandwidth
    bandwidth_mult, #Multiplier of bandwidth
    bandwidth_adj,  #Adjust of bandwidth
    grid,           #Size of grid
    thread_number,  #Number of thread
    IC,             #IC to be compared
    neighborhood
    ):

    '''
    Calculate individual p-values for IC
    '''

    origtarget = subcomb.iloc[0].tolist()
    targets = []
    for i in range(10000):
        targets.append(random.sample(origtarget, len(origtarget)))

    res = runIndividualPvalIC(np.ascontiguousarray(np.asarray(targets)), 
                  np.asarray(subcomb.iloc[1].values.astype(int).tolist()),
                  k, 10000, size, bandwidth, bandwidth_mult, bandwidth_adj, 
                  grid,thread_number, neighborhood)

    countless = 0
    for r in res:
        if r > res[0]:
            countless += 1
    if countless == 0:
        countless = 1
    return (countless / 10000)
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef np.double_t[:] runIndividualPvalIC(
    np.double_t[:,:] ys_in,     #List of targets 
    np.int_t[:] x_in,           #List of features
    int k,                      #K value for kernel
    int n,                      #Number of result
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    int grid,                   #Size of grid
    int thread_number,          #Number of thread
    int neighborhood
    ): 
    
    '''
    Function to run individual IC for p-values 
    '''

    cdef int i
    cdef np.double_t[:] res = np.zeros(n)
    cdef np.double_t[:,:] y = np.asarray(ys_in)
    cdef np.int_t[:] x = np.asarray(x_in)

    cdef np.double_t miny
    cdef np.double_t deltay

    for i in prange(n, nogil=True,num_threads=thread_number):
        miny = findmin(y[i,:], size)
        deltay = findmax(y[i,:], size) - findmin(y[i,:], size)
        res[i] = binaryInformationCoefficient_cython(y[i,:], x, k, size, bandwidth, bandwidth_mult, 
                                                     bandwidth_adj, neighborhood, deltay, miny)
    i = 0
    return res
   

    
def calcBootstrap(
    subcomb,        #Combined dataframe
    cythonseed,     #Seed to calculate
    k,              #K value for kernel
    size,           #Number of sample
    bandwidth,      #Value of bandwidth
    bandwidth_mult, #Multiplier of bandwidth
    bandwidth_adj,  #Adjust of bandwidth
    grid,           #Size of grid
    thread_number,  #Number of thread
    IC,             #IC to be compared
    neighborhood
    ):

    '''
    Calculate variance of CIC by bootstrapping subset of sample
    '''

    origfeature = subcomb.iloc[1].tolist()
    origtarget = subcomb.iloc[0].tolist()
    features = []
    targets = []
    i = 0
    while i < 100:
        subsubcomb = subcomb.sample(math.floor(len(subcomb.columns.tolist())*0.62),axis='columns')
        if sum(np.asarray(subsubcomb.iloc[1].values.astype(int).tolist())) > 2:
            targets.append(np.asarray(subsubcomb.iloc[0].values.astype(float).tolist()))
            features.append(np.asarray(subsubcomb.iloc[1].values.astype(int).tolist()))
            i += 1

    res = runBootstrap(np.ascontiguousarray(np.asarray(targets)), cythonseed, 
                       np.asarray(features), k, 100, math.floor(len(subcomb.columns.tolist())*0.62), 
                       bandwidth, bandwidth_mult, bandwidth_adj, grid,thread_number, neighborhood)
    
    ninety = [np.percentile(np.asarray(res), 5), np.percentile(np.asarray(res), 95)]
    diff = max(IC - ninety[0],ninety[1] - IC)
    
    return diff
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef np.double_t[:] runBootstrap(
    np.double_t[:,:] ys_in,     #List of targets 
    np.int_t[:] z_in,           #Condition feature to calculate CIC
    np.int_t[:,:] xs_in,        #List of features
    int k,                      #K value for kernel
    int n,                      #Number of result
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    int grid,                   #Size of grid
    int thread_number,          #Number of thread
    int neighborhood
    ):
    
    '''
    Function to run bootstrap(CIC) with different pair of target and features 
    '''

    cdef int i
    cdef np.double_t[:] res = np.zeros(n)
    cdef np.double_t[:,:] y = np.asarray(ys_in)
    cdef np.int_t[:,:] x = np.asarray(xs_in)
    cdef np.int_t[:] z = np.asarray(z_in)

    cdef np.double_t miny
    cdef np.double_t deltay

    for i in prange(n, nogil=True,num_threads=thread_number):
        miny = findmin(y[i,:], size)
        deltay = findmax(y[i,:], size) - findmin(y[i,:], size)
        res[i] = ConditionalInformationCoefficient_cython(y[i,:], x[i,:], z, k, size, bandwidth, 
                                                          bandwidth_mult, bandwidth_adj, 
                                                          neighborhood, deltay, miny)
    i = 0
    return res
    
def calcBootstrapIC(
    subcomb,        #Combined dataframe
    k,              #K for kernel
    size,           #Number of sample
    bandwidth,      #Value of bandwidth
    bandwidth_mult, #Multiplier for bandwidth
    bandwidth_adj,  #Adjust of bandwidth
    grid,           #Grid size
    thread_number,  #Number of thread
    IC,             #IC to be compared
    neighborhood
    ):

    '''
    Calculate variance of IC by bootstrapping subset of sample
    '''

    #Save original target and picked feature
    origfeature = subcomb.iloc[1].tolist()
    origtarget = subcomb.iloc[0].tolist()
    features = []
    targets = []
    i = 0

    #Make 100 pair of target and feature by subsetting 62% of original one
    while i < 100:
        subsubcomb = subcomb.sample(math.floor(len(subcomb.columns.tolist())*0.62),axis='columns')

        #Save only if there are more than 3 occurence
        if sum(np.asarray(subsubcomb.iloc[1].values.astype(int).tolist())) > 2:
            targets.append(np.asarray(subsubcomb.iloc[0].values.astype(float).tolist()))
            features.append(np.asarray(subsubcomb.iloc[1].values.astype(int).tolist()))
            i += 1

    #Calculate IC for each pair (can be ran parallelly)
    res = runBootstrapIC(np.ascontiguousarray(np.asarray(targets)), 
                         np.asarray(features).astype(int), k, 100, 
                         math.floor(len(subcomb.columns.tolist())*0.62), bandwidth, 
                         bandwidth_mult, bandwidth_adj, grid, thread_number, neighborhood)
        
    #get value for 90 percentile and extract larger side
    ninety = [np.percentile(np.asarray(res), 5), np.percentile(np.asarray(res), 95)]
    diff = max(IC - ninety[0],ninety[1] - IC)

    return diff
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef np.double_t[:] runBootstrapIC(
    np.double_t[:,:] ys_in,     #List of targets 
    np.int_t[:,:] xs_in,        #List of features
    int k,                      #K value for kernel
    int n,                      #Number of result
    int size,                   #Number of sample
    np.double_t bandwidth,      #Value of bandwidth
    np.double_t bandwidth_mult, #Multiplier of bandwidth
    np.double_t bandwidth_adj,  #Adjust of bandwidth
    int grid,                   #Size of grid
    int thread_number,          #Number of thread
    int neighborhood
    ):

    '''
    Function to run bootstrap(IC) with different pair of target and features 
    '''

    cdef int i
    cdef np.double_t[:] res = np.zeros(n)
    cdef np.double_t[:,:] y = np.asarray(ys_in)
    cdef np.int_t[:,:] x = np.asarray(xs_in)
    cdef np.double_t miny
    cdef np.double_t deltay

    for i in prange(n, nogil=True,num_threads=thread_number):
        miny = findmin(y[i,:], size)
        deltay = findmax(y[i,:], size) - findmin(y[i,:], size)
        res[i] = binaryInformationCoefficient_cython(y[i,:], x[i,:], k, size, bandwidth, 
                                                     bandwidth_mult, bandwidth_adj, 
                                                     neighborhood, deltay, miny)
    i = 0

    return res
    

def plotCluster(
    plotcomb,       #Dataframe used to plot
    CICs,           #List of CIC values
    size,           #Number of sample
    cythonseed,     #Seed to plot
    seedcmap,       #Cmap for seed
    featurecmap,    #Cmap for feature
    prefix,         #Prefix for files
    figure_format,  #Format for figure
    pVals,          #List of p-values
    num_top,        #Number of top feature
    bootstraps,     #List of bootstraps
    nitr,           #Itration number for this figure
    out_folder,     #Folder to put output file
    target_name,    #Name of target
    seed_name,      #Name of original seed
    locusdic,
    gmt
    ):
    
    '''
    Function to save intermediate figures with top ranking features in each loop while clustering them into 
    '''

    #Check if the bootstrap, clustering, and p-value are calculated
    plotpval = False
    if_bootstrap = False
    if len(pVals) != 0:
        plotpval = True
    if len(bootstraps) != 0:
        if_bootstrap = True
    comball = plotcomb.copy()
    comball['CICs'] = CICs 
    if plotpval == True:
        comball['pVals'] = list(pVals)
    if if_bootstrap == True:
        comball['bootstraps'] = list(bootstraps)
        
    #Replace new line otherwise there's error
    plotcomb2 = plotcomb.copy()
    plotcomb2.index = plotcomb2.index.str.replace("\n", ",")
        
    #Run NMF with different k
    res, nmf_results, cophenetic_correlation_coefficients = ccal.explore_components(
                    df = plotcomb2.iloc[1:,].T,
                    ks = range(2,12),
                    n_clustering = 10,
                    random_seed = 19981218,  
                    directory_path = out_folder + prefix + '_NMF/',
                    verbose = 0,
                    algorithm = 'ls')
    
    #Pick best k value based on cophenetic correlation coefficients
    diff = -1
    diffindex = 0
    for i in range(len(cophenetic_correlation_coefficients.index.tolist()[:-1])):
        if cophenetic_correlation_coefficients.iloc[i] - cophenetic_correlation_coefficients.iloc[i+1] > diff:
            diff = cophenetic_correlation_coefficients.iloc[i] - cophenetic_correlation_coefficients.iloc[i+1]
            diffindex = i

    #get grouping result with optimal k
    group = nmf_results.loc[cophenetic_correlation_coefficients.index.tolist()[diffindex]].tolist()
    comball['group'] = [0]+group
    numgroup = len(set(group))
    
    #Prepare canvas for result figure
    fig = plt.figure()
    fig.set_figheight(len(CICs)/2.0+1+numgroup/2.0)
    fig.set_figwidth(12)

    #Label target name at left of target heatmap
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(0, 0), colspan=20,rowspan=5)
    ax.set_axis_off()
    ax.text(0.9,0.5, 'Target:'+str(target_name), ha='right', va='center')

    #Plot heatmap for target
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(0, 20), colspan=90,rowspan=5)
    ax = sns.heatmap(plotcomb.iloc[[0]].to_numpy(), cmap='bwr', annot=False, yticklabels=False,
                     xticklabels=False, cbar=False, center=plotcomb.iloc[0].mean())

    #Label for IC or CIC and p-values on right of target heatmap.
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(0, 110), colspan=12,rowspan=5)
    ax.set_axis_off()
    if if_bootstrap == True:
        ax.text(0.5,0.5,'CIC(Δ)',ha='center', va='center', color = 'blue')
    else:
        ax.text(0.5,0.5,'CIC',ha='center', va='center', color = 'blue')
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(0, 122), colspan=10,rowspan=5)
    ax.set_axis_off()
    if plotpval == True:
        ax.text(0.5,0.5,'p-value',ha='center', va='center')

    #Label seed name at left of seed heatmap
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(5, 0), colspan=20,rowspan=5)
    ax.set_axis_off()
    if seed_name != None:
        if len(seed_name) > 1:
            ax.text(0.9,0.5, 'Seed:'+str(seed_name[0])+',...', ha='right', va='center')
        else:
            ax.text(0.9,0.5, 'Seed:'+str(seed_name[0]), ha='right', va='center')
    else:
        ax.text(0.9,0.5, 'Seed', ha='right', va='center')

    #Plot seed heatmap
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(5, 20), colspan=90,rowspan=5)
    ax = sns.heatmap(np.array([cythonseed]), cmap=seedcmap, annot=False, yticklabels=False,
                     xticklabels=False, cbar=False)

    #Blanks
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(5, 110), colspan=12,rowspan=5)
    ax.set_axis_off()

    #Subset dataframe to plot portion
    plotsubdf = comball.iloc[1:]
    plotsubdf.sort_values(by='group', ascending=False, inplace=True)
    pos = 2
    groupindex = 1
    
    #Loop for plot features by group
    for g in plotsubdf['group'].unique():

        #Put group id
        ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(pos*5, 20), 
                              colspan=90,rowspan=5)
        ax.set_axis_off()
        ax.text(0.5,0.5,"Group"+str(groupindex),ha='center', va='center')

        #Subset df with group id and sort by CIC value
        gsubdf = plotsubdf[plotsubdf['group'] == g]
        pos += 1
        groupindex += 1
        gsubdf.sort_values(by='CICs', ascending=False, inplace=True)

        #Loop for plot features in specific group
        for j in range(len(gsubdf.index.tolist())):

            #Label name of picked feature in this round
            ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(pos*5, 0), 
                                  colspan=20,rowspan=5)
            ax.set_axis_off()

            if (gmt is not None) and (gsubdf.index.tolist()[j] in gmt.index.tolist()):
                if locusdic != None:
                    ax.text(0.9,0.5,gsubdf.index.tolist()[j] +' '+ locusdic[gsubdf.index.tolist()[j]]+
                        '(' + str(sum(gmt.loc[gsubdf.index.tolist()[j]].notna())) + ',' +
                        str(plotcomb.loc[gsubdf.index.tolist()[j]].tolist().count(1)) + ')',
                        ha='right', va='center')
                else:
                    ax.text(0.9,0.5,gsubdf.index.tolist()[j]+
                        '(' + str(sum(gmt.loc[gsubdf.index.tolist()[i]].notna())) + ',' +
                        str(plotcomb.loc[gsubdf.index.tolist()[i]].tolist().count(1)) + ')',
                        ha='right', va='center')
            else:
                if locusdic != None:
                    ax.text(0.9,0.5,gsubdf.index.tolist()[i] +' '+ locusdic[gsubdf.index.tolist()[i]],
                            ha='right', va='center')
                else:
                    ax.text(0.9,0.5,gsubdf.index.tolist()[i], ha='right', va='center')

            #Plot heatmap for picked feature
            ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(pos*5, 20), 
                                  colspan=90,rowspan=5)
            ax = sns.heatmap(np.asarray([gsubdf.iloc[j,:size].tolist()]).astype(int),
                             cmap=featurecmap, annot=False, yticklabels=False, xticklabels=False, 
                             cbar=False)

            #Label CIC or IC for this feature
            ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(pos*5, 110), 
                                  colspan=12,rowspan=5)
            ax.set_axis_off()
            if if_bootstrap == True:
                ax.text(0.5,0.5,"%.3f(%s)"%(gsubdf['CICs'].iloc[j],
                        int(1000*gsubdf['bootstraps'].iloc[j])),
                        ha='center', va='center', color = 'blue')
            else:
                ax.text(0.5,0.5,"%.3f"%(gsubdf['CICs'].iloc[j]),
                    ha='center', va='center', color = 'blue')

            #Label p-values if plotpval is True
            ax = plt.subplot2grid(shape=(5*(len(CICs)+2+numgroup),132), loc=(pos*5, 122), 
                                  colspan=10, rowspan=5)
            ax.set_axis_off()
            if plotpval == True:
                pval = gsubdf['pVals'].iloc[j]
                if pval > 1:
                    pval = float(1)/float(pval)
                    pvallist = ("%e"%pval).split('e')
                    ax.text(0.5,0.5,"<%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),
                            ha='center', va='center')
                else:
                    pvallist = ("%e"%pval).split('e')
                    ax.text(0.5,0.5,"%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),
                            ha='center', va='center')
            pos += 1

    #Save Plot            
    plt.savefig(out_folder + prefix+'itr'+str(nitr)+'_Top'+str(num_top)+'ResultClustered.'+
                figure_format, format=figure_format)
    plt.close()

    if (gmt is not None) and (all(elem in gmt.index.tolist() for elem in plotsubdf.index.tolist()) == True):
        gmt.loc[plotsubdf.index.tolist()].to_csv(out_folder + prefix+'_itr'+str(nitr)+'_Top'+str(num_top)+'.gmt',sep='\t',header=False)

    #Save report
    comball.sort_values(by='group', ascending=False).to_csv(out_folder + prefix+'itr'+str(nitr)+'_Top'+
                                                            str(num_top)+'ResultClustered.txt',sep='\t')
    

def pileRank(
    rank,       #Rank for features
    num_top,    #Number of top feature
    prefix,     #Prefix for generated file
    out_folder, #Folder to put output file
    nitr        #Itration number for this figure
    ):

    sub = []
    ICs = []
    old = 0
    n = 0
    total = []
    for i in rank:
        new = float(i[1])
        if new == old:
            sub.append(i)
            continue
        else:
            if n != 0:
                total.append(sub)
                old = new
                n = n + 1
                sub = []
                sub.append(i)
            else:
                sub = []
                sub.append(i)
                old = new
                n = n + 1
            ICs.append(old)
        if n == num_top:
            break

    total.append(sub)
    pileres = 'CIC\tGeneNames\n'
    for pile in total:
        pileres = pileres + "%.3f"%(float(pile[0][1]))+'\t'
        for gene in pile:
            if '\n' in gene[0]:
                pileres = pileres + gene[0].replace('\n','(')+')' + '\t'
            else:
                pileres = pileres + gene[0] + '\t'
        pileres = pileres[:-1]+'\n'
    f = open(out_folder + prefix+'_'+str(nitr)+'_top'+str(num_top)+".txt", "w")
    f.write(pileres[:-1])
    f.close()

    return total, ICs

    
def savetopfig(
    plotcomb,       #Dataframe used to plot
    CICs,           #List of CIC values
    cythonseed,     #Seed to plot
    seedcmap,       #Cmap for seed
    featurecmap,    #Cmap for feature
    prefix,         #Prefix for result files
    figure_format,  #Format for figure
    pVals,          #List of p-values
    num_top,        #Number of top features
    bootstraps,     #List of bootstrap values
    pilesize,       #List of size piled
    nitr,           #Itration number for this figure
    out_folder,     #Name of output folder
    target_name,    #Name of target
    seed_name,      #Name of original seed
    gmt,
    locusdic
    ):

    '''
    Function to save intermediate figures with top ranking features in each loop
    '''

    #Check if the bootstrap, clustering, and p-value are calculated
    plotpval = False
    piled = False
    if_bootstrap = False
    if len(pilesize) != 0:
        piled = True
    if len(pVals) != 0:
        plotpval = True
    if len(bootstraps) != 0:
        if_bootstrap = True

    #Prepare canvas for result figure
    fig = plt.figure()
    fig.set_figheight(len(CICs)/2.0+1)
    fig.set_figwidth(14.2)

    #Label target name at left of target heatmap
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=(0, 0), colspan=30,rowspan=5)
    ax.set_axis_off()
    ax.text(0.9,0.5, 'Target:'+str(target_name), ha='right', va='center')

    #Plot heatmap for target
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=(0, 30), colspan=90,rowspan=5)
    ax = sns.heatmap(plotcomb.iloc[[0]].to_numpy(), cmap='bwr', annot=False, yticklabels=False,
                     xticklabels=False, cbar=False, center=plotcomb.iloc[0].mean())

    #Label for IC or CIC and p-values on right of target heatmap.
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=(0, 120), colspan=12,rowspan=5)
    ax.set_axis_off()
    if if_bootstrap == True:
        ax.text(0.5,0.5,'CIC(Δ)',ha='center', va='center',color='blue')
    else:
        ax.text(0.5,0.5,'CIC',ha='center', va='center',color='blue')
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=(0, 132), colspan=10,rowspan=5)
    ax.set_axis_off()
    if plotpval == True:
        ax.text(0.5,0.5,'p-value',ha='center', va='center')

    #Label seed name at left of seed heatmap
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=(5, 0), colspan=30,rowspan=5)
    ax.set_axis_off()
    if seed_name != None:
        if len(seed_name) > 1:
            ax.text(0.9,0.5, 'Seed:'+str(seed_name[0])+',...', ha='right', va='center')
        else:
            ax.text(0.9,0.5, 'Seed:'+str(seed_name[0]), ha='right', va='center')
    else:
        ax.text(0.9,0.5, 'Seed', ha='right', va='center')

    #Plot seed heatmap
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=(5, 30), colspan=90,rowspan=5)
    ax = sns.heatmap(np.array([cythonseed]), cmap=seedcmap, annot=False, yticklabels=False,
                     xticklabels=False, cbar=False)

    #Blanks
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=(5, 120), colspan=12,rowspan=5)
    ax.set_axis_off()
    ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=(5, 132), colspan=10,rowspan=5)
    ax.set_axis_off()

    #Loop to plot top features one by one
    for i in range(1,len(CICs)):

        #Label name of picked feature in this round
        ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=((i+1)*5, 0), colspan=30,rowspan=5)
        ax.set_axis_off()
        if piled == True and pilesize[i-1] > 1:
            ax.text(0.9,0.5,plotcomb.index.tolist()[i] +' '+ locusdic[plotcomb.index.tolist()[i]] +
                    '(' + str(pilesize[i-1]) + ',' +
                    str(plotcomb.loc[plotcomb.index.tolist()[i]].tolist().count(1)) + 
                    ')', ha='right', va='center')
        else:
            if (gmt is not None) and (plotcomb.index.tolist()[i] in gmt.index.tolist()):
                if locusdic != None:
                    ax.text(0.9,0.5,plotcomb.index.tolist()[i] +' '+ locusdic[plotcomb.index.tolist()[i]]+
                        '(' + str(sum(gmt.loc[plotcomb.index.tolist()[i]].notna())) + ',' +
                        str(plotcomb.loc[plotcomb.index.tolist()[i]].tolist().count(1)) + ')',
                        ha='right', va='center')
                else:
                    ax.text(0.9,0.5,plotcomb.index.tolist()[i]+
                        '(' + str(sum(gmt.loc[plotcomb.index.tolist()[i]].notna())) + ',' +
                        str(plotcomb.loc[plotcomb.index.tolist()[i]].tolist().count(1)) + ')',
                        ha='right', va='center')
            else:
                if locusdic != None:
                    ax.text(0.9,0.5,plotcomb.index.tolist()[i] +' '+ 
                        locusdic[plotcomb.index.tolist()[i]] + '(' + 
                        str(plotcomb.loc[plotcomb.index.tolist()[i]].tolist().count(1)) + ')',
                        ha='right', va='center')
                else:
                    ax.text(0.9,0.5,plotcomb.index.tolist()[i] + '(' + 
                        str(plotcomb.loc[plotcomb.index.tolist()[i]].tolist().count(1)) + ')', 
                        ha='right', va='center')

        #Plot heatmap for picked feature
        ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=((i+1)*5, 30), colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.iloc[i].tolist()]).astype(int), cmap=featurecmap,
                         annot=False, yticklabels=False, xticklabels=False, cbar=False)

        #Label CIC or IC for this feature
        ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=((i+1)*5, 120), colspan=12,rowspan=5)
        ax.set_axis_off()
        if if_bootstrap == True:
            ax.text(0.5,0.5,"%.3f(%s)"%(CICs[i],int(float(bootstraps[i]*1000))), ha='center', 
                    va='center',color='blue')
        else:
            ax.text(0.5,0.5,"%.3f"%(CICs[i]), ha='center', va='center',color='blue')

        #Label p-values if plotpval is True
        ax = plt.subplot2grid(shape=(5*(len(CICs)+2),142), loc=((i+1)*5, 132), colspan=10,rowspan=5)
        ax.set_axis_off()
        if plotpval == True:
            if pval > 1:
                pval = float(1)/float(pval)
                pvallist = ("%e"%pval).split('e')
                ax.text(0.5,0.5,"<%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),
                        ha='center', va='center')
            else:
                pvallist = ("%e"%pval).split('e')
                ax.text(0.5,0.5,"%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),
                        ha='center', va='center')

    #Save Plot
    plt.savefig(out_folder + prefix+'_itr'+str(nitr)+'_Top'+str(num_top)+'_Result.'+figure_format,
                format=figure_format)
    plt.close()
    
    #Set new line of gene locus  to ,
    #plotcomb.index = plotcomb.index.str.replace("\n", ",")
    
    #Adjust length of CIC/IC
    plotcomb['CICs'] = CICs
    
    #Put bootstrap and p-values to report
    if if_bootstrap == True:
        for i in range(len(bootstraps)):
            if bootstraps[i] != '':
                bootstraps[i] = 1000*bootstraps[i]
        plotcomb['bootstrap'] = bootstraps
    if plotpval == True:
        plotcomb['pVals'] = pVals

    if (gmt is not None) and (all(elem in gmt.index.tolist() for elem in plotcomb.index.tolist()) == True):
        gmt.loc[plotcomb.index.tolist()].to_csv(out_folder + prefix+'_itr'+str(nitr)+'_Top'+str(num_top)+'.gmt',sep='\t',header=False)

    #Save report
    plotcomb.to_csv(out_folder + prefix+'_itr'+str(nitr)+'_Top'+str(num_top)+'_Result.txt',sep='\t')
        
    
def saveresfigWithPval(
    plotcomb,       #Dataframe used to plot
    seedids,        #List of name of feature picked 
    CICs,           #List of CIC\IC values
    seedlists,      #List of seeds
    seedcmap,       #Cmap for seed heatmap
    featurecmap,    #Cmap for feature heatmap
    prefix,         #Prefix for file generated
    figure_format,  #Format for result figure
    pVals,          #List of p-values
    bootstraps,     #List of ootstrap values
    out_folder,     #Name of output folder
    target_name,    #Name of target
    seed_name,      #Name of original seed
    locusdic,
    gmt
    ):

    '''
    Function to save result summary figure with p-values. 
    '''

    #Check if bootstrap value is calculated
    if_bootstrap = False
    if len(bootstraps) != 0:
        if_bootstrap = True
        
    #Create dataframe to write down report
    reportdf = pd.DataFrame(columns=plotcomb.columns.tolist())
    
    #Prepare canvas for result figure
    fig = plt.figure()
    fig.set_figheight(len(seedids)+1)
    fig.set_figwidth(15.4)

    #Add Target row for report
    reportdf.loc['Target'] = plotcomb.iloc[0].tolist()

    #Label target name at left of target heatmap
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(0, 0), colspan=20,rowspan=5)
    ax.set_axis_off()
    ax.text(0.9,0.5, 'Target:'+str(target_name), ha='right', va='center')

    #Plot heatmap for target
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(0, 20), colspan=90,rowspan=5)
    ax = sns.heatmap(plotcomb.iloc[[0]].to_numpy(), cmap='bwr', annot=False, yticklabels=False,
                     xticklabels=False, cbar=False, center=plotcomb.iloc[0].mean())

    #Label for IC, p-value for IC, CIC, p-value for CIC on right of target heatmap.
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(0, 110), colspan=12,rowspan=5)
    ax.set_axis_off()
    if if_bootstrap == True:
        ax.text(0.5,0.5,'IC(Δ)',ha='center', va='center')
    else:
        ax.text(0.5,0.5,'IC',ha='center', va='center')
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(0, 122), colspan=10,rowspan=5)
    ax.set_axis_off()
    ax.text(0.5,0.5,'p-value',ha='center', va='center')
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(0, 132), colspan=12,rowspan=5)
    ax.set_axis_off()
    if if_bootstrap == True:
        ax.text(0.5,0.5,'CIC(Δ)',ha='center', va='center',color = 'blue')
    else:
        ax.text(0.5,0.5,'CIC',ha='center', va='center',color = 'blue')
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(0, 144), colspan=10,rowspan=5)
    ax.set_axis_off()
    ax.text(0.5,0.5,'p-value',ha='center', va='center',color = 'blue')
    
    #Add seed row for report
    reportdf.loc['Seed'] = seedlists[0]

    #Label seed name at left of seed heatmap
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(5, 0), colspan=20,rowspan=5)
    ax.set_axis_off()
    if seed_name != None:
        if len(seed_name) > 1:
            ax.text(0.9,0.5, 'Seed:'+str(seed_name[0])+',...', ha='right', va='center')
        else:
            ax.text(0.9,0.5, 'Seed:'+str(seed_name[0]), ha='right', va='center')
    else:
        ax.text(0.9,0.5, 'Null Seed', ha='right', va='center')
    
    #Plot seed heatmap
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(5, 20), colspan=90,rowspan=5)
    ax = sns.heatmap(np.array([seedlists[0]]),cmap=seedcmap,annot=False,yticklabels=False,
                     xticklabels=False,cbar=False)

    #CIC for seed if seed exists
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(5, 110), colspan=12,rowspan=5)
    ax.set_axis_off()
    if isinstance(CICs[0],float) == True and if_bootstrap == True:
        ax.text(0.5,0.5,"%.3f(%s)"%(CICs[0], int(bootstraps[0]*1000)), ha='center', va='center')
    elif isinstance(CICs[0],float) == True:
        ax.text(0.5,0.5,"%.3f"%(CICs[0]), ha='center', va='center')

    #p-value for this seed if seed exists
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(5, 122), colspan=10,rowspan=5)
    ax.set_axis_off()    
    if isinstance(pVals[0],float) == True:
        pval = pVals[0]
        pvallist = ("%e"%pval).split('e')
        ax.text(0.5,0.5,"%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),
                ha='center', va='center')

    #Blanks
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(5, 132), colspan=12,rowspan=5)
    ax.set_axis_off()
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(5, 144), colspan=10,rowspan=5)
    ax.set_axis_off()
        
    #Loop to plot best feature and new seeds.
    for i in range(1,len(seedids)+1):

        #Label name of picked feature in this round
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(((i)*2)*5, 0), 
                              colspan=20,rowspan=5)
        ax.set_axis_off()
        if (gmt is not None) and (seedids[(i-1)] in gmt.index.tolist()):
            if locusdic != None: 
                ax.text(0.9,0.5,seedids[(i-1)] +' '+ locusdic[seedids[(i-1)]] + 
                    '(' + str(sum(gmt.loc[seedids[(i-1)]].notna())) + ',' +
                    str(plotcomb.loc[seedids[(i-1)]].tolist().count(1)) + ')', 
                    ha='right', va='center')
            else:
                ax.text(0.9,0.5,seedids[(i-1)] + 
                    '(' + str(sum(gmt.loc[seedids[(i-1)]].notna())) + ',' +
                    str(plotcomb.loc[seedids[(i-1)]].tolist().count(1)) + ')', 
                    ha='right', va='center')
        else:
            if locusdic != None: 
                ax.text(0.9,0.5,seedids[(i-1)] +' '+ locusdic[seedids[(i-1)]] + 
                    '(' +str(plotcomb.loc[seedids[(i-1)]].tolist().count(1)) + ')', 
                    ha='right', va='center')
            else:
                ax.text(0.9,0.5,seedids[(i-1)] + '(' +
                    str(plotcomb.loc[seedids[(i-1)]].tolist().count(1)) + ')', 
                    ha='right', va='center')

        #Plot heatmap for best feature
        reportdf.loc[seedids[i-1]] = plotcomb.loc[seedids[i-1]].tolist()
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(((i)*2)*5, 20), 
                              colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.loc[seedids[i-1]].tolist()]).astype(int),
                         cmap=featurecmap,annot=False,yticklabels=False, xticklabels=False,cbar=False)

        #Label CIC for this feature        
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(((i)*2)*5, 132), 
                              colspan=12,rowspan=5)
        ax.set_axis_off()
        if if_bootstrap == True:
            ax.text(0.5,0.5,"%.3f(%s)"%(CICs[(i-1)*2+1], int(1000*bootstraps[(i-1)*2+1])), 
                    ha = 'center', va = 'center', color = 'blue')
        else:
            ax.text(0.5,0.5,"%.3f"%(CICs[(i-1)*2+1]), ha='center', va='center',color = 'blue')

        #Label p-value for picked feature
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(((i)*2)*5, 144), 
                              colspan=10,rowspan=5)
        ax.set_axis_off()
        pval = pVals[(i-1)*2+1]
        if pval > 1:
            pval = float(1)/float(pval)
            pvallist = ("%e"%pval).split('e')
            ax.text(0.5,0.5,"<%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),ha = 'center', 
                    va = 'center',color = 'blue')
        else:
            pvallist = ("%e"%pval).split('e')
            ax.text(0.5,0.5,"%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),ha = 'center', 
                    va = 'center',color = 'blue')
        
        #Label for new seed
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(((i)*2+1)*5, 0), 
                              colspan=20,rowspan=5)
        ax.set_axis_off()
        ax.text(0.9,0.5,'New Seed', ha='right', va='center')

        #Put new seed to report
        reportdf.loc['New Seed'+str(i)] = seedlists[i]

        #Plot heatmap for new seed 
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(((i)*2+1)*5, 20), 
                              colspan=90,rowspan=5)
        ax = sns.heatmap(np.array([seedlists[i]]),cmap=seedcmap,annot=False,yticklabels=False,
                         xticklabels=False,cbar=False)

        #Label IC for new seed
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(((i)*2+1)*5, 110), 
                              colspan=12,rowspan=5)
        ax.set_axis_off()
        if if_bootstrap == True:
            ax.text(0.5,0.5,"%.3f(%s)"%(CICs[(i-1)*2+2],int(1000*bootstraps[(i-1)*2+2])), 
                    ha='center', va='center')
        else:
            ax.text(0.5,0.5,"%.3f"%(CICs[(i-1)*2+2]), ha='center', va='center')

        #Label p-value for new seed 
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,154), loc=(((i)*2+1)*5, 122), 
                              colspan=10,rowspan=5)
        ax.set_axis_off()
        pval = pVals[(i-1)*2+2]
        if pval > 1:
            pval = float(1)/float(pval)
            pvallist = ("%e"%pval).split('e')
            ax.text(0.5,0.5,"<%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),
                    ha='center', va='center')
        else:
            pvallist = ("%e"%pval).split('e')
            ax.text(0.5,0.5,"%.1fx${10^{%s}}$"%(float(pvallist[0]),int(pvallist[1])),
                    ha='center', va='center')
        
    #Save plot
    plt.savefig(out_folder + prefix+'_Result.'+figure_format,format=figure_format)
    plt.close()
    
    #Set new line of gene locus  to ,
    reportdf.index = reportdf.index.str.replace("\n", ",")

    #Adjust length of CIC/IC
    reportdf['CICs'] = [''] + CICs

    #Put bootstrap and p-values to report
    if if_bootstrap == True:
        for i in range(len(bootstraps)):
            if bootstraps[i] != '':
                bootstraps[i] = 1000*bootstraps[i]
        reportdf['bootstrap'] = [''] + bootstraps
    reportdf['pVals'] = [''] + pVals

    if (gmt is not None) and (all(elem in gmt.index.tolist() for elem in seedids) == True):
        gmt.loc[seedids].to_csv(out_folder + prefix+'_Result.gmt',sep='\t',header=False)
    
    #Save report
    reportdf.to_csv(out_folder + prefix+'_Result.txt',sep='\t')
    
def saveresfig(
    plotcomb,       #Dataframe used to plot 
    seedids,        #List of name of feature picked 
    CICs,           #List of CIC\IC values
    seedlists,      #List of seeds
    seedcmap,       #Cmap for seed heatmap
    featurecmap,    #Cmap for feature heatmap
    prefix,         #Prefix for file generated
    figure_format,  #Format for result figure
    bootstraps,     #List of ootstrap values
    out_folder,     #Name of output folder
    target_name,    #Name of target
    seed_name,      #Name of original seed
    locusdic,
    gmt
    ):

    '''
    Function to save result summary figure without p-values. 
    '''

    #Check if bootstrap value is calculated
    if_bootstrap = False
    if len(bootstraps) != 0:
        if_bootstrap = True
    
    #Create dataframe to write down report
    reportdf = pd.DataFrame(columns=plotcomb.columns.tolist())
    
    #Prepare canvas for result figure
    fig = plt.figure()
    fig.set_figheight(len(seedids)+1)
    fig.set_figwidth(13.4)

    #Add Target row for report
    reportdf.loc['Target'] = plotcomb.iloc[0].tolist()

    #Target name at left of target heatmap
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(0, 0), colspan=20,rowspan=5)
    ax.set_axis_off()
    ax.text(0.9,0.5, 'Target:'+str(target_name), ha='right', va='center')

    #Plot heatmap for target
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(0, 20), colspan=90,rowspan=5)
    ax = sns.heatmap(plotcomb.iloc[[0]].to_numpy(), cmap='bwr', annot=False, yticklabels=False,
                     xticklabels=False, cbar=False, center=plotcomb.iloc[0].mean())

    #Label for IC and CIC on right of target heatmap.
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(0, 110), colspan=12,rowspan=5)
    ax.set_axis_off()
    if if_bootstrap == True:
        ax.text(0.5,0.5,'IC(Δ)',ha='center', va='center')
    else:
        ax.text(0.5,0.5,'IC',ha='center', va='center')
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(0, 122), colspan=12,rowspan=5)
    ax.set_axis_off()
    if if_bootstrap == True:
        ax.text(0.5,0.5,'CIC(Δ)',ha='center', va='center',color = 'blue')
    else:
        ax.text(0.5,0.5,'CIC',ha='center', va='center',color = 'blue')
    
    #Add seed row for report
    reportdf.loc['Seed'] = seedlists[0]

    #Label Seed name at left of seed heatmap
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(5, 0), colspan=20,rowspan=5)
    ax.set_axis_off()
    if seed_name != None:
        if len(seed_name) > 1:
            ax.text(0.9,0.5, 'Seed:'+str(seed_name[0])+',...', ha='right', va='center')
        else:
            ax.text(0.9,0.5, 'Seed:'+str(seed_name[0]), ha='right', va='center')
    else:
        ax.text(0.9,0.5, 'Seed', ha='right', va='center')

    #Plot seed heatmap
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(5, 20), colspan=90,rowspan=5)
    ax = sns.heatmap(np.array([seedlists[0]]),cmap=seedcmap,annot=False,yticklabels=False,
                     xticklabels=False,cbar=False)

    #CIC for seed if seed exists
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(5, 110), colspan=12,rowspan=5)
    ax.set_axis_off()
    if isinstance(CICs[0],float) == True and if_bootstrap == True:
        ax.text(0.5,0.5,"%.3f(%s)"%(CICs[0], int(bootstraps[0]*1000)), ha='center', va='center')
    elif isinstance(CICs[0],float) == True:
        ax.text(0.5,0.5,"%.3f"%(CICs[0]), ha='center', va='center')
    ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(5, 122), colspan=12,rowspan=5)
    ax.set_axis_off()
        
    #Loop to plot best feature and new seeds.
    for i in range(1,len(seedids)+1):

        #Label name of picked feature in this round
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(((i)*2)*5, 0), 
                              colspan=20,rowspan=5)
        ax.set_axis_off()
        if (gmt is not None) and (seedids[(i-1)] in gmt.index.tolist()):
            if locusdic != None: 
                ax.text(0.9,0.5,seedids[(i-1)] +' '+ locusdic[seedids[(i-1)]] + 
                    '(' + str(sum(gmt.loc[seedids[(i-1)]].notna())) + ',' +
                    str(plotcomb.loc[seedids[(i-1)]].tolist().count(1)) + ')', 
                    ha='right', va='center')
            else:
                ax.text(0.9,0.5,seedids[(i-1)] + 
                    '(' + str(sum(gmt.loc[seedids[(i-1)]].notna())) + ',' +
                    str(plotcomb.loc[seedids[(i-1)]].tolist().count(1)) + ')', 
                    ha='right', va='center')
        else:
            if locusdic != None: 
                ax.text(0.9,0.5,seedids[(i-1)] +' '+ locusdic[seedids[(i-1)]] + 
                    '(' +str(plotcomb.loc[seedids[(i-1)]].tolist().count(1)) + ')', 
                    ha='right', va='center')
            else:
                ax.text(0.9,0.5,seedids[(i-1)] + '(' +
                    str(plotcomb.loc[seedids[(i-1)]].tolist().count(1)) + ')', 
                    ha='right', va='center')

        #Add best feature row in report
        reportdf.loc[seedids[i-1]] = plotcomb.loc[seedids[i-1]].tolist()

        #Plot heatmap for best feature
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(((i)*2)*5, 20), 
                              colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.loc[seedids[i-1]].tolist()]).astype(int),
                         cmap=featurecmap, annot=False, yticklabels=False, xticklabels=False, cbar=False)

        #Label CIC for this feature        
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(((i)*2)*5, 122), 
                              colspan=12,rowspan=5)
        ax.set_axis_off()
        if if_bootstrap == True:
            ax.text(0.5,0.5,"%.3f(%s)"%(CICs[(i-1)*2+1],int(1000*bootstraps[(i-1)*2+1])), 
                    ha='center', va='center', color = 'blue')
        else:
            ax.text(0.5,0.5,"%.3f"%(CICs[(i-1)*2+1]), ha='center', va='center',color = 'blue')

        # Label new seed
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(((i)*2+1)*5, 0), 
                              colspan=20,rowspan=5)
        ax.set_axis_off()
        ax.text(0.9,0.5,'New Seed', ha='right', va='center')

        #Put new seed into 
        reportdf.loc['New Seed'+str(i)] = seedlists[i]

        #Plot heatmap for new seed 
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(((i)*2+1)*5, 20), 
                              colspan=90,rowspan=5)
        ax = sns.heatmap(np.array([seedlists[i]]),cmap=seedcmap,annot=False,yticklabels=False,
                         xticklabels=False,cbar=False)

        #Label IC for new seed
        ax = plt.subplot2grid(shape=(5*(len(seedids)+1)*2,134), loc=(((i)*2+1)*5, 110), 
                              colspan=12,rowspan=5)
        ax.set_axis_off()
        if if_bootstrap == True:
            ax.text(0.5,0.5,"%.3f(%s)"%(CICs[(i-1)*2+2],int(1000*bootstraps[(i-1)*2+2])), 
                    ha='center', va='center')
        else:
            ax.text(0.5,0.5,"%.3f"%(CICs[(i-1)*2+2]), ha='center', va='center')

    #Save figure 
    plt.savefig(out_folder + prefix+'_Result.'+figure_format,format=figure_format)
    plt.close()
    
    #Set new line of gene locus  to ,
    reportdf.index = reportdf.index.str.replace("\n", ",")

    #Adjust length of CIC/IC
    reportdf['CICs'] = [''] + CICs

    #Put bootstrap to report
    if if_bootstrap == True:
        for i in range(len(bootstraps)):
            if bootstraps[i] != '':
                bootstraps[i] = 1000*bootstraps[i]
        reportdf['bootstrap'] = [''] + bootstraps
    
    if (gmt is not None) and (all(elem in gmt.index.tolist() for elem in seedids) == True):
        gmt.loc[seedids].to_csv(out_folder + prefix+'_Result.gmt',sep='\t',header=False)

    #Save report
    reportdf.to_csv(out_folder + prefix+'_Result.txt',sep='\t')
    
def topMatches(
    comb,           #Dataframe contain target and features
    y,              #Target
    grid,           #Grid size
    k,              #K for kernel
    size,           #Number of sample
    bandwidth,      #Bandwidth size
    bandwidth_mult, #Multiplier for bandwidth
    bandwidth_adj,  #Adjust for bandwidth
    thread_number,  #Number of parallel thread
    neighborhood,
    seed = 1        #Seed value 
    ):

    '''
    Functionn to make rank for IC/CIC values. 
    '''

    rank = []
    if isinstance(seed, int):
        results = rankIC(y, np.array(comb.iloc[1:].values.astype(int).tolist()),
                         k, int(len(comb.index)-1), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid,thread_number, neighborhood)
        rank = []
        for i in range(len(results)):
            rank.append([comb.index[i+1],results[i]])
        return rank
    else:
        results = rankCIC(y, seed, np.array(comb.iloc[1:].values.astype(int).tolist()),
                         k, int(len(comb.index)-1), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid,thread_number, neighborhood)
        rank = []
        for i in range(len(results)):
            rank.append([comb.index[i+1],results[i]])
        return rank
    
def seedcomball(
    comb,       #Dataframe 
    seed_name   #List of seed name
    ):
    
    '''
    combine all seed with given seed names
    '''

    currentseed = comb.loc[seed_name[0]].tolist()
    if len(seed_name) == 1:
        return currentseed
    for subseed in seed_name[1:]:
        currentseed = seedCombine(currentseed,comb.loc[subseed].tolist())
    return currentseed

def readInput(
    target_file,    #Name of target file
    feature_file,   #Name of feature file
    seed_file,      #Name of seed file
    seed_name,      #List of seed name(s)
    target_name,    #Target name
    direction,      #Direction to sort target
    gene_locus,     #Name of gene locus file
    low_threshold,  #Lower threshold to filter feature
    high_threshold, #Higher threshold to filter feature
    subset,         #List to subset target file if no subset, indicated 'no'
    gene_set,
    gene_separator,
    thread_number,
    tissue_file
    ):
    
    print('seed_file is' + seed_file)

    '''
    Function to read input files to generate combined dataframe
    '''
    
    print('Start reading input files...')

    #Read target file with given target name or index, replace - with _ in column name
    if isinstance(target_name,int) == True:
        target = pd.read_csv(target_file,skiprows=[0,1],sep='\t',index_col=0)
        target = target.drop(columns=target.columns[0]).iloc[[target_name]]
        target.columns = target.columns.str.replace("-", "_")
    else:
        target = pd.read_csv(target_file,skiprows=[0,1],sep='\t',index_col=0)
        target = target.drop(columns=target.columns[0]).loc[[target_name]]
        target.columns = target.columns.str.replace("-", "_")


    #Read seed file and feature file and extract seed. If same, read only one
    if tissue_file:
        if seed_file == feature_file:
            feature = pd.read_csv(feature_file,skiprows=[0,1],sep='\t',index_col=0)
            feature = feature.drop(columns=feature.columns[0])
            feature.columns = feature.columns.str.replace("-", "_")
            patient = pd.read_csv(tissue_file,skiprows=[0,1],sep='\t',index_col=0)
            patient = patient.drop(columns=patient.columns[0])
            patient.columns = patient.columns.str.replace("-", "_")
            commoncol = list(set(target.columns)&set(feature.columns)&set(patient.columns))
            comb = pd.concat([target,feature],  join='inner')
            comb = comb[commoncol]
        else:
            feature = pd.read_csv(feature_file,skiprows=[0,1],sep='\t',index_col=0)
            feature = feature.drop(columns=feature.columns[0])
            seed_df = pd.read_csv(seed_file,skiprows=[0,1],sep='\t',index_col=0)
            seed_df = seed_df.drop(columns=seed_df.columns[0]).loc[seed_name]
            patient = pd.read_csv(tissue_file,skiprows=[0,1],sep='\t',index_col=0)
            patient = patient.drop(columns=patient.columns[0])
            patient.columns = patient.columns.str.replace("-", "_")
            feature.columns = feature.columns.str.replace("-", "_")
            seed_df.columns = seed_df.columns.str.replace("-", "_")
            seed_df = seed_df.dropna(axis=1)
            commoncol = list(set(target.columns)&set(seed_df.columns)&set(feature.columns)&set(patient.columns))
            comb = pd.concat([target,seed_df,feature],  join='inner')
            comb = comb[commoncol]
    else:
        if seed_file == feature_file:
            feature = pd.read_csv(feature_file,skiprows=[0,1],sep='\t',index_col=0)
            feature = feature.drop(columns=feature.columns[0])
            feature.columns = feature.columns.str.replace("-", "_")
            comb = pd.concat([target,feature],  join='inner')
        else:
            feature = pd.read_csv(feature_file,skiprows=[0,1],sep='\t',index_col=0)
            feature = feature.drop(columns=feature.columns[0])
            seed_df = pd.read_csv(seed_file,skiprows=[0,1],sep='\t',index_col=0)
            seed_df = seed_df.drop(columns=seed_df.columns[0]).loc[seed_name]
            seed_df = seed_df.dropna(axis=1)
            feature.columns = feature.columns.str.replace("-", "_")
            seed_df.columns = seed_df.columns.str.replace("-", "_")
            comb = pd.concat([target,seed_df,feature],  join='inner')

    #If subset list is passed, extract subset only
    if subset != 'no':
        if(len(list(set(comb.columns.tolist())&set(subset)))) == 0:
            print('subset not exist! Please check your subset list.')
            sys.exit(1)
        comb = comb[list(set(comb.columns.tolist())&set(subset))]

    #Sort whole dataframe by target by direction
    if direction == 'neg':
        comb = comb.sort_values(by = comb.index[0], axis = 1)
    else:
        comb = comb.sort_values(by = comb.index[0], axis = 1,ascending=False)

    comb = comb.dropna(axis=1, how='all')
    comb = comb.dropna(axis=0, how='all')
    
    comb = comb.dropna(axis=1)
    comb = comb.dropna(axis=0)

    #remove row by low and high threshold
    if low_threshold < 1 and low_threshold != 0:
        low_threshold = len(comb.iloc[0].tolist()) * low_threshold
    if high_threshold < 1 and high_threshold != 0:
        high_threshold = len(comb.iloc[0].tolist()) * high_threshold
    rmrow = []
    for i in comb.index.tolist()[1:]:
        if (sum(comb.loc[i].tolist()) <= low_threshold or sum(comb.loc[i].tolist()) >= high_threshold) and ((seed_name == None) or ((seed_name != None) and (i not in seed_name))):
            rmrow.append(i)
    comb = comb.drop(index=rmrow)
    seedIndex = []
    
    #Save index of seed    
    if seed_name != None:
        for i in seed_name:
            seedIndex.append(comb.index.tolist().index(i))
    

    if gene_locus != 'None':
        locus = pd.read_csv(gene_locus,sep='\t')
        locus = locus.dropna(axis=0)
        locusdic = {}
        for name in comb.index.tolist()[1:]:
            genename = name.split(gene_separator)[0]
            if genename in locus['Name'].tolist():
                locusdic[name] = locus[locus['Name']==genename]['Location'].tolist()[0]
            else:
                locusdic[name] = ''
    else:
        locusdic = None

    #Set first seed by extracting specific index part
    if seed_name != None:
        newseed_names = []
        for i in seedIndex:
            newseed_names.append(comb.index.tolist()[i])
        currentseed = seedcomball(comb,newseed_names)
        comb = comb.drop(index=newseed_names)
        comb.loc['firstseed'] = currentseed
        if gene_set != None:
            inset_list = []
            for feature_name in comb.index.tolist()[2:]:
                inset_list.append(check_inset(gene_set,feature_name,gene_separator))
            inset_list = [True,True]+inset_list
            comb = comb.iloc[inset_list]
    else:
        if gene_set != None:
            inset_list = []
            for feature_name in comb.index.tolist()[1:]:
                inset_list.append(check_inset(gene_set,feature_name,gene_separator))
            inset_list = [True]+inset_list
            comb = comb.iloc[inset_list]
            

    print('Done reading input files')
    
    return comb,locusdic

def check_inset(gene_set,feature_name,gene_separator):
    if feature_name.split(gene_separator)[0] in gene_set:
        return True
    else:
        return False

