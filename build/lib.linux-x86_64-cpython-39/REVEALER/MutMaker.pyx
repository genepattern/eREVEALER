#-f -a --compile-args=-DCYTHON_TRACE_NOGIL=1 
cimport cython
cython: infer_types=True
cython: profile=True
import warnings
warnings.filterwarnings("ignore")

from scipy import stats

import numpy as np
cimport numpy as np
import math
cdef np.double_t EPS = np.finfo(float).eps
import sys
import pandas as pd
import os
import tarfile

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as clr
import argparse

from libc.float cimport DBL_EPSILON
from libc.limits cimport INT_MAX
from libc.math cimport exp, log, sqrt, M_PI, fabs, isnan, fmin, fmax, round, isnan
from libc.stdlib cimport rand, malloc, free, srand
from libc.stdio cimport printf
from cython.parallel import prange

cdef extern from "limits.h":
    int RAND_MAX
cdef int RANDOM_SEED = 20121020

cdef struct density:
    np.double_t d
    np.uint32_t* cnt
    
cdef np.double_t DELMAX=1000

from scipy.stats import norm
import seaborn as sns
from tqdm.notebook import tqdm
import time
import random

import pickle
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef plotfreq(gct_output_file_prefix, frequency_threshold, phenotype, key, genedf, figformat, 
               direction, out_folder, featurecmap, y, k, bandwidth, thread_number, 
               bandwidth_mult, bandwidth_adj, grid, neighborhood, size):

    """
    Function to plot features separated with target with frequency and weight labeled.
    """

    counts = []
    weights = list(np.asarray(rankIC(y, np.array(genedf.values.astype(int).tolist()),
                         k, int(len(genedf.index)), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid, thread_number, neighborhood)))
#     weights = []
    for i in genedf.index.tolist():
        counts.append(genedf.loc[i].sum())
#         weight_vec = phenotype.iloc[0] * genedf.loc[i]
#         weight = weight_vec.sum()/genedf.loc[i].sum()
#         weights.append(weight)
    genedf['counts'] = counts
    genedf['weights'] = weights
    
    if direction == 'pos':
        genedf = genedf.sort_values(['counts', 'weights'], ascending=[False, False])
    elif direction == 'neg':
        genedf = genedf.sort_values(['counts', 'weights'], ascending=[False, True])
    
    plotcomb = genedf.iloc[:,:-2]
    
    fig = plt.figure()
    fig.set_figheight(len(genedf.index.tolist())/2.0+1)
    fig.set_figwidth(11.6)
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(0, 10), colspan=90,rowspan=5)
    ax = sns.heatmap(phenotype.iloc[[0]].to_numpy(),cmap='bwr',annot=False,yticklabels=False,xticklabels=False,cbar=False,
                     center=phenotype.iloc[0].mean())
    ax.set_ylabel('Target',rotation=0,labelpad=45)
    ax.yaxis.set_label_coords(-0.1,0.3)
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(0, 100), colspan=8,rowspan=5)
    ax.set_axis_off()
    ax.text(0.5,0.5,'Count',ha='center', va='center')
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(0, 108), colspan=8,rowspan=5)
    ax.set_axis_off()
    ax.text(0.5,0.5,'Weight',ha='center', va='center')
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(5, 10), colspan=90,rowspan=5)
    ax.set_axis_off()
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(5, 100), colspan=8,rowspan=5)
    ax.set_axis_off()
    ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=(5, 108), colspan=8,rowspan=5)
    ax.set_axis_off()
    
    for i in range(0,len(counts)):
        ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=((i+2)*5, 10), colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.iloc[i].tolist()]).astype(int),cmap=featurecmap,annot=False,
                         yticklabels=False, xticklabels=False,cbar=False)
        if len(plotcomb.index.tolist()[i]) > 21:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=6)
        elif len(plotcomb.index.tolist()[i]) > 18:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=8)
        else:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45)
        ax.yaxis.set_label_coords(-0.1,0.3)
        ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=((i+2)*5, 100), colspan=8,rowspan=5)
        ax.set_axis_off()
        ax.text(0.5,0.5,"%d"%(genedf['counts'].tolist()[i]),ha='center', va='center')
        ax = plt.subplot2grid(shape=(5*(len(counts)+2),116), loc=((i+2)*5, 108), colspan=8,rowspan=5)
        ax.set_axis_off()
        ax.text(0.5,0.5,"%.3f"%(genedf['weights'].tolist()[i]),ha='center', va='center')

    plt.savefig(out_folder+gct_output_file_prefix+'_'+key+'_Top'+str(frequency_threshold)+'_Heatmap.'+figformat,format=figformat)
    plt.close()
    
    return genedf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef plotweight(gct_output_file_prefix, weight_threshold, phenotype, key, genedf, figformat, direction, 
                 out_folder, featurecmap, y, k, bandwidth, bandwidth_mult, bandwidth_adj, grid, thread_number, 
                 neighborhood, size):

    """
    Function to plot features with target, labeled with wieght.
    """

    weights = []
#    for i in genedf.index.tolist():
#        print(str(stats.pearsonr(phenotype.iloc[0].tolist(), genedf.loc[i].tolist())[0]))
#         weight_vec = phenotype.iloc[0] * genedf.loc[i]
#         weight = weight_vec.sum()/genedf.loc[i].sum()
#         weights.append(weight)


    weights = list(np.asarray(rankIC(y, np.array(genedf.values.astype(int).tolist()),
                         k, int(len(genedf.index)), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid, thread_number, neighborhood)))
    
    genedf['weights'] = weights
    if direction == 'pos':
        genedf = genedf.sort_values(['weights'], ascending=[False])
    elif direction == 'neg':
        genedf = genedf.sort_values(['weights'], ascending=[True])
    
    plotcomb = genedf.iloc[:,:-2]
    fig = plt.figure()
    fig.set_figheight(len(genedf.index.tolist())/2.0+1)
    fig.set_figwidth(10.8)
    ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=(0, 10), colspan=90,rowspan=5)
    ax = sns.heatmap(phenotype.iloc[[0]].to_numpy(),cmap='bwr',annot=False,yticklabels=False,xticklabels=False,cbar=False,
                     center=phenotype.iloc[0].mean())
    ax.set_ylabel('Target',rotation=0,labelpad=45)
    ax.yaxis.set_label_coords(-0.1,0.3)
    ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=(0, 100), colspan=8,rowspan=5)
    ax.set_axis_off()
    ax.text(0.5,0.5,'Weight',ha='center', va='center')

    ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=(5, 10), colspan=90,rowspan=5)
    ax.set_axis_off()
    ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=(5, 100), colspan=8,rowspan=5)
    ax.set_axis_off()
    
    for i in range(0,len(weights)):
        ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=((i+2)*5, 10), colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.iloc[i].tolist()]).astype(int),cmap=featurecmap,annot=False,
                         yticklabels=False, xticklabels=False,cbar=False)
        if len(plotcomb.index.tolist()[i].split('\n')[0]) > 21:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=6)
        elif len(plotcomb.index.tolist()[i].split('\n')[0]) > 18:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=8)
        else:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45)
        ax.yaxis.set_label_coords(-0.1,0.3)
        ax = plt.subplot2grid(shape=(5*(len(weights)+2),108), loc=((i+2)*5, 100), colspan=8,rowspan=5)
        ax.set_axis_off()
        ax.text(0.5,0.5,"%.3f"%(genedf['weights'].tolist()[i]),ha='center', va='center')
        

    plt.savefig(out_folder+gct_output_file_prefix+'_'+key+'_Match'+str(weight_threshold)+'_Heatmap.'+figformat,format=figformat)
    plt.close()
    
    df2 = pd.DataFrame([list(y)+[0]], columns=genedf.columns)
    
    with open(out_folder+gct_output_file_prefix+'_'+key+'_Match'+str(weight_threshold)+'.gct', mode = "w") as output_file:
        output_file.writelines("#1.2\n{}\t{}\n".format(genedf.shape[0], genedf.shape[1]))
        pd.concat([df2,genedf]).to_csv(output_file, sep= '\t')
    
    return genedf



def plotclass(gct_output_file_prefix, key, genedf, figformat, out_folder, featurecmap):

    """
    Function to plot features separated with class with target.
    """

    plotcomb = genedf
    
    fig = plt.figure()
    fig.set_figheight(len(plotcomb.index.tolist())/2.0)
    fig.set_figwidth(10)
    
    for i in range(0,len(plotcomb.index.tolist())):
        ax = plt.subplot2grid(shape=(5*(len(plotcomb.index.tolist())+2),100), loc=((i)*5, 10), colspan=90,rowspan=5)
        ax = sns.heatmap(np.asarray([plotcomb.iloc[i].tolist()]).astype(int),cmap=featurecmap,annot=False,
                         yticklabels=False, xticklabels=False,cbar=False)
        if len(plotcomb.index.tolist()[i]) > 21:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=6)
        elif len(plotcomb.index.tolist()[i]) > 18:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45,fontsize=8)
        else:
            ax.set_ylabel(plotcomb.index.tolist()[i],rotation=0,labelpad=45)
        ax.yaxis.set_label_coords(-0.1,0.3)
        

    plt.savefig(out_folder+gct_output_file_prefix+'_'+key+'_Class_Heatmap.'+figformat,format=figformat)
    plt.close()
    
    return genedf



def drawTarget(comb):

    """
    Function to draw continuous target.
    """

    f, ax = plt.subplots(figsize=(10, 1))
    ax = sns.heatmap(comb.iloc[[0]].to_numpy(),cmap='bwr',annot=False,yticklabels=False,xticklabels=False,
                     cbar=False,center=comb.iloc[0].mean())
    return f

def drawFeature(comb,featurecmap,seedName=None,seedID=None):

    """
    Function to draw binary feature.
    """

    if seedName != None:
        f, ax = plt.subplots(figsize=(10, 1))
        ax = sns.heatmap(comb.loc[[seedName]].to_numpy(),cmap=featurecmap,annot=False,yticklabels=False,
                         xticklabels=False,cbar=False)
        return f
    else:
        f, ax = plt.subplots(figsize=(10, 1))
        ax = sns.heatmap(comb.iloc[[seedID]].to_numpy(),cmap=featurecmap,annot=False,yticklabels=False,
                         xticklabels=False,cbar=False)
        return f
    
def drawSeed(seed,seedcmap):

    """
    Function to combine all seed from comb that are in seedName.
    """

    f, ax = plt.subplots(figsize=(10, 1))
    ax = sns.heatmap([seed],cmap=seedcmap,annot=False,yticklabels=False,xticklabels=False,cbar=False)
    return f



def seedcomball(comb,seedName):

    """
    Function to combine all seed from comb that are in seedName.
    """

    currentseed = comb.loc[seedName[0]].tolist()
    if len(seedName) == 1:
        return currentseed
    for subseed in seedName[1:]:
        currentseed = seedCombine(currentseed,comb.loc[subseed].tolist())
    return currentseed

def seedCombine(currentseed,newseed):

    """
    Function to combine two seed.
    """

    seed = []
    for i in range(len(currentseed)):
        if currentseed[i] == 1 or newseed[i] == 1:
            seed.append(1)
        else:
            seed.append(0)
    return seed


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef produce_mutation_file(
    maf_input_file = None, # Input file with maf format.
    gct_input_file = None, # Input file with gct format.
    gct_output_file_prefix = 'Mut', # Prefix for output file
    phenotype_file = None, # Phenotype required if mode is freq or weight
    class_list = ['Nonsense_Mutation','In_Frame_Del','Silent','Frame_Shift_Ins','Missense_Mutation','Splice_Site',
                  'Frame_Shift_Del','De_novo_Start_OutOfFrame','Nonstop_Mutation','In_Frame_Ins','Start_Codon_SNP',
                  'Start_Codon_Del','Stop_Codon_Ins','Start_Codon_Ins','Stop_Codon_Del','Intron','IGR',"5'Flank",
                  "3'UTR","5'UTR",'Mut_All'], # list of class
    class_seperator = '_', # Separator between gene name and later information
    phenotype_name = 0, # Name of Phenotype, can be int or string
    file_separator = '\t', # Separator for file
    protein_change_identifier = 'Protein_Change', # Identifier for protein change column
    mode = 'class', # Mode to run the program 
    direction = 'pos', # Direction of features matching phenotype
    frequency_threshold = 5, # Threshold for frequency
    weight_threshold = 0.7, # Threshold for weight
    gene_list = None, # Gene list if only part of gene is ran
    name_match = True, # Indicate if column name is perfect matching
    make_figure = False, # Indicate if heatmap for each gene is generated
    figure_format='pdf', # Format of figure
    out_folder='.', # Folder to put results
    ratio = float(1/3), # Ratio of selected features by weight that is acceptable
    verbose = 1,
    sample_list = None,
    total_ratio = 0.4, # Percentage of sample
    if_gmt = True,
    k = 5,
    bandwidth_mult = 0.95,
    bandwidth_adj = 0.65,
    grid = 34,
    thread_number = 1,
    neighborhood = 4,
    gzip = True,
    combine = False # If allele figures are combined
    ):

    """
    Function to create mutation gct file with given file using given mode.
    """

    seedcmap = clr.LinearSegmentedColormap.from_list('custom greys', [(.9,.9,.9),(0.5,0.5,0.5)], N=256)
    featurecmap = clr.LinearSegmentedColormap.from_list('custom greys', 
                                                        [(<double>(176)/<double>(255),
                                                        <double>(196)/<double>(255),
                                                        <double>(222)/<double>(255)),
                                                        (0,0,<double>(139)/<double>(255))], N=256)

    if out_folder[-1] != '/':
        out_folder = out_folder + '/'

    # if output folder does not exist, make it
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Check if input file is passed.
    if maf_input_file == None and gct_input_file == None:
        print('Please indicate input file.')
        sys.exit(1)
        
    cdef int size
    cdef np.double_t[:] y
    cdef np.double_t miny
    cdef np.double_t deltay
    cdef np.double_t bandwidth

    # This part is ran if input file is maf file
    if maf_input_file != None:
        #read file
        if verbose > 0:
            print('Reading input file...')
        ds= pd.read_csv(maf_input_file, sep=file_separator, header=0, index_col=None, dtype=str)
        ds=ds.loc[:,['Hugo_Symbol','Variant_Classification', 'Tumor_Sample_Barcode', protein_change_identifier]]
        ds = ds[ds['Variant_Classification'].notna()]
        ds = ds[ds[protein_change_identifier].notna()]

        # This part is ran when mode is class or all
        if mode == 'mutall' or mode == 'all':
            print('Start making gct by class.')
            if sample_list == None:
                if verbose > 0:
                    print('Start getting sample information.')
                # Make list of sample and its unique index
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
                sample_list = list(sample_set)
            else:
                if verbose > 0:
                    print('Start getting sample information.')
                # Make list of sample and its unique index
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
                sample_set = set(sample_list)&sample_set
                sample_list = list(sample_set)

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            geneclasspair = {}

            for gene in gene_list:
                geneclasspair[gene] = [gene+'_Mut_All']

            allpairlist = []
            for pair in geneclasspair.keys():
                allpairlist = allpairlist + geneclasspair[pair]
            
            countgene = 0
            geneclassallelepair = {}
            if verbose > 0:
                print('Start collecting allele information for each feature.')
            for gene in gene_list:
                countgene += 1
                if verbose > 1:
                    print(gene + ' ' + str(countgene)+'/'+str(len(gene_list)))
                for classification in ds[ds['Hugo_Symbol'] == gene]['Variant_Classification'].unique().tolist():
                    for allele in ds[(ds['Hugo_Symbol'] == gene) & (ds['Variant_Classification'] == classification)][protein_change_identifier].unique().tolist():
                        if gene not in geneclassallelepair.keys():
                            geneclassallelepair[gene] = {}
                            geneclassallelepair[gene][gene+'_Mut_All'] = [gene+'_'+allele]
                        else:
                            geneclassallelepair[gene][gene+'_Mut_All'].append(gene+'_'+allele)
                    geneclassallelepair[gene][gene+'_Mut_All'] = list(set(geneclassallelepair[gene][gene+'_Mut_All']))

            restable = pd.DataFrame(0,index=allpairlist,columns=sample_list)

            if verbose > 0:
                print('Start creating mutation dataframe.')
            for i in ds[ds['Hugo_Symbol'].isin(gene_list)].index.tolist():
                restable.loc[ds.loc[i]['Hugo_Symbol']+'_Mut_All',ds.loc[i]['Tumor_Sample_Barcode']] = 1

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                if verbose > 0:
                    print('Start generating figures.')
                if combine == True:
                    newgenedf = plotclass(gct_output_file_prefix, 'all', restable,
                                              figure_format, out_folder, featurecmap)
                else:
                    for gene in gene_list:
                        newgenedf = plotclass(gct_output_file_prefix, gene, restable.loc[geneclasspair[gene]],
                                              figure_format, out_folder, featurecmap)

            if verbose > 0:
                print('Start removing feature with more than total_ratio.')
            #Remove genes with more than total_ratio are mutated
            for gene in gene_list:
                nummut = seedcomball(restable.loc[geneclasspair[gene]],restable.loc[geneclasspair[gene]].index.tolist()).count(1)
                if nummut > len(sample_list)*total_ratio:
                    restable.drop(labels=geneclasspair[gene],inplace=True)
                    del geneclasspair[gene]
                    del geneclassallelepair[gene]


            if if_gmt == True:
                if verbose > 0:
                    print('Start generating gmt file.')
                countgene = 0
                gmtdf = pd.DataFrame()
                for gene in geneclassallelepair.keys():
                    countgene += 1
                    if verbose > 1:
                        print(gene + ' ' + str(countgene)+'/'+str(len(gene_list)))
                    for classallele in geneclassallelepair[gene].keys():
                        gmtsubdf = pd.DataFrame()
                        gmtsubdf[classallele] = ['na'] + geneclassallelepair[gene][classallele]
                        gmtdf = pd.concat([gmtdf,gmtsubdf],axis=1)

                gmtdf = gmtdf.T
                gmtdf.to_csv(out_folder+gct_output_file_prefix + '_Mut_All.gmt', sep= '\t',header=False)

            if verbose > 0:
                print('Writing Mut All result to gct.')
            # Prepare writing to gct file
            restable.insert(0, "Description", ['na']*len(restable.index))
            restable.index.name = "Name"
            restable.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_Mut_All.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(restable.shape[0], restable.shape[1] - 1))
                restable.to_csv(output_file, sep= '\t')
            

        # This part is ran when mode is class or all
        if mode == 'class' or mode == 'all':
            if verbose > 0:
                print('Start making gct by class.')    

            # Make list of sample and its unique index
            if sample_list == None:
                if verbose > 0:
                    print('Start getting sample information.')
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
                sample_list = list(sample_set)
            else:
                if verbose > 0:
                    print('Start getting sample information.')
                # Make list of sample and its unique index
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
                sample_set = set(sample_list)&sample_set
                sample_list = list(sample_set)

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            # Get allele information
            geneclasspair = {}
            geneclassallelepair = {}
            countgene = 0
            maxsublength = 0
            if verbose > 0:
                print('Start collecting allele information for each feature.')
            for gene in gene_list:
                countgene += 1
                if verbose > 1:
                    print(gene + ' ' + str(countgene)+'/'+str(len(gene_list)))
                geneclasspair[gene] = []
                geneclassallelepair[gene] = {}
                geneclassallelepair[gene][gene+'_Mut_All'] = []
                for classification in ds[ds['Hugo_Symbol'] == gene]['Variant_Classification'].unique().tolist():
                    geneclassallelepair[gene][gene+'_'+classification] = []
                    for allele in ds[(ds['Hugo_Symbol'] == gene) & (ds['Variant_Classification'] == classification)][protein_change_identifier].unique().tolist():
                        geneclassallelepair[gene][gene+'_'+classification].append(gene+'_'+allele)
                        geneclassallelepair[gene][gene+'_Mut_All'].append(gene+'_'+allele)
                    geneclasspair[gene].append(gene+'_'+classification)
                    geneclassallelepair[gene][gene+'_Mut_All'] = list(set(geneclassallelepair[gene][gene+'_Mut_All']))
                    if maxsublength < len(geneclassallelepair[gene][gene+'_Mut_All']):
                        maxsublength = len(geneclassallelepair[gene][gene+'_Mut_All'])
                geneclasspair[gene] = list(set(geneclasspair[gene])) + [gene+'_Mut_All']

            allpairlist = []
            for pair in geneclasspair.keys():
                allpairlist = allpairlist + geneclasspair[pair]
                    
            restable = pd.DataFrame(0,index=allpairlist,columns=sample_list)

            if verbose > 0:
                print('Start creating mutation dataframe.')
            for i in ds[ds['Hugo_Symbol'].isin(gene_list)].index.tolist():
                restable.loc[ds.loc[i]['Hugo_Symbol']+'_'+ds.loc[i]['Variant_Classification'],ds.loc[i]['Tumor_Sample_Barcode']] = 1
                restable.loc[ds.loc[i]['Hugo_Symbol']+'_Mut_All',ds.loc[i]['Tumor_Sample_Barcode']] = 1

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                if combine == True:
                    newgenedf = plotclass(gct_output_file_prefix, 'all', restable,
                                              figure_format, out_folder, featurecmap)
                else:
                    if verbose > 0:
                        print('Start generating figures.')
                    countgene = 0
                    for gene in gene_list:
                        countgene += 1
                        if verbose > 1:
                            print(gene + ' ' + str(countgene)+'/'+str(len(gene_list)))
                        newgenedf = plotclass(gct_output_file_prefix, gene, restable.loc[geneclasspair[gene]],
                                              figure_format, out_folder, featurecmap)

            #Remove gene with more than total_ratio are mutated
            if verbose > 0:
                print('Start removing feature with more than total_ratio.')
            for gene in gene_list:
                nummut = seedcomball(restable.loc[geneclasspair[gene]],restable.loc[geneclasspair[gene]].index.tolist()).count(1)
                if nummut > len(sample_list)*total_ratio:
                    restable.drop(labels=geneclasspair[gene],inplace=True)
                    del geneclasspair[gene]
                    del geneclassallelepair[gene]

            if if_gmt == True:
                if verbose > 0:
                    print('Start generating gmt file.')
                countgene = 0
                gmtdf = pd.DataFrame()
                for gene in geneclassallelepair.keys():
                    countgene += 1
                    if verbose > 1:
                        print(gene + ' ' + str(countgene)+'/'+str(len(gene_list)))
                    for classallele in geneclassallelepair[gene].keys():
                        # gmtsubdf = pd.DataFrame()
                        gmtdf[classallele] = ['na'] + geneclassallelepair[gene][classallele] + ([np.nan]*(maxsublength-len(geneclassallelepair[gene][classallele])))
                        #gmtdf = pd.concat([gmtdf,gmtsubdf],axis=1)

                gmtdf = gmtdf.T
                gmtdf.to_csv(out_folder+gct_output_file_prefix + '_class.gmt', sep= '\t',header=False)

            print('Writing class result to gct.')
            # Prepare writing to gct file
            restable.insert(0, "Description", ['na']*len(restable.index))
            restable.index.name = "Name"
            restable.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_class.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(restable.shape[0], restable.shape[1] - 1))
                restable.to_csv(output_file, sep= '\t')

        # This part is ran when mode is freq or all
        if mode == 'freq' or mode == 'all':
            if verbose > 0:
                print('Start making gct by frequency.')

            # Read Phenotype and only keep intersecting rows.
            phenotype = pd.read_csv(phenotype_file,skiprows=[0,1],sep='\t',index_col=0).drop(columns=['Description'])
            phenotype.columns = phenotype.columns.str.replace("-", "_")
            ds['Tumor_Sample_Barcode'].replace('-','_',regex=True,inplace=True)
            if sample_list != None:
                if verbose > 0:
                    print('Start getting sample information.')
                sample_list_new = []
                for i in sample_list:
                    sample_list_new.append(i.replace('-','_'))
                sample_list = sample_list_new

            # If name is not matching by default, check for subset, if matching, just take intersect
            # This one has to be clear since taking subset to match sometimes cause problem
            if name_match == False:
                if sample_list != None:
                    idlist = sample_list
                else:
                    idlist = ds['Tumor_Sample_Barcode'].unique().tolist()
                newcolnames = [] 
                for i in phenotype.columns.tolist():
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound')
                phenotype.columns = newcolnames

                newcolnames = []
                for i in sample_list:
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound_sl')
                sample_list = newcolnames

            # Take indicated row of phenotype to use
            if verbose > 0:
                print('Extracting matching phenotype.')
            if isinstance(phenotype_name,int):
                phenotype = phenotype.iloc[[phenotype_name]]
            else:
                phenotype = phenotype.loc[[phenotype_name]]

            # Take intersecting columns
            if verbose > 0:
                print('Extracting columns.')
            if sample_list == None:
                sample_list = list(set(ds['Tumor_Sample_Barcode'].unique().tolist())&set(phenotype.columns.tolist()))
            else:
                sample_list = list(set(sample_list)&set(phenotype.columns.tolist()))
            phenotype = phenotype[sample_list]


            # Sort according to direction
            if verbose > 0:
                print('Sorting Phenotype.')
            if direction == 'neg':
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1)
            else:
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1,ascending=False)

            # Normalize phenotype
            if verbose > 0:
                print('Normalizing phenotype.')
            phenotype.iloc[0] = (np.array(phenotype.iloc[0].tolist()) - np.array(phenotype.iloc[0].tolist()).mean())/np.array(phenotype.iloc[0].tolist()).std()
            sample_set = phenotype.columns.tolist()

            ds = ds[ds['Tumor_Sample_Barcode'].isin(list(sample_list))]

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if verbose > 0:
                print('Extracting genes.')
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            geneallelepair = {}
            countgene = 0
            if verbose > 0:
                print('Collecting allele information for each gene.')
            for gene in gene_list:
                countgene += 1
                if verbose > 1:
                    print(gene + ' ' + str(countgene)+'/'+str(len(gene_list)))
                for allele in ds[ds['Hugo_Symbol'] == gene][protein_change_identifier].unique().tolist():
                    if gene in geneallelepair.keys():
                        geneallelepair[gene].append(gene+'_'+allele)
                    else:
                        geneallelepair[gene]=[gene+'_'+allele]
                geneallelepair[gene] = list(set(geneallelepair[gene]))

            allpairlist = []
            for pair in geneallelepair.keys():
                allpairlist = allpairlist + geneallelepair[pair]

            restable = pd.DataFrame(0,index=allpairlist,columns=phenotype.columns.tolist())
            if verbose > 0:
                print('Ceating mutation dataframe.')
            for i in ds[ds['Hugo_Symbol'].isin(gene_list)].index.tolist():
                restable.loc[ds.loc[i]['Hugo_Symbol']+'_'+ds.loc[i][protein_change_identifier],ds.loc[i]['Tumor_Sample_Barcode']] = 1

            size = len(phenotype.iloc[0])
            y = np.ascontiguousarray(np.asarray(phenotype.iloc[0].tolist()))
            miny = findmin(y, size)
            deltay = findmax(y, size) - findmin(y, size)
            bandwidth = calc_bandwidth(y,size)
                
            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                if verbose > 0:
                    print('Generating figures.')
                if combine == True:
                    newgenedf = plotfreq(gct_output_file_prefix, weight_threshold, phenotype, 'all',
                                         restable, figure_format, direction,
                                         out_folder, featurecmap, y, k, bandwidth, thread_number, 
                                         bandwidth_mult, bandwidth_adj, grid, neighborhood, size)
                else:
                    for gene in gene_list:
                        newgenedf = plotfreq(gct_output_file_prefix, weight_threshold, phenotype, gene,
                                         restable.loc[geneallelepair[gene]], figure_format, direction,
                                         out_folder, featurecmap, y, k, bandwidth, thread_number, 
                                         bandwidth_mult, bandwidth_adj, grid, neighborhood, size)

            #Remove gene with more than total_ratio are mutated
            if verbose > 0:
                print('Removing feature with more than total_ratio.')
            for gene in gene_list:
                nummut = seedcomball(restable.loc[geneallelepair[gene]],restable.loc[geneallelepair[gene]].index.tolist()).count(1)
                if nummut > len(phenotype.columns)*total_ratio:
                    restable.drop(labels=geneallelepair[gene],inplace=True)
                    del geneallelepair[gene]

            if verbose > 0:
                print('Calculating IC for all feature.')
            weights = list(np.asarray(rankIC(phenotype, np.array(restable.values.astype(int).tolist()),
                         k, int(len(restable.index)), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid, thread_number, neighborhood)))
            
            counts = []
            for i in restable.index.tolist():
                counts.append(restable.loc[i].sum())
                
            restable['counts'] = counts
            restable['weights'] = weights

            subset = []

            if verbose > 0:
                print('Filtering by frequency and breaking tie with weight.')
            for gene in geneallelepair.keys():
                if direction == 'pos':
                    genedf = restable.loc[geneallelepair[gene]].sort_values(['counts', 'weights'], ascending=[False, False])
                    if frequency_threshold > len(genedf.index.tolist()):
                        newindex = genedf.index.tolist()
                        subset = subset + newindex
                        geneallelepair[gene] = newindex
                    else:
                        newindex = genedf.iloc[:frequency_threshold].index.tolist()
                        subset = subset + newindex
                        geneallelepair[gene] = newindex
                else:
                    genedf = restable.loc[geneallelepair[gene]].sort_values(['counts', 'weights'], ascending=[False, True])
                    if frequency_threshold > len(genedf.index.tolist()):
                        newindex = genedf.index.tolist()
                        subset = subset + newindex
                        geneallelepair[gene] = newindex
                    else:
                        newindex = genedf.iloc[:frequency_threshold].index.tolist()
                        subset = subset + newindex
                        geneallelepair[gene] = newindex

            restable = restable.loc[subset]

            if verbose > 0:
                print('Generating mutation dataframe.')
            resultdf = pd.DataFrame(columns = phenotype.columns)
            for gene in geneallelepair.keys():
                resultdf.loc[gene+'_frequency_'+str(weight_threshold)] = seedcomball(restable.loc[geneallelepair[gene]].iloc[:,:-2],restable.loc[geneallelepair[gene]].iloc[:,:-2].index.tolist())

            if if_gmt == True:
                gmtdf = pd.DataFrame()
                for gene in geneallelepair.keys():
                    gmtsubdf = pd.DataFrame()
                    gmtsubdf[gene+'_frequency_'+str(weight_threshold)] = ['na'] + geneallelepair[gene]
                    gmtdf = pd.concat([gmtdf,gmtsubdf],axis=1)

                gmtdf = gmtdf.T
                gmtdf.to_csv(out_folder+gct_output_file_prefix + '_freqency_'+str(frequency_threshold)+'.gmt', sep= '\t',header=False)

            print('Writing result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_class_'+str(frequency_threshold)+'.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1]))
                resultdf.to_csv(output_file, sep= '\t')

        if mode == 'weight' or mode == 'all' or mode == 'weight_filter':
            print('start making gct by weight')

            # Read Phenotype and only keep intersecting rows.
            phenotype = pd.read_csv(phenotype_file,skiprows=[0,1],sep='\t',index_col=0).drop(columns=['Description'])
            phenotype.columns = phenotype.columns.str.replace("-", "_")
            ds['Tumor_Sample_Barcode'].replace('-','_',regex=True,inplace=True)
            if sample_list != None:
                sample_list_new = []
                for i in sample_list:
                    sample_list_new.append(i.replace('-','_'))
                sample_list = sample_list_new
            # If name is not matching by default, check for subset, if matching, just take intersect
            # This one has to be clear since taking subset to match sometimes cause problem
            print('here')
            if name_match == False:
                print('start matching name')
                # if sample_list != None:
                #     idlist = sample_list
                # else:
                idlist = ds['Tumor_Sample_Barcode'].unique().tolist()
                newcolnames = [] 
                for i in phenotype.columns.tolist():
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound')
                phenotype.columns = newcolnames
                print(newcolnames)

                newcolnames = []
                for i in sample_list:
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound_sl')
                sample_list = newcolnames
                print(newcolnames)

            # Take indicated row of phenotype to use
            if isinstance(phenotype_name,int):
                phenotype = phenotype.iloc[[phenotype_name]]
            else:
                phenotype = phenotype.loc[[phenotype_name]]

            # Take intersecting columns
            if sample_list == None:
                sample_list = list(set(ds['Tumor_Sample_Barcode'].unique().tolist())&set(phenotype.columns.tolist()))
            else:
                sample_list = list(set(sample_list)&set(phenotype.columns.tolist()))
            phenotype = phenotype[sample_list]

            # Sort according to direction
            if direction == 'neg':
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1)
            else:
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1,ascending=False)

            # Normalize phenotype
            # phenotype.iloc[0] = (np.array(phenotype.iloc[0].tolist()) - np.array(phenotype.iloc[0].tolist()).mean())/np.array(phenotype.iloc[0].tolist()).std()
            sample_set = phenotype.columns.tolist()

            ds = ds[ds['Tumor_Sample_Barcode'].isin(list(sample_list))]

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            geneallelepair = {}
            for gene in gene_list:
                for allele in ds[ds['Hugo_Symbol'] == gene][protein_change_identifier].unique().tolist():
                    if gene in geneallelepair.keys():
                        geneallelepair[gene].append(gene+'_'+allele)
                    else:
                        geneallelepair[gene]=[gene+'_'+allele]
                geneallelepair[gene] = list(set(geneallelepair[gene]))

            allpairlist = []
            for pair in geneallelepair.keys():
                allpairlist = allpairlist + geneallelepair[pair]

            restable = pd.DataFrame(0,index=allpairlist,columns=phenotype.columns.tolist())
            for i in ds[ds['Hugo_Symbol'].isin(gene_list)].index.tolist():
                restable.loc[ds.loc[i]['Hugo_Symbol']+'_'+ds.loc[i][protein_change_identifier],ds.loc[i]['Tumor_Sample_Barcode']] = 1

            size = len(phenotype.iloc[0])
            y = np.ascontiguousarray(np.asarray(phenotype.iloc[0].tolist()))
            miny = findmin(y, size)
            deltay = findmax(y, size) - findmin(y, size)
            bandwidth = calc_bandwidth(y,size)
            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                if combine == True:
                    newgenedf = plotweight(gct_output_file_prefix, weight_threshold, phenotype, 'all',
                                           restable, figure_format,
                                           direction, out_folder, featurecmap, y, k, bandwidth, bandwidth_mult, 
                                           bandwidth_adj, grid, thread_number, neighborhood, size)
                else:
                    for gene in gene_list:
                        newgenedf = plotweight(gct_output_file_prefix, weight_threshold, phenotype, gene,
                                           restable.loc[geneallelepair[gene]], figure_format,
                                           direction, out_folder, featurecmap, y, k, bandwidth, bandwidth_mult, 
                                           bandwidth_adj, grid, thread_number, neighborhood, size)

            #Remove gene with more than 40% are mutated
            for gene in gene_list:
                nummut = seedcomball(restable.loc[geneallelepair[gene]],
                                     restable.loc[geneallelepair[gene]].index.tolist()).count(1)
                if nummut > len(phenotype.columns)*total_ratio:
                    restable.drop(labels=geneallelepair[gene],inplace=True)
                    del geneallelepair[gene]

            weights = list(np.asarray(rankIC(y, np.array(restable.values.astype(int).tolist()),
                         k, int(len(restable.index)), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid, thread_number, neighborhood)))

            restable['weights'] = weights
            subset = []
            delgene = []
            if mode == 'weight':
                for gene in geneallelepair.keys():
                    genedf = restable.loc[geneallelepair[gene]]
                    if direction == 'pos':
                        newindex = genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index.tolist()
                        if len(newindex) != 0:
                            subset = subset + newindex
                            geneallelepair[gene] = newindex
                        else:
                            delgene.append(gene)
                    else:
                        newindex = genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights').index.tolist()
                        if len(newindex) != 0:
                            subset = subset + newindex
                            geneallelepair[gene] = newindex
                        else:
                            delgene.append(gene)
                            
            elif mode == 'weight_filter':
                for gene in geneallelepair.keys():
                    genedf = restable.loc[geneallelepair[gene]]
                    if direction == 'pos':
                        if (len(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index) != 0) and (seedcomball(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False),genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index.tolist()).count(1) >= (ratio * seedcomball(genedf,genedf.index.tolist()).count(1))):
                            newindex = genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index.tolist()
                            subset = subset + newindex
                            geneallelepair[gene] = newindex
                        else:
                            delgene.append(gene)
                    else:
                        if (len(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights').index) != 0) and (seedcomball(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights'),genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights').index.tolist()).count(1) >= (ratio * seedcomball(genedf,genedf.index.tolist()).count(1))):
                            newindex = genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights').index.tolist()
                            subset = subset + newindex
                            geneallelepair[gene] = newindex
                        else:
                            delgene.append(gene)

            for gene in delgene:
                del geneallelepair[gene]

            restable = restable.loc[subset]
            resultdf = pd.DataFrame(columns = phenotype.columns)
            for gene in geneallelepair.keys():
                if mode == 'weight': 
                    resultdf.loc[gene+'_weight_'+str(weight_threshold)] = seedcomball(restable.loc[geneallelepair[gene]].iloc[:,:-1],restable.loc[geneallelepair[gene]].iloc[:,:-1].index.tolist())
                elif mode == 'weight_filter': 
                    resultdf.loc[gene+'_weight_filter_'+str(weight_threshold)] = seedcomball(restable.loc[geneallelepair[gene]].iloc[:,:-1],restable.loc[geneallelepair[gene]].iloc[:,:-1].index.tolist())

            if if_gmt == True:
                gmtdf = pd.DataFrame()
                for gene in geneallelepair.keys():
                    gmtsubdf = pd.DataFrame()
                    if mode == 'weight': 
                        gmtsubdf[gene+'_weight_'+str(weight_threshold)] = ['na'] + geneallelepair[gene]
                    elif mode == 'weight_filter': 
                        gmtsubdf[gene+'_weight_filter_'+str(weight_threshold)] = ['na'] + geneallelepair[gene]
                    gmtdf = pd.concat([gmtdf,gmtsubdf],axis=1)

                gmtdf = gmtdf.T
                gmtdf.to_csv(out_folder+gct_output_file_prefix + '_weight_'+str(weight_threshold)+'.gmt', sep= '\t',header=False)

            print('Writing match result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_weight_'+str(weight_threshold)+'.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1]))
                resultdf.to_csv(output_file, sep= '\t')

        if mode == 'allele' or mode == 'all':

            print('start making gct by allele')

            ds['Tumor_Sample_Barcode'].replace('-','_',regex=True,inplace=True)

            if sample_list == None:
                # Make list of sample and its unique index
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
                sample_list = list(sample_set)
            else:
                sample_set = set(sample_list)

            if name_match == False:
                print('start matching name')

                idlist = ds['Tumor_Sample_Barcode'].unique().tolist()
                newcolnames = []
                for i in sample_list:
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound_sl')
                sample_list = newcolnames
                print(newcolnames)

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            geneallelepair = {}

            for gene in gene_list:
                for allele in ds[ds['Hugo_Symbol'] == gene][protein_change_identifier].unique().tolist():
                    if gene in geneallelepair.keys():
                        geneallelepair[gene].append(gene+'_'+allele)
                    else:
                        geneallelepair[gene]=[gene+'_'+allele]
                geneallelepair[gene] = list(set(geneallelepair[gene]))

            allpairlist = []
            for pair in geneallelepair.keys():
                allpairlist = allpairlist + geneallelepair[pair]

            restable = pd.DataFrame(0,index=allpairlist,columns=sample_list,dtype='uint8')

            for i in ds[ds['Hugo_Symbol'].isin(gene_list)].index.tolist():
                restable.loc[ds.loc[i]['Hugo_Symbol']+'_'+ds.loc[i][protein_change_identifier],ds.loc[i]['Tumor_Sample_Barcode']] = 1

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                if combine == True:
                    newgenedf = plotclass(gct_output_file_prefix, 'all', restable,
                                              figure_format, out_folder, featurecmap)
                else:
                    for gene in gene_list:
                        plotclass(gct_output_file_prefix,gene,restable.loc[geneallelepair[gene]],figure_format,out_folder)

            counts = []
            for i in restable.index.tolist():
                counts.append(restable.loc[i].sum())

            restable['counts'] = counts
            restable = restable[restable['counts'] >= frequency_threshold]

            print('writing class result to gct')
            # Prepare writing to gct file
            restable.insert(0, "Description", ['na']*len(restable.index))
            restable.index.name = "Name"
            restable.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_allele.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(restable.shape[0], restable.shape[1] - 1))
                restable.iloc[:,:-1].to_csv(output_file, sep= '\t')

        if mode == 'comb':
            print('start making gct by class')

            if sample_list == None:
                # Make list of sample and its unique index
                sample_set=set()
                for i in ds['Tumor_Sample_Barcode']:
                    sample_set.add(i)
                sample_list = list(sample_set)
            else:
                sample_set = set(sample_list)

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            if gene_list == None:
                gene_list = ds['Hugo_Symbol'].unique().tolist()
            else:
                inte = []
                allgene = ds['Hugo_Symbol'].unique().tolist()
                for i in gene_list:
                    if i in allgene:
                        inte.append(i)
                gene_list = inte

            # Exit if gene in gene list is not in maf file
            if len(gene_list) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)

            genepair = {}

            for gene in gene_list:
                for classification in ds[ds['Hugo_Symbol'] == gene]['Variant_Classification'].unique().tolist():
                    if gene in genepair.keys():
                        genepair[gene].append(gene+'_'+classification)
                    else:
                        genepair[gene]=[gene+'_'+classification]
                genepair[gene] = list(set(genepair[gene]))

            for gene in gene_list:
                for allele in ds[ds['Hugo_Symbol'] == gene][protein_change_identifier].unique().tolist():
                    if gene in genepair.keys():
                        genepair[gene].append(gene+'_'+allele)
                    else:
                        genepair[gene]=[gene+'_'+allele]
                genepair[gene] = list(set(genepair[gene]))

            allpairlist = []
            for pair in genepair.keys():
                allpairlist = allpairlist + genepair[pair]

            restable = pd.DataFrame(0,index=allpairlist,columns=sample_list)

            for i in ds[ds['Hugo_Symbol'].isin(gene_list)].index.tolist():
                restable.loc[ds.loc[i]['Hugo_Symbol']+'_'+ds.loc[i][protein_change_identifier],ds.loc[i]['Tumor_Sample_Barcode']] = 1
                restable.loc[ds.loc[i]['Hugo_Symbol']+'_'+ds.loc[i]['Variant_Classification'],ds.loc[i]['Tumor_Sample_Barcode']] = 1

            if make_figure == True:
                if combine == True:
                    newgenedf = plotclass(gct_output_file_prefix, 'all', restable,
                                              figure_format, out_folder, featurecmap)
                else:
                    for gene in gene_list:
                        newgenedf = plotclass(gct_output_file_prefix,gene,restable.loc[genepair[gene]],figure_format,out_folder)

            #Remove gene with more than total_threshold are mutated
            for gene in gene_list:
                nummut = seedcomball(restable.loc[genepair[gene]],restable.loc[genepair[gene]].index.tolist()).count(1)
                if nummut > len(phenotype.columns)*total_ratio:
                    restable.drop(labels=genepair[gene],inplace=True)
                    del genepair[gene]

            print('writing class result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_all.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')

    else:
        ingct = pd.read_csv(gct_input_file,skiprows=[0,1],sep='\t',index_col=0)
        ingct = ingct.drop(columns=ingct.columns[0])

        if mode == 'class' or mode == 'all':
            subindex = []
            newclass_list = []

            # Add separater in front of each class
            for i in class_list:
                newclass_list.append(class_seperator+i)

            # Check if end of index match any class, if match, put into subindex
            for ind in ingct.index.tolist():
                for oneclass in newclass_list:
                    if len(oneclass) < len(ind) and oneclass == ind[-len(oneclass):]:
                        subindex.append(ind)

            # Remove duplicate and subset dataframe
            subindex = list(set(subindex))
            resultdf = ingct.loc[subindex]

            print('writing class result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to file
            with open(gct_output_file_prefix + '_class.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')

        if mode == 'freq' or mode == 'all':
            print('freq')

            newclass_list = []
            subindex = []

            # Add separater in front of each class
            for i in class_list:
                newclass_list.append(class_seperator+i)

            # Check if end of index match any class, if no match, put into subindex
            for ind in ingct.index.tolist():
                for oneclass in newclass_list:
                    if len(oneclass) < len(ind) and oneclass == ind[-len(oneclass):]:
                        subindex.append(ind)

            # Remove duplicate and subset dataframe
            subindex = list(set(subindex))
            ingct = ingct.loc[~ingct.index.isin(subindex)]

            # Read Phenotype and only keep intersecting rows.
            phenotype = pd.read_csv(phenotype_file,skiprows=[0,1],sep='\t',index_col=0)
            phenotype = phenotype.drop(columns=phenotype.columns[0])
            phenotype.columns = phenotype.columns.str.replace("-", "_")
            if sample_list != None:
                sample_list_new = []
                for i in sample_list:
                    sample_list_new.append(i.replace('-','_'))
                sample_list = sample_list_new

            # If name is not matching by default, check for subset, if matching, just take intersect
            # This one has to be clear since taking subset to match sometimes cause problem
            if name_match == False:
                if sample_list != None:
                    idlist = sample_list
                else:
                    idlist = ingct.columns.tolist()
                newcolnames = [] 
                for i in phenotype.columns.tolist():
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound')

                phenotype.columns = newcolnames

            # Take indicated row of phenotype to use
            if isinstance(phenotype_name,int):
                phenotype = phenotype.iloc[[phenotype_name]]
            else:
                phenotype = phenotype.loc[[phenotype_name]]

            # Take intersecting columns
            if sample_list == None:
                sample_list = list(set(ingct.columns.tolist())&set(phenotype.columns.tolist()))
            else:
                sample_list = list(set(sample_list)&set(phenotype.columns.tolist()))
            phenotype = phenotype[sample_list]

            # Sort according to direction
            if direction == 'neg':
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1)
            else:
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1,ascending=False)

            # Normalize phenotype
            phenotype.iloc[0] = (np.array(phenotype.iloc[0].tolist()) - np.array(phenotype.iloc[0].tolist()).mean())/np.array(phenotype.iloc[0].tolist()).std()
            ingct = ingct[phenotype.columns.tolist()]

            #Remove row with 0 value
            rmrow = []
            for i in ingct.index.tolist():
                if sum(ingct.loc[i]) == 0:
                    rmrow.append(i)
            ingct = ingct.drop(index=rmrow)

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            # Gene is taken as element before first class separator, may not work perfectly, has to restrict input format 
            genenamedic = {}
            if gene_list == None:
                for i in ingct.index.tolist():
                    genename = i[:i.find(class_seperator)]
                    if genename not in list(genenamedic.keys()):
                        genenamedic[genename] = [i]
                    else:
                        genenamedic[genename].append(i)
            else:
                for i in ingct.index.tolist():
                    genename = i[:i.rfind(class_seperator)]
                    if genename not in gene_list:
                        continue
                    if genename not in list(genenamedic.keys()):
                        genenamedic[genename] = [i]
                    else:
                        genenamedic[genename].append(i)

            # If no gene in dictionary, exit
            if len(genenamedic.keys()) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)
                
            size = len(phenotype.iloc[0])
            y = np.ascontiguousarray(np.asarray(phenotype.iloc[0].tolist()))
            miny = findmin(y, size)
            deltay = findmax(y, size) - findmin(y, size)
            bandwidth = calc_bandwidth(y,size)

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                if combine == True:
                    newgenedf = plotfreq(gct_output_file_prefix, weight_threshold, phenotype, 'all', 
                                           ingct, figure_format, direction,
                                           out_folder, featurecmap, y, k, bandwidth, bandwidth_mult, 
                                           bandwidth_adj, grid, thread_number, neighborhood, size)
                else:
                    for gene in genenamedic.keys():
                        newgenedf = plotfreq(gct_output_file_prefix, weight_threshold, phenotype, gene, 
                                               ingct.loc[genenamedic[gene]], figure_format, direction,
                                               out_folder, featurecmap, y, k, bandwidth, bandwidth_mult, 
                                               bandwidth_adj, grid, thread_number, neighborhood, size)

            #Remove gene with more than 40% are mutated
            for gene in list(genenamedic.keys()):
                nummut = seedcomball(ingct[gene],ingct[gene].index.tolist()).count(1)
                if nummut > len(phenotype.columns)*total_ratio:
                    ingct.drop(labels=genenamedic[gene],inplace=True)
                    del genenamedic[gene]

            counts = []
            weights = list(np.asarray(rankIC(phenotype, np.array(ingct.values.astype(int).tolist()),
                         k, int(len(ingct.index)), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid, thread_number, neighborhood)))
            
            for i in ingct.index.tolist():
                counts.append(ingct.loc[i].sum())
#                 weight_vec = phenotype.iloc[0] * ingct.loc[i]
#                 weight = weight_vec.sum()/ingct.loc[i].sum()
#                 weights.append(weight)

            ingct['counts'] = counts
            ingct['weights'] = weights

            subset = []

            for gene in genenamedic.keys():
                if direction == 'pos':
                    genedf = ingct.loc[genenamedic[gene]].sort_values(['counts', 'weights'], ascending=[False, False])
                    if frequency_threshold > len(genedf.index.tolist()):
                        newindex = genedf.index.tolist()
                        subset = subset + newindex
                        genenamedic[gene] = newindex
                    else:
                        newindex = genedf.iloc[:frequency_threshold].index.tolist()
                        subset = subset + newindex
                        genenamedic[gene] = newindex
                else:
                    genedf = ingct.loc[genenamedic[gene]].sort_values(['counts', 'weights'], ascending=[False, True])
                    if frequency_threshold > len(genedf.index.tolist()):
                        newindex = genedf.index.tolist()
                        subset = subset + newindex
                        genenamedic[gene] = newindex
                    else:
                        newindex = genedf.iloc[:frequency_threshold].index.tolist()
                        subset = subset + newindex
                        genenamedic[gene] = newindex

            ingct = ingct.loc[subset]

            resultdf = pd.DataFrame(columns = phenotype.columns)
            for gene in genenamedic.keys():
                resultdf.loc[gene+'_frequency_'+str(weight_threshold)] = seedcomball(ingct.loc[genenamedic[gene]].iloc[:,:-2],ingct.loc[genenamedic[gene]].iloc[:,:-2].index.tolist())

            if if_gmt == True:
                gmtdf = pd.DataFrame()
                for gene in genenamedic.keys():
                    gmtsubdf = pd.DataFrame()
                    gmtsubdf[gene+'_frequency_'+str(weight_threshold)] = ['na'] + genenamedic[gene]
                    gmtdf = pd.concat([gmtdf,gmtsubdf],axis=1)

                gmtdf = gmtdf.T
                gmtdf.to_csv(out_folder+gct_output_file_prefix + '_frequency_'+str(frequency_threshold)+'.gmt', sep= '\t',header=False)


            print('writing top result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_frequency_threshold_'+str(frequency_threshold)+'.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')

        if mode == 'weight' or mode == 'all' or mode == 'weight_filter':
            print('Start running by weight')
            ingct = pd.read_csv(gct_input_file,skiprows=[0,1],sep='\t',index_col=0)
            ingct = ingct.drop(columns=ingct.columns[0])

            newclass_list = []
            subindex = []

            # Add separater in front of each class
            for i in class_list:
                newclass_list.append(class_seperator+i)

            # Check if end of index match any class, if no match, put into subindex
            for ind in ingct.index.tolist():
                for oneclass in newclass_list:
                    if len(oneclass) < len(ind) and oneclass == ind[-len(oneclass):]:
                        subindex.append(ind)

            # Remove duplicate and subset dataframe
            subindex = list(set(subindex))
            ingct = ingct.loc[~ingct.index.isin(subindex)]

            # Read Phenotype and only keep intersecting rows.
            phenotype = pd.read_csv(phenotype_file,skiprows=[0,1],sep='\t',index_col=0)
            phenotype = phenotype.drop(columns=phenotype.columns[0])
            phenotype.columns = phenotype.columns.str.replace("-", "_")
            if sample_list != None:
                sample_list_new = []
                for i in sample_list:
                    sample_list_new.append(i.replace('-','_'))
                sample_list = sample_list_new

            # If name is not matching by default, check for subset, if matching, just take intersect
            # This one has to be clear since taking subset to match sometimes cause problem
            if name_match == False:
                if sample_list != None:
                    idlist = sample_list
                else:
                    idlist = ingct.columns.tolist()
                newcolnames = [] 
                for i in phenotype.columns.tolist():
                    iffind = False
                    for j in idlist:
                        if i in j:
                            newcolnames.append(j)
                            iffind = True
                            break
                    if iffind == False:
                        newcolnames.append('notfound')

                phenotype.columns = newcolnames

            # Take indicated row of phenotype to use
            if isinstance(phenotype_name,int):
                phenotype = phenotype.iloc[[phenotype_name]]
            else:
                phenotype = phenotype.loc[[phenotype_name]]

            # Take intersecting columns
            if sample_list == None:
                sample_list = list(set(ingct.columns.tolist())&set(phenotype.columns.tolist()))
            else:
                sample_list = list(set(sample_list)&set(phenotype.columns.tolist()))
            phenotype = phenotype[sample_list]

            # Sort according to direction
            if direction == 'neg':
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1)
            else:
                phenotype = phenotype.sort_values(by = phenotype.index[0], axis = 1,ascending=False)

            # Normalize phenotype
            phenotype.iloc[0] = (np.array(phenotype.iloc[0].tolist()) - np.array(phenotype.iloc[0].tolist()).mean())/np.array(phenotype.iloc[0].tolist()).std()

            ingct = ingct[phenotype.columns.tolist()]

            #Remove row with 0 value
            rmrow = []
            for i in ingct.index.tolist():
                if sum(ingct.loc[i]) == 0:
                    rmrow.append(i)
            ingct = ingct.drop(index=rmrow)

            # Make gene list with all gene if no gene list input, find intersection if gene list is passed
            # Gene is taken as element before first class separator, may not work perfectly, has to restrict input format 
            genenamedic = {}
            if gene_list == None:
                for i in ingct.index.tolist():
                    genename = i[:i.find(class_seperator)]
                    if genename not in list(genenamedic.keys()):
                        genenamedic[genename] = [i]
                    else:
                        genenamedic[genename].append(i)
            else:
                for i in ingct.index.tolist():
                    genename = i[:i.rfind(class_seperator)]
                    if genename not in gene_list:
                        continue
                    if genename not in list(genenamedic.keys()):
                        genenamedic[genename] = [i]
                    else:
                        genenamedic[genename].append(i)

            # If no gene in dictionary, exit
            if len(genenamedic.keys()) == 0:
                print('Indicataed gene not present in file.')
                sys.exit(1)
                
            size = len(phenotype.iloc[0])
            y = np.ascontiguousarray(np.asarray(phenotype.iloc[0].tolist()))
            miny = findmin(y, size)
            deltay = findmax(y, size) - findmin(y, size)
            bandwidth = calc_bandwidth(y,size)

            # If make figure is True, make one figure with all instances by gene
            if make_figure == True:
                if combine == True:
                    newgenedf = plotweight(gct_output_file_prefix, weight_threshold, phenotype, 'all', 
                                           ingct, figure_format, direction,
                                           out_folder, featurecmap, y, k, bandwidth, bandwidth_mult, 
                                           bandwidth_adj, grid, thread_number, neighborhood, size)

                else:
                    for gene in genenamedic.keys():
                        newgenedf = plotweight(gct_output_file_prefix, weight_threshold, phenotype, gene, 
                                           ingct.loc[genenamedic[gene]], figure_format, direction,
                                           out_folder, featurecmap, y, k, bandwidth, bandwidth_mult, 
                                           bandwidth_adj, grid, thread_number, neighborhood, size)

            #Remove gene with more than 40% are mutated
            for gene in list(genenamedic.keys()):
                nummut = seedcomball(ingct[gene],ingct[gene].index.tolist()).count(1)
                if nummut > len(phenotype.columns)*total_ratio:
                    ingct.drop(labels=genenamedic[gene],inplace=True)
                    del genenamedic[gene]

            weights = list(np.asarray(rankIC(phenotype, np.array(ingct.values.astype(int).tolist()),
                         k, int(len(ingct.index)), size, bandwidth,
                         bandwidth_mult, bandwidth_adj, grid, thread_number, neighborhood)))

            ingct['weights'] = weights

            subset = []

            if mode == 'weight':
                for gene in genenamedic.keys():
                    genedf = ingct.loc[genenamedic[gene]]
                    if direction == 'pos':
                        newindex = genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index.tolist()
                        subset = subset + newindex
                        genenamedic[gene] = newindex
                    else:
                        newindex = genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights').index.tolist()
                        subset = subset + newindex
                        genenamedic[gene] = newindex

            elif mode == 'weight_filter':
                for gene in genenamedic.keys():
                    genedf = ingct.loc[genenamedic[gene]]
                    if direction == 'pos':
                        if (len(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index) != 0) and (seedcomball(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False),genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index.tolist()).count(1) >= (ratio * seedcomball(genedf,genedf.index.tolist()).count(1))):
                            newindex = genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights',ascending=False).index.tolist()
                            subset = subset + newindex
                            genenamedic[gene] = newindex
                    else:
                        if (len(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights').index) != 0) and (seedcomball(genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights'),genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights').index.tolist()).count(1) >= (ratio * seedcomball(genedf,genedf.index.tolist()).count(1))):
                            newindex = genedf[genedf['weights'] >= weight_threshold].sort_values(by='weights').index.tolist()
                            subset = subset + newindex
                            genenamedic[gene] = newindex

            ingct = ingct.loc[subset]

            resultdf = pd.DataFrame(columns = phenotype.columns)
            for gene in genenamedic.keys():
                if mode == 'weight': 
                    resultdf.loc[gene+'_weight_'+str(weight_threshold)] = seedcomball(ingct.loc[genenamedic[gene]].iloc[:,:-1],ingct.loc[genenamedic[gene]].iloc[:,:-1].index.tolist())
                elif mode == 'weight_filter': 
                    resultdf.loc[gene+'_weight_filter_'+str(weight_threshold)] = seedcomball(ingct.loc[genenamedic[gene]].iloc[:,:-1],ingct.loc[genenamedic[gene]].iloc[:,:-1].index.tolist())

            if if_gmt == True:
                gmtdf = pd.DataFrame()
                for gene in genenamedic.keys():
                    gmtsubdf = pd.DataFrame()
                    if mode == 'weight': 
                        gmtsubdf[gene+'_weight_'+str(weight_threshold)] = ['na'] + genenamedic[gene]
                    elif mode == 'weight_filter': 
                        gmtsubdf[gene+'_weight_filter_'+str(weight_threshold)] = ['na'] + genenamedic[gene]
                    gmtdf = pd.concat([gmtdf,gmtsubdf],axis=1)

                gmtdf = gmtdf.T
                gmtdf.to_csv(out_folder+gct_output_file_prefix + '_weight_'+str(weight_threshold)+'.gmt', sep= '\t',header=False)

            print('writing top result to gct')
            # Prepare writing to gct file
            resultdf.insert(0, "Description", ['na']*len(resultdf.index))
            resultdf.index.name = "Name"
            resultdf.columns.name = None

            # Write to gct file
            with open(out_folder+gct_output_file_prefix + '_weight_'+str(weight_threshold)+'.gct', mode = "w") as output_file:
                output_file.writelines("#1.2\n{}\t{}\n".format(resultdf.shape[0], resultdf.shape[1] - 1))
                resultdf.to_csv(output_file, sep= '\t')

    if gzip == True:
        with tarfile.open(gct_output_file_prefix+'.tar.gz', "w:gz") as tar:
            tar.add(out_folder, arcname=os.path.basename(out_folder))