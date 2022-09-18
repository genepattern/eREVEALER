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

cdef extern from "limits.h":
    int RAND_MAX
cdef int RANDOM_SEED = 20121020

cdef struct density:
    np.double_t d
    np.uint32_t* cnt
    
cdef np.double_t DELMAX=1000

import matplotlib.pyplot as plt
import plotly.express as px
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
cpdef int calculategridsize(
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
    cdef np.double_t  pX_0, pX_1, pysum, py0sum, py1sum
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
    
    return grid

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.int_t[:] calcGrids(
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

    cdef np.int_t[:] res = np.zeros(n,dtype=int)
    cdef np.double_t[:] y = np.asarray(y_in)
    cdef np.int_t[:,:] x = np.asarray(xs_in)

    cdef int i
    cdef int maxseedid = 0
    cdef np.double_t maxCIC = 0.0
    cdef np.double_t miny = findmin(y, size)
    cdef np.double_t deltay = findmax(y, size) - findmin(y, size)
    
    for i in prange(n, nogil=True,num_threads=thread_number):
        res[i] = calculategridsize(y, x[i,:],k,size, bandwidth,bandwidth_mult,
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
cpdef runCheckGrid(target_file='no', # gct file for target(continuous or binary)
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
                  neighborhood = 4
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
        if verbose != 0:
            print('All core Used')
        thread_number = mp.cpu_count()
    report = report + 'Number of thread used: ' + str(thread_number) + '\n'
    
    if seed_file == 'no':
        seed_file = feature_file
    if isinstance(feature_df,str):
        comb,locusdic = readInput(target_file = target_file, feature_file = feature_file, 
                         seed_file = seed_file, seed_name = seed_name, target_name = target_name,
                         direction = direction, gene_locus = gene_locus, normalize = normalize,
                         low_threshold = low_threshold, high_threshold = high_threshold,
                         grid = grid, subset = subset, gene_set = gene_set, 
                         gene_separator = gene_separator, thread_number = thread_number)
    else:
        comb = pd.concat([target_df,seed_df,feature_df], join='inner')
        rmrow=[]
        for i in comb.index.tolist()[2:]:
            if sum(comb.loc[i].tolist()) <= low_threshold or \
               sum(comb.loc[i].tolist()) >= high_threshold:
                rmrow.append(i)
        comb = comb.drop(index=rmrow)
        origcomb = comb.copy()
        seed_name = ['firstseed']
        locusdic = None
    if verbose != 0:
        print('Time used to read input: %s second(s)'%(float(time.time() - start)))
    
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
            gridlist = REVEALERInner(prefix = prefix[ind], comb = comb, 
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
                                                           neighborhood = neighborhood)
        
    elif mode == 'single':
        gridlist = REVEALERInner(prefix = prefix, comb = comb, grid = grid, 
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
                                                       neighborhood = neighborhood)
        
    return gridlist

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
    neighborhood
    ):

    seedcmap = clr.LinearSegmentedColormap.from_list('custom greys', [(.9,.9,.9),(0.5,0.5,0.5)], 
                                                     N=256)
    featurecmap = clr.LinearSegmentedColormap.from_list('custom greys', 
                                                        [(<double>(176)/<double>(255),
                                                        <double>(196)/<double>(255),
                                                        <double>(222)/<double>(255)),
                                                        (0,0,<double>(139)/<double>(255))], N=256)


    newpheno = []
    for i in comb.iloc[0].tolist():
        newpheno.append(np.sign(i)*(abs(i)**alpha))
    comb.iloc[0] = newpheno
    
    if verbose != 0:
        if seed_name != None:
            print("Number of features that pass the threshold is: "+str(len(comb.index)-2))
            print("Number of sample is: " + str(len(comb.columns)))
        else:
            print("Number of features that pass the threshold is: "+str(len(comb.index)-1))
            print("Number of sample is: " + str(len(comb.columns)))

    if seed_name != None:
        report = report + 'Number of features passing threshold is: ' + str(len(comb.index)-2)+'\n'
    else:
        report = report + 'Number of features passing threshold is: ' + str(len(comb.index)-1)+'\n'
    report = report + "Number of sample is: " + str(len(comb.columns)) + '\n'

    start= time.time()
    cdef int size = len(comb.iloc[0])
    cdef np.double_t[:] y = np.ascontiguousarray(np.asarray(comb.iloc[0].tolist()))
    cdef np.double_t miny = findmin(y, size)
    cdef np.double_t deltay = findmax(y, size) - findmin(y, size)
    i = 0
    cdef np.double_t bandwidth = calc_bandwidth(y,size)                                    
                                                
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
        print('bandwidth: '+str(bandwidth))

    report = report + 'Universal bandwidth is: ' + str(bandwidth) + '\n'

    savecomb = comb.copy()
    seedlists = []

    gridlist = np.array(calcGrids(y, np.array(comb.iloc[1:].values.astype(int).tolist()),
                             k, int(len(comb.index)-1), size, bandwidth,
                             bandwidth_mult, bandwidth_adj, direction,grid,thread_number, neighborhood))
    
    plt.hist(gridlist)
    plt.title('Grid Distribution of '+prefix+'when k = '+str(k))
    plt.savefig(out_folder + prefix+'_grid_k'+str(k)+'.'+
                figure_format, format=figure_format)
    plt.close()
    return gridlist
        

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
    normalize,      #Name of normalization method
    low_threshold,  #Lower threshold to filter feature
    high_threshold, #Higher threshold to filter feature
    grid,           #Gird size
    subset,         #List to subset target file if no subset, indicated 'no'
    gene_set,
    gene_separator,
    thread_number
    ):

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
    if low_threshold < 1:
        low_threshold = len(comb.iloc[0].tolist()) * low_threshold
    if high_threshold < 1:
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
