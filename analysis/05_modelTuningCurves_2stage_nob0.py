# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:25:10 2024

@author: sammirc
"""

import numpy as np
import scipy as sp
import pandas as pd
import sklearn as skl
from sklearn import *
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import time
%matplotlib

import progressbar
progressbar.streams.flush()

loc = 'workstation'
if loc == 'workstation':
    wd = 'C:/Users/sammirc/Desktop/postdoc/tuningcurves'
    # sys.path.insert(0, op.join(wd, 'analysis', 'tools'))
elif loc == 'laptop':
    wd = '/Users/sammichekroud/Desktop/postdoc/tuningcurves'
sys.path.insert(0, op.join(wd, 'analysis', 'tools'))
os.chdir(wd)
from funcs import getSubjectInfo
import TuningCurveFuncs
from TuningCurveFuncs import createFeatureBins, visualise_FeatureBins


subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
nsubs = subs.size

#get bin structure for data we want to load in
weightTrials = True #whether trial contributions to mean activity for a reference bin are weighted by distance from the bin centre
# binstep, binwidth = 4, 11 #4 degree jumps, 22 degree full width
# binstep, binwidth = 4, 16 #4 degree jumps, 32 degree full width
# binstep, binwidth = 4, 22 #4 degree jumps, 44 degree full width
binstep, binwidth = 15, 10 #jumps of 15 degrees, 20 degree full width 
binstep, binwidth = 15, 15 #jumps of 15 degrees, 30 degree full width
# binstep, binwidth = 15, 22 #jumps of 15 degrees, 44 degree full width
_, binmids, binstarts, binends = createFeatureBins(binstep = binstep, binwidth = binwidth,
                                                       feature_start = -90+binstep, feature_end = 90)
thetas = np.cos(np.radians(binmids))
binmidsrad = np.deg2rad(binmids)

visualise_bins = False
if visualise_bins:
    visualise_FeatureBins(binstarts, binmids, binends)
times = np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy'))
#some parameters to control what the model is doing
smooth_alphas = True


subcount = 0
for i in subs:
    subcount += 1
    print(f'\n\nworking on participant {subcount}/{nsubs}')
    sub = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    
    if smooth_alphas:
        smooth_sigma = 3
        print('modelling with smooth alpha values')
    elif not smooth_alphas:
        smooth_sigma = None
        print('modelling with raw alpha values')
    
    data = np.load(op.join(wd, 'data', 'tuningcurves',
                           f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'))
    [nitems, ntrials, nbins, ntimes] = data.shape
    
    bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv')) #read in associated behavioural data
    
    data = data * -1 #sign flip mahalanobis distances so that larger (more positive) values reflect more similar representations
    d = data.copy()
    dm = data.copy()
    dz = data.copy()
    #demean the vector of distances at each time point, separately for each item and trial
    # for iitem in range(nitems):
    #     for itrl in range(ntrials):
    #         for tp in range(ntimes):
    #             dm[iitem, itrl,:, tp] = np.subtract(d[iitem, itrl, :, tp], np.mean(d[iitem, itrl, :, tp]))#.mean()) #demean
    
    #if this loop above gets prohibitively slow, then you can also do this below: (it's different because finite precision isnt associative even if the math operation is) 
    imean = np.mean(data.copy(), axis=2, keepdims=True)
    dm = np.subtract(d.copy(), imean)
    
    #we also want to zscore the single trial distances to have a copy that has [mean, var] [0, 1]
    dz = sp.stats.zscore(dz, axis=2) #zscore along the feature bin axis
    
    # first, fit a fixed cosine-only model to estimate the shape of the best fitting cosine - cos(alpha * theta)
    # this is done on the standardised (z-scored) distances which preserves the *shape* at each time point but standardises amplitude
    # we should fit this separately for each 'decoded' item in the stimulus array
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # the code below is pretty slow when dealing with larger amounts of data, so probably want to find a way of parallelising
    # at *least* parallelising each decoded item independently, but probably parallelising
    # per time point instead to allow more core use
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    print('estimating alpha on z-scored distances, per trial and timepoint')
    tic = time.time()
    alphas = np.zeros(shape = [nitems, ntrials, ntimes]) * np.nan #we get one alpha value for cosine width across distances, per tp and trial
    for iitem in range(nitems):
        for itrl in range(ntrials):
            for tp in range(ntimes):
                dztp = dz[iitem, itrl, :, tp].copy() #get z-scored distances across reference bins (standardised tuning curve for this trial)
                tpa = TuningCurveFuncs.fitAlpha(binmidsrad, dztp, bounds = ([0], [1])) #constrain alpha such that 0 <= alpha <= 1
                alphas[iitem, itrl, tp] = tpa[0]
    toc=time.time()
    print(f'- - alpha fitting took {int(divmod(toc-tic, 60)[0])}m{round(divmod(toc-tic, 60)[1])}s')
    
    
    if smooth_alphas:
        alphas = sp.ndimage.gaussian_filter1d(alphas, sigma = smooth_sigma) #defaults to smooth over the last axis, which here is time
    
    print('estimating beta on demeaned distances, per stimulus, trial and timepoint')
    tic=time.time()
    betas = np.zeros(shape = [nitems, ntrials, ntimes]) * np.nan
    for iitem in range(nitems):
        for itrl in range(ntrials):
            for tp in range(ntimes):
                dtp = dm[iitem, itrl, :, tp].copy() #get demeaned, inverted distances across bins
                
                tpfit = sp.optimize.curve_fit(TuningCurveFuncs.b1_cosine, #fitting cosine model with fixed alpha and only B1, using demeaned distances
                                              xdata = np.multiply(binmidsrad, alphas[iitem, itrl, tp]), #pre-multiply theta by previously estimated alpha
                                              ydata = dtp,
                                              p0 = [1], #initial guess is the mean distance + B1 scaling of 1 (0+1)
                                              maxfev = 5000, method = 'trf', nan_policy='omit')[0]
                betas[iitem, itrl, tp] = tpfit[0]
    toc=time.time()
    print(f'- - beta fitting took {int(divmod(toc-tic, 60)[0])}m{round(divmod(toc-tic, 60)[1])}s')
    
    # ta = alphas.copy().mean(1).mean(0) #average across trials
    # tb = betas.copy().mean(1).mean(0) #average across trials
    
    # fig=plt.figure(figsize = [6, 12])
    # ax=fig.add_subplot(211)
    # ax.plot(times, ta)
    # ax=fig.add_subplot(212)
    # ax.plot(times, sp.ndimage.gaussian_filter1d(tb, sigma=3))
    
    #save alphas and betas
    np.save(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'fixedalpha_b1only',
                    f's{i}_ParamFits_Alpha_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{smooth_sigma}_no_bzero.npy'), arr = alphas)
    np.save(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'fixedalpha_b1only',
                    f's{i}_ParamFits_Beta_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{smooth_sigma}_no_bzero.npy'), arr = betas)
    
    
    
    
    
    
    