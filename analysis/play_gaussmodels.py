# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:07:45 2024

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
from scipy import stats
import time
import multiprocessing as mp
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
# binstep 4  options for binwidth: 11, 16, 22
# binstep 15 options for binwidth: 10, 15, 22
binstep, binwidth = 4, 22
# binstep, binwidth = 15, 22
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
    dminmax = data.copy()
   
    #if this loop above gets prohibitively slow, then you can also do this below: (it's different because finite precision isnt associative even if the math operation is) 
    imean = np.mean(data.copy(), axis=2, keepdims=True)
    dm = np.subtract(d.copy(), imean) #demean the distances across orientation bins
    
    #we also want to zscore the single trial distances to have a copy that has [mean, var] [0, 1]
    dz = sp.stats.zscore(dz, axis=2) #zscore along the feature bin axis
    
    #minmax scale so distances are between 0 and 1 now (closer to 1 = closest)
    imin = np.min(data.copy(), axis=2, keepdims = True)
    imax = np.max(data.copy(), axis=2, keepdims = True)
    dminmax = d.copy()
    dminmax = np.divide(np.subtract(dminmax, imin), np.subtract(imax, imin)) #scales distances between 0 and 1
    
    
    sigmas = np.zeros(shape = [nitems, ntrials, 2, ntimes]) * np.nan #2 because fitting 2 params, b0 and sigma
    alphas = np.zeros(shape = [nitems, ntrials, ntimes]) * np.nan
    for iitem in range(nitems):
        bar = progressbar.ProgressBar(initial_value = 0, max_value = ntrials)
        for itrl in range(ntrials):
            bar.update(itrl)
            for tp in range(ntimes):
                tpfit = TuningCurveFuncs.fitAlpha(binmidsrad,
                                                  dminmax[iitem, itrl, :, tp].copy(),
                                                  bounds = ([0], [3]))
                alphas[iitem, itrl, tp] = tpfit[0]

    am = alphas.mean(1).mean(0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times, am, lw = 1.5)
    ax.axvline(0, ls = 'dashed', lw = 1, color = '#bdbdbd')
    # ax.axvline(0, ls = 'dashed', lw = 1, color = '#bdbdbd')
    
    
    d = data.mean(1).mean(0) #average across trials then items
    #%%
    tp=61
    # tp=1
    for tp in [1, 61]:
        print(f'running on tp = {tp}')
        tmp = d.copy()[:,tp] #average across trials then items
        tmpdz = sp.stats.zscore(d.copy()[:,tp], ddof=0) #zscore the above
        tmpdm = np.subtract(tmp.copy(), tmp.mean())
        tmpdminmax = np.divide(tmp.copy()-tmp.min(), tmp.max()-tmp.min())
    
        fig = plt.figure(figsize=[15,3])
        fig.suptitle(f'tp = {tp}')
        ax=fig.add_subplot(141); ax.plot(binmids, tmp, color='k', label = 'raw inverted'); ax.legend(loc='lower center')
        ax=fig.add_subplot(142); ax.plot(binmids, tmpdz, color='k', label = 'zscored'); ax.legend(loc='lower center')
        ax=fig.add_subplot(143); ax.plot(binmids, tmpdm, color='k', label = 'demeaned'); ax.legend(loc='lower center')
        ax=fig.add_subplot(144); ax.plot(binmids, tmpdminmax, color='k', label = 'minmax scaled'); ax.legend(loc='lower center')
        # ax.plot(binmids, tmpdm2, color='r', label = 'tp0')
        
        #cosine modelling
        #estimating alpha
        alphafit1 = sp.optimize.curve_fit(TuningCurveFuncs.basic_cosine_demean, 
                                          xdata = binmidsrad, 
                                          ydata = tmpdz, p0 = [1],
                                          bounds = ([0], [3]))[0]
        print(f'fit 1 [cos(alpha*theta) - cos(alpha*theta).mean()], alpha = {alphafit1.round(3)}')
        
        alphafit2 = sp.optimize.curve_fit(TuningCurveFuncs.basic_cosine,
                                          xdata = binmidsrad, ydata = tmpdz, p0 = [1],
                                          bounds = ([0], [3]))[0]
        print(f'fit 2 (basic cosine), alpha = {alphafit2.round(3)}')
        
        alphafit3 = sp.optimize.curve_fit(TuningCurveFuncs.b0_cosine,
                                          xdata = binmidsrad, ydata = tmpdz,
                                          bounds = ([-np.inf, 0], [np.inf, 3]))[0]
        print(f'fit 3 [b0 + cos(alpha*theta)], alpha = {alphafit3.round(3)[1]}')
        fitted3 = np.add(alphafit3[0], np.cos(binmidsrad * alphafit3[1]))
        
        alphafit4 = sp.optimize.curve_fit(TuningCurveFuncs.b0_cosine,
                                          xdata = binmidsrad, ydata = tmpdminmax, #p0 = [0.5],
                                          bounds = ([-np.inf, 0], [np.inf, 3]))[0]
        # print(f'fit 4 [cos(alpha*theta)] on minmax data, alpha = {alphafit4.round(3)}\n')
        print(f'fit 4 [b0 + cos(alpha*theta)] on minmax data, alpha = {alphafit4.round(3)}\n')
        fitted4   = np.add(alphafit4[0], np.cos(alphafit4[1]*binmidsrad))
        
        #this highlights (in the second panel) that actually rescaling the distances between [0,1], preserving shape
        #allows the curve fit to actually incorporate data along the full length of orientations as all values are positive
        #demeaning the data along Y in the fit doesn't actually accommodate this fully
        fig = plt.figure(figsize=[8,3])
        fig.suptitle(f'tp = {tp}')
        ax = fig.add_subplot(121)
        ax.plot(binmidsrad, tmpdz, lw = 1.5, color = 'k', label='zscored distances')
        ax.plot(binmidsrad, np.cos(binmidsrad * alphafit1), lw = 1, color = 'r', label = 'fit1 demeaned cos')
        ax.plot(binmidsrad, np.cos(binmidsrad * alphafit2), lw = 1, color = 'g', label = 'fit2 basic cos')
        ax.plot(binmidsrad, fitted3, lw = 1, color = 'blue', label = 'b0 cosine fit')
        ax.legend(loc='lower center', fontsize = 8, frameon=False)
        ax=fig.add_subplot(122)
        ax.plot(binmidsrad, tmpdminmax, lw = 1.5, color = 'k', label = 'minmax distances')
        ax.plot(binmidsrad, fitted4, lw = 1, color = 'r', label = 'basic cos minmax')
        ax.legend(loc='lower center', fontsize = 8, frameon=False)
#%%
    
    
    
    fit1 = TuningCurveFuncs.fit_width(tmpdm, binmids)
    fit2 = TuningCurveFuncs.fit_width(tmpdminmax, binmids)
    fit3 = TuningCurveFuncs.fitAlpha(binmidsrad, tmpdminmax, bounds = ([0], [3]))
    
    fitted1 = TuningCurveFuncs.gaussfunc(binmids, mu = 0, sigma = fit1)
    fitted2 = TuningCurveFuncs.gaussfunc(binmids, mu = 0, sigma = fit2)
    fitted3 = np.cos(fit3[0]* binmidsrad)
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(binmids, tmpdm, color = 'k', lw = 1.5)
    ax.plot(binmids, fitted1, color = 'r', lw = 1)
    ax = fig.add_subplot(122)
    ax.plot(binmids, tmpdminmax, color = 'k', lw = 1.5)
    ax.plot(binmids, fitted2, color = 'r', lw = 1)
    ax.plot(binmids, fitted3, color = 'b', lw = 1)
    
    
    cosalpha = TuningCurveFuncs.fitAlpha(binmids, tmp, bounds = ([0], [1]))
    
    def gaussfunc_zeromean(x, sigma):
        return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-((x-0)**2/(2 * sigma**2)))
    
    fitsigma = sp.optimize.curve_fit(gaussfunc_zeromean,
                                     xdata = binmids,
                                     ydata = tmpdz)[0]
    
    fitted = gaussfunc_zeromean(binmids, sigma = curvefit[0])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(binmids, tmpdz, color='k', lw = 1.5)
    ax.plot(binmids, fitted, color='r', lw = 1)
    
    
    def gaussfunc_fullbeta(x, B0, B1, sigma):
        return B0 + B1 * (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-((x-0)**2/(2 * sigma**2)))
    
    def gaussfunc_b1only(x, B1, sigma):
        return B1 * (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-((x-0)**2/(2 * sigma**2)))
   
    #estimate beta
    b1fit = sp.optimize.curve_fit(lambda x, B1: gaussfunc_b1only(x, B1, fitsigma),
                                  xdata = binmids,
                                  ydata = tmpdm)[0]
    
    betafit = sp.optimize.curve_fit(lambda x, B0, B1: gaussfunc_fullbeta(x, B0, B1, fitsigma),
                                    xdata = binmids,
                                    ydata = tmp)[0]
    
    fullfit = np.add(betafit[0], np.multiply(betafit[1], sp.stats.norm.pdf(binmids, loc = 0, scale = fitsigma)))
    fitb1only = np.multiply(b1fit, sp.stats.norm.pdf(binmids, loc = 0, scale = fitsigma))
    
    fig = plt.figure(figsize = [12, 4])
    ax = fig.add_subplot(121)
    ax.plot(binmids, tmp, lw = 1.5, color='k', label = 'raw distances')
    ax.plot(binmids, fullfit, lw = 1, color = 'red', label = 'incl B0')
    ax.legend(loc='lower left', frameon=False)
    ax=fig.add_subplot(122)
    ax.plot(binmids, tmpdm, lw = 1.5, color = 'k', label = 'demeaned distances')
    ax.plot(binmids, fitb1only, lw = 1, color = 'blue', label = 'b1 only')
    ax.legend(loc='lower left', frameon=False)
    
    