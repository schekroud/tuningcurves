#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:25:35 2024

@author: sammichekroud
"""
import numpy as np
import scipy as sp
import pandas as pd
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import time
import multiprocessing as mp
import statsmodels.api as sma


loc = 'laptop'
if loc == 'workstation':
    wd = 'C:/Users/sammirc/Desktop/postdoc/tuningcurves'
    # sys.path.insert(0, op.join(wd, 'analysis', 'tools'))
elif loc == 'laptop':
    wd = '/Users/sammichekroud/Desktop/postdoc/tuningcurves'
sys.path.insert(0, op.join(wd, 'analysis', 'tools'))
os.chdir(wd)
from funcs import getSubjectInfo
import TuningCurveFuncs as tcf


subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
nsubs = subs.size

#get bin structure for data we want to load in
weightTrials = True #whether trial contributions to mean activity for a reference bin are weighted by distance from the bin centre
# binstep 4  options for binwidth: 11, 16, 22
# binstep 15 options for binwidth: 10, 15, 22
binstep, binwidth = 4, 22
# binstep, binwidth = 15, 22
_, binmids, binstarts, binends = tcf.createFeatureBins(binstep = binstep, binwidth = binwidth,
                                                       feature_start = -90+binstep, feature_end = 90)
thetas = np.cos(np.radians(binmids))
binmidsrad = np.deg2rad(binmids)

times = np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy'))
#some parameters to control what the model is doing
smooth_alphas=True

i = 4
for i in subs:
    data = np.load(op.join(wd, 'data', 'tuningcurves',
                           f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'))
    [nitems, ntrials, nbins, ntimes] = data.shape
    # bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv')) #read in associated behavioural data
    
    # #save data as matlab structure
    # sp.io.savemat(op.join(wd, 'data', 'tuningcurves',
    #                        f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.mat'),
    #               mdict = dict(distances=data))
    
    data = data * -1 #sign flip mahalanobis distances so that larger (more positive) values reflect more similar representations
    d = data.copy()
    dm = data.copy()
    dz = data.copy()
    dminmax = data.copy()
       
    imean = np.mean(data.copy(), axis=2, keepdims=True) #get mean distance per item, trial and time point
    dm = np.subtract(d.copy(), imean) #demean the distances across orientation bins
    
    #we also want to zscore the single trial distances to have a copy that has [mean, var] [0, 1]
    dz = sp.stats.zscore(dz, axis=2) #zscore along the feature bin axis
    
    #minmax scale so distances are between 0 and 1 now (closer to 1 = closest)
    imin = np.min(data.copy(), axis=2, keepdims = True)
    imax = np.max(data.copy(), axis=2, keepdims = True)
    dminmax = d.copy()
    dminmax = np.divide(np.subtract(dminmax, imin), np.subtract(imax, imin)) #scales distances between 0 and 1
    
    
    tmpd = d.mean(axis=1).mean(0)
    
    y = d.mean(1).mean(0)
    dmy = y - y.mean()
    zy = sp.stats.zscore(y, ddof=0)
    yminmax = (y-y.min())/(y.max()-y.min())
    
    def fmin_func(params, x, binmids):
        '''
        x       - distances for this time point and trial
        params  - parameters to try on this iteration
        binmids - radian values for bin middles being used  
        '''
        [b1, alpha] = params
        fitted = (b1 * np.cos(alpha * binmids))
        resids = np.subtract(x, fitted)
        ssr = np.sum(np.power(resids,2))
        return ssr
    
    def bcosfit(thetas, b1, alpha):
        return b1*np.cos(thetas*alpha)
    def bcosfit_demeaned(thetas, b1, alpha):
        return b1 * (np.cos(thetas*alpha) - np.cos(thetas*alpha).mean())
    
    
    results = np.zeros(shape = [2, ntimes]) * np.nan
    results2 = np.zeros(shape = [2, ntimes]) * np.nan
    
    for tp in range(ntimes):
        iy = y[:,tp]
        yminmax = (iy - iy.min())/(iy.max()-iy.min())
        
        res = sp.optimize.fmin(fmin_func,
                         x0 = [1, 0.5], #initial parameter guesses
                         args = (yminmax, binmidsrad), disp=0)
        results[:, tp] = res
        
        res2 = sp.optimize.curve_fit(bcosfit,
                                     xdata = binmidsrad,
                                     ydata = yminmax,
                                     #p0 = [1, 0],
                                     method='trf')[0]
        results2[:,tp] = res2
        
    
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(times, results[1], color = 'k', lw = 2, label = 'fmin'); ax.set_ylabel('alpha')
    ax.plot(times, results2[1], color = 'r', lw = 1, label = 'curve fit')
    ax.plot(times, sp.ndimage.gaussian_filter1d(results[1], 3), color='g', lw = 1, label = 'smoothed')
    ax.legend()
    ax = fig.add_subplot(212)
    ax.plot(times, results[0], color = 'k', lw = 2, label = 'fmin'); ax.set_ylabel('b1')
    ax.plot(times, results2[0], color='r', lw = 1, label = 'curve fit')
    ax.legend()
    
    
    alphas = results[1].copy()
    #alphas = sp.ndimage.gaussian_filter1d(alphas, 2)
    
    
    glmfit = np.zeros(shape = [2, ntimes]) * np.nan
    cfit  = np.zeros(shape = [ntimes]) * np.nan
    for tp in range(ntimes):
        iy = y[:,tp]
        iy = iy - iy.mean()
        ia = alphas[tp].copy()
        
        desmat = np.cos(binmidsrad*ia) - np.cos(binmidsrad*ia).mean()
        # gl = sma.GLM(endog = iy, exog = desmat, family = sma.families.Gaussian())
        gl = sma.regression.linear_model.OLS(endog = iy, exog = desmat, hasconst =  False)
        glfit = gl.fit()
        ib, it = glfit.params[0], glfit.tvalues[0]
        glmfit[:,tp] = [ib, it]
        
        icfit = sp.optimize.curve_fit(lambda thetas, b1: bcosfit(thetas, b1, ia),
                              xdata = binmidsrad,
                              ydata = iy,
                              p0 = [1], method='trf')[0]
        cfit[tp] = icfit[0]
        
    
    fig = plt.figure()
    ax  = fig.add_subplot(311)
    ax.plot(times, glmfit[0], lw = 1, color='b', label = 'beta'); ax.set_ylabel('beta')
    ax.axhline(0, ls = 'dashed', lw = 0.5, color='k')
    ax.set_ylim([-0.02, 0.02])
    ax = fig.add_subplot(312)
    ax.plot(times, glmfit[1], lw = 1, color = 'r', label = 't'); ax.set_ylabel('t-value')
    ax.axhline(0, ls = 'dashed', lw = 0.5, color='k')
    ax = fig.add_subplot(313)
    ax.plot(times, cfit, lw = 1, color= 'k', label = 'cfit beta')
    
    
    
    
    #%%
    
    
    
    
    
    
    #set up objective function to minimise over
    
    def cosfmin(x, thetas, alpha, beta):
        fitted = beta * np.cos(thetas*alpha)
        ssr = np.power(np.subtract(x, fitted),2).sum()
        return ssr
    
    def cosfmin(x, thetas, alpha, beta):
        fitted = beta * np.cos(thetas*alpha)
        ssr = np.power(np.subtract(x, fitted),2).sum()
        return ssr
        
    
    
    alphas = np.zeros(shape = [nitems, ntrials, ntimes]) * np.nan
    for iitem in range(nitems):
        for itrl in range(ntrials):
            for tp in range(ntimes):
                fit = sp.optimize.fmin(cosfmin,
                                       x0 = [0.5, 1],
                                       args = (dminmax[iitem, itrl, :,tp], binmidsrad))[0]
                alphas[iitem, itrl, tp] = fit
                
    if smooth_alphas:
        alphas = sp.ndimage.gaussian_filter1d(alphas, sigma = 3)
    am = alphas.mean(0) #average across items
    ams = sp.ndimage.gaussian_filter1d(am, 3)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(times,ams.T, alpha = 0.3, lw = 0.5)
    ax.fill_between(times,
                    np.add(ams.mean(0), sp.stats.sem(ams, axis=0)),
                    np.subtract(ams.mean(0), sp.stats.sem(ams, axis=0)),
                    color='k', alpha = 0.3)
    ax.plot(times, ams.mean(0), lw = 1, color='k')
    
    def lvl2_cosfmin(x, thetas, b1):
        dmcos = np.cos(thetas)
        fitted = np.multiply(b1, dmcos)
        ssr = np.power(np.subtract(x, fitted),2).sum()
        return ssr
    
    betas = np.zeros(shape = [nitems, ntrials, ntimes]) * np.nan
    for iitem in range(nitems):
        for itrl in range(ntrials):
            for tp in range(ntimes):
                bfit = sp.optimize.fmin(lvl2_cosfmin,
                                        x0 = [1],
                                        args = (dm[iitem, itrl,:,tp], np.multiply(binmidsrad,alphas[iitem, itrl, tp])),
                                        disp=0)[0]
                betas[iitem, itrl, tp] = bfit
    
    bm = betas.mean(0) #average across items
    bms = sp.ndimage.gaussian_filter1d(bm, 3)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(times,bms.T, alpha = 0.3, lw = 0.5)
    ax.fill_between(times,
                    np.add(bms.mean(0), sp.stats.sem(bms, axis=0)),
                    np.subtract(bms.mean(0), sp.stats.sem(bms, axis=0)),
                    color='k', alpha = 0.7)
    ax.plot(times, bms.mean(0), lw = 1, color='k')
    
    
    fig = plt.figure();
    ax = fig.add_subplot(211)
    ax.plot(times, bms.mean(0))
    ax.axhline(0, ls = 'dashed', color='k')
    ax = fig.add_subplot(212)
    ax.plot(times, ams.mean(0))
    
    
    tmp =np.multiply(bms, ams)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times, tmp.mean(0))
    
    
    
    
    
    
    
    