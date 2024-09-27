# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:51:06 2024

@author: sammirc
"""
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as skl
import multiprocessing as mp
import statsmodels.api as sma
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
import TuningCurveFuncs as tcf
from TuningCurveFuncs import minmax


subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
nsubs = subs.size

#get bin structure for data we want to load in
weightTrials = True #whether trial contributions to mean activity for a reference bin are weighted by distance from the bin centre
# binstep 4  options for binwidth: 11, 16, 22
# binstep 15 options for binwidth: 10, 15, 22
binstep, binwidth = 4, 22
# binstep, binwidth = 4, 11
# binstep, binwidth = 15, 22
# binstep, binwidth = 15, 15
_, binmids, binstarts, binends = tcf.createFeatureBins(binstep = binstep, binwidth = binwidth,
                                                       feature_start = -90+binstep, feature_end = 90)
thetas = np.cos(np.radians(binmids))
binmidsrad = np.deg2rad(binmids)

visualise_bins = False
if visualise_bins:
    tcf.visualise_FeatureBins(binstarts, binmids, binends)
times = np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'times_500hz.npy'))
#some parameters to control what the model is doing
smooth_alphas = True
smooth_distances = False #smooth distances before parameter estimation
smoothsig = 10

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
    
    data = np.load(op.join(wd, 'data', 'tuningcurves', '500hz',
                           f's{i}_TuningCurve_mahaldists_500Hz_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'))
    [nitems, ntrials, nbins, ntimes] = data.shape
    
    bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv')) #read in associated behavioural data
    
    if smooth_distances:
        data = sp.ndimage.gaussian_filter1d(data, sigma = smoothsig) #smooth with gaussian of 10 sample sd, which is 20ms
    
    data = data * -1 #sign flip mahalanobis distances so that larger (more positive) values reflect more similar representations
    d = data.copy()
    dm = data.copy()
    dminmax = data.copy()
    
    #demean the vector of distances at each time point, separately for each item and trial
    imean = np.mean(data.copy(), axis=2, keepdims=True)
    dm = np.subtract(d.copy(), imean)
    
    # we also want a version of the data that is minmax scaled across orientation bins
    # for that, we need a few things:
    dmin = d.min(axis=2, keepdims=True)
    dmax = d.max(axis=2, keepdims=True)
    dminmax = np.divide(np.subtract(dminmax, dmin), np.subtract(dmax, dmin)) #scales bin distances linearly between 0 and 1
    
    #alpha was already previously estimated, so we can just read it in here
    if op.exists(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'twostage_alphaminmaxfit',
                         f's{i}_ParamFits_Alpha_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{smooth_sigma}.npy')):
        print(f'\n- -  loading in previously estimated tuning curve precisions')
        alphas = np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'twostage_alphaminmaxfit',
                f's{i}_ParamFits_Alpha_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{smooth_sigma}.npy'))
    else:
        #first, fit a B1*cos(precision*theta) model to the rescaled data to get best fitting alpha
        #note that multiprocessing this per time point doesn't seem to actually make it any faster **at all** (its slower actually)
        print('estimating tuning curve precision, per trial and timepoint')
        tic = time.time()
        # alphas = np.zeros(shape = [nitems, ntrials, ntimes]) * np.nan #we get one alpha value for cosine width across distances, per tp and trial
        step1params = np.zeros(shape = [nitems, ntrials, 2, ntimes]) * np.nan #2 as we are fitting 2 params: b1 (we dont care) and alpha (we care)
        #b1 is only fit here to allow a flexible fit of alpha that captures the data shape properly
        for iitem in range(nitems):
            bar = progressbar.ProgressBar(min_value=0, max_value = ntrials, initial_value=0)
            for itrl in range(ntrials):
                bar.update(itrl)
                for tp in range(ntimes):
                    iminmax = dminmax[iitem, itrl,:,tp].copy()
                    res = sp.optimize.curve_fit(lambda x, B1, alpha: tcf.fullCosineModel(x, 0, B1, alpha), #fixes B0 at 0 (removes from model, keeps everything else)
                                                      xdata = binmidsrad,
                                                      ydata = iminmax,
                                                      p0 = [1, 1], #initialise both parameters at 1
                                                      bounds = ([-np.inf, 0], [np.inf, 3]), #b1 unbounded, alpa between 0.001 and 3
                                                      maxfev = 5000, method = 'trf', nan_policy='omit')#[0]
                    iparams = res[0]
                    ise = np.sqrt(np.diag(res[1])) #sqrt of the variance in the parameter estimate (equivalent to glmfit.bse)
                    it  = iparams/ise
                    step1params[iitem, itrl, :, tp] = [iparams[1], it[1]] #store the parameter estimate and t-value for precision
        toc=time.time()
        print(f'- - tuning curve precision modelling took {int(divmod(toc-tic, 60)[0])}m{round(divmod(toc-tic, 60)[1])}s')
        
        #note, constraining alpha is only essential for the estimation of amplitude as you cannot fit a glm where cos(theta) is a vector of ones
        #it's better to constrain this in the precision estimation step though so that the t-value is accurate
        
        if smooth_alphas:
            step1params = sp.ndimage.gaussian_filter1d(step1params, sigma = smoothsig) #smooth with sd gaussian blur
                
        
        alphas = step1params[:,:,0,:].copy() #get just the parameter estimate for tuning curve precision
        #we aren't going to smooth this parameter estimate as we smoothed the distances instead
        
    constrain_alpha = True #because we constrain in the precision estimation step instead
    
    
    #at the second stage, we want to fit tuning curve amplitude, with both a glm and an optimisation routine
    print('estimating tuning curve amplitude, per stimulus, trial and timepoint')
    tic=time.time()
    betas = np.zeros(shape = [nitems, ntrials, ntimes]) * np.nan
    glmfit = np.zeros(shape = [nitems, ntrials, 2, ntimes]) * np.nan #2 because storing beta & t-value
    optfit = np.zeros(shape = [nitems, ntrials, 2, ntimes]) * np.nan #2 because storing best fit param & t value for it
    for iitem in range(nitems):
        bar = progressbar.ProgressBar(min_value=0, max_value = ntrials, initial_value=0)
        for itrl in range(ntrials):
            bar.update(itrl)
            for tp in range(ntimes):
                #get the demeaned (inverse) distances across orientation bins for this timepoint
                iy = d[iitem, itrl, :, tp].copy() #get raw, inverted distances
                iypos = np.add(iy, np.abs(iy.min())) #get distances in an exclusively positive way. preserves shape of distances across orientations
                
                ia = alphas[iitem, itrl, tp]
                if constrain_alpha:
                    ia = max(ia, 0.001) #constrain alpha so that it is never lower than 0.001, which can prevent model fitting
                
                desmat = minmax(np.cos(binmidsrad*ia)) #minmax scale the cosine that is weighted by the pre-estimated alpha value

                
                gl = sma.GLM(endog = iypos, exog = desmat, family = sma.families.Gaussian())
                glfit = gl.fit()
                ib, it = glfit.params[0], glfit.tvalues[0] #get the fitted beta weight and t-value
                glmfit[iitem, itrl, :, tp] = [ib, it] #store them
                
                #now estimate the same tuning curve amplitude parameter but using an optimisation method instead of a glm
                res = sp.optimize.curve_fit(lambda x, B1: tcf.fullCosineModel(x, 0, B1, ia), #fix b0 at zero (demeaned) and fix alpha at pre-estimated alpha)
                                            xdata = binmidsrad,
                                            ydata = iypos,
                                            p0 = [1], bounds = ([-np.inf], [np.inf]),
                                            maxfev = 5000, method='trf', nan_policy='omit')
                glfitted = glfit.fittedvalues;
                optfitted = res[0]*np.cos(binmidsrad*ia)
                opt_mse = np.power(np.subtract(iy, optfitted),2).sum()
                gl_mse  = glfit.deviance #np.power(np.subtract(iy, glfitted),2).sum()
                
                optfitse = np.sqrt(np.diag(res[1])) #sqrt of the variance in the parameter estimate (equiv to glmfit.bse)
                optfitT  = res[0]/optfitse
                
                #store the optimised amplitude parameter (beta) and its t-value
                optfit[iitem, itrl, :, tp] = np.array([res[0], optfitT]).flatten()
    toc=time.time()
    print(f'- - amplitude modelling took {int(divmod(toc-tic, 60)[0])}m{round(divmod(toc-tic, 60)[1])}s')
    
    
    # #compare optimize fit vs glmfit of second level
    # #average across items and trials
    # glfit_gm  = np.nanmean(np.nanmean(glmfit.copy(),axis=1), axis=0) #average across trials then items, has shape [2 x ntimes] for [beta, tvalue]
    # optfit_gm = optfit.copy().mean(1).mean(0) 
    # alphafit  = alphas.mean(1).mean(0)
    
    # fig = plt.figure(figsize = [9, 6])
    # ax = fig.add_subplot(321); ax.plot(times, glfit_gm[0], lw = 1, color='k'); ax.set_ylabel('beta'); ax.set_title('glmfit')
    # ax = fig.add_subplot(322); ax.plot(times, optfit_gm[0], lw = 1, color='b'); ax.set_ylabel('beta'); ax.set_title('optimisation fit')
    # ax = fig.add_subplot(323); ax.plot(times, glfit_gm[1], lw = 1, color='k'); ax.set_ylabel('t-value')
    # ax = fig.add_subplot(324); ax.plot(times, optfit_gm[1], lw = 1, color='b'); ax.set_ylabel('t-value')
    # ax = fig.add_subplot(3,1,3); ax.plot(times, alphafit, lw = 2, color='g'); ax.set_ylabel('alpha')
    # fig.tight_layout()
    
    
    
    if i==4:
        #for the first participant, write to file what model parameters are used
        model_settings = dict() 
        model_settings['smooth_alphas'] = smooth_alphas
        model_settings['smooth_distances'] = smooth_distances
        model_settings['smoothing'] = smoothsig
        model_settings['constrain_alpha'] = constrain_alpha
        model_settings['minmax_stage1'] = True
        model_settings['minmax_designmatrix_stage2'] = True
        
        np.save(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'twostage_alphaminmaxfit', 'model_settings.npy'), model_settings)
    
    #save tuning curve model parameters
    np.save(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'twostage_alphaminmaxfit',
                f's{i}_ParamFits_precision_binstep{binstep}_binwidth{binwidth}.npy'), arr = step1params)
    
    np.save(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'twostage_alphaminmaxfit',
            f's{i}_ParamFits_amplitude_binstep{binstep}_binwidth{binwidth}_glmfit.npy'), arr = glmfit)
    
    np.save(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'twostage_alphaminmaxfit',
            f's{i}_ParamFits_amplitude_binstep{binstep}_binwidth{binwidth}_optfit.npy'), arr = optfit)
    