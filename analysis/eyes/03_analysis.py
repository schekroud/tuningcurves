#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:16:08 2024

@author: sammichekroud
"""
import numpy as np
import pandas as pd
import scipy as sp
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import statsmodels as sm
import statsmodels.api as sma
%matplotlib
import glmtools as glm


loc = 'workstation'
if loc == 'laptop':
    #eyefuncdir = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools'
    eyefuncdir = '/Users/sammichekroud/Desktop/postdoc/tools'
    wd         = '/Users/sammichekroud/Desktop/postdoc/wmconfidence' #working on confidence data, but in postdoc dir
elif loc == 'workstation':
    eyefuncdir = 'C:/Users/sammirc/Desktop/postdoc/tools/'
    wd         =  'C:/Users/sammirc/Desktop/postdoc/tuningcurves'
os.chdir(wd)
sys.path.insert(0, eyefuncdir)
#import eyefuncs_v2 as eyes
import eyefuncs as eyes

eyedir = op.join(wd, 'data', 'eyes')
bdir   = op.join(wd, 'data', 'datafiles')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,             13, 14, 15,     17,         20, 21, 22,     24, 25, 26])

nsubs = subs.size


#set some params here
nregs = 3
b = np.zeros(shape = [nsubs, nregs, 3000]) * np.nan
t = np.zeros(shape = [nsubs, nregs, 3000]) * np.nan

#ppts x tuning curve timepoints x regressors x
b = np.zeros(shape = [nsubs, 175, nregs, 1750]) * np.nan
t = np.zeros(shape = [nsubs, 175, nregs, 1750]) * np.nan


subcount=-1
for sub in subs:
    subcount +=1
    # if not op.exists(op.join(eyedir, 'preprocessed', f'EffDS{sub}_preprocessed.pickle')): #dont do this if it already exists!
    print(f'\n- - - working on ppt {sub} - - -')
    data = eyes.io.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_pupil_arraylocked.pickle'))
    # data = data.apply_baseline([-0.5, -0.2])
    trlcheck = np.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_arraylocked_nanperc.npy')) #this always has as many trials as the eyetracking data does
    #load in the parameter fits for this participant
    binstep, binwidth = 4, 22
    smooth_alphas, smooth_sigma = True, 3
    
    alpha = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'fixedalpha_b1only',
                    f's{sub}_ParamFits_Alpha_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{smooth_sigma}_no_bzero.npy'))
    b1 = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'fixedalpha_b1only',
                    f's{sub}_ParamFits_Beta_binstep{binstep}_binwidth{binwidth}_smoothedAlpha_{smooth_alphas}{smooth_sigma}_no_bzero.npy'))
    tcbdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{sub}_TuningCurve_metadata.csv')) #read in associated behavioural data
    modeltimes = np.round(np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy')), 2)
    
    #average parameter estimates across items, as we dont have pupillometry separately for each item
    alpha = alpha.mean(0)
    b1    = b1.mean(0)
    
    #need to align the trials in the eyetracking data with the trials in the tuning curve data:
    keeptrleyes = trlcheck<=30
    tcbdata = tcbdata.assign(trlid = np.where(tcbdata.session=='a', tcbdata.trialnum, tcbdata.trialnum+256))
    data.metadata = data.metadata.assign(trlid = np.where(data.metadata.session=='a', data.metadata.trialnum, data.metadata.trialnum+256))

    #first, drop eyetracking trials with bad eyetracking
    data.metadata = data.metadata.loc[keeptrleyes]
    data.data = data.data[keeptrleyes]
    data.event_id = data.event_id[keeptrleyes]
    
    #align trials between the now-cleaned eyetracking and the eeg tuning curves
    tctrls = tcbdata.trlid.to_numpy()
    eyetrls = data.metadata.trlid.to_numpy()
    sametrls = np.intersect1d(tctrls, eyetrls)
    
    #make booleans for indexing
    tckeep = np.isin(tctrls, sametrls)
    ekeep  = np.isin(eyetrls, sametrls)
    
    #get the right tuning curve data
    alpha = alpha[tckeep]
    b1    = b1[tckeep]
    tcbdata = tcbdata.loc[tckeep]
    
    #get the right eyetracking data
    data.metadata = data.metadata.loc[ekeep]
    data.data = data.data[ekeep]
    data.event_id = data.event_id[ekeep]
    
    allb = np.zeros(shape = [modeltimes.size, nregs, data.times.size])
    allt = np.zeros(shape = [modeltimes.size, nregs, data.times.size])
    print('- - - - -  running glm across tuning curve timepoints - - - - -')
    for itp in range(modeltimes.size):
        #get parameter estimates for this timepoint, across trials    
        ia = alpha[:, itp]
        ib1 = b1[:,itp]
    
        error = data.metadata.absrdif.to_numpy()
        err = np.power(error, 0.5)
        acc = np.multiply(error, -1) #flip sign, so more positive = more accurate
        
        DC = glm.design.DesignConfig()
        # if iglm == 0:
        DC.add_regressor(name = 'mean', rtype = 'Constant')
        # DC.add_regressor(name ='accuracy', rtype = 'Parametric', datainfo = 'acc', preproc = 'z')
        # DC.add_regressor(name = 'accuracy', rtype = 'Parametric', datainfo = 'error', preproc = None)
        DC.add_regressor(name = 'alpha', rtype = 'Parametric', datainfo = 'alpha', preproc = 'z')
        DC.add_regressor(name = 'b1', rtype = 'Parametric', datainfo = 'beta1', preproc = 'z')
        DC.add_simple_contrasts()
        
        #create glmdata object
        glmdata = glm.data.TrialGLMData(data = data.data.squeeze(), time_dim = 1, sample_rate = 1000,
                                        #add in metadata that's used to construct the design matrix
                                        acc=acc, error = error,
                                        alpha = ia, beta1 = ib1
                                        )
    
        glmdes = DC.design_from_datainfo(glmdata.info)
        
        model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model 
            
        betas = model.betas.copy()
        copes = model.copes.copy()
        tstats = model.tstats.copy()
        allb[itp] = betas.copy()
        allt[itp] = tstats.copy()
    
    
    # plotb = 1
    # fig = plt.figure(figsize=[15, 4])
    # ax = fig.add_subplot(121)
    # plotd = allb[:,plotb,:].copy().squeeze()
    # plot = ax.imshow(plotd, aspect = 'auto', interpolation = 'gaussian', cmap='RdBu_r', origin = 'lower',
    #           extent = [data.times.min(), data.times.max(), modeltimes.min(), modeltimes.max()])
    # ax.set_ylabel('tuning curve time (s)')
    # ax.set_xlabel('time relative to array onset')
    # ax.set_title(f'regressor = {model.regressor_names[plotb]}')
    # fig.colorbar(plot)
    # ax = fig.add_subplot(122)
    # plotd = allb[:,plotb+1,:].copy().squeeze()
    # plot = ax.imshow(plotd, aspect = 'auto', interpolation = 'gaussian', cmap='RdBu_r', origin = 'lower',
    #           extent = [data.times.min(), data.times.max(), modeltimes.min(), modeltimes.max()])
    # ax.set_ylabel('tuning curve time (s)')
    # ax.set_xlabel('time relative to array onset')
    # ax.set_title(f'regressor = {model.regressor_names[plotb+1]}')
    # fig.colorbar(plot)
    
    
    # fig.colorbar(plot)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plotd =  allt[modeltimes==0.3, 1:].squeeze()
    # ax.plot(data.times, plotd.T, lw = 1, label = model.regressor_names[1:])
    # ax.axhline(0, lw = 0.5, ls = 'dashed', color = 'k')
    # ax.legend()
    
        # fig = plt.figure(figsize = [6, 3])
        # ax = fig.add_subplot(111)
        # ax.plot(data.times, tstats.T, label = model.regressor_names)
        # ax.legend()
        
    b[subcount] = allb.copy()
    t[subcount] = allt.copy()
    
    #save individual betas and tstats
    # np.save(op.join(wd, 'data', 'glms', 'glm1', f'wmc_s{sub:02d}_glm1_betas.npy'), arr = allb)
    # np.save(op.join(wd, 'data', 'glms', 'glm1', f'wmc_s{sub:02d}_glm1_betas.npy', arr = allt)
    
#%%
plt.close('all')
bm = np.nanmean(b, axis=0)
bsem = sp.stats.sem(b, axis=0, ddof = 0, nan_policy='omit')
tm = np.nanmean(t, axis=0)


fig = plt.figure(figsize = [16, 4])
for ireg in range(3):
    if ireg >=1:
        vmin, vmax = -10, 10
    else:
        vmin, vmax = -600, 600
    ax = fig.add_subplot(1, 3, ireg+1)
    p = ax.imshow(bm[:,ireg], aspect='auto', interpolation='None', cmap = 'RdBu_r', origin = 'lower', vmin = vmin, vmax = vmax,
              extent = [data.times.min(), data.times.max(), modeltimes.min(), modeltimes.max()])
    ax.set_title(f'regressor: {model.regressor_names[ireg]}')
    fig.colorbar(p)

fig = plt.figure(figsize = [16, 4])
for ireg in range(3):
    ax = fig.add_subplot(1, 3, ireg+1)
    p = ax.imshow(tm[:,ireg], aspect='auto', interpolation='None', cmap = 'RdBu_r', origin = 'lower', 
              extent = [data.times.min(), data.times.max(), modeltimes.min(), modeltimes.max()])
    ax.set_title(f'regressor: {model.regressor_names[ireg]}')
    fig.colorbar(p)

mstarts = [-0.3, 0.0, 0.1, 0.2, 0.3, 0.4]
mends   = [-0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
nplots = len(mstarts)
cols = ['#3182bd', '#d95f0e']
fig = plt.figure(figsize = [16, 4])
for iplot, istart, iend in zip(range(nplots), mstarts, mends):
    # print(iplot, istart, iend)
    ax = fig.add_subplot(1, nplots, iplot+1)
    tinds = np.logical_and(modeltimes >= istart, modeltimes <= iend)
    plotd = bm[tinds, 1:].copy().mean(0) #average across the time window
    plotsem = bsem[tinds, 1:].copy().mean(0)
    for ireg in range(2):
        ax.plot(data.times, plotd[ireg], label = model.regressor_names[1:][ireg], lw = 1, c = cols[ireg])
        ax.fill_between(data.times,
                        np.add(plotd[ireg], plotsem[ireg]),
                        np.subtract(plotd[ireg], plotsem[ireg]),
                        edgecolor=None, alpha = 0.3, color = cols[ireg])
    ax.axhline(0, lw = 1, ls = 'dashed', color='k')
    ax.axvline(0, lw = 1, ls = 'dashed', color='k')
    ax.set_title(f'{istart} < tc time < {iend}', fontsize = 8)
    ax.set_ylim([-12, 10])
    ax.set_xlabel('eyetracker time rel to array(s)', fontsize = 8)
    ax.legend(loc = 'lower right', frameon=False)
fig.tight_layout()



