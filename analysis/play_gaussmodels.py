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
from TuningCurveFuncs import createFeatureBins


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

times = np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy'))
#some parameters to control what the model is doing

i = 4
data = np.load(op.join(wd, 'data', 'tuningcurves',
                       f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'))
[nitems, ntrials, nbins, ntimes] = data.shape
# bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv')) #read in associated behavioural data

#save data as matlab structure
sp.io.savemat(op.join(wd, 'data', 'tuningcurves',
                       f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.mat'),
              mdict = dict(distances=data))

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
#%%
d = data.mean(1).mean(0) #average across trials then items
tp=61
# tp=1
for tp in [1, 61]:
    print(f'running on tp = {tp}')
    tmp = d.copy()[:,tp] #average across trials then items
    tmpdz = sp.stats.zscore(d.copy()[:,tp], ddof=0) #zscore the above
    tmpdm = np.subtract(tmp.copy(), tmp.mean())
    tmpdminmax = np.divide(tmp.copy()-tmp.min(), tmp.max()-tmp.min())

    # fig = plt.figure(figsize=[15,3])
    # fig.suptitle(f'tp = {tp}')
    # ax=fig.add_subplot(141); ax.plot(binmids, tmp, color='k', label = 'raw inverted'); ax.legend(loc='lower center')
    # ax=fig.add_subplot(142); ax.plot(binmids, tmpdz, color='k', label = 'zscored'); ax.legend(loc='lower center')
    # ax=fig.add_subplot(143); ax.plot(binmids, tmpdm, color='k', label = 'demeaned'); ax.legend(loc='lower center')
    # ax=fig.add_subplot(144); ax.plot(binmids, tmpdminmax, color='k', label = 'minmax scaled'); ax.legend(loc='lower center')
    # ax.plot(binmids, tmpdm2, color='r', label = 'tp0')
    
    #cosine modelling
    #estimating alpha
    alphafit1 = sp.optimize.curve_fit(tcf.basic_cosine_demean, 
                                      xdata = binmidsrad, 
                                      ydata = tmpdz, p0 = [1],
                                      bounds = ([0], [3]))[0]
    print(f'fit 1 [cos(alpha*theta) - cos(alpha*theta).mean()], alpha = {alphafit1.round(3)}')
    fitted1 = np.cos(binmidsrad*alphafit1) - np.cos(binmidsrad*alphafit1).mean()
    
    alphafit2 = sp.optimize.curve_fit(tcf.basic_cosine,
                                      xdata = binmidsrad, ydata = tmpdz, p0 = [1],
                                      bounds = ([0], [3]))[0]
    print(f'fit 2 (basic cosine), alpha = {alphafit2.round(3)}')
    fitted2 = np.cos(binmidsrad*alphafit2)
    
    alphafit3 = sp.optimize.curve_fit(tcf.b0_cosine,
                                      xdata = binmidsrad, ydata = tmpdz,
                                      bounds = ([-np.inf, 0], [np.inf, 3]))[0]
    print(f'fit 3 [b0 + cos(alpha*theta)], alpha = {alphafit3.round(3)[1]}')
    fitted3 = np.add(alphafit3[0], np.cos(binmidsrad * alphafit3[1]))
    
    alphafit4 = sp.optimize.curve_fit(tcf.b0_cosine,
                                      xdata = binmidsrad, ydata = tmpdminmax, #p0 = [0.5],
                                      bounds = ([-np.inf, 0], [np.inf, 3]))[0]
    # print(f'fit 4 [cos(alpha*theta)] on minmax data, alpha = {alphafit4.round(3)}\n')
    print(f'fit 4 [b0 + cos(alpha*theta)] on minmax data, alpha = {alphafit4.round(3)}\n')
    fitted4   = np.add(alphafit4[0], np.cos(alphafit4[1]*binmidsrad))
    
    #this highlights (in the second panel) that actually rescaling the distances between [0,1], preserving shape
    #allows the curve fit to actually incorporate data along the full length of orientations as all values are positive
    #demeaning the data along Y in the fit doesn't actually accommodate this fully
    fig = plt.figure(figsize=[15,3])
    fig.suptitle(f'tp = {tp}')
    ax = fig.add_subplot(141); ax.plot(binmidsrad, tmpdz, lw = 1.5, color='k');
    ax.plot(binmidsrad, fitted1, lw = 1, color = 'r', label = 'cos - cos.mean()'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    ax.set_title(f'alpha = {alphafit1.round(3)[0]}')
    ax = fig.add_subplot(142); ax.plot(binmidsrad, tmpdz, lw = 1.5, color='k');
    ax.plot(binmidsrad, fitted2, lw = 1, color = 'r', label = 'cos'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    ax.set_title(f'alpha = {alphafit2.round(3)[0]}')
    ax = fig.add_subplot(143); ax.plot(binmidsrad, tmpdz, lw = 1.5, color='k');
    ax.plot(binmidsrad, fitted3, lw = 1, color = 'r', label = 'b0+cos(alpha*theta)'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    ax.set_title(f'alpha = {alphafit3.round(3)[1]}')
    ax = fig.add_subplot(144); ax.plot(binmidsrad, tmpdminmax, lw = 1.5, color='k');
    ax.plot(binmidsrad, fitted4, lw = 1, color = 'r', label = 'b0+cos(alpha*theta) minmax'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    ax.set_title(f'alpha = {alphafit4.round(3)[1]}')
#%%
#similar but gaussian modelling
tp=61
# tp=1
for tp in [1, 61]:
    print(f'running on tp = {tp}')
    tmp = d.copy()[:,tp] #average across trials then items
    tmpdz = sp.stats.zscore(d.copy()[:,tp], ddof=0) #zscore the above
    tmpdm = np.subtract(tmp.copy(), tmp.mean())
    tmpdminmax = np.divide(tmp.copy()-tmp.min(), tmp.max()-tmp.min())
    
    # xdat = sp.stats.norm.pdf(binmidsrad)
    xdat = binmidsrad
    #gaussian modelling, estimating the width of the distribution that describes the distances across orientation
    
    #first, model with just a gaussian with mean 0, and estimate sigma
    fit1 = sp.optimize.curve_fit(lambda x, sigma: tcf.gaussfunc(x, 0, sigma), #fix mu at 0 as middle bin is centred on zero
                                 xdata = xdat,
                                 ydata = tmpdz, method = 'trf')[0]
    print(f'fit 1: [Z(d) ~ gauss(x)], sigma = {fit1.round(3)[0]}')
    fitted = tcf.gaussfunc(xdat, 0, fit1[0])
    
    fit2 = sp.optimize.curve_fit(lambda x, sigma, b0: tcf.gauss_fullmodel(x, 0, sigma, b0, 1),#fix mu = 0, b1=1 (removed cos multiplier))
                                 xdata = xdat,
                                 ydata = tmpdz, method = 'trf')[0]
    print(f'fit 2: [b0 + (Z(d) ~ gauss(x))], sigma = {fit2.round(3)[0]}')   
    fitted2 = fit2[1] + tcf.gaussfunc(xdat, mu = 0, sigma = fit2[0])
                                 
    
    #second, model with a gaussian that is demeaned over Y, and estimate sigma
    fit3 = sp.optimize.curve_fit(lambda x, sigma: tcf.gauss_demeaned(x, 0, sigma), #fix mu at 0 as middle bin is centred on zero
                                 xdata = xdat,
                                 ydata = tmpdz, method = 'trf')[0]
    fitted3 = tcf.gaussfunc(xdat, 0, fit3[0]) - tcf.gaussfunc(xdat, 0, fit3[0]).mean()
    print(f'fit 3: [Z(d) ~ gauss(x) - gauss(x).mean()], sigma = {fit3.round(3)[0]}')                             
    
    fit4 = sp.optimize.curve_fit(lambda x, sigma, b0: tcf.gauss_demeaned_fullmodel(x, 0, sigma, b0, 1), #fix mu = 0, b1=1 (removed cos multiplier))
                                 xdata = xdat, 
                                 ydata = tmpdz, method = 'trf')[0]
    fitted4 = fit4[1] + (tcf.gauss_demeaned(xdat, 0, fit4[0]) - tcf.gauss_demeaned(xdat, 0, fit4[0]).mean())
    print(f'fit 4: [b0 + (Z(d) ~ gauss(x) - gauss(x).mean() )], sigma = {fit4.round(3)[0]}')                             
    
                         
    
    fig = plt.figure(figsize=[15,3])
    fig.suptitle(f'tp = {tp}')
    ax = fig.add_subplot(141); ax.plot(binmids, tmpdz, lw = 1.5, color='k');
    ax.plot(binmids, fitted, lw = 1, color = 'r', label = 'gauss(x)'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    ax.set_title(f'sigma = {fit1.round(5)[0]}', fontsize=8)
    ax = fig.add_subplot(142); ax.plot(binmids, tmpdz, lw = 1.5, color='k');
    ax.plot(binmids, fitted2, lw = 1, color = 'r', label = 'b0 + gauss(x)'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    ax.set_title(f'sigma = {fit2.round(5)[0]}', fontsize=8)
    ax = fig.add_subplot(143); ax.plot(binmids, tmpdz, lw = 1.5, color='k');
    ax.plot(binmids, fitted3, lw = 1, color = 'r', label = 'g(x)-g(x).mean() fit'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    ax.set_title(f'sigma = {fit3.round(5)[0]}', fontsize=8)
    ax = fig.add_subplot(144); ax.plot(binmids, tmpdz, lw = 1.5, color='k');
    ax.plot(binmids, fitted4, lw = 1, color = 'r', label = 'b0 + (g(x)-g(x).mean()) fit'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    ax.set_title(f'sigma = {fit4.round(5)[0]}', fontsize=8)    
    
    # ax = fig.add_subplot(143); ax.plot(binmidsrad, tmpdz, lw = 1.5, color='k');
    # ax.plot(binmidsrad, fitted3, lw = 1, color = 'r', label = 'b0+cos(alpha*theta)'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    # ax.set_title(f'alpha = {alphafit3.round(3)[1]}')
    # ax = fig.add_subplot(144); ax.plot(binmidsrad, tmpdminmax, lw = 1.5, color='k');
    # ax.plot(binmidsrad, fitted4, lw = 1, color = 'r', label = 'b0+cos(alpha*theta) minmax'); ax.legend(loc='lower center', frameon=False, fontsize=8)
    # ax.set_title(f'alpha = {alphafit4.round(3)[1]}')
    
    
    
    
    
    
    

    