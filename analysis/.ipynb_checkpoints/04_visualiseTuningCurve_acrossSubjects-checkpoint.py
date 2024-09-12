# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:40:19 2024

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
from TuningCurveFuncs import makeTuningCurve, getTuningCurve_FullSpace, createFeatureBins, visualise_FeatureBins

os.chdir(wd)
subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#1,2,3,10,19 all have only 1 session. 23 something wrong in the second session, completely unusable eeg data due to serious noise
nsubs = subs.size
#set params for what file to load in per subject
binstep  = 15
binwidth = 22

times = np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy'))
ntimes = times.size
nitems = 2 #two items are presented in the array, we 'decode' both

nbins, binmids, binstarts, binends = createFeatureBins(binstep = binstep, binwidth = binwidth,
                                                       feature_start = -75, feature_end = 90)

alldata = np.zeros(shape = [subs.size, nitems, binmids.size, ntimes]) * np.nan #2 because 2 items decoded
subcount = -1
for i in subs:
    subcount +=1
    print(f'working on ppt {subcount+1}/{subs.size}')
    
    #read in single subject data
    data = np.load(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}.npy'))
    bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv'))
    [nitems, ntrials, nbins, ntimes] = data.shape
    
    data = data * -1 #invert this, so more positive (larger) values = closer (mahalnobis distances are small when test is close to train)
    meandata = np.nanmean(data, axis=1)
    alldata[subcount] = meandata
    
#%%

#for each participant and decoded item, demean across bins at each time point
gmean_tp = alldata.copy()
for isub in range(nsubs):
    for iitem in range(nitems):
        for tp in range(ntimes):
            gmean_tp[isub, iitem, :, tp] = np.subtract(gmean_tp[isub, iitem, :, tp], gmean_tp[isub, iitem, :, tp].mean())

gmean_ave = gmean_tp.mean(axis=1) #average across items
gmean_ave = gmean_ave.mean(0) #average across participants now
#%%

fig = plt.figure(figsize = [6, 4])
ax = fig.add_subplot(111)
ax.imshow(gmean_ave, aspect='auto', origin = 'lower', cmap = 'RdBu_r', interpolation='none',
         extent = [times.min(), times.max(), binmids.min(), binmids.max()])
ax.set_ylabel(f'orientation')
ax.set_xlabel('time relative to array onset')


tstarts = [-0.3, 0,   0.2, 0.4, 0.6, 0.8]
tends   = [ 0,   0.2, 0.4, 0.6, 0.8, 1.0]
plotcount = np.arange(len(tstarts))+1
fig = plt.figure(figsize = [15, 3])
for x in zip(plotcount, tstarts, tends):
    ax = fig.add_subplot(1,6,x[0])
    tinds = np.logical_and(np.greater_equal(times, x[1]), np.less_equal(times, x[2]))
    tmpdat = np.nanmean(gmean_ave[:,tinds], axis=1) #average across time
    ax.plot(binmids, tmpdat)
    ax.set_xlabel('orientation')
    ax.set_title(f'distances {x[1]}$\\rightarrow${x[2]}s')
    ax.set_ylim([-0.05, 0.05])
fig.tight_layout()


