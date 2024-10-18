# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:22:02 2024

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
figdir = op.join(wd, 'figures')

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
times = np.round(np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'times_500hz.npy')), decimals=2)

#%% visualise the bin structure across trials
binwidth = int((binends[0]-binstarts[0])/2)
nanglesdeg = np.arange(binstarts.min(), binends.max()+1, 1)
nangles = np.deg2rad(nanglesdeg) #conv to radians
tmpcurves = np.zeros(shape = [binstarts.size, nangles.size]) * np.nan #create empty thing to populate to show angle curves for bins
tmpbins   = np.zeros_like(tmpcurves)
binangs   = np.zeros(shape = [binstarts.size, (binwidth*2)+1]) #store angles for each bin
for ibin in range(binstarts.size):
    istart, iend = binstarts[ibin], binends[ibin] #get the angles
    istartrad, iendrad = np.deg2rad(istart), np.deg2rad(iend)
    # iangles = np.arange(istart, iend)
    iangles = nanglesdeg[np.argwhere(nanglesdeg==istart).squeeze():np.argwhere(nanglesdeg==iend).squeeze()+1]
    binangs[ibin] = iangles
    # iangles = np.deg2rad(iangles)
    imid = binmids[ibin]
    imidrad = np.deg2rad(binmids[ibin])
    # ibinmask = np.isin(nangles, np.arange(istart, iend)).astype(int)
    ibinmask = np.logical_and(np.greater_equal(nangles, istartrad), np.less_equal(nangles, iendrad))
    tmpbins[ibin] = ibinmask
    ianglesrad = np.deg2rad(iangles)
    # tmp  = np.cos(np.radians(((iangles-imid)/binwidth*2))* np.pi)
    tmp = np.cos( ((ianglesrad-imidrad)/np.deg2rad(binwidth*2)*np.pi) )
    icurve = ibinmask.copy().astype(float) * np.nan;
    icurve[ibinmask==1] = tmp
    tmpcurves[ibin] = icurve

#visualise the weightings that will be applied to different angles within each bin
fig = plt.figure(figsize = [12,3]);
ax = fig.add_subplot(111)
for ibin in range(binstarts.size):
    ax.plot(np.deg2rad(binangs[ibin]), np.squeeze(tmpcurves[ibin, np.where(~np.isnan(tmpcurves[ibin]))]), lw=1  )
ax.set_xlabel('orientation (radians)')
ax.set_ylabel('weighted contribution to bin mean (AU)')
fig.suptitle('contribution of individual orientations to activity in reference bins')
fig.tight_layout()
fig.savefig(op.join(figdir, f'trialcontribution_bins_weighted_binstep{binstep}binwidth{binwidth}.pdf'), dpi=300, format = 'pdf')
#%% get examples of distances for a single trial to highlight the process?

i = subs[0];
loc = 'workstation';
sub = dict(loc = loc, id = i)
param = getSubjectInfo(sub)

data = np.load(op.join(wd, 'data', 'tuningcurves', '500hz',
                       f's{i}_TuningCurve_mahaldists_500Hz_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'))

bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv')) #read in associated behavioural data

#average distances across items
data = data.mean(0) #now has [trials x bins x times]
[ntrials, nbins, ntimes] = data.shape

#%%
plt.close('all')
tps = np.logical_and(times >= 0.05, times <= 0.25)
plotdat = data.copy()[trl,:,tps].T.mean(-1)
plottrls = 10
trlshift = 190
fig = plt.figure(figsize = [16,2])
for itrl in range(plottrls):
    ax=fig.add_subplot(1,plottrls, itrl+1)
    plotdat = data.copy()[itrl+trlshift, :, tps].T.mean(-1)
    ax.plot(binmids, plotdat)
    ax.set_title(f'trial {itrl+trlshift+1}')
fig.tight_layout()

#good trials: 52, 78, 83, 96, 103 !!, 108!!, 132, 155!, 158 (broad), 200 !!
# example trials with no real orientation preference: 116? 130, 121?, 156, 199

goodtrls = [52, 78, 83, 96, 103, 108, 132, 155, 158, 200]
badtrls  = [116, 130, 121, 156, 199]

fig = plt.figure(figsize=[16,4])
for it in range(len(goodtrls)):
    ax=fig.add_subplot(2,len(goodtrls), it+1)
    plotdat = data.copy()[goodtrls[it]-1, :, tps].T.mean(-1)
    ax.plot(binmids, plotdat)
    ax.set_title(f'trial {goodtrls[it]}')
    ax2 = fig.add_subplot(2, len(goodtrls), it+1+len(goodtrls))
    ax2.plot(binmids, plotdat*-1)
fig.tight_layout()

badtrls  = [116, 130, 121, 156, 199]

fig = plt.figure(figsize=[16,4])
for it in range(len(badtrls)):
    ax=fig.add_subplot(2,len(badtrls), it+1)
    plotdat = data.copy()[badtrls[it]-1, :, tps].T.mean(-1)
    ax.plot(binmids, plotdat)
    ax.set_title(f'trial {badtrls[it]}')
    ax2 = fig.add_subplot(2, len(badtrls), it+1+len(badtrls))
    ax2.plot(binmids, plotdat*-1)
fig.tight_layout()


#%%
exampletrls = [103, 96, 156]
fig = plt.figure(figsize=[9,8], layout='constrained')
subfig2 = fig.subfigures(1, 3)[0]
subfig3 = fig.subfigures(1, 3)[1]
subfig4 = fig.subfigures(1, 3)[2]


for it in range(len(exampletrls)):
    dists = data.copy()[exampletrls[it]-1, :, tps].T.mean(-1)
    invdists = dists*-1
    dminmax = minmax(invdists)
    dpos = np.add(invdists, np.abs(np.min(invdists)))
    ax2 = subfig2.add_subplot(len(exampletrls),1, it+1)
    ax2.plot(binmids, invdists, lw = 1, color='b')
    ax3 = subfig3.add_subplot(len(exampletrls),1, it+1)
    ax3.plot(binmids, dminmax, lw = 1, color = 'r')
    #step 1 fit precision
    res = sp.optimize.curve_fit(lambda x, B1, alpha: tcf.fullCosineModel(x, 0, B1, alpha), #fixes B0 at 0 (removes from model, keeps everything else)
                                      xdata = binmidsrad,
                                      ydata = dminmax,
                                      p0 = [1, 1], #initialise both parameters at 1
                                      bounds = ([-np.inf, 0], [np.inf, 3]), #b1 unbounded, alpa between 0.001 and 3
                                      maxfev = 5000, method = 'trf', nan_policy='omit')#[0]
    iparams = res[0]
    fitted = np.multiply(iparams[0], np.cos(binmidsrad*iparams[1]))
    ax3.plot(binmids, fitted, lw = 1, ls ='dashed', color='g', label = f'prec: {np.round(iparams[1],decimals=2)}')
    ax3.legend(loc='upper left', frameon=False, fontsize = 8, handlelength=1)
    
    #step2, fit amplitude on the positive distances
    ia = max(iparams[1], 0.001) #constrain alpha so that it is never lower than 0.001, which can prevent model fitting
    # desmat = minmax(np.cos(binmidsrad*ia)) #minmax scale the cosine that is weighted by the pre-estimated alpha value
    desmat = np.cos(binmidsrad*ia)
    gl = sma.GLM(endog = dpos, exog = desmat, family = sma.families.Gaussian())
    glfit = gl.fit(); ib = glfit.params[0] #get glmfit amplitude estimate
    ampfit = desmat*ib
    ax4 = subfig4.add_subplot(len(exampletrls), 1, it+1)
    ax4.plot(binmids, dpos, lw = 1, color = '#756bb1')
    ax4.plot(binmids, ampfit, lw = 1, ls = 'dashed', color = 'k', label = f'amp: {np.round(ib,decimals=2)}')
    ax4.legend(loc='upper left', frameon=False, fontsize = 8, handlelength=1)
    ax4.set_ylim([0, 0.13])
    
    if it == 0:
        ax2.set_title('negative distances', fontsize=10)
        ax3.set_title('re-scaled similarity', fontsize=10)
        ax4.set_title('positive similarity', fontsize=10)
    
    for a in [ax2, ax3, ax4]:
        a.set_xticks(np.arange(-90, 91, 45))
        a.tick_params(axis='x', which='major', labelsize=10)
        a.set_xticks(np.arange(-90, 90, 15), minor=True)
        
    ax2.set_ylabel('pattern similarity (AU)')

ax2.set_xlabel('orientation')
ax3.set_xlabel('orientation')
ax4.set_xlabel('orientation')
fig.savefig(op.join(figdir, 'trialexamples_modellingapproach.pdf'), dpi = 300, format = 'pdf')
# subfig2.suptitle('similarity', fontsize=10)
# subfig3.suptitle('re-scaled similarity', fontsize=10)
# subfig4.suptitle('positive similarity', fontsize = 10)


#%%


data = data * -1 #sign flip mahalanobis distances so that larger (more positive) values reflect more similar representations
d = data.copy()
dm = data.copy()
dminmax = data.copy()


