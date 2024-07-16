# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:16:07 2024

@author: sammirc
"""
import numpy   as np
import scipy   as sp
import pandas  as pd
import seaborn as sns
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from copy import deepcopy
import time
%matplotlib
mne.viz.set_browser_backend('qt')
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

def wrap(x):
    return (x+180)%360 - 180
def wrap90(x):
    return (x+90)%180 - 90

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#1,2,3,10,19 all have only 1 session. 23 something wrong in the second session, completely unusable eeg data due to serious noise
nsubs = subs.size

use_vischans_only  = True
use_pca            = False #using pca after selecting only visual channels is bad for a classifier, it seems
smooth_singletrial = True

# the distance between, and width of bins, shapes how many bins we have,
# and we need this info for preallocating arrays, lets do this here
# set up orientation bins

binstep = 15
binwidth = 22 #if you don't want overlap between bins, binwidth should be exactly half the binstep
# binstep, binwidth = 4, 11
binstep, binwidth = 4, 16
# binstep, binwidth = 4, 22
nbins, binmids, binstarts, binends = createFeatureBins(binstep = binstep, binwidth = binwidth,
                                                       feature_start = -90+binstep, feature_end = 90)
thetas = np.cos(np.radians(binmids))

visualise_bins = False
if visualise_bins:
    visualise_FeatureBins(binstarts, binmids, binends)

subcount = 0
for i in subs:
    subcount += 1
    print(f'\n\nworking on participant {subcount}/{nsubs}')
    sub = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)

    epochs = mne.read_epochs(fname = op.join(param['path'], 'eeg', param['substr'],
                                   f'{param["substr"]}_arraylocked_AllTrials_Preprocessed-epo.fif'), preload=True)
    dropchans = ['RM', 'LM', 'HEOG', 'VEOG', 'Trigger'] #drop some irrelevant channels
    epochs.drop_channels(dropchans) #61 scalp channels left
    epochs.crop(tmin = -0.5, tmax = 1.25)
    epochs.resample(100) #resample to reduce computation time
    times = epochs.times.copy()

    if use_vischans_only:
        #drop all but the posterior visual channels
        epochs = epochs.pick(picks = [
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO3',  'POz', 'PO4', 'PO8', 
                            'O1', 'Oz', 'O2'])
    data   = epochs._data.copy() #get data array
    [ntrials, nfeatures, ntimes] = data.shape #get data shape (trials x channels x times)
    #smooth single-trial data to denoise a bit
    if smooth_singletrial:
        data = sp.ndimage.gaussian_filter1d(data.copy(), sigma = int(20 * epochs.info['sfreq']/1000)) #smooth by gaussian with 10ms sd blur
        
    #zero-centre (deman) the scalp topography at each time point (preserves shape)
    demean_tpwise = True
    if demean_tpwise:
        for trl in range(ntrials):
            for tp in range(ntimes):
                data[trl,:,tp]  = np.subtract(data[trl, :, tp], data[trl, :, tp].mean())
                
    #get behavioural data
    bdata = epochs.metadata.copy()
    orientations = np.array([bdata.ori1.to_numpy(), bdata.ori2.to_numpy()]) #2d array, [norientations, ntrials], orientations are [left, right] items
    trls = np.arange(ntrials)    
    
    weightTrials=False
    nruns = 2 #two runs, first for left item, second for right item
    tc_all = np.empty(shape = [nruns, ntrials, nbins, ntimes]) * np.nan
    tic = time.time()
    for irun in range(nruns):
        orisuse = orientations[irun].copy() #get orientations for stimuli on the particular side of the display to decode
        tc = np.zeros(shape = [ntrials, nbins, ntimes]) * np.nan
        bar = progressbar.ProgressBar(min_value=0, max_value = ntimes, initial_value=0)
        for tp in range(ntimes):
            bar.update(tp)
            dists = getTuningCurve_FullSpace(data[:,:,tp], orisuse,
                                    binstep = binstep, binwidth = binwidth, weight_trials = weightTrials, feature_start=-90+binstep, feature_end=90)
            tc[:,:,tp] = dists
        tc_all[irun] = tc.copy()
    toc = time.time()
    print(f'decoding took {int(divmod(toc-tic, 60)[0])}m{round(divmod(toc-tic, 60)[1])}s')
    #save data
    np.save(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'), tc_all)
    
    if not op.exists(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv')):
        bdata.to_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv'), index=False)
    if i == 4:
        np.save(op.join(wd, 'data', 'tuningcurves', 'times.npy'), times) #save times

#%%
    
# tmp = tc_all.copy() * -1
# tmpdm = tc_all.copy() * -1
# #demean tuning curve across bins
# for irun in range(nruns):
#     for trl in range(ntrials):
#         for tp in range(ntimes):
#             tmpdm[irun, trl,:,tp] = np.subtract(tmpdm[irun, trl,:,tp], tmpdm[irun, trl,:,tp].mean())
            
# tmpcos = tmp.copy()
# tmpcos = tmpcos.mean(0).mean(0)
# for tp in range(ntimes):
#     tmpcos[:,tp] = np.multiply(tmpcos[:,tp], -thetas)
# tmp2 = tmpdm.copy().mean(1) #average across trials
# fig = plt.figure(figsize = [15, 6])
# for irun in range(nruns):
#     ax=fig.add_subplot(1, 2, irun+1)
#     plot = ax.imshow(tmp2[irun], aspect='auto', origin = 'lower', interpolation = 'none', cmap = 'RdBu_r',
#                      extent = [times.min(), times.max(), binmids.min(), binmids.max()])
#     fig.colorbar(plot)

# #plot average tuning curve across both items
# tmp3 = tmp2.copy().mean(0) #average across items
# fig = plt.figure(figsize = [6,4])
# ax = fig.add_subplot(111)
# plot = ax.imshow(tmp3, aspect= 'auto', origin ='lower', interpolation='none', cmap = 'RdBu_r',
#                  extent = [times.min(), times.max(), binmids.min(), binmids.max()])
# fig.colorbar(plot)

# tmpcos = tmp3.copy()
# for tp in range(ntimes):
#     tmpcos[:,tp] = np.multiply(tmpcos[:,tp], thetas)
# fig = plt.figure(figsize = [6,4])
# ax = fig.add_subplot(111)
# plot = ax.imshow(tmpcos, aspect= 'auto', origin ='lower', interpolation='none', cmap = 'RdBu_r',
#                  extent = [times.min(), times.max(), binmids.min(), binmids.max()])
# fig.colorbar(plot)

# tstarts = [-0.3, 0,   0.2, 0.4, 0.6, 0.8]
# tends   = [ 0,   0.2, 0.4, 0.6, 0.8, 1.0]
# plotcount = np.arange(len(tstarts))+1
# fig = plt.figure(figsize = [15, 3])
# for x in zip(plotcount, tstarts, tends):
#     ax = fig.add_subplot(1,6,x[0])
#     tinds = np.logical_and(np.greater_equal(times, x[1]), np.less_equal(times, x[2]))
#     tmpdat = np.nanmean(tmp3[:,tinds], axis=1) #average across time
#     ax.plot(binmids, tmpdat)
#     ax.set_xlabel('orientation')
#     ax.set_title(f'distances {x[1]}$\\rightarrow${x[2]}s')
#     ax.set_ylim([-0.01, 0.01])
# fig.tight_layout()