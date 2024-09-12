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
# %matplotlib inline

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
# binstep  = 15
# binwidth = 22
# binstep, binwidth = 4, 11
# #binstep, binwidth = 4, 16
# binstep, binwidth = 4, 22


times = np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy'))
ntimes = times.size
nitems = 2 #two items are presented in the array, we 'decode' both

# nbins, binmids, binstarts, binends = createFeatureBins(binstep = binstep, binwidth = binwidth,
#                                                        feature_start = -90+binstep, feature_end = 90)

weightTrials = True
alld = np.zeros(shape = [nsubs, ntimes]) *np.nan
subcount = -1
for i in subs:
    subcount += 1
    print(f'working on ppt {subcount+1}/{nsubs}')
    for binstep, binwidth in [(4,11), (4,16), (4,22), (15,10), (15,15), (15,22)]:
        print(f'- getting cosine-convolved evidence for binstep {binstep}, binwidth {binwidth}')
        data = np.load(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_mahaldists_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'))
        bdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{i}_TuningCurve_metadata.csv'))
        [nitems, ntrials, nbins, ntimes] = data.shape

        nbins, binmids, binstarts, binends = createFeatureBins(binstep = binstep, binwidth = binwidth,
                                                       feature_start = -90+binstep, feature_end = 90)
    
        dnew = np.zeros(shape = [nitems, ntrials, ntimes]) * np.nan #creating a summary metric across bins
        #convolving distances with the cosine of their relative orientation, then averaging across orientation bins to get 'cosine amplitude', or 'decoding accuracy'
        for iitem in range(nitems):
            for itrl in range(ntrials):
                for it in range(ntimes):
                    id = data[iitem, itrl, :, it].copy()
                    idm = id - id.mean()
                    idcos = np.multiply(np.cos(binmids), idm).mean() *-1
                    dnew[iitem, itrl, it] = idcos
        # dnew = sp.ndimage.gaussian_filter1d(dnew, sigma = 1) #lightly smooth with a gaussian kernel with sd 2 samples (20ms)
        
        #save this as a new array
        np.save(op.join(wd, 'data','tuningcurves', f's{i}_CosineConvolvedEvidence_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'), arr=dnew)
    
