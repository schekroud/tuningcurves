# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:16:30 2024

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
#mne.viz.set_browser_backend('qt')
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

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#1,2,3,10,19 all have only 1 session. 23 something wrong in the second session, completely unusable eeg data due to serious noise
nsubs = subs.size
baselined = False
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
    epochs.crop(tmin = -1, tmax = 1.25)
    
    if baselined:
        epochs = epochs.apply_baseline((-0.5, -0.3))
    
    epochs.resample(100) #resample to reduce computation time
    
    vischans = [
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO3',  'POz', 'PO4', 'PO8', 
                        'O1', 'Oz', 'O2']
    
    
    #set up time frequency decomposition
    freqs = np.arange(1, 31, 1) #from 1 to 40
    n_cycles = freqs * 0.3 #use a 300ms (0.3s) time window for estimation
    
    # tfr = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = n_cycles,
    #                                         use_fft = True, return_itc = False, average = False, n_jobs = 4)
    
    tfr = epochs.compute_tfr(method = 'multitaper', freqs = freqs, n_cycles = n_cycles, use_fft = False, return_itc = False, average = False, n_jobs = -1)
    tfr.save(fname = op.join(param['path'], 'eeg', f's{i:02d}/wmConfidence_s{i:02d}_arraylocked_preproc-tfr.h5'), overwrite = True)
    tfr.crop(fmin = 8, fmax = 12)
    tfr.save(fname = op.join(param['path'], 'eeg', f's{i:02d}/wmConfidence_s{i:02d}_arraylocked_preproc_alpha-tfr.h5'), overwrite = True)
    
    
    
    