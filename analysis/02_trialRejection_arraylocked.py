# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:15:13 2024

@author: sammirc
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from copy import deepcopy
%matplotlib
mne.viz.set_browser_backend('qt')

loc = 'workstation'
if loc == 'workstation':
    wd = 'C:/Users/sammirc/Desktop/postdoc/tuningcurves'
    # sys.path.insert(0, op.join(wd, 'analysis', 'tools'))
elif loc == 'laptop':
    wd = '/Users/sammichekroud/Desktop/postdoc/tuningcurves'
sys.path.insert(0, op.join(wd, 'analysis', 'tools'))
os.chdir(wd)
from funcs import getSubjectInfo, gesd, plot_AR

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#1,2,3,10,19 all have only 1 session. 23 something wrong in the second session, completely unusable eeg data due to serious noise
nsubs = subs.size

subcount = 0
for i in subs:
    subcount += 1
    print(f'working on participant {subcount}/{nsubs}')
    sub = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    
    #do trial rejection separately for each experimental session, then combine into one object for analysis later
    for part in [1,2]:
        partstr = ['a', 'b'][part-1]
        epochs = mne.read_epochs(fname = op.join(param['path'], 'eeg', param['substr'], f'{param["substr"]}{partstr}_arraylocked_icacleaned-epo.fif'),
                                  preload=True)
        epochs.shift_time(tshift = -0.025, relative = True) #shift based on photodiode timings, 25ms shift
        epochs = epochs.apply_baseline((-0.2, 0)) #baseline just prior to stim1 onset
        epochs.metadata = epochs.metadata.assign(session = partstr)
    
        #run gesd
        #do trial rejection from the two files separately before concatenating events
        _, keeps = plot_AR(deepcopy(epochs).pick_types(eeg=True),
                           method = 'gesd',
                           zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
        keeps = keeps.flatten() #indices of trials to be kept
    
        discards = np.ones(len(epochs), dtype = 'bool')
        discards[keeps] = False #vector that is False if kept, True if discarded
        epochs = epochs.drop(discards) #first we'll drop trials with excessive noise in the EEG
        
        #write to file which trials are discarded from the eeg
        
        #go through any remaining trials to look for excessively noisy trials to discard
        # epoched.plot(events = epoched.events, n_epochs = 3, n_channels = 64, scalings = dict(eeg=40e-6))
        #epoched.interpolate_bads()
        
        #save the epoched data, combined with metadata, to file
        epochs.save(fname = op.join(param['path'],
                                    'eeg',
                                    param['substr'],
                                    f'{param["substr"]}{partstr}_arraylocked_preprocessed-epo.fif'), overwrite=True)
        #save the resulting behavioural data too ? probably not needed as metadata is saved with the individual dataset
        
        
        del(epochs)
        plt.close('all')

subcount=0
for i in subs:
    subcount += 1
    print(f'\ncombining sessions for participant {subcount}/{nsubs}')
    sub = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    
    #trial rejection has happened separately on each dataset, so we can now combine into one object for subsequent analysis
    epochs = mne.read_epochs(fname = op.join(param['path'], 'eeg', param['substr'], f'{param["substr"]}a_arraylocked_preprocessed-epo.fif'), preload=True)
    epochs2 = mne.read_epochs(fname = op.join(param['path'], 'eeg', param['substr'], f'{param["substr"]}b_arraylocked_preprocessed-epo.fif'), preload=True)
    
    allEpochs = mne.concatenate_epochs([epochs, epochs2])
    
    allEpochs.save(fname = op.join(param['path'], 
                                   'eeg',
                                   param['substr'],
                                   f'{param["substr"]}_arraylocked_AllTrials_Preprocessed-epo.fif'), overwrite=True)
    
    #may need to save the behaviouraldata for this separately after making sure its in the right order, if mne doesn't join the dataframes properly
    
    
    
    
    
    
    
    
    