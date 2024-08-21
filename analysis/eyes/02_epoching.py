#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:12:01 2024

@author: sammichekroud
"""
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
%matplotlib

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
if not op.exists(op.join(eyedir, 'epoched')):
    os.mkdir(op.join(eyedir, 'epoched'))

bdir   = op.join(wd, 'data', 'datafiles')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])

nsubs = subs.size
#set some params here
for sub in subs:
    # if not op.exists(op.join(eyedir, 'preprocessed', f'EffDS{sub}_preprocessed.pickle')): #dont do this if it already exists!
    print(f'\n- - - working on ppt {sub} - - -')
    epoch_list = []
    for part in ['a', 'b']: #two recordings per subject
        if not np.logical_and(sub==14, part == 'b'):
            #ppt 14 session 2 data is corrupted on disk, need to get from pc again
            eyename = op.join(eyedir, 'preprocessed', f'wmc_s{sub}{part}_preprocessed.pickle')
            raw = eyes.io.load(eyename)
            
            for iblock in range(raw.nblocks):
                if raw.data[iblock].binocular:
                    raw.data[iblock].average_channels(channels = ['pupil_corrected_l', 'pupil_corrected_r'],
                                                      new_name = 'pupil_corrected')
            
            # fig = plt.figure(figsize = [12,8])
            # ax = fig.add_subplot(211)
            # ax.plot(raw.data[0].time, raw.data[0].pupil_clean_l, lw = 1, color ='#3182bd')
            # ax.plot(raw.data[0].time, raw.data[0].modelled_l, lw = 1.5, color = 'k')
            # ax = fig.add_subplot(212)
            # ax.plot(raw.data[0].time, raw.data[0].pupil_corrected_l, color = 'r', lw = 0.5)
            
            
            tmin, tmax = -0.5, 1.25
            
            epoched = eyes.utils.epochs(raw, tmin = tmin, tmax = tmax,
                                        channels = ['pupil_corrected'],
                                        triggers = ['trig1', 'trig2'])
            bname = op.join(bdir, 'preprocessed_data', f'wmConfidence_S{sub:02d}{part}_allData_preprocessed.csv')
            df = pd.read_csv(bname, index_col=False)
            if 'Unnamed: 0' in df.columns:
                df.drop(columns = 'Unnamed: 0', inplace = True)
            if raw.nblocks != 8: #at least one block has been dropped from the raw data before epoching
                df = df.query('block in @raw.blocks') #strip down to just these usable blocks then
            
            epoched.metadata = df
            
            epoch_list.append(epoched)
    if sub != 14:
        allepochs = eyes.epochs.concatenate_epochs(epoch_list)
    elif sub == 14:
        allepochs = deepcopy(epoch_list[0])
    
    eyes.io.save(allepochs,
                 fname = op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_pupil_arraylocked.pickle')
                 )
    
    
    
    
    
    
    
    
    
    
    
    
    

    