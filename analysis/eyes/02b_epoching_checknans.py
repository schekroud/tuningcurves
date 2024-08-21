# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:45:01 2024

@author: sammirc
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
fig = plt.figure(figsize = [15, 15])
subcount = 0
for sub in subs:
    subcount += 1
    # if not op.exists(op.join(eyedir, 'preprocessed', f'EffDS{sub}_preprocessed.pickle')): #dont do this if it already exists!
    print(f'\n- - - working on ppt {sub} - - -')
    epoch_list = []
    for part in ['a', 'b']: #two recordings per subject
        if not np.logical_and(sub==14, part == 'b'):
            #ppt 14 session 2 data is corrupted on disk, need to get from pc again
            eyename = op.join(eyedir, 'preprocessed', f'wmc_s{sub}{part}_preprocessed.pickle')
            raw = eyes.io.load(eyename)
            
            lnan = np.zeros(raw.nblocks) * np.nan
            rnan = np.zeros(raw.nblocks) * np.nan
            for iblock in range(raw.nblocks):
                if raw.data[iblock].binocular:
                    linan = np.isnan(getattr(raw.data[iblock], 'pupil_nan_l')).sum()
                    rinan = np.isnan(getattr(raw.data[iblock], 'pupil_nan_r')).sum()
                    lnan[iblock] = linan
                    rnan[iblock] = rinan
            
            nancheck = np.subtract(rnan, lnan) #get the relative number of nans. if positive, right eye has more nans. if negative, left eye has more
            binoccheck = [x.binocular for x in raw.data] #get whether each recorded block is binocular
            
            #where the data are binocular, we want to change the name (align across ppts and blocks) of the eye with the least missing data
            for iblock in range(raw.nblocks):
                if binoccheck[iblock]:
                    if not np.isnan(nancheck[iblock]):
                        if nancheck[iblock] > 0:
                            raw.data[iblock].drop_eye('right')
                        elif nancheck[iblock] < 0:
                            raw.data[iblock].drop_eye('left')
            tmin, tmax = -0.5, 1.25
            
            epoched = eyes.utils.epochs(raw, tmin = tmin, tmax = tmax,
                                        channels = ['pupil_nan'],
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
    
    ax = fig.add_subplot(5,4, subcount)
    ax.imshow(allepochs.data.squeeze(), aspect = 'auto',
              extent = [epoched.times.min(), epoched.times.max(), 0, len(allepochs.data)], origin = 'lower')
    ax.set_title(f'ppt {sub}')
    
    #find what % of a trial epoch is nans
    d = allepochs.data.squeeze()
    isnan = np.isnan(d).sum(axis=1)
    isnanperc = np.divide(isnan, epoched.times.size)*100
    
    np.save(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_arraylocked_nanperc.npy'), arr = isnanperc)

fig.tight_layout()
fig.savefig(op.join(wd, 'figures', 'arraylocked_nantimes.pdf'), dpi = 300, format = 'pdf')
eyes.io.save(allepochs,
             fname = op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_pupil_arraylocked_checknans.pickle')
             )