#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:43:33 2024

@author: sammichekroud
"""
import numpy as np
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
if not op.exists(op.join(eyedir, 'preprocessed')):
    os.mkdir(op.join(eyedir, 'preprocessed'))

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
nsubs = subs.size

#set some params here
for sub in subs:
    # if not op.exists(op.join(eyedir, 'preprocessed', f'EffDS{sub}_preprocessed.pickle')): #dont do this if it already exists!
    print(f'\n- - - working on ppt {sub} - - -')
    for part in ['a', 'b']: #two recordings per subject
        fname = 'WMCS%02d%s.asc'%(sub, part)
        eyename = op.join(eyedir, fname)
        
        if op.exists(eyename):
            data = eyes.io.parse_eyes(eyename)
    
            data.blocks = np.arange(8)+1 #8 blocks
            
            data.nan_missingdata() #replace missing data (coded as 0) to nan
            data.identify_blinks(buffer = 0.075)
            # d = deepcopy(data)
            
            #if you want to visualise the eyes in each block to see where it is missing blocks of data
            #uncomment below and run
            # fig = plt.figure(figsize = [15, 6])
            # for iplot in range(data.nblocks):
            #     ax = fig.add_subplot(2, 4, iplot+1)
            #     ax.plot(data.data[iplot].trackertime, data.data[iplot].pupil_l, color='b', lw = 1, alpha=0.5, label = 'left')
            #     ax.plot(data.data[iplot].trackertime, data.data[iplot].pupil_r, color='r', lw = 1, alpha=0.5, label = 'right')
            #     ax.set_title(f'block {data.blocks[iplot]}', size=10)
            #     ax.legend(frameon=False)
            # fig.suptitle(f'participant {sub} part {part}')
            
            if sub == 11:
                if part == 'a':
                #lost recording of both eyes in the last block, drop this here:
                    data.blocks = np.delete(data.blocks, 7) #log that the last block was deleted
                    del(data.data[7])
                    data.nblocks -=1
                elif part == 'b':
                    data.blocks = np.delete(data.blocks, 2) #log that block 3 was deleted
                    del(data.data[2])
                    data.nblocks -=1
                    
            if sub == 12:
                if part == 'a':
                    data.data[6].drop_eye('left') #lost tracking of the left eye in this entire block
                if part == 'b': #missing eyes in block 6 and 7
                    data.blocks = np.delete(data.blocks, [5, 6])
                    data.data = np.delete(data.data, [5,6]).tolist()
                    data.nblocks -= 2
            if sub == 24:
                if part == 'b':
                    data.data[-1].drop_eye('right') #right eye very noisy in this block (lots of missing data), so we drop it
            
            data.interpolate_blinks()
            data.smooth_pupil() #smooth pupul with 50ms gaussian blur
            data.cubicfit()
            
            eyes.io.save(data,
                            fname = op.join(eyedir, 'preprocessed', f'wmc_s{sub}{part}_preprocessed.pickle'))
        else:
            print(f"couldn't find file {fname}, skipping")
    
    
#%%