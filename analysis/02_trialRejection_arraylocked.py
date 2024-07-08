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
%matplotlib
mne.viz.set_browser_backend('qt')

loc = 'workstation'
if loc == 'workstation':
    wd = 'C:/Users/sammirc/Desktop/postdoc/tuningcurves'
    sys.path.insert(0, op.join(wd, 'analysis', 'tools'))

os.chdir(wd)
from funcs import getSubjectInfo

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22, 23,     24, 25, 26])
#1,2,3,10,19 all have only 1 session. 23 something wrong in the second session, completely unusable eeg data serious noise
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
        epoched = mne.read_epochs(fname = op.join(param['path'], 'eeg', param['substr'], f'{param["substr"]}{partstr}_arraylocked_icacleaned-epo.fif'),
                                  preload=True)