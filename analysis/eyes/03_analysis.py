#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:16:08 2024

@author: sammichekroud
"""
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import statsmodels as sm
import statsmodels.api as sma
%matplotlib
import glmtools as glm


loc = 'laptop'
if loc == 'laptop':
    #eyefuncdir = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools'
    eyefuncdir = '/Users/sammichekroud/Desktop/postdoc/tools'
    wd = '/Users/sammichekroud/Desktop/postdoc/wmconfidence' #working on confidence data, but in postdoc dir
os.chdir(wd)
sys.path.insert(0, eyefuncdir)
#import eyefuncs_v2 as eyes
import eyefuncs as eyes

eyedir = op.join(wd, 'data', 'eyes')
bdir   = op.join(wd, 'data', 'datafiles')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])

nsubs = subs.size


#set some params here
for sub in subs:
    # if not op.exists(op.join(eyedir, 'preprocessed', f'EffDS{sub}_preprocessed.pickle')): #dont do this if it already exists!
    print(f'\n- - - working on ppt {sub} - - -')
    data = eyes.io.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_pupil_arraylocked.pickle'))
    
    # data = data.apply_baseline([-0.2, 0])
    # fig = plt.figure(figsize = [6, 4])
    # ax = fig.add_subplot(211)
    # ax.plot(data.times, data.data.squeeze().mean(0))
    # ax = fig.add_subplot(212)
    # ax.plot(d.times, d.data.squeeze().mean(0))
    
    error = data.metadata.absrdif.to_numpy()
    acc = np.multiply(error, -1) #flip sign, so more positive = more accurate
    
    DC = glm.design.DesignConfig()
    # if iglm == 0:
    DC.add_regressor(name ='accuracy', rtype = 'Parametric', datainfo = 'acc', preproc = 'z')
    DC.add_regressor(name = 'mean', rtype = 'Constant')
    DC.add_simple_contrasts()
    
    #create glmdata object
    glmdata = glm.data.TrialGLMData(data = data.data.squeeze(), time_dim = 1, sample_rate = 1000,
                                    #add in metadata that's used to construct the design matrix
                                    acc=acc
                                    )

    glmdes = DC.design_from_datainfo(glmdata.info)
    
    print('- - - - -  running glm - - - - -')
    model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model 
        
    betas = model.betas.copy()
    copes = model.copes.copy()
    tstats = model.tstats.copy()
    
    fig = plt.figure(figsize = [6, 4])
    ax = fig.add_subplot(111)
    ax.plot(data.times, tstats.T, label = model.regressor_names)
    ax.legend()
    