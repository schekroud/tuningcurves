# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:47:17 2024

@author: sammirc
"""
import numpy as np
import pandas as pd
import scipy as sp
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import statsmodels as sm
import statsmodels.api as sma
# %matplotlib
import glmtools as glm


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
bdir   = op.join(wd, 'data', 'datafiles')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,             13, 14, 15,     17,         20, 21, 22,     24, 25, 26])
nsubs = subs.size
#set some params here
modeltimes = np.round(np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy')), 2)

for sub in subs:
    print(f'\n- - - working on ppt {sub} - - -')
    data = eyes.io.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_pupil_arraylocked.pickle'))
    # data = data.apply_baseline([-0.5, -0.2])
    trlcheck = np.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_arraylocked_nanperc.npy')) #this always has as many trials as the eyetracking data does
    #load in the parameter fits for this participant
    binstep, binwidth = 4, 22
    smooth_alphas, smooth_sigma = True, 3
    weightTrials = True
    
    #read in cosine convolved decoding evidence
    evidence = np.load(op.join(wd, 'data','tuningcurves', f's{sub}_CosineConvolvedEvidence_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'))
    
    
    tcbdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{sub}_TuningCurve_metadata.csv')) #read in associated behavioural data
    
    #average evidence across items, as we dont have pupil responses to each separate item
    evidence = np.nanmean(evidence, axis=0)
    
    
    #need to align the trials in the eyetracking data with the trials in the tuning curve data:
    keeptrleyes = trlcheck<=30
    tcbdata = tcbdata.assign(trlid = np.where(tcbdata.session=='a', tcbdata.trialnum, tcbdata.trialnum+256))
    data.metadata = data.metadata.assign(trlid = np.where(data.metadata.session=='a', data.metadata.trialnum, data.metadata.trialnum+256))

    #first, drop eyetracking trials with bad eyetracking
    data.metadata = data.metadata.loc[keeptrleyes]
    data.data = data.data[keeptrleyes]
    data.event_id = data.event_id[keeptrleyes]
    
    #align trials between the now-cleaned eyetracking and the eeg tuning curves
    tctrls = tcbdata.trlid.to_numpy()
    eyetrls = data.metadata.trlid.to_numpy()
    sametrls = np.intersect1d(tctrls, eyetrls)
    
    #make booleans for indexing
    tckeep = np.isin(tctrls, sametrls)
    ekeep  = np.isin(eyetrls, sametrls)
    
    #get the right tuning curve data
    evidence = evidence[tckeep]
    tcbdata = tcbdata.loc[tckeep]
    
    #get the right eyetracking data
    data.metadata = data.metadata.loc[ekeep]
    data.data = data.data[ekeep]
    data.event_id = data.event_id[ekeep]
    
    nparams = 2 #only using a constant and one of the 
    allb = np.zeros(shape = [modeltimes.size, nparams, data.times.size])
    allt = np.zeros(shape = [modeltimes.size, nparams, data.times.size])
    print('- - - - -  running glm across tuning curve timepoints - - - - -')
    for itp in range(modeltimes.size):
        #get parameter estimates for this timepoint, across trials   
        idec = evidence[:, itp]
        
        DC = glm.design.DesignConfig()
        DC.add_regressor(name = 'mean', rtype = 'Constant')
        DC.add_regressor(name = 'evidence', rtype = 'Parametric', datainfo = 'ev', preproc = 'z')
        DC.add_simple_contrasts()
        
        #create glmdata object
        glmdata = glm.data.TrialGLMData(data = data.data.squeeze(), time_dim = 1, sample_rate = 1000,
                                        ev = idec
                                        )
    
        glmdes = DC.design_from_datainfo(glmdata.info)
        
        model = glm.fit.OLSModel(glmdes, glmdata) #fit the actual model 
            
        betas = model.betas.copy()
        copes = model.copes.copy()
        tstats = model.tstats.copy()
        allb[itp] = betas.copy()
        allt[itp] = tstats.copy()
    
    #allb and allt have shape [modeltimes x params x eyetimes]. tranpose these to make it easier later on
    allb = allb.transpose(1, 0, 2)
    allt = allt.transpose(1, 0, 2)
    
    # fig = plt.figure(figsize = [15,5])
    # for ireg in range(nparams):
    #     ax = fig.add_subplot(1, nparams, ireg+1)
    #     tmp = allt[ireg].squeeze()
    #     if ireg == 0:
    #         vmin, vmax = None, None
    #     else:
    #         vmin, vmax = -2, 2
    #     plot = ax.imshow(tmp, aspect='auto', interpolation='none', cmap = 'RdBu_r', origin='lower',
    #                       vmin = vmin, vmax = vmax,
    #               extent = [data.times.min(), data.times.max(), modeltimes.min(), modeltimes.max()])
    #     fig.colorbar(plot)
    #     ax.axvline(0, lw=1, ls='dashed', color='k'); 
    #     ax.axhline(0, lw=1, ls='dashed', color='k'); 
    #     ax.set_title(model.regressor_names[ireg])
    #     ax.set_ylabel('tuning curve time (s)')
    #     ax.set_xlabel('pupil time rel to array onset (s)')
    

    #save individual betas and tstats
    np.save(op.join(wd, 'data', 'glms', 'glm4_evidence', f'wmc_s{sub:02d}_glm4_betas.npy'), arr = allb)
    np.save(op.join(wd, 'data', 'glms', 'glm4_evidence', f'wmc_s{sub:02d}_glm4_tvalues.npy'), arr = allt)
    if sub == 4:
        np.save(op.join(wd, 'data', 'glms', 'glm4_evidence', 'eyetracker_times.npy'), arr = data.times)
        np.save(op.join(wd, 'data', 'glms', 'glm4_evidence', f'regressor_names.npy'), arr = model.regressor_names)
