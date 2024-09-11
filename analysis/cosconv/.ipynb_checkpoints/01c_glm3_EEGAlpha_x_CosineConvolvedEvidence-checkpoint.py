# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:06:04 2024

@author: sammirc
"""
import numpy as np
import pandas as pd
import scipy as sp
from copy import deepcopy
import os
import os.path as op
import sys
import mne
from matplotlib import pyplot as plt
import statsmodels as sm
import statsmodels.api as sma
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
# sys.path.insert(0, eyefuncdir)
sys.path.insert(0, op.join(wd, 'analysis', 'tools'))

#import eyefuncs_v2 as eyes
from funcs import getSubjectInfo

eyedir = op.join(wd, 'data', 'eyes')
bdir   = op.join(wd, 'data', 'datafiles')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
# subs = np.array([         4, 5, 6, 7, 8, 9,             13, 14, 15,     17,         20, 21, 22,     24, 25, 26])
nsubs = subs.size
#set some params here
modeltimes = np.round(np.load(op.join(wd, 'data', 'tuningcurves', 'times.npy')), 2)

for sub in subs:
    print(f'\n- - - working on ppt {sub} - - -')
    #load in the parameter fits for this participant
    binstep, binwidth = 4, 22
    smooth_alphas, smooth_sigma = True, 3
    weightTrials = True
    
    #read in cosine convolved decoding evidence
    evidence = np.load(op.join(wd, 'data','tuningcurves', f's{sub}_CosineConvolvedEvidence_binstep{binstep}_binwidth{binwidth}_weightTrials{weightTrials}.npy'))
    
    
    tcbdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{sub}_TuningCurve_metadata.csv')) #read in associated behavioural data
    
    #average evidence across items, as we dont have pupil responses to each separate item
    evidence = np.nanmean(evidence, axis=0)
        
    #read in the eeg alpha data
    isub = dict(loc = loc, id = sub)
    param = getSubjectInfo(isub)
    
    tfr = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'eeg', f's{sub:02d}/wmConfidence_s{sub:02d}_arraylocked_preproc_alpha-tfr.h5'))
    #dont need to align these data types, as tfr is done on the same trials as the tuning curve estimation (both on preprocessed/kept epochs)
    
    #crop data
    tfr = tfr.crop(tmin = -0.7, tmax = 1)
    eegtimes = tfr.times.copy()
    vischans = [
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO3',  'POz', 'PO4', 'PO8', 
                        'O1', 'Oz', 'O2']
    tfr = tfr.pick(picks=vischans)
    tfdata = tfr._data.copy()   
    
    logdata = True
    if logdata:
        tfdata = np.multiply(10, np.log10(tfdata))
        logtxt = '_logpower'
    else:
        logtxt = ''
    tfdata = tfdata.mean(2).mean(1) #average across frequencies then channels
    
    #create a trial regressor (zscored) to account for potential drift of alpha over time
    tcbdata = tcbdata.assign(trlid = np.where(tcbdata.session.eq('a'), tcbdata.trialnum, tcbdata.trialnum + 256))
    trlz = np.arange(512)+1 #identifier with max number of trials
    trlz = sp.stats.zscore(trlz, ddof=0)
    trl_z = trlz[tcbdata.trlid-1] #account for 0 indexing of the array
    
    
    nparams = 3 #only using a constant, sensory evidence, and a trial marker regressor
    allb = np.zeros(shape = [modeltimes.size, nparams, eegtimes.size])
    allt = np.zeros(shape = [modeltimes.size, nparams, eegtimes.size])
    print('- - - - -  running glm across tuning curve timepoints - - - - -')
    for itp in range(modeltimes.size):
        #get sensory evidence at this time point
        idec = evidence[:,itp].copy()
        
        DC = glm.design.DesignConfig()
        DC.add_regressor(name = 'mean', rtype = 'Constant')
        DC.add_regressor(name = 'evidence', rtype = 'Parametric', datainfo = 'ev', preproc = 'z')
        DC.add_regressor(name = 'trlid', rtype = 'Parametric', datainfo = 'trlidz', preproc = None)
        DC.add_simple_contrasts()
        
        #create glmdata object
        glmdata = glm.data.TrialGLMData(data = tfdata, time_dim = 1, sample_rate = 1000,
                                        ev = idec, trlidz = trl_z
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
    #     tmp = allb[ireg].squeeze()
    #     if ireg == 0:
    #         vmin, vmax = None, None
    #     else:
    #         vmin, vmax = -2, 2
    #     plot = ax.imshow(tmp, aspect='auto', interpolation='none', cmap = 'RdBu_r', origin='lower',
    #                       vmin = vmin, vmax = vmax,
    #               extent = [eegtimes.min(), eegtimes.max(), modeltimes.min(), modeltimes.max()])
    #     fig.colorbar(plot)
    #     ax.axvline(0, lw=1, ls='dashed', color='k'); 
    #     ax.axhline(0, lw=1, ls='dashed', color='k'); 
    #     ax.set_title(model.regressor_names[ireg])
    #     ax.set_ylabel('tuning curve time (s)')
    #     ax.set_xlabel('eeg time rel to array onset (s)')
    
    

    #save individual betas and tstats
    np.save(op.join(wd, 'data', 'glms', 'eeg', 'glm3_evidence', f'wmc_s{sub:02d}_glm3{logtxt}_betas.npy'), arr = allb)
    np.save(op.join(wd, 'data', 'glms', 'eeg', 'glm3_evidence', f'wmc_s{sub:02d}_glm3{logtxt}_tvalues.npy'), arr = allt)
    if sub == 4:
        np.save(op.join(wd, 'data', 'glms', 'eeg', 'glm3_evidence', 'eeg_times.npy'), arr = eegtimes)
        np.save(op.join(wd, 'data', 'glms', 'eeg', 'glm3_evidence', f'regressor_names.npy'), arr = model.regressor_names)
