# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 18:39:23 2024

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
modeltimes = np.round(np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'times_500hz.npy')), 3)

for sub in subs:
    for fittype in ['opt', 'glm']:
        # if not op.exists(op.join(eyedir, 'preprocessed', f'EffDS{sub}_preprocessed.pickle')): #dont do this if it already exists!
        print(f'\n- - - working on ppt {sub} - - -')
        data = eyes.io.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_pupil_arraylocked.pickle'))
        # data = data.apply_baseline([-0.5, -0.2])
        trlcheck = np.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_arraylocked_nanperc.npy')) #this always has as many trials as the eyetracking data does
        #load in the parameter fits for this participant
        binstep, binwidth = 4, 22
        
        #read in modelled tuning curve parameters
        prec = np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'smooth_alphas',
                f's{sub}_ParamFits_precision_binstep{binstep}_binwidth{binwidth}.npy'))
        
        if fittype == 'glm':
            amp = np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'smooth_alphas',
                        f's{sub}_ParamFits_amplitude_binstep{binstep}_binwidth{binwidth}_glmfit.npy'))
            fittext = 'b1glmfit'
        elif fittype == 'opt':
            amp = np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'smooth_alphas',
                        f's{sub}_ParamFits_amplitude_binstep{binstep}_binwidth{binwidth}_optfit.npy'))
            fittext = 'b1optfit'
        
        
        tcbdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{sub}_TuningCurve_metadata.csv')) #read in associated behavioural data
        
        #average parameter estimates across items, as we dont have pupillometry separately for each item
        prec = prec.mean(0) #average across items in the array
        prec = prec[:,0].copy() #take just the parameter estimate for precision, as the other level of that dimension is its t-value

        use_b = True
        if use_b:
            paramind = 0
            addtext = 'modelAmplitudeBeta'
        elif not use_b:
            paramind = 1
            addtext = 'modelAmplitudeTvalue'
            
        amp = amp.mean(0)[:, paramind] #get the beta/tvalue of amplitude depending on what you want to model
        
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
        prec = prec[tckeep]
        amp    = amp[tckeep]
        tcbdata = tcbdata.loc[tckeep]
        
        #get the right eyetracking data
        data.metadata = data.metadata.loc[ekeep]
        data.data = data.data[ekeep]
        data.event_id = data.event_id[ekeep]
        
        nparams = 3 #only using a constant and one of the 
        allb = np.zeros(shape = [modeltimes.size, nparams, data.times.size])
        allt = np.zeros(shape = [modeltimes.size, nparams, data.times.size])
        print('- - - - -  running glm across tuning curve timepoints - - - - -')
        for itp in range(modeltimes.size):
            #get parameter estimates for this timepoint, across trials    
            ia = prec[:, itp]
            ib1 = amp[:,itp]
            
            DC = glm.design.DesignConfig()
            DC.add_regressor(name = 'mean', rtype = 'Constant')
            DC.add_regressor(name = 'precision', rtype = 'Parametric', datainfo = 'precisions', preproc = 'z')
            DC.add_regressor(name = 'amplitude', rtype = 'Parametric', datainfo = 'amplitudes', preproc = 'z')
            DC.add_simple_contrasts()
            
            #create glmdata object
            glmdata = glm.data.TrialGLMData(data = data.data.squeeze(), time_dim = 1, sample_rate = 1000,
                                            precisions = ia, amplitudes = ib1
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
        
        # fig=plt.figure();
        # ax=fig.add_subplot(111); 
        # ax.plot(modeltimes, tmp.T, label=['precision', 'amplitude']);
        # ax.axvline(0, lw=1, ls='dashed', color='k'); 
        # ax.axhline(0, lw=1, ls='dashed', color='k'); 
        # ax.legend()
        

        #save individual betas and tstats
        np.save(op.join(wd, 'data', 'glms', '500hz', 'smooth_alphas', 'eye_glm', f'wmc_s{sub:02d}_glm3_betas_{fittext}_{addtext}.npy'), arr = allb)
        np.save(op.join(wd, 'data', 'glms', '500hz', 'smooth_alphas', 'eye_glm', f'wmc_s{sub:02d}_glm3_tvalues_{fittext}_{addtext}.npy'), arr = allt)
        if sub == 4:
            np.save(op.join(wd, 'data', 'glms', '500hz', 'smooth_alphas', 'eye_glm', 'eyetracker_times.npy'), arr = data.times)
            np.save(op.join(wd, 'data', 'glms', '500hz', 'smooth_alphas', 'eye_glm', f'regressor_names.npy'), arr = model.regressor_names)