# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:12:29 2024

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
    for fittype in ['opt', 'glm']:
        # if not op.exists(op.join(eyedir, 'preprocessed', f'EffDS{sub}_preprocessed.pickle')): #dont do this if it already exists!
        print(f'\n- - - working on ppt {sub} - - -')
        # data = eyes.io.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_pupil_arraylocked.pickle'))
        # data = data.apply_baseline([-0.5, -0.2])
        # trlcheck = np.load(op.join(eyedir, 'epoched', f'wmc_s{sub:02d}_arraylocked_nanperc.npy')) #this always has as many trials as the eyetracking data does
        #load in the parameter fits for this participant
        binstep, binwidth = 4, 22
        smooth_alphas, smooth_sigma = True, 3
        
        #read in modelled tuning curve parameters
        alpha = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'twostage_alphaminmaxfit_b1desmatminmax',
                f's{sub}_ParamFits_precision_binstep{binstep}_binwidth{binwidth}_smoothedprec.npy'))
        
        
        ampglm = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'twostage_alphaminmaxfit_b1desmatminmax',
            f's{sub}_ParamFits_amplitude_binstep{binstep}_binwidth{binwidth}_smoothedprec_glmfit.npy'))
        
        ampopt = np.load(op.join(wd, 'data', 'tuningcurves', 'parameter_fits', 'twostage_alphaminmaxfit_b1desmatminmax',
            f's{sub}_ParamFits_amplitude_binstep{binstep}_binwidth{binwidth}_smoothedprec_optfit.npy'))
        #note here precision is modelled in the same way, but amplitude is modelled with a constrained alpha (min 0.001), and minmax-scaled design matrix for the glmfit. optimised fit doesnt have to scale the design matrix
        
        tcbdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{sub}_TuningCurve_metadata.csv')) #read in associated behavioural data
        
        #average parameter estimates across items, as we dont have pupillometry separately for each item
        # alpha = alpha.mean(0)
        # b1    = b1.mean(0)
        alpha = alpha.mean(0) #average across items in the array
        use_b = False
        if use_b:
            paramind = 0
            addtext = 'modelAmplitudeBeta'
        elif not use_b:
            paramind = 1
            addtext = 'modelAmplitudeTvalue'

        if fittype == 'opt':
            amp    = ampopt.mean(0)[:,paramind] #b1 opt has shape [nitems x ntrials x params x time] where params are [beta, tvalue]
            fittext = 'b1optfit'
        elif fittype == 'glm':
            amp = ampglm.mean(0)[:, paramind]        
            fittext = 'b1glmfit'
        
        smoothamp = True
        if smoothamp:
            #smooth amplitude estimate lightly over time   
            amp = sp.ndimage.gaussian_filter1d(amp, sigma = 2) #lightly smooth with gaussian blur to reduce single trial noise
            amptxt = '_smoothamp'
        elif not smoothamp:
            amptxt = ''
            
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
        
        
        nparams = 3 #only using a constant and one of the 
        nparams = 4
        allb = np.zeros(shape = [modeltimes.size, nparams, eegtimes.size])
        allt = np.zeros(shape = [modeltimes.size, nparams, eegtimes.size])
        print('- - - - -  running glm across tuning curve timepoints - - - - -')
        for itp in range(modeltimes.size):
            #get parameter estimates for this timepoint, across trials    
            ia  = alpha[:, itp]
            ib1 = amp[:,itp]
            
            DC = glm.design.DesignConfig()
            DC.add_regressor(name = 'mean', rtype = 'Constant')
            DC.add_regressor(name = 'precision', rtype = 'Parametric', datainfo = 'alpha', preproc = 'z')
            DC.add_regressor(name = 'amplitude', rtype = 'Parametric', datainfo = 'beta1', preproc = 'z')
            DC.add_regressor(name = 'trlid', rtype = 'Parametric', datainfo = 'trlidz', preproc = None)
            DC.add_simple_contrasts()
            
            #create glmdata object
            glmdata = glm.data.TrialGLMData(data = tfdata, time_dim = 1, sample_rate = 100, #TF data as input has 100hz sampling rate
                                            alpha = ia, beta1 = ib1, trlidz = trl_z
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
        np.save(op.join(wd, 'data', 'glms', 'eeg', 'glm2', f'wmc_s{sub:02d}_glm2{logtxt}_betas_{fittext}_{addtext}{amptxt}.npy'), arr = allb)
        np.save(op.join(wd, 'data', 'glms', 'eeg', 'glm2', f'wmc_s{sub:02d}_glm2{logtxt}_tvalues_{fittext}_{addtext}{amptxt}.npy'), arr = allt)
        if sub == 4:
            np.save(op.join(wd, 'data', 'glms', 'eeg', 'glm2', 'eeg_times.npy'), arr = eegtimes)
            np.save(op.join(wd, 'data', 'glms', 'eeg', 'glm2', f'regressor_names.npy'), arr = model.regressor_names)