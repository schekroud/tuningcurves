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

loc = 'workstation'
if loc == 'laptop':
    #eyefuncdir = '/Users/sammichekroud/Desktop/postdoc/student_projects/EffortDifficulty/analysis/tools'
    eyefuncdir = '/Users/sammichekroud/Desktop/postdoc/tools'
    wd         = '/Users/sammichekroud/Desktop/postdoc/wmconfidence' #working on confidence data, but in postdoc dir
elif loc == 'workstation':
    eyefuncdir = 'C:/Users/sammirc/Desktop/postdoc/tools/'
    wd         =  'C:/Users/sammirc/Desktop/postdoc/tuningcurves'
    funcdir    = op.join(wd, 'analysis', 'tools')
    sys.path.insert(0, funcdir)
os.chdir(wd)
# sys.path.insert(0, eyefuncdir)
sys.path.insert(0, op.join(wd, 'analysis', 'tools'))

from funcs import getSubjectInfo, clusterperm_test

eyedir = op.join(wd, 'data', 'eyes')
bdir   = op.join(wd, 'data', 'datafiles')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26]) #eeg ppts
# subs = np.array([         4, 5, 6, 7, 8, 9,             13, 14, 15,     17,         20, 21, 22,     24, 25, 26]) # eyetracking ppts
nsubs = subs.size
#set some params here
modeltimes = np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'times_500hz.npy'))

nparams=2 #how many parameters will we fit in later models
b_prec = np.zeros(shape = [nsubs, nparams, modeltimes.size]) * np.nan
b_amp = b_prec.copy()
t_prec, t_amp = b_prec.copy(), b_prec.copy()
b_glm3 = np.zeros(shape = [nsubs, 3, modeltimes.size]) * np.nan
t_glm3 = b_glm3.copy()

subcount = -1
print('looping over participants and modelling')
for sub in subs:
    subcount += 1 
    # print(f'- - - working on ppt {sub} - - -')
    #load in the parameter fits for this participant
    binstep, binwidth = 4, 22
    # binstep, binwidth = 15, 22
    smooth_alphas, smooth_sigma = True, 3
    
    #read in modelled tuning curve parameters
    prec = np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'all_smoothing',
                f's{sub}_ParamFits_precision_binstep{binstep}_binwidth{binwidth}.npy'))
    
    ampglm = np.load(op.join(wd, 'data', 'tuningcurves', '500hz', 'parameter_fits', 'all_smoothing',
                f's{sub}_ParamFits_amplitude_binstep{binstep}_binwidth{binwidth}_glmfit.npy'))
    
    #note here precision is modelled in the same way, but amplitude is modelled with a constrained alpha (min 0.001), and minmax-scaled design matrix for the glmfit. optimised fit doesnt have to scale the design matrix
    tcbdata = pd.read_csv(op.join(wd, 'data', 'tuningcurves', f's{sub}_TuningCurve_metadata.csv')) #read in associated behavioural data
    #metadata doesnt have the behavioural preproc output
    tcbdata = tcbdata.assign(trlid = np.where(tcbdata.session.eq('a'), tcbdata.trialnum, tcbdata.trialnum+256))
    
    #so lets grab the file that does
    isub = dict(loc = loc, id = sub); param = getSubjectInfo(isub)
    bdata = pd.read_csv(op.join(param['path'], 'datafiles', f's{sub:02d}', f'wmConfidence_S{sub:02d}_gathered_preprocessed.csv')) #has full behavioural preproc done
    bdata = bdata.assign(trlid = np.where(bdata.session.eq('a'), bdata.trialnum, bdata.trialnum+256)) #get trialnumber
    bdata = bdata.query('trlid in @tcbdata.trlid') #take just the trials used in the tuning curve decomposition
    
    #drop trials with slow RTs (more than 2.5sds from the ppt mean)
    dt = bdata.DT.to_numpy()
    meandt = dt.mean() #mean decision time
    sd_dt  = dt.std() #std deviation of decision times
    dtcheck = np.logical_or(np.greater(dt, meandt + 3*sd_dt), np.less(dt, meandt - 3*sd_dt)) #mark as 1 if discard
    dtcheck = np.greater(dt, meandt + 3*sd_dt)
    bdata = bdata.assign(DTcheck = dtcheck) #reassign dtcheck so it's across all trials of the participant, not within cue condition
    keeptrls = dtcheck == 0 #mark if keeping a trial

    #discard trials that are too slow
    prec  = prec[:,keeptrls]
    ampglm = ampglm[:,keeptrls]
    bdata  = bdata.iloc[keeptrls]    

    # # exclude trials with over 60 degrees of error
    # keeptrls = bdata.absrdif.le(60).to_numpy()
    # prec = prec[:, keeptrls]
    # ampglm = ampglm[:,keeptrls]
    # bdata = bdata.query('absrdif <= 60')
    
    [nitems, ntrials, _, ntimes] = prec.shape #get some feature dimensions
    
    #get things
    pitem = bdata.pside.to_numpy() #vector describing which side the subsequently probed item was on. 0 = left, 1 = right item
    cued  = bdata.cue.to_numpy()
    cuecond = np.where(cued == 0, -1, 1) #[1, -1] contrast regressor for [cued, neutral] as this influences later response error
    cuecond_z = sp.stats.zscore(cuecond, ddof=0)
    
    error = bdata.absrdif.to_numpy() #absolute angular error on a given trial
    logerr = np.log(np.where(error == 0, 0.1, error))
    logerr_z = sp.stats.zscore(logerr, ddof=0)
    rt    = bdata.DT.to_numpy() #decision time - time taken to move the mouse to start reporting the orientation
    
    er_z = sp.stats.zscore(error, ddof = 0) #zscore error
    logrt = np.log(rt) #log transform reaction times (note that we aren't cutting out any trials yet)
    
    precision = np.nanmean(prec, axis=0)[:, 0] #average tuning curve precision across items in the array, take just the parameter estimate of precision not the t-value
    amplitude = np.nanmean(ampglm, axis=0)[:,0] #take just the parameter estimate for amplitude fit, average across items in the array
    
    #get precision and amplitude of the item that was actually probed - lets us test if precision/amplitude of the representation of the item that was actually reported is relevant for behaviour
    itemprec = np.zeros(shape = [ntrials, ntimes]) * np.nan
    itemamp = np.zeros(shape = [ntrials, ntimes]) * np.nan
    for itrl in range(ntrials):
        ipside = pitem[itrl]
        itemprec[itrl] = prec[ipside, itrl, 0]
        itemamp[itrl]  = ampglm[ipside, itrl, 0] #keep just the parameter estimate

    #empty vectors to store data
    # nparams = 2
    betas_prec = np.zeros(shape = [nparams, ntimes]) * np.nan
    tvals_prec = np.zeros(shape = [nparams, ntimes]) * np.nan
    betas_amp  = np.zeros(shape = [nparams, ntimes]) * np.nan
    tvals_amp  = np.zeros(shape = [nparams, ntimes]) * np.nan

    betas_glm3 = np.zeros(shape = [3, ntimes]) * np.nan
    tvals_glm3 = np.zeros(shape = [3, ntimes]) * np.nan
    
    #fitting the same design matrix at each time point, so just make it once
    dm = np.array([np.ones(er_z.size), er_z]).T
    dm = pd.DataFrame(dm, columns = ['constant', 'error'])
    
    for tp in range(modeltimes.size): #loop over time points
        #get item specific tuning curve parameter estimates
        iprec, iamp = itemprec[:,tp], itemamp[:,tp]
        iprec, iamp = sp.stats.zscore(iprec, ddof=0), sp.stats.zscore(iamp, ddof=0)

        #model precision
        gl1 = sma.GLM(endog = iprec, exog = dm, family = sma.families.Gaussian())
        gl1fit = gl1.fit()

        #model amplitude
        gl2 = sma.GLM(endog = iamp, exog = dm, family = sma.families.Gaussian())
        gl2fit = gl2.fit()

        gl3 = sma.GLM(endog = np.where(error == 0, 0.1, error),
                      exog = np.array([np.ones(ntrials), iprec, iamp]).T,
                      family = sma.families.Gaussian())
        gl3fit = gl3.fit()
        
        #store parameter estimates
        betas_prec[:, tp] = gl1fit.params.to_numpy()
        betas_amp[:, tp]  = gl2fit.params.to_numpy()
        betas_glm3[:, tp] = gl3fit.params

        #store individual ppt t-values for effects
        tvals_prec[:, tp] = gl1fit.tvalues.to_numpy()
        tvals_amp[:, tp]  = gl2fit.tvalues.to_numpy()
        tvals_glm3[:, tp] = gl3fit.tvalues
    
    b_prec[subcount] = betas_prec
    b_amp[subcount]  = betas_amp
    t_prec[subcount] = tvals_prec
    t_amp[subcount]  = tvals_amp
    b_glm3[subcount] = betas_glm3
    t_glm3[subcount] = tvals_glm3
print('done fitting on all participants')

#save across subject betas in one file
np.save(op.join(wd, 'data', 'glms', '500hz', 'all_smoothing', 'beh_glm', f'TCParamsXbehaviour_beta_precision_binstep{binstep}_binwidth{binwidth}.npy'),  arr = b_prec)
np.save(op.join(wd, 'data', 'glms', '500hz', 'all_smoothing', 'beh_glm', f'TCParamsXbehaviour_beta_amplitude_binstep{binstep}_binwidth{binwidth}.npy'),  arr = b_amp)
np.save(op.join(wd, 'data', 'glms', '500hz', 'all_smoothing', 'beh_glm', f'TCParamsXbehaviour_tstat_precision_binstep{binstep}_binwidth{binwidth}.npy'), arr = t_prec)
np.save(op.join(wd, 'data', 'glms', '500hz', 'all_smoothing', 'beh_glm', f'TCParamsXbehaviour_tstat_amplitude_binstep{binstep}_binwidth{binwidth}.npy'), arr = t_amp)
