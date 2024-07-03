#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:01:40 2023

@author: sammi
"""
import os
import os.path as op
import numpy as np
import scipy as sp
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt
import mne

def getSubjectInfo(subject):
    
    param = {}
    
    if subject['loc']   == 'workstation':
        param['path']   = 'F:\wmconfidence'
        datdir = 'F:\wmconfidence'
        eegdir = op.join(datdir, 'eeg')
        
    substr = 's%02d'%subject['id']
    param['substr'] = substr
    files = os.listdir(op.join(eegdir, f'{substr}'))
    
    #get paths for the raw data for each session
    afname = [x for x in files if f'{substr}a' in x and x[-4:] == '.cdt'][0]
    param['raw1'] = op.join(eegdir, f'{substr}', afname)
    
    if subject['id'] not in [1, 2, 3, 10, 19]: #these particiants only have one session
        bfname = [x for x in files if f'{substr}b' in x and x[-4:] == '.cdt'][0]
        param['raw2'] = op.join(eegdir, f'{substr}', bfname)
        
    #if subject['id'] in []: #has three recordings for the two sessions
    
    if subject['id'] == 4:
        param['badchans'] = []
    if subject['id'] == 5:
        param['badchans'] = []
    if subject['id'] == 6:
        param['badchans'] = []
    if subject['id'] == 7:
        param['badchans'] = []
    if subject['id'] == 8:
        param['badchans'] = []
    if subject['id'] == 9:
        param['badchans'] = ['T8']
    if subject['id'] == 10:
        param['badchans'] = []
    if subject['id'] == 11:
        param['badchans'] = ['T8', 'P1']
    if subject['id'] == 12:
        param['badchans'] = ['T7']
    if subject['id'] == 13:
        param['badchans'] = ['T8']
    if subject['id'] == 14:
        param['badchans'] = ['T7', 'TP8']
    if subject['id'] == 15:
        param['badchans'] = []
    if subject['id'] == 16:
        param['badchans'] = []
    if subject['id'] == 17:
        param['badchans'] = ['TP7', 'TP8']
    if subject['id'] == 18:
        param['badchans'] = ['TP7']
    if subject['id'] == 19:
        param['badchans'] = []
    if subject['id'] == 20:
        param['badchans'] = []
    if subject['id'] == 21:
        param['badchans'] = ['T8', 'T7', 'F6', 'FP2']
    if subject['id'] == 22:
        param['badchans'] = []
    if subject['id'] == 23:
        param['badchans'] = []
    if subject['id'] == 24:
        param['badchans'] = ['FT8', 'T7', 'T8']
    if subject['id'] == 25:
        param['badchans'] = ['T8', 'TP7']
    if subject['id'] == 26:
        param['badchans'] = ['T8', 'F3', 'F5', 'T7', 'FT8']


    return param

    
def gesd(x, alpha = .05, p_out = .1, outlier_side = 0):
    import numpy as np
    import scipy.stats
    import copy

    '''
    Detect outliers using Generalizes ESD test
    based on the code from Romesh Abeysuriya implementation for OSL
      
    Inputs:
    - x : Data set containing outliers - should be a np.array 
    - alpha : Significance level to detect at (default = .05)
    - p_out : percent of max number of outliers to detect (default = 10% of data set)
    - outlier_side : Specify sidedness of the test
        - outlier_side = -1 -> outliers are all smaller
        - outlier_side = 0 -> outliers could be small/negative or large/positive (default)
        - outlier_side = 1 -> outliers are all larger
        
    Outputs
    - idx : Logicial array with True wherever a sample is an outlier
    - x2 : input array with outliers removed
    
    For details about the method, see
    B. Rosner (1983). Percentage Points for a Generalized ESD Many-outlier Procedure, Technometrics 25(2), pp. 165-172.
    http://www.jstor.org/stable/1268549?seq=1
    '''

    if outlier_side == 0:
        alpha = alpha/2


    if type(x) != np.ndarray:
        x = np.asarray(x)

    n_out = int(np.ceil(len(x)*p_out))

    if any(~np.isfinite(x)):
        #Need to find outliers only in non-finite x
        y = np.where(np.isfinite(x))[0] # these are the indexes of x that are finite
        idx1, x2 = gesd(x[np.isfinite(x)], alpha, n_out, outlier_side)
        # idx1 has the indexes of y which were marked as outliers
        # the value of y contains the corresponding indexes of x that are outliers
        idx = [False] * len(x)
        idx[y[idx1]] = True

    n      = len(x)
    temp   = x
    R      = np.zeros((1, n_out))[0]
    rm_idx = copy.deepcopy(R)
    lam    = copy.deepcopy(R)
    
    
    for j in range(0,int(n_out)):
        i = j+1
        if outlier_side == -1:
            rm_idx[j] = np.nanargmin(temp)
            sample    = np.nanmin(temp)
            R[j]      = np.nanmean(temp) - sample
        elif outlier_side == 0:
            rm_idx[j] = int(np.nanargmax(abs(temp-np.nanmean(temp))))
            R[j]      = np.nanmax(abs(temp-np.nanmean(temp)))
        elif outlier_side == 1:
            rm_idx[j] = np.nanargmax(temp)
            sample    = np.nanmax(temp)
            R[j]      = sample - np.nanmean(temp)

        R[j] = R[j] / np.nanstd(temp)
        temp[int(rm_idx[j])] = np.nan

        p = 1-alpha/(n-i+1)
        t = scipy.stats.t.ppf(p,n-i-1)
        lam[j] = ((n-i) * t) / (np.sqrt((n-i-1+t**2)*(n-i+1)))

    #And return a logical array of outliers
    idx = np.zeros((1,n))[0]
    idx[np.asarray(rm_idx[range(0,np.max(np.where(R>lam))+1)],int)] = np.nan
    idx = ~np.isfinite(idx)

    x2 = x[~idx]


    return idx, x2


def plot_AR(epochs, method = 'gesd', zthreshold = 1.5, p_out = .1, alpha = .05, outlier_side = 1):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import scipy.stats
    from matplotlib import pyplot as plt

    #Get data, variance, number of trials, and number of channels
    dat     = epochs.get_data()
    var     = np.var(dat, 2)
    ntrials = np.shape(dat)[0]
    nchan   = len(epochs.ch_names)

    #set up the axis for the plots
    x_epos  = range(1,ntrials+1)
    y_epos  = np.mean(var,1)
    y_chans = range(1,nchan+1)
    x_chans = np.mean(var,0)

    #scale the variances
    y_epos  = [x * 10**6 for x in y_epos]
    x_chans = [x * 10**6 for x in x_chans]

    #Get the zScore
    zVar = scipy.stats.zscore(y_epos)

    #save everything in the dataFrame
    df_epos           = pd.DataFrame({'var': y_epos, 'epochs': x_epos, 'zVar': zVar})
    df_chans          = pd.DataFrame({'var': x_chans, 'chans': y_chans})

    # Apply the artefact rejection method
    if method == 'gesd':
        try:
            idx,x2            = gesd(y_epos, p_out=p_out, alpha=alpha, outlier_side=outlier_side) #use the gesd to find outliers (idx is the index of the outlier trials)
        except:
            print('***** gesd failed here, no trials removed *****')
            idx = []
        keepTrials        = np.ones((1,ntrials))[0]
        keepTrials[idx]   = 0
        title = 'Generalized ESD test (alpha=' + str(alpha) + ', p_out=' + str(p_out) + ', outlier_side=' + str(outlier_side) + ')'
    elif method == 'zScore':
        keepTrials        = np.where(df_epos['zVar'] > zthreshold, 0, 1)
        title = 'ZVarience threshold of ' + str(zthreshold)
    elif method == 'none':
        title = 'no additional artefact rejection '
        keepTrials        = np.ones((1,ntrials))[0]

    df_epos['keepTrial'] = keepTrials
    df_keeps = df_epos[df_epos['keepTrial'] == 1]
    print(str(ntrials - len(df_keeps)) + ' trials discarded')

    # get the clean data
    keep_idx    = np.asarray(np.where(keepTrials),int)
    clean_dat    = np.squeeze(dat[keep_idx])

    #recalculate the var for chan
    clean_var    = np.var(clean_dat, 2)
    x_chans_c    = np.mean(clean_var,0)
    x_chans_c    = [x * 10**6 for x in x_chans_c]

    df_chans_c   = pd.DataFrame({'var': x_chans_c, 'chans': y_chans})


    # Plot everything
    fig, axis = plt.subplots(2, 2, figsize=(12, 12))
    axis[0,0].set_ylim([0, max(y_epos) + min(y_epos)*2])
    axis[0,1].set_xlim([0, max(x_chans)+ min(x_chans)*2])
    axis[1,0].set_ylim([0, max(df_keeps['var'])+ min(df_keeps['var'])*2])
    axis[1,1].set_xlim([0, max(x_chans_c)+ min(x_chans_c)*2])

    axis[0,0].set_title(title)
    sns.scatterplot(x = 'epochs', y = 'var', hue = 'keepTrial', hue_order = [1,0], ax = axis[0,0], data = df_epos)
    sns.scatterplot(x = 'var', y = 'chans', ax = axis[0,1], data = df_chans)
    sns.scatterplot(x = 'epochs', y = 'var', ax = axis[1,0], data =df_keeps)
    sns.scatterplot(x = 'var', y = 'chans', ax = axis[1,1], data = df_chans_c)



    return axis, keep_idx


def streaks(array):
    '''
    finds streaks of 1/0s in an array
    '''
    x = np.zeros(array.size).astype(int)
    count = 0
    for ind in range(len(x)):
        if array[ind]:
            count += 1
        else:
            count = 0
        x[ind] = count
    
    return x

def streaks_numbers(array):
    '''
    

    Parameters
    ----------
    array : np.array
        array of numbers you want to find streaks in

    Returns
    -------
    numpy array. when a new value is found vs previous, the value of x is 1.
    
    It increases incrementally with each repeated number to count how many values have the same value as the previous

    '''
    
    x = np.zeros(array.size).astype(int)
    count = 0
    for ind in range(len(x)):
        if array[ind] == array[ind-1]:
            count += 1 #continuing the sequence
        else: #changed
            count = 1
        x[ind] = count
    
    return x

def gauss_smooth(array, sigma = 2):
    return sp.ndimage.gaussian_filter1d(array, sigma = sigma, axis=1) #smooths across time, given a 2d array of trials x time

def get_difficulty_sequences(subdat):
    '''
    

    Parameters
    ----------
    subdat : pandas dataframe
        dataframe containing participant behaviour. must contain column called `blocknumber` to separate blocks, and `trialdifficulty` to know difficulty on each trial of the task

    Returns
    -------
    pandas dataframe
        contains two new columns:
            sinceDifficultySwitch - number of trials since the difficulty changed
            untilDifficultySwitch - how many trials are left until the difficulty will change

    '''
    
    data = subdat.copy()
    nruns = data.blocknumber.unique().size #get how many task blocks, as difficulty starts fresh each block
    df = pd.DataFrame() #make a new dataframe to append to, and return at the end
    
    for run in np.add(range(nruns),1): #loop over blocks
        rundat = data.copy().query('blocknumber == @run') #get data for this block
        
        #getting streaks in difficulty sequence
        streaks1 = streaks_numbers(rundat.trialdifficulty.to_numpy()) #trials since difficulty changed
        streaks2 = np.flip(streaks_numbers(np.flip(rundat.trialdifficulty.to_numpy()))) #trials until difficulty changes
        rundat = rundat.assign(
            sinceDifficultySwitch = streaks1,
            untilDifficultySwitch = streaks2)
        df = pd.concat([df, rundat])
    
    return df
        
def clusterperm_test(data, labels, of_interest, times, tmin = None, tmax = None, out_type = 'indices', n_permutations = 'Default', tail = 0, threshold = None, n_jobs = 2):
    '''
    function to run permutation tests on a time-series (eg. alpha timecourse).
    
    Inputs:
        data            - the data array of interest (e.g. betas/copes/tstats)
        labels          - the labels (names) of regressors/copes/tstats. length of this should match an axis
        of_interest     - the regressor/contrast of interest
        times           - array showing time labels for each data point (useful if wanting to crop bits)
        tmin, tmax      - timewindow for period to run cluster test over (if not whole signal)
        out_type        - specify output type. defaults to indices, can set to mask if you really want
        tail            - specify whether you want to do one tailed or two tailed. 0 = two-tailed, 1/-1 = one-tailed
        threshold       - cluster forming threshold. Default = None (t-threshold chosen by mne). can specify a float, where data values more extreme than this threshold will be used to form clusters
    
    '''
    import scipy as sp
    from scipy import ndimage
    from copy import deepcopy
    
    iid = np.where(labels == of_interest)[0] #get location of the regressor/cope we want
    dat = np.squeeze(data.copy()[:,iid,:])
    nsubs = len(dat)
    
    #set defaults assuming no restriction of cluster timewindow
    twin_minid = 0 #first timepoint
    twin_maxid = None #last timepoint
    if tmin != None or tmax != None: #some specified time window
        if tmin != None: #get onset of time window
            twin_minid = np.where(times == tmin)[0][0]
        elif tmin == None:
            twin_minid = times.min()
            
        if tmax != None: #get offset of time window
            twin_maxid = np.where(times == tmax)[0][0]
        elif tmax == None:
            twin_maxid = times.max()
    
    if twin_maxid != None:
        twin_times = times[twin_minid:twin_maxid + 1]
        data_twin  = dat.copy()[:, twin_minid:twin_maxid + 1]
    else:
        twin_times = times[twin_minid:]
        data_twin  = dat.copy()[:, twin_minid:]
    
    if n_permutations != 'Default':
        t, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(data_twin, out_type=out_type, n_permutations = n_permutations, tail = tail, threshold = threshold, n_jobs = n_jobs)
    else:
        t, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(data_twin, out_type=out_type, tail = tail, threshold = threshold, n_jobs = n_jobs)
    
    return t, clusters, cluster_pv, H0
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    