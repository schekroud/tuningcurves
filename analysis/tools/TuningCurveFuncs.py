#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:56:12 2024

@author: sammichekroud
"""
import numpy as np
import sklearn as skl
import sklearn.metrics
import scipy as sp
from scipy import optimize
from matplotlib import pyplot as plt

def wrap(x):
    return (x+180)%360 - 180
def wrap90(x):
    return (x+90)%180 - 90

def createFeatureBins(binstep, binwidth, feature_start = -90, feature_end = 90):
    binmids = np.arange(feature_start, feature_end+1, binstep) #get the centre angle of each bin, in degrees
    binstarts, binends = np.subtract(binmids, binwidth), np.add(binmids, binwidth) #get start and end points of each bin, in degrees
    nbins   = binstarts.size #get how many bins these bin parameters create 
    
    return nbins, binmids, binstarts, binends

def visualise_FeatureBins(binstarts, binmids, binends):
    
    '''
    all angles should be in degrees, to begin with. The visualisation function will convert to radians 
    '''

    #get binwidth
    binwidth = int((binends[0]-binstarts[0])/2)
    nangles = np.arange(binstarts.min(), binends.max()+1, 1)
    nangles = np.deg2rad(nangles) #conv to radians
    tmpcurves = np.zeros(shape = [binstarts.size, nangles.size]) * np.nan #create empty thing to populate to show angle curves for bins
    tmpbins   = np.zeros_like(tmpcurves)
    binangs   = np.zeros(shape = [binstarts.size, binwidth*2]) #store angles for each bin
    for ibin in range(binstarts.size):
        istart, iend = binstarts[ibin], binends[ibin] #get the angles
        istartrad, iendrad = np.deg2rad(binstarts[ibin]), np.deg2rad(binends[ibin])
        iangles = np.arange(istart, iend)
        binangs[ibin] = iangles
        # iangles = np.deg2rad(iangles)
        imid = binmids[ibin]
        imidrad = np.deg2rad(binmids[ibin])
        # ibinmask = np.isin(nangles, np.arange(istart, iend)).astype(int)
        ibinmask = np.logical_and(np.greater_equal(nangles, istartrad), np.less(nangles, iendrad))
        tmpbins[ibin] = ibinmask
        ianglesrad = np.deg2rad(iangles)
        # tmp  = np.cos(np.radians(((iangles-imid)/binwidth*2))* np.pi)
        tmp = np.cos( ((ianglesrad-imidrad)/np.deg2rad(binwidth*2)*np.pi) )
        icurve = ibinmask.copy().astype(float) * np.nan;
        icurve[ibinmask==1] = tmp
        tmpcurves[ibin] = icurve
    
    #visualise the weightings that will be applied to different angles within each bin
    fig = plt.figure(); ax = fig.add_subplot(111)
    for ibin in range(binstarts.size):
        ax.plot(np.deg2rad(binangs[ibin]), np.squeeze(tmpcurves[ibin, np.where(~np.isnan(tmpcurves[ibin]))])  )
    ax.set_xlabel('orientation (radians)')
    ax.set_ylabel('weighted contribution to bin mean (AU)')
    fig.suptitle('contribution of individual orientations to activity in reference bins')
    
    #visualise what angles are possible in each bin
    fig = plt.figure(figsize = [12, 3]);
    ax = fig.add_subplot(111)
    # # ax.plot(nangles, sp.ndimage.gaussian_filter1d(tmpbins,3).T)
    ax.plot(nangles, tmpbins.T)
    ax.set_xlabel('orientation (radians)')
    ax.set_ylabel('Orientation in reference bin?')
    ax.set_yticks(ticks = [0, 1], labels = ['no', 'yes'])

def makeTuningCurve(data, orientations, binstep, binwidth, weight_trials=True, feature_start = -90, feature_end = 90):
    
    '''
    this function is designed to be run per timepoint. as a result, inputs should look like:
        
    data:         array,      shape = [ntrials, nchannels], the EEG data being used
    orientations: array-like, shape = [ntrials], vector containing the stimulus orientation associated with each trial
    nbins:        integer, the number of bins to create the tuning curve with
    binstep:      integer, number of degrees you advance along the feature (orientation) space for each bin centre
    binwidth:     integer, the width of the feature (orientation) bin. this width is applied both sides (so true width = binwidth * 2)
    weight_trials: Bool, whether or not trials within a bin should have their contribution to the mean activity weighted by their distance from bin centre
    featurestart: integer, start point of the feature (orientation) space in degrees. (default -90)
    featureend:   integer, end point of the feature (orientation) space in degrees. (default 90, this is INCLUSIVE)
    
    '''
    
    [ntrials, nfeatures] = data.shape #get shape
    trls = np.arange(ntrials) #vector for trial number
    
    #set up feature (orientation) space bins
    nbins, binmids, binstarts, binends = createFeatureBins(binstep, binwidth, feature_start, feature_end)
    
    d = np.zeros([ntrials, nbins]) * np.nan
    
    for trl in trls: #loop over trials
        
        xtrain = data[np.setdiff1d(trls, trl)]
        xtest  = data[trl]
        
        oritrain = orientations[np.setdiff1d(trls, trl)]
        oritest  = orientations[trl]
        
        reloris     = wrap90(np.subtract(oritrain, oritest)) #get the orientation of all training trials relative to the orientation of the test (left out) trial
        relorisrad  = np.deg2rad(reloris) #convert to radians
        
        m       = np.zeros([nbins, nfeatures])   * np.nan #array to store mean activity across all trials within a feature bin
        inbin   = np.zeros([len(xtrain), nbins]) * np.nan #array to store whether a trial is in a feature bin
        
        icov = np.linalg.pinv(np.cov(xtrain.T)) #get inverse covariance matrix for the training trials, across channels (shape = [nfeatures, nfeatures])
        mahaldist = skl.metrics.DistanceMetric.get_metric('mahalanobis', VI = icov) #set up mahalanobis distance metric
        
        #loop over bins, get (weighted) activity in each bin
        for b in range(nbins):
            istart  = np.radians(binstarts[b])
            iend    = np.radians(binends[b])
            imid    = np.radians(binmids[b])
            
            
            #if non-overlapping bins, then the binwidth is exactly half the binstep
            if binwidth == int(binstep/2):
                trlcheck = np.logical_and(np.greater_equal(relorisrad, istart), np.less(relorisrad, iend)) #make sure we are checking if *less* than the bin end, as we dont want overlap with the next feature bin
            else:
                trlcheck = np.logical_and(np.greater_equal(relorisrad, istart), np.less_equal(relorisrad, iend)) #if you are doing overlapping bins this doesn't matter
            
            binoris = relorisrad[trlcheck] #get the orientations for trials that contribute to this feature bin
            bindata = xtrain[trlcheck]     #get the EEG data for trials that contribute to this feature bin
            
            
            if weight_trials: #weight trial contribution to the mean based on distance from the centre of the feature bin
                #this means that trials at the edges of the bin (less similar to the bin centre) contribute less than trials closer to the desired orientation
                w = np.cos( ((binoris-imid)/np.radians(binwidth*2))*np.pi ) #cosine weighting based on distance from bin centre (scaled by pi)
            else:
                w = np.ones(binoris.size) #if not weighting, just multiply everything by 1
                
            bindata = np.multiply(bindata, w.reshape(-1, 1)) #reshape weights to allow multiplication
            
            if bindata.size >0: #just prevents warnings churning out for means of empty slices (where no data in a bin because the relative orientations dont exist)
                m[b,:] = np.nanmean(bindata, axis=0)
        
        nanbins = np.isnan(m).sum(axis=1)>0 #find bins where there are no data (due to design of orientations used in the task)
        idists  = mahaldist.pairwise(xtest.reshape(1,-1), m[~nanbins]) #get mahalanobis distance between test trial activity and activity in all non-nan feature bins 
        d[trl, ~nanbins] = idists
    
    return d #return the mahalanobis distance between each trial and all other trials in feature space bins
            
            
def cosmodel(thetas, B0, B1, alpha):
    return B0 + (B1 * np.cos(alpha * thetas))

def cosmodel2(thetas, B1, alpha):
    return B1 * np.cos(alpha * thetas)

def fmin_func(params, x, binmids):
    '''
    x       - distances for this time point and trial
    params  - parameters to try on this iteration
    binmids - radian values for bin middles being used  
    '''
    [b0, b1, alpha] = params
    fitted = b0 + (b1 * np.cos(alpha * binmids))
    resids = np.subtract(x, fitted)
    ssr = np.sum(np.power(resids,2))
    return ssr

def fmin_func2(params, x, binmids):
    '''
    x       - distances for this time point and trial
    params  - parameters to try on this iteration
    binmids - radian values for bin middles being used  
    '''
    x_mean = np.nanmean(x) #get the across bin mean distance
    [b0, b1, alpha] = params
    fitted = x_mean + b0 + (b1 * np.cos(alpha * binmids)) #model this
    resids = np.subtract(x, fitted)
    ssr = np.sum(np.power(resids,2))
    return ssr

def getCosineFit(angles, data, fitType = 'curve', bounds = None, p0 = None, method = 'trf', disp = 0, maxfev = 5000):
    '''
    
    use `sp.optimize.curve_fit`, using non-linear squares to find the optimal cosine fit parameters to tuning curve data.
    This function fits a 3 parameter model: B0 + B1*cos(alpha*theta)
    - B0 models the average mahalanobis across bins
    - B1 models the amplitude of the cosine fit, reflecting strength of 'decoding' or concordance with circular space
    - alpha models the width of the cosine fit,  reflecting the uncertainty in the 'decoding' or concordance with circular space
    
    angles  - centre angle of the feature (orientation) bin (binmids from previous function calls in the pipeline)
    data    - array of length nbins. This is distance data for a single trial at a single time point
              mahalanobis distances between the left out (test) trial and average activity for trials in all other feature bins
              (note this is effectively the singla trial distance from the trial to activity relating to other orientations)
    fitType - whether to model data using an optimised curve fit (scipy) or fminsearch
    p0      - initial guess for parameters to fit if using curve fit option
    method  - optimisation algorithm to use for curve fit
    disp    - whether or not to display convergence messages with fminsearch fitting
             
             
    returns:
        - parameters that optimise the model fit
        - angles used for the model fit
        - data used for the model fit
    '''
    
    isnan = np.isnan(data)
    imids = angles[~isnan]
    idat = data[~isnan]
    
    if fitType == 'curve': #use scipy inbuilt curve optimisation
        if bounds != None    :
            fitparams = sp.optimize.curve_fit(cosmodel, imids, idat, bounds = bounds,
                                              p0 = p0, #add initial parameter guess if supplied
                                              maxfev = maxfev, method = method)[0] #get just the optimal params for this cosine fit
        else:
            fitparams = sp.optimize.curve_fit(cosmodel, imids, idat,
                                              p0 = p0, #add initial parameter guess if supplied
                                              maxfev = maxfev, method = method)[0] #get just the optimal params for this cosine fit
    if fitType == 'curve2':
        fitparams = sp.optimize.curve_fit(cosmodel2, imids, idat,
                                          p0 = p0,
                                          maxfev = maxfev, method = method)[0]
    elif fitType == 'fmin':
        fitparams = sp.optimize.fmin(func = fmin_func, #function to minimize
                                     x0 = [idat.mean(), 1, 1], #initial parameter guesses
                                     args = (idat, imids),
                                     disp = disp) #whether or not to display fminsearch convergence messages
    elif fitType == 'fmin2':
        fitparams = sp.optimize.fmin(func = fmin_func2, #function to minimize
                                     x0 = [1, 1, 1],
                                     args = (idat, imids), disp = disp)
    return fitparams, imids, idat


def basic_cosine(thetas, alpha):
    return np.cos(alpha * thetas)

def fixedAlphaCosine(thetas, B0, B1):
    '''
    thetas - angles for bins, already pre-multiplied by our alpha scaling factor
    '''
    
    return B0 + B1*np.cos(thetas)

def b1_cosine(thetas, B1):
    '''
    thetas - angles for bins, already pre-multiplied by alpha scaling factor
    '''
    return B1 * np.cos(thetas)

def fullCosineModel(thetas, B0, B1, alpha):
    return B0 + (B1 * np.cos(alpha * thetas))

def fitAlpha(thetas, distances, p0 = None, bounds = None):
    '''
    thetas - feature values (orientation) for the centre of each bin
    distances - z-scored, inverse (x -1) mahalanobis distances of test trial to each reference bin
    '''
    if bounds == None:
        fitparams = sp.optimize.curve_fit(basic_cosine, thetas, distances,
                                          p0 = p0, maxfev = 5000, method = 'trf', nan_policy='omit')
    else:
        fitparams = sp.optimize.curve_fit(basic_cosine, thetas, distances,
                                          p0 = p0, bounds = bounds,
                                          maxfev = 5000, method = 'trf', nan_policy='omit')
    
    return fitparams[0] #return the optimized recovered parameters



