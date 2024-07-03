# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:02:47 2024

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
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22, 23,     24, 25, 26]) #1,2,3,10,19 all have only 1 session. 23 something wrong? double check
nsubs = subs.size

#some parameters for epoching

events_array = [1, 2]
tmin, tmax = -1, 1.5 #250ms array duration, 1s ISI then cue appears. The last 250ms here is cue presentation, but need window if we want to TF
baseline = None

event_id = {'1' : 1, '2':2,                 #array
            '11':11,'12':12,'13':13,'14':14,#cue
            '21':21,'22':22,'23':23,'24':24,#probe
            '31':31,'32':32,'33':33,'34':34,#space
            '41':41,'42':42,'43':43,'44':44,#click
            '51':51,'52':52,'53':53,'54':54,#confprobe
            '61':61,'62':62,'63':63,'64':64,#space
            '71':71,'72':72,'73':73,'74':74,#click
            '76':76,'77':77,'78':78,'79':79, #feedback
            '254':254,'255':255}

subcount = 0
for i in subs:
    subcount += 1
    print(f'working on participant {subcount}/{nsubs}')
    sub = dict(loc = loc, id = i)
    param = getSubjectInfo(sub)
    
    if i not in [1, 2, 3, 10]:
        for part in [1,2]:
            partstr = ['a', 'b'][part-1]
            raw = mne.io.read_raw_curry(param[f'raw{part}'], preload=True)
            raw = mne.add_reference_channels(raw, ref_channels = 'LM', copy = False) #left mastoid was active reference, add it in as empty channel
            raw.set_eeg_reference(ref_channels = ['LM','RM']) #re-reference to average mastoid
            raw.filter(1, 40, n_jobs = 3)
            
            #set channel types for better handling later
            raw.set_channel_types(mapping = {
                'VEOG':'eog',
                'HEOG':'eog',
                'LM':'misc',
                'RM':'misc',
                'Trigger':'misc'
                })
            
            raw.set_montage('easycap-M1', on_missing='raise', match_case=False) #apply montage
            raw.info['bads'] = param['badchans']
            raw.interpolate_bads()            

            
            events, _ = mne.events_from_annotations(raw, event_id)
            
            epoched = mne.Epochs(raw, events, events_array, tmin, tmax, baseline, reject_by_annotation=False, preload=True) #epoch around array1 presentation
            
            #load in the behavioural data here
            if part == 1:
                bdir = op.join(param['path'], 'datafiles', param['substr'], 'a')
            elif part == 2:
                bdir = op.join(param['path'], 'datafiles', param['substr'], 'b')
            bfiles = os.listdir(bdir)
            
            #gather all the single-block datafiles into one file for this session
            df = pd.DataFrame()
            for ifile in bfiles:
                tmp = pd.read_csv(op.join(bdir, ifile))
                df = pd.concat([df, tmp])
            bdata = df.copy().sort_values(by = 'trialnum')
            
            epoched.metadata = bdata #attach behavioural data to the eeg data
            
            # epoched.plot(picks='eeg', scalings = dict(eeg = 50e-6), n_epochs = 4, n_channels = 61)
            
            if i == 11 and part == 2:
                #here there is a single trial with large enough variance to cause big issues with the ica
                #drop this trial already
                epoched.drop(1)
            
            #run ica on the epoched data
            
            ica = mne.preprocessing.ICA(n_components = 0.95, method = 'infomax').fit(epoched, picks = 'eeg', reject_by_annotation=True)
            eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name = ['HEOG', 'VEOG'])
            eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
            ica.plot_scores(eog_scores, eog_inds)
                    
            ica.plot_components(inst=epoched, contours = 0)
            print('\n\n- - - - - - - subject %d part %s, %d components to remove: - - - - - - -\n\n'%(i, partstr, len(eog_inds)))
            
            comps2rem = input('components  remove: ') #need to separate component numbers by comma and space
            comps2rem = list(map(int, comps2rem.split(', ')))
            np.savetxt(fname = op.join(param['path'], 'removed_comps', 's%02d%s_removedcomps_arraylocked.txt'%(i, partstr)),
                       X = comps2rem, fmt = '%i') #record what components were removed
            ica.exclude.extend(comps2rem) #mark components for removal
            ica.exclude = np.unique(ica.exclude).tolist()
            ica.apply(inst=epoched)
            
            epoched.save(fname = op.join(param['path'], 'eeg', param['substr'], f'{param["substr"]}{partstr}_arraylocked_icacleaned-epo.fif'),
                         fmt = 'double', overwrite = True)
            
            plt.close('all')
            del(raw)
            del(epoched)
            del(ica)
            del(eog_epochs)
    if i in [1, 2, 3, 10]:
        
        raw = mne.io.read_raw_curry(param[f'raw1'], preload=True)
        raw = mne.add_reference_channels(raw, ref_channels = 'LM', copy = False) #left mastoid was active reference, add it in as empty channel
        raw.set_eeg_reference(ref_channels = ['LM','RM']) #re-reference to average mastoid
        raw.filter(1, 40, n_jobs = 3)
        
        #set channel types for better handling later
        raw.set_channel_types(mapping = {
            'VEOG':'eog',
            'HEOG':'eog',
            'LM':'misc',
            'RM':'misc',
            'Trigger':'misc'
            })
        
        raw.set_montage('easycap-M1', on_missing='raise', match_case=False) #apply montage
        
        events, _ = mne.events_from_annotations(raw, event_id)
        
        epoched = mne.Epochs(raw, events, events_array, tmin, tmax, baseline, reject_by_annotation=False, preload=True) #epoch around array1 presentation
        
        #load in the behavioural data here
        
        bdir = op.join(param['path'], 'datafiles', param['substr'])
        bfiles = os.listdir(bdir)
        bfiles = [x for x in bfiles if '_block_' in x]
        
        #gather all the single-block datafiles into one file for this session
        df = pd.DataFrame()
        for ifile in bfiles:
            tmp = pd.read_csv(op.join(bdir, ifile))
            df = pd.concat([df, tmp])
        bdata = df.copy().sort_values(by = 'trialnum')
        
        epoched.metadata = bdata #attach behavioural data to the eeg data
        
        # epoched.plot(picks='eeg', scalings = dict(eeg = 50e-6), n_epochs = 4, n_channels = 61)
        
        #run ica on the epoched data
        
        ica = mne.preprocessing.ICA(n_components = 0.95, method = 'infomax').fit(epoched, picks = 'eeg', reject_by_annotation=True)
        eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name = ['HEOG', 'VEOG'])
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
        ica.plot_scores(eog_scores, eog_inds)
                
        ica.plot_components(inst=epoched, contours = 0)
        print('\n\n- - - - - - - subject %d, %d components to remove: - - - - - - -\n\n'%(i, len(eog_inds)))
        
        comps2rem = input('components  remove: ') #need to separate component numbers by comma and space
        comps2rem = list(map(int, comps2rem.split(', ')))
        np.savetxt(fname = op.join(param['path'], 'removed_comps', 's%02d_removedcomps_arraylocked.txt'%i),
                   X = comps2rem, fmt = '%i') #record what components were removed
        ica.exclude.extend(comps2rem) #mark components for removal
        ica.apply(inst=epoched)
        
        epoched.save(fname = op.join(param['path'], 'eeg', param['substr'], f'{param["substr"]}_arraylocked_icacleaned-epo.fif'),
                     fmt = 'double', overwrite = True)
        
        del(raw)
        del(epoched)
        del(ica)
        del(eog_epochs)
            
            
            
            