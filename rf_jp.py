#!/usr/bin/env python

import numpy as np
import pickle

import matlab_wrapper
#Here goes the Matlab root 
matlab = matlab_wrapper.MatlabSession(matlab_root = '/n/sw/centos6/matlab-R2013a')
 
def rf_jp(X_out):
        
    matlab.put('data',X_out)
    matlab.eval('RF_outlierness')
    joint = matlab.get('joint')
    

    [delta_out, delta_norm, thresh] = pickle.load(open('RF_JP.pk', 'rb'))

    scores = 1 - joint

    y_scores = scores

    scores = np.zeros([len(y_scores)])
    
    for i, y_i in enumerate(y_scores):
        if y_i > thresh:
            scores[i] = 1.0/(1.0+np.exp(-1.0/delta_out*(y_i-thresh)))
        else:
            scores[i] = 1.0/(1.0+np.exp(-1.0/delta_norm*(y_i-thresh)))

    
    #scores = 1.0/(1.0+np.exp(-1.0/delta*(scores-thresh)))

    return scores  
     


